"""
The runner: one job at a time, the only writer of status.json. Standard library only.

  python3 -m compute_bot.runner     (COMPUTE_HOME, COMPUTE_PYTHON, COMPUTE_THREADS)

A job runs in its own process group with nice 19 inherited from the service, so the bots and game servers on the
laptop always come first. Too hot → the group gets SIGSTOP until the CPU cools down. A job that was running when
the runner stopped (reboot, update) is started again; md_step resumes GROMACS from its checkpoint.
"""
from __future__ import annotations

import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import jobs, updater

APP = Path(__file__).resolve().parent.parent
BIN = Path(__file__).resolve().parent / "bin"
PROGRESS_EVERY = 20  # seconds between progress updates in status.json
KILL_GRACE = 10  # seconds between SIGTERM and SIGKILL on cancel


def read_cpu_temp():
    """Hottest CPU sensor in °C, or None when the machine has none we can read."""
    readings = []
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = Path(hwmon, "name").read_text().strip()
        except OSError:
            continue
        if name in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
            readings += glob.glob(f"{hwmon}/temp*_input")
    if not readings:
        readings = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    temps = []
    for path in readings:
        try:
            temps.append(int(Path(path).read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def log(message):
    print(message, flush=True)


class Runner:
    def __init__(self, store, python, threads, read_temp=read_cpu_temp, clock=time.time):
        self.store, self.python, self.threads = store, python, threads
        self.read_temp, self.clock = read_temp, clock
        self.job = self.proc = self.log_file = None
        self.kill_at = None
        self.last_progress = 0.0

    # --- lifecycle -------------------------------------------------------------------------------

    def recover(self):
        for job, status in self.store.statuses().items():
            again = jobs.recover(status)
            if again != status:
                log(f"job {job}: {status.state} at start → {again.state}")
                self.store.set_status(job, again)

    def tick(self):
        if self.proc is None:
            self.start_next()
        else:
            self.watch()

    @property
    def idle(self):
        return self.proc is None

    # --- starting --------------------------------------------------------------------------------

    def environment(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TG_")}  # a job never sees the token
        python_bin = str(Path(self.python).parent)
        env["PATH"] = os.pathsep.join([str(BIN), python_bin, env.get("PATH", "/usr/bin:/bin")])
        env["PYTHONPATH"] = str(APP)
        threads = str(self.threads)
        env.update(COMPUTE_THREADS=threads, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                   MKL_NUM_THREADS=threads, MPLBACKEND="Agg", GMX_MAXBACKUP="-1", PYTHONUNBUFFERED="1")
        prefix = Path(self.python).parent.parent
        if (prefix / "conda-meta").exists():
            env["CONDA_PREFIX"] = str(prefix)
        return env

    def start_next(self):
        statuses = self.store.statuses()
        for job, status in statuses.items():
            if status.state == "queued" and self.store.cancel_requested(job):
                self.store.set_status(job, status._replace(state="cancelled", finished=self.clock()))
                statuses[job] = self.store.status(job)
        job = jobs.next_job(statuses)
        if job is None:
            return
        spec = self.store.spec(job)
        status = statuses[job]
        cmd = jobs.command(spec, self.python)
        self.log_file = open(self.store.log_path(job), "ab")
        self.log_file.write(f"\n=== próba {status.attempt + 1}: {' '.join(cmd)}\n".encode())
        self.log_file.flush()
        try:
            self.proc = subprocess.Popen(cmd, cwd=self.store.work(job), stdout=self.log_file,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=self.environment(), start_new_session=True)
        except OSError as e:
            self.log_file.write(f"Nie mogę uruchomić: {e}\n".encode())
            self.log_file.close()
            self.store.set_status(job, status._replace(state="failed", attempt=status.attempt + 1,
                                                       finished=self.clock(), progress=str(e)))
            return
        self.job, self.kill_at = job, None
        self.store.set_status(job, status._replace(state="running", attempt=status.attempt + 1,
                                                   started=status.started or self.clock()))
        log(f"job {job} started: {cmd}")

    # --- while it runs -----------------------------------------------------------------------------

    def signal(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def watch(self):
        job, status = self.job, self.store.status(self.job)
        code = self.proc.poll()
        if code is not None:
            return self.finish(code)
        now = self.clock()
        if self.store.cancel_requested(job):
            if self.kill_at is None:
                self.signal(signal.SIGCONT)
                self.signal(signal.SIGTERM)
                self.kill_at = now + KILL_GRACE
            elif now >= self.kill_at:
                self.signal(signal.SIGKILL)
            return
        action = jobs.thermal_action(self.read_temp(), paused=status.state == "paused")
        if action == "stop":
            self.signal(signal.SIGSTOP)
            self.store.set_status(job, status._replace(state="paused"))
            log(f"job {job} paused: too hot")
            return
        if action == "cont":
            self.signal(signal.SIGCONT)
            status = status._replace(state="running")
            self.store.set_status(job, status)
        if now - self.last_progress >= PROGRESS_EVERY:
            self.last_progress = now
            progress = jobs.progress_from_log(self.store.log_tail(job, 20000))
            if progress and progress != status.progress:
                self.store.set_status(job, status._replace(progress=progress))

    def finish(self, code):
        job, status = self.job, self.store.status(self.job)
        self.log_file.close()
        if self.store.cancel_requested(job):
            state = "cancelled"
        else:
            state = "done" if code == 0 else "failed"
        progress = jobs.progress_from_log(self.store.log_tail(job, 20000)) or status.progress
        self.store.set_status(job, status._replace(state=state, exit_code=code, finished=self.clock(),
                                                   progress=progress))
        log(f"job {job} {state} ({code})")
        self.job = self.proc = self.log_file = None
        self.last_progress = 0.0




def main():
    home = Path(os.environ.get("COMPUTE_HOME", "."))
    python = os.environ.get("COMPUTE_PYTHON", sys.executable)
    threads = int(os.environ.get("COMPUTE_THREADS") or os.cpu_count() or 1)
    runner = Runner(jobs.Store(home), python, threads)
    runner.recover()
    log(f"runner started: {home}, {threads} threads, {python}")
    checked = 0.0
    while True:
        runner.tick()
        if runner.idle and time.time() - checked > 60:
            checked = time.time()
            if updater.restart_wanted(home):
                log("newer version installed, restarting")
                return 0
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
