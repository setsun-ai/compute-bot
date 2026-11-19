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
import shutil
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
MIN_FREE = 2 * 1024 ** 3  # a new job starts only with this much free disk
HEARTBEAT_EVERY = 30  # seconds between writes of runner.json
PSTATE = Path("/sys/devices/system/cpu/intel_pstate")  # made writable for the group by deploy/compute-bot-cpu.conf
NIGHT_PERF = 50  # % of the top clock at night, turbo off: about a third of the heat, so the fans stay slow


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


def read_fans():
    """Fan speeds in rpm (dell_smm and the like); empty where the machine doesn't show them."""
    speeds = []
    for path in sorted(glob.glob("/sys/class/hwmon/hwmon*/fan*_input")):
        try:
            speeds.append(int(Path(path).read_text()))
        except (OSError, ValueError):
            pass
    return speeds


def read_boot_id():
    """Changes with every boot of the machine: tells a reboot (a crash, too) from a restart of the service."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


class CpuLimit:
    """A slower, cooler CPU at night through intel_pstate; does nothing where the files can't be written."""

    def __init__(self, folder=PSTATE):
        self.folder = Path(folder)
        self.limited = None  # unknown at start: the first set() writes, so a limit left by a crash goes away

    @property
    def usable(self):
        return all(os.access(self.folder / name, os.W_OK) for name in ("no_turbo", "max_perf_pct"))

    def set(self, limited):
        if limited == self.limited or not self.usable:
            return
        values = {"no_turbo": 1, "max_perf_pct": NIGHT_PERF} if limited else {"no_turbo": 0, "max_perf_pct": 100}
        try:
            for name, value in values.items():
                (self.folder / name).write_text(f"{value}\n")
        except OSError as e:
            return log(f"cpu limit: {e}")
        self.limited = limited
        log(f"cpu limit {'on' if limited else 'off'}")


def log(message):
    print(message, flush=True)


class Runner:
    def __init__(self, store, python, threads, read_temp=read_cpu_temp, clock=time.time, cpu=None,
                 boot_id=read_boot_id):
        self.store, self.python, self.threads = store, python, threads
        self.read_temp, self.clock = read_temp, clock
        self.cpu = cpu or CpuLimit()
        self.boot = boot_id()
        self.previous = store.runner_state()  # the last heartbeat before this start
        self.restart = None  # {"after": last heartbeat, "at": now, "jobs": [put back]} when the machine rebooted
        self.night = False
        self.job = self.proc = self.log_file = None
        self.kill_at = None
        self.last_progress = self.last_beat = 0.0
        self.blocked = ""
        self.free_space = lambda: shutil.disk_usage(store.root).free

    # --- lifecycle -------------------------------------------------------------------------------

    def recover(self):
        put_back = []
        for job, status in self.store.statuses().items():
            again = jobs.recover(status)
            if again != status:
                log(f"job {job}: {status.state} at start → {again.state}")
                self.store.set_status(job, again)
                put_back.append(job)
        old_boot = self.previous.get("boot")
        if old_boot and self.boot and old_boot != self.boot:
            self.restart = {"after": self.previous.get("time", 0), "at": self.clock(), "jobs": put_back}
            log(f"the machine rebooted; last heartbeat before: {self.restart['after']}")

    def tick(self):
        now = self.clock()
        night = jobs.night_now(self.store.night(), now, time.localtime(now).tm_hour)
        changed, self.night = night != self.night, night
        if self.proc is None:
            self.start_next()
        else:
            self.watch()
        self.cpu.set(self.night and self.proc is not None)
        self.heartbeat(force=changed)

    def heartbeat(self, force=False):
        now = self.clock()
        if force or now - self.last_beat >= HEARTBEAT_EVERY:
            self.last_beat = now
            self.store.set_runner_state({"time": now, "job": self.job, "blocked": self.blocked,
                                         "free": self.free_space(), "threads": self.threads, "boot": self.boot,
                                         "restart": self.restart, "night": self.night,
                                         "cpu_limit": self.cpu.usable})

    def set_blocked(self, reason):
        if reason != self.blocked:
            self.blocked = reason
            self.heartbeat(force=True)

    @property
    def idle(self):
        return self.proc is None

    # --- starting --------------------------------------------------------------------------------

    def environment(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TG_")}  # a job never sees the token
        python_bin = str(Path(self.python).parent)
        tools = str(self.store.root / "tools" / "bin")  # NAMD and others the owner installed (compute_bot.tools)
        env["PATH"] = os.pathsep.join([str(BIN), tools, python_bin, env.get("PATH", "/usr/bin:/bin")])
        env["COMPUTE_HOME"] = str(self.store.root)
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
        job = jobs.next_job(statuses, self.store.priorities())
        if job is None:
            return self.set_blocked("")
        if self.store.hold:
            return self.set_blocked("hold")
        if self.free_space() < MIN_FREE:
            return self.set_blocked("disk")
        self.set_blocked("")
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
        held = self.store.hold
        if held and status.state != "paused":
            self.signal(signal.SIGSTOP)
            self.store.set_status(job, status._replace(state="paused", pause="hold"))
            return log(f"job {job} paused: hold")
        if status.pause == "hold" and not held:
            self.signal(signal.SIGCONT)
            status = status._replace(state="running", pause="")
            self.store.set_status(job, status)
        if status.pause != "hold":
            hot, cool = (jobs.NIGHT_HOT, jobs.NIGHT_COOL) if self.night else (jobs.HOT, jobs.COOL)
            action = jobs.thermal_action(self.read_temp(), paused=status.state == "paused", hot=hot, cool=cool)
            if action == "stop":
                self.signal(signal.SIGSTOP)
                reason = "night" if self.night else "heat"
                self.store.set_status(job, status._replace(state="paused", pause=reason))
                return log(f"job {job} paused: {reason}")
            if action == "cont":
                self.signal(signal.SIGCONT)
                status = status._replace(state="running", pause="")
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
                                                   progress=progress, pause=""))
        log(f"job {job} {state} ({code})")
        self.job = self.proc = self.log_file = None
        self.last_progress = 0.0
        self.heartbeat(force=True)




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
