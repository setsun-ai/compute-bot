"""The runner with real processes: jobs are short shell scripts."""
import sys
import time

from compute_bot import jobs
from compute_bot.runner import NIGHT_PERF, CpuLimit, Runner


class Thermometer:
    def __init__(self, temp=50.0):
        self.temp = temp

    def __call__(self):
        return self.temp


def make(tmp_path, temp=None):
    store = jobs.Store(tmp_path)
    runner = Runner(store, python=sys.executable, threads=3, read_temp=temp or Thermometer())
    return store, runner


def script(store, body):
    return store.create("script", "run.sh", "x.zip", {}, {"run.sh": body.encode()}).id


def run_until(runner, store, job, states, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        runner.tick()
        if store.status(job).state in states:
            return store.status(job)
        time.sleep(0.05)
    raise AssertionError(f"job {job} is {store.status(job)}")


def test_job_runs_to_done_with_its_output_in_the_log(tmp_path):
    store, runner = make(tmp_path)
    job = script(store, "echo hello; echo oops >&2; echo PROGRESS: half; echo 42 > out.txt")
    status = run_until(runner, store, job, jobs.FINAL)
    assert status.state == "done" and status.exit_code == 0 and status.attempt == 1
    assert status.started <= status.finished
    assert status.progress == "half"
    assert "hello" in store.log_tail(job) and "oops" in store.log_tail(job)
    assert (store.work(job) / "out.txt").read_text() == "42\n"


def test_failing_job(tmp_path):
    store, runner = make(tmp_path)
    job = script(store, "exit 3")
    status = run_until(runner, store, job, jobs.FINAL)
    assert (status.state, status.exit_code) == ("failed", 3)


def test_jobs_run_one_at_a_time_in_order(tmp_path):
    store, runner = make(tmp_path)
    first = script(store, "sleep 0.3")
    second = script(store, "true")
    run_until(runner, store, first, {"running"})
    runner.tick()
    assert store.status(second).state == "queued"
    run_until(runner, store, second, jobs.FINAL)
    assert store.status(first).finished <= store.status(second).started


def test_cancel_kills_the_whole_job(tmp_path):
    store, runner = make(tmp_path)
    job = script(store, "sleep 30 & wait")
    run_until(runner, store, job, {"running"})
    store.request_cancel(job)
    status = run_until(runner, store, job, jobs.FINAL, timeout=15)
    assert status.state == "cancelled"
    assert runner.proc is None


def test_cancel_before_start(tmp_path):
    store, runner = make(tmp_path)
    job = script(store, "touch ran")
    store.request_cancel(job)
    runner.tick()
    assert store.status(job).state == "cancelled"
    assert not (store.work(job) / "ran").exists()


def test_a_job_interrupted_by_a_restart_runs_again(tmp_path):
    store, runner = make(tmp_path)
    job = script(store, "echo run >> runs.txt")
    store.set_status(job, jobs.Status("running", attempt=1, started=1.0))
    runner.recover()
    status = run_until(runner, store, job, jobs.FINAL)
    assert status.state == "done" and status.attempt == 2


def test_too_hot_stops_the_job_and_cool_lets_it_go_on(tmp_path):
    temp = Thermometer(95.0)
    store, runner = make(tmp_path, temp)
    job = script(store, "sleep 0.5; echo done > out.txt")
    run_until(runner, store, job, {"paused"})
    time.sleep(0.8)  # stopped: the sleep doesn't end
    runner.tick()
    assert store.status(job).state == "paused"
    temp.temp = 60.0
    status = run_until(runner, store, job, jobs.FINAL)
    assert status.state == "done"


def test_the_job_sees_its_threads_helpers_and_python(tmp_path):
    store, runner = make(tmp_path)
    job = script(store, 'echo "$COMPUTE_THREADS $OMP_NUM_THREADS" > env.txt; command -v md_step > which.txt')
    run_until(runner, store, job, jobs.FINAL)
    assert (store.work(job) / "env.txt").read_text() == "3 3\n"
    assert (store.work(job) / "which.txt").read_text().strip().endswith("compute_bot/bin/md_step")


def test_python_jobs_use_the_environment_python_and_see_the_app(tmp_path):
    store, runner = make(tmp_path)
    code = b"import sys, compute_bot.jobs; open('py.txt', 'w').write(sys.executable)"
    job = store.create("python", "a.py", "a.py", {}, {"a.py": code}).id
    status = run_until(runner, store, job, jobs.FINAL)
    assert status.state == "done", store.log_tail(job)
    assert (store.work(job) / "py.txt").read_text() == sys.executable


def test_a_job_never_sees_the_bot_token(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_TOKEN", "123:secret")
    store, runner = make(tmp_path)
    job = script(store, 'echo "token=${TG_TOKEN:-none}" > env.txt')
    run_until(runner, store, job, jobs.FINAL)
    assert (store.work(job) / "env.txt").read_text() == "token=none\n"


def test_hold_stops_the_job_and_starts_nothing_until_released(tmp_path):
    store, runner = make(tmp_path)
    first = script(store, "sleep 0.4; echo ok > out.txt")
    second = script(store, "true")
    run_until(runner, store, first, {"running"})
    store.set_hold(True)
    status = run_until(runner, store, first, {"paused"})
    assert status.pause == "hold"
    time.sleep(0.6)
    runner.tick()
    assert store.status(second).state == "queued"
    store.set_hold(False)
    run_until(runner, store, second, jobs.FINAL)
    assert (store.work(first) / "out.txt").exists()


def test_no_new_job_without_disk_space_and_the_runner_says_why(tmp_path):
    store, runner = make(tmp_path)
    runner.free_space = lambda: 500 * 1024 ** 2
    job = script(store, "true")
    runner.tick()
    assert store.status(job).state == "queued"
    assert store.runner_state()["blocked"] == "disk"
    runner.free_space = lambda: 50 * 1024 ** 3
    run_until(runner, store, job, jobs.FINAL)
    assert store.runner_state()["blocked"] == ""


def test_runner_leaves_a_heartbeat(tmp_path):
    store, runner = make(tmp_path)
    runner.tick()
    state = store.runner_state()
    assert state["time"] > 0 and state["job"] is None


def test_fetch_downloads_then_runs_and_skips_what_is_already_there(tmp_path):
    import http.server
    import threading

    payload = b"col\n1\n" * 1000
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/data.csv"
    store, runner = make(tmp_path)
    body = b"import os; open('seen.txt', 'w').write(str(os.path.getsize('data.csv')))"
    job = store.create("python", "a.py", "a.py", {}, {"a.py": body}, fetch=[{"url": url, "name": "data.csv"}]).id
    status = run_until(runner, store, job, jobs.FINAL)
    assert status.state == "done", store.log_tail(job)
    assert (store.work(job) / "seen.txt").read_text() == str(len(payload))
    store.set_status(job, jobs.Status("queued", attempt=1))
    run_until(runner, store, job, jobs.FINAL)
    assert len(hits) == 1  # the second run found the file complete
    server.shutdown()


# --- quiet nights -------------------------------------------------------------------------------------

def night_runner(tmp_path, temp=None, pstate=True):
    """A runner whose whole day is night, with a fake intel_pstate folder it may write to."""
    store = jobs.Store(tmp_path / "home")
    folder = tmp_path / "pstate"
    if pstate:
        folder.mkdir()
        (folder / "no_turbo").write_text("0\n")
        (folder / "max_perf_pct").write_text("100\n")
    runner = Runner(store, python=sys.executable, threads=3, read_temp=temp or Thermometer(),
                    cpu=CpuLimit(folder))
    store.set_night({"hours": None, "until": time.time() + 3600})
    return store, runner, folder


def test_at_night_a_running_job_gets_a_slower_cpu_and_the_day_gets_it_back(tmp_path):
    store, runner, pstate = night_runner(tmp_path)
    job = script(store, "sleep 0.4")
    run_until(runner, store, job, {"running"})
    runner.tick()
    assert (pstate / "no_turbo").read_text().strip() == "1"
    assert int((pstate / "max_perf_pct").read_text()) == NIGHT_PERF
    assert store.runner_state()["night"] is True
    run_until(runner, store, job, jobs.FINAL)
    runner.tick()
    assert (pstate / "no_turbo").read_text().strip() == "0"
    assert (pstate / "max_perf_pct").read_text().strip() == "100"


def test_morning_lifts_the_limit_while_the_job_still_runs(tmp_path):
    store, runner, pstate = night_runner(tmp_path)
    job = script(store, "sleep 1")
    run_until(runner, store, job, {"running"})
    runner.tick()
    store.set_night(jobs.NO_NIGHT)
    runner.tick()
    assert (pstate / "max_perf_pct").read_text().strip() == "100"
    assert store.runner_state()["night"] is False


def test_at_night_a_lower_temperature_pauses_the_job_quietly(tmp_path):
    temp = Thermometer(jobs.NIGHT_HOT + 1)
    store, runner, _ = night_runner(tmp_path, temp)
    job = script(store, "sleep 0.3")
    status = run_until(runner, store, job, {"paused"})
    assert status.pause == "night"
    temp.temp = jobs.NIGHT_COOL - 1
    assert run_until(runner, store, job, jobs.FINAL).state == "done"


def test_the_same_temperature_by_day_does_not_pause(tmp_path):
    store, runner = make(tmp_path, Thermometer(jobs.NIGHT_HOT + 1))
    job = script(store, "sleep 0.2")
    run_until(runner, store, job, {"running"})
    time.sleep(0.1)
    runner.tick()
    assert store.status(job).state != "paused"


def test_without_access_to_the_cpu_setting_the_runner_says_so_and_still_runs(tmp_path):
    store, runner, _ = night_runner(tmp_path, pstate=False)
    job = script(store, "true")
    assert run_until(runner, store, job, jobs.FINAL).state == "done"
    runner.heartbeat(force=True)
    assert store.runner_state()["cpu_limit"] is False


# --- after a reboot ---------------------------------------------------------------------------------------

def test_a_new_boot_is_reported_with_the_last_sign_of_life_and_the_jobs_put_back(tmp_path):
    store = jobs.Store(tmp_path)
    store.set_runner_state({"time": 1000.0, "boot": "old-boot", "job": 1})
    job = script(store, "true")
    store.set_status(job, jobs.Status("running", attempt=1, started=900.0))
    runner = Runner(store, python=sys.executable, threads=3, read_temp=Thermometer(), boot_id=lambda: "new-boot")
    runner.recover()
    runner.heartbeat(force=True)
    state = store.runner_state()
    assert state["boot"] == "new-boot"
    assert state["restart"]["after"] == 1000.0 and state["restart"]["jobs"] == [job]


def test_a_runner_restart_without_a_reboot_is_not_a_reboot(tmp_path):
    store = jobs.Store(tmp_path)
    store.set_runner_state({"time": 1000.0, "boot": "same"})
    runner = Runner(store, python=sys.executable, threads=3, read_temp=Thermometer(), boot_id=lambda: "same")
    runner.recover()
    runner.heartbeat(force=True)
    assert store.runner_state()["restart"] is None


def test_a_script_that_hides_mdrun_still_shows_how_far_the_run_is(tmp_path):
    store, runner = make(tmp_path)
    start = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(time.time() - 100))
    log = ("   integrator                     = md\n   nsteps                         = 25000\n"
           f"Started mdrun on rank 0 {start}\n\n           Step           Time\n              0        0.00000\n\n"
           "           Step           Time\n           5000       10.00000\n")
    job = script(store, f"mkdir -p sys/a; echo PROGRESS: przebieg 2/30; cat > sys/a/nvt.log << 'X'\n{log}X\nsleep 1")
    run_until(runner, store, job, {"running"})
    time.sleep(0.3)
    runner.last_progress = 0
    runner.tick()
    progress = store.status(job).progress
    assert progress.startswith("przebieg 2/30 · ")
    assert "sys/a/nvt" in progress and "20%" in progress and "etap jeszcze ~" in progress


def test_the_runner_keeps_the_peak_temperature_and_the_time_spent_paused(tmp_path):
    temp = Thermometer(95.0)
    store, runner = make(tmp_path, temp)
    job = script(store, "sleep 0.3")
    run_until(runner, store, job, {"paused"})
    time.sleep(0.5)
    runner.tick()
    temp.temp = 60.0
    status = run_until(runner, store, job, jobs.FINAL)
    assert status.temp_max == 95.0 and status.paused_s >= 0.4
