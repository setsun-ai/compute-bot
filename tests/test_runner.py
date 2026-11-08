"""The runner with real processes: jobs are short shell scripts."""
import sys
import time

from compute_bot import jobs
from compute_bot.runner import Runner


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
