"""The bot as its owner uses it: files and commands go in, what Telegram would show comes out."""
import io
import json
import zipfile

import pytest

from compute_bot import jobs
from compute_bot.bot import Bot

OWNER = 100


class FakeApi:
    def __init__(self):
        self.calls, self.downloads = [], {}

    def __call__(self, method, fields=None, files=None, timeout=60):
        self.calls.append((method, fields or {}, files or {}))
        return {"message_id": len(self.calls)}

    def download(self, file_id):
        return self.downloads[file_id]

    def sent(self, method):
        return [c for c in self.calls if c[0] == method]

    def texts(self):
        return [c[1].get("text") or c[1].get("caption") or "" for c in self.calls]

    def last_text(self):
        return self.texts()[-1]


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    api, clock = FakeApi(), Clock()
    store = jobs.Store(tmp_path)
    bot = Bot(store, api, OWNER, tmp_path / "bot.json", clock=clock, system_info=lambda: "CPU 50 °C")
    return bot, api, store, clock


def message(text=None, document=None, caption=None, sender=OWNER):
    msg = {"message_id": 7, "chat": {"id": sender, "type": "private"}, "from": {"id": sender}}
    if text is not None:
        msg["text"] = text
    if document is not None:
        msg["document"] = document
    if caption is not None:
        msg["caption"] = caption
    return {"update_id": 1, "message": msg}


def upload(bot, api, name, data, caption=None, sender=OWNER):
    api.downloads[name] = data
    bot.handle_update(message(document={"file_id": name, "file_name": name, "file_size": len(data)},
                              caption=caption, sender=sender))


def tap(bot, data):
    bot.handle_update({"update_id": 2, "callback_query": {"id": "q", "from": {"id": OWNER}, "data": data,
                                                          "message": {"message_id": 9, "chat": {"id": OWNER}}}})


def zip_bytes(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return out.getvalue()


# --- making jobs -----------------------------------------------------------------------------------

def test_zip_with_run_script_is_queued(env):
    bot, api, store, _ = env
    upload(bot, api, "1E0L.zip", zip_bytes({"1E0L/run.sh": "md_step md", "1E0L/md.tpr": "x"}))
    (spec,) = store.specs()
    assert (spec.kind, spec.entry) == ("script", "run.sh")
    assert "#1" in api.last_text() and "kolejce" in api.last_text()


def test_strangers_are_ignored(env):
    bot, api, store, _ = env
    upload(bot, api, "a.zip", zip_bytes({"run.sh": "rm -rf ~"}), sender=666)
    bot.handle_update(message("/jobs", sender=666))
    assert store.specs() == [] and api.calls == []


def test_table_without_a_word_asks_and_a_tap_makes_the_job(env):
    bot, api, store, _ = env
    upload(bot, api, "dane.xlsx", b"PK-xlsx", caption="max_desc=3")
    assert store.specs() == []
    buttons = json.loads(api.sent("sendMessage")[-1][1]["reply_markup"])["inline_keyboard"]
    choices = [b["callback_data"] for row in buttons for b in row]
    assert any(c.endswith(":qsar") for c in choices) and any(c.endswith(":omics") for c in choices)
    tap(bot, next(c for c in choices if c.endswith(":qsar")))
    (spec,) = store.specs()
    assert (spec.kind, spec.entry, spec.params["max_desc"]) == ("recipe", "qsar", 3)
    assert (store.work(1) / "dane.xlsx").read_bytes() == b"PK-xlsx"


def test_bad_parameter_is_explained_and_nothing_is_queued(env):
    bot, api, store, _ = env
    upload(bot, api, "dane.xlsx", b"x", caption="qsar max_desc=lots")
    assert store.specs() == []
    assert "max_desc" in api.last_text()


def test_md_from_a_pdb_code(env):
    bot, api, store, _ = env
    bot.handle_update(message("/md 1E0L T=600 ns=0.5"))
    (spec,) = store.specs()
    assert spec.entry == "md" and spec.params["pdb"] == "1E0L" and spec.params["T"] == 600.0


def test_recipe_that_needs_a_file_says_so(env):
    bot, api, store, _ = env
    bot.handle_update(message("/qsar"))
    assert store.specs() == [] and "plik" in api.last_text()


def test_too_big_file_is_explained(env):
    bot, api, store, _ = env
    bot.handle_update(message(document={"file_id": "x", "file_name": "big.zip", "file_size": 30 * 1024 ** 2}))
    assert store.specs() == [] and "20 MB" in api.last_text()


# --- watching jobs ------------------------------------------------------------------------------------

def finished_job(store, files, state="done", code=0):
    job = store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"}).id
    for rel, data in files.items():
        path = store.work(job) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    store.log_path(job).write_text("line 1\nERROR: kolumna nie istnieje\n")
    store.set_status(job, jobs.Status(state, attempt=1, started=1.0, finished=61.0, exit_code=code))
    return job


def test_start_and_finish_are_announced_once_with_results(env):
    bot, api, store, _ = env
    job = store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"}).id
    store.set_status(job, jobs.Status("running", attempt=1, started=1.0))
    bot.tick()
    assert "▶️" in api.last_text()
    png = b"\x89PNG fake"
    for rel, data in {"summary.txt": "R² = 0.91".encode(), "wyniki/plot.png": png,
                      "highlights.txt": b"wyniki/plot.png\n"}.items():
        (store.work(job) / rel).parent.mkdir(parents=True, exist_ok=True)
        (store.work(job) / rel).write_bytes(data)
    store.set_status(job, jobs.Status("done", attempt=1, started=1.0, finished=3601.0, exit_code=0))
    bot.tick()
    assert any("R² = 0.91" in t for t in api.texts())
    assert api.sent("sendPhoto")[0][2]["photo"][1] == png
    archive = api.sent("sendDocument")[-1][2]["document"]
    names = zipfile.ZipFile(io.BytesIO(archive[1])).namelist()
    assert archive[0] == "wyniki_0001.zip" and "wyniki/plot.png" in names and "log.txt" in names
    count = len(api.calls)
    bot.tick()
    assert len(api.calls) == count  # nothing twice


def test_failure_shows_the_end_of_the_log(env):
    bot, api, store, _ = env
    finished_job(store, {}, state="failed", code=1)
    bot.tick()
    assert any("kolumna nie istnieje" in t for t in api.texts())


def test_resume_after_restart_and_heat_pause_are_announced(env):
    bot, api, store, _ = env
    job = store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"}).id
    store.set_status(job, jobs.Status("running", attempt=1))
    bot.tick()
    store.set_status(job, jobs.Status("paused", attempt=1))
    bot.tick()
    assert "gorąco" in api.last_text()
    store.set_status(job, jobs.Status("running", attempt=2))
    bot.tick()
    assert "wznowione" in api.last_text()


def test_progress_report_every_few_hours(env):
    bot, api, store, clock = env
    job = store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"}).id
    store.set_status(job, jobs.Status("running", attempt=1, started=clock.now, progress="NPT 40%"))
    bot.tick()
    count = len(api.calls)
    clock.now += 3600
    bot.tick()
    assert len(api.calls) == count
    clock.now += 3 * 3600
    bot.tick()
    assert "NPT 40%" in api.last_text()


def test_big_files_stay_behind_and_get_fetches_one(env):
    bot, api, store, _ = env
    job = finished_job(store, {"small.txt": b"ok", "traj.xtc": b"x" * 2_000_000})
    bot.zip_budget = 1_000_000
    bot.tick()
    names = zipfile.ZipFile(io.BytesIO(api.sent("sendDocument")[-1][2]["document"][1])).namelist()
    assert "small.txt" in names and "traj.xtc" not in names
    assert any("traj.xtc" in t and "/get" in t for t in api.texts())
    bot.handle_update(message(f"/get {job} traj.xtc"))
    assert api.sent("sendDocument")[-1][2]["document"][0] == "traj.xtc"


# --- commands -------------------------------------------------------------------------------------------

def test_jobs_list_and_cancel(env):
    bot, api, store, _ = env
    upload(bot, api, "a.zip", zip_bytes({"run.sh": "sleep 100"}))
    bot.handle_update(message("/jobs"))
    assert "#1" in api.last_text() and "a.zip" in api.last_text()
    bot.handle_update(message("/cancel 1"))
    assert store.cancel_requested(1)


def test_retry_makes_a_new_job_from_the_original_files(env):
    bot, api, store, _ = env
    upload(bot, api, "a.zip", zip_bytes({"run.sh": "exit 1"}))
    (store.work(1) / "junk.txt").write_text("from the failed run")
    store.set_status(1, jobs.Status("failed", attempt=1, exit_code=1))
    bot.handle_update(message("/retry 1"))
    assert [s.id for s in store.specs()] == [1, 2]
    assert (store.work(2) / "run.sh").exists() and not (store.work(2) / "junk.txt").exists()


def test_help_for_a_recipe_lists_its_parameters(env):
    bot, api, store, _ = env
    bot.handle_update(message("/help qsar"))
    assert "max_desc=" in api.last_text()


def test_system_info(env):
    bot, api, store, _ = env
    bot.handle_update(message("/system"))
    assert "CPU 50 °C" in api.last_text()


def test_report_interval_is_kept(env, tmp_path):
    bot, api, store, clock = env
    bot.handle_update(message("/co 1"))
    again = Bot(store, api, OWNER, tmp_path / "bot.json", clock=clock, system_info=lambda: "")
    assert again.state["report_hours"] == 1.0
