"""The bot as its owner uses it: files and commands go in, what Telegram would show comes out."""
import io
import json
import zipfile

import pytest

from compute_bot import actions, jobs
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


def buttons(api, index=-1):
    markup = api.sent("sendMessage")[index][1].get("reply_markup") or api.sent("editMessageText")[-1][1]["reply_markup"]
    return [b for row in json.loads(markup)["inline_keyboard"] for b in row]


def last_buttons(api):
    calls = [c for c in api.calls if c[0] in ("sendMessage", "editMessageText") and c[1].get("reply_markup")]
    return [b for row in json.loads(calls[-1][1]["reply_markup"])["inline_keyboard"] for b in row]


def press(bot, api, label_start):
    button = next(b for b in last_buttons(api) if b["text"].startswith(label_start))
    tap(bot, button["callback_data"])


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
    assert store.specs() == []  # a card first
    press(bot, api, "▶️")
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
    press(bot, api, "▶️")
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


# --- quality of life ------------------------------------------------------------------------------------

def test_pause_and_resume_everything(env):
    bot, api, store, _ = env
    bot.handle_update(message("/pauza"))
    assert store.hold and "/wznow" in api.last_text()
    bot.handle_update(message("/wznow"))
    assert not store.hold


def test_top_moves_a_job_up(env):
    bot, api, store, _ = env
    for name in ("a.zip", "b.zip"):
        upload(bot, api, name, zip_bytes({"run.sh": "true"}))
    bot.handle_update(message("/top 2"))
    assert store.priorities() == {2: 1}


def test_retry_with_changed_parameters(env):
    bot, api, store, _ = env
    bot.handle_update(message("/md 1E0L T=600"))
    press(bot, api, "▶️")
    store.set_status(1, jobs.Status("done", attempt=1))
    bot.handle_update(message("/retry 1 T=500 ns=2"))
    spec = store.spec(2)
    assert (spec.params["T"], spec.params["ns"], spec.params["pdb"]) == (500.0, 2.0, "1E0L")


def test_remove_needs_a_confirmation_tap(env):
    bot, api, store, _ = env
    finished_job(store, {"big.xtc": b"x" * 10})
    bot.handle_update(message("/usun 1"))
    assert store.ids() == [1]
    markup = json.loads(api.sent("sendMessage")[-1][1]["reply_markup"])
    tap(bot, markup["inline_keyboard"][0][0]["callback_data"])
    assert store.ids() == []


def test_remove_old_finished_jobs(env):
    bot, api, store, clock = env
    old = finished_job(store, {})
    store.set_status(old, store.status(old)._replace(finished=clock.now - 8 * 86400))
    recent = finished_job(store, {})
    store.set_status(recent, store.status(recent)._replace(finished=clock.now - 3600))
    running = store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"}).id
    store.set_status(running, jobs.Status("running", attempt=1))
    bot.handle_update(message("/usun stare"))
    assert f"#{old}" in api.last_text() and f"#{recent}" not in api.last_text()
    markup = json.loads(api.sent("sendMessage")[-1][1]["reply_markup"])
    tap(bot, markup["inline_keyboard"][0][0]["callback_data"])
    assert store.ids() == [recent, running]


def test_link_makes_a_job_that_fetches_its_file(env):
    bot, api, store, _ = env
    bot.handle_update(message("/url https://drive.google.com/file/d/ABC/view omics group=grupa"))
    press(bot, api, "▶️")
    (spec,) = store.specs()
    assert (spec.kind, spec.entry, spec.params["group"]) == ("recipe", "omics", "grupa")
    assert spec.fetch == [{"url": "https://drive.google.com/file/d/ABC/view", "name": "pobrany"}]


def test_quiet_hours_send_without_sound(env):
    bot, api, store, _ = env
    bot.local_hour = lambda: 2
    bot.handle_update(message("/jobs"))
    assert api.sent("sendMessage")[-1][1]["disable_notification"] is True
    bot.local_hour = lambda: 14
    bot.handle_update(message("/jobs"))
    assert "disable_notification" not in api.sent("sendMessage")[-1][1]
    bot.handle_update(message("/cisza off"))
    bot.local_hour = lambda: 2
    bot.handle_update(message("/jobs"))
    assert "disable_notification" not in api.sent("sendMessage")[-1][1]


def test_silent_runner_with_work_waiting_is_reported_once_and_its_return_too(env):
    bot, api, store, clock = env
    store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"})
    store.set_runner_state({"time": clock.now - 600, "job": None, "blocked": ""})
    bot.tick()
    bot.tick()
    assert sum("runner" in t.lower() and "nie" in t for t in api.texts()) == 1
    store.set_runner_state({"time": clock.now, "job": None, "blocked": ""})
    bot.tick()
    assert "znowu" in api.last_text()


def test_full_disk_is_reported_once(env):
    bot, api, store, clock = env
    store.create("script", "run.sh", "x.zip", {}, {"run.sh": b"true"})
    store.set_runner_state({"time": clock.now, "job": None, "blocked": "disk", "free": 10 ** 9})
    bot.tick()
    bot.tick()
    assert sum("dysk" in t for t in api.texts()) == 1


def test_results_come_with_buttons(env):
    bot, api, store, _ = env
    finished_job(store, {"summary.txt": b"ok"})
    bot.tick()
    summary_msg = next(c for c in api.sent("sendMessage") if "ok" in c[1]["text"])
    data = [b["callback_data"] for row in json.loads(summary_msg[1]["reply_markup"])["inline_keyboard"] for b in row]
    assert "j:1:retry" in data and "j:1:usun" in data


# --- the parameter card --------------------------------------------------------------------------------

def test_a_pdb_opens_a_card_and_nothing_runs_before_start(env):
    bot, api, store, _ = env
    upload(bot, api, "1e0l.pdb", b"ATOM", caption="T=600")
    assert store.specs() == []
    text = api.last_text()
    assert "T" in text and "600" in text and "▶️" in "".join(b["text"] for b in last_buttons(api))


def test_tapping_a_parameter_and_a_value_changes_it(env):
    bot, api, store, _ = env
    upload(bot, api, "1e0l.pdb", b"ATOM")
    press(bot, api, "ns")
    press(bot, api, "10")
    assert "10" in api.calls[-1][1]["text"]
    press(bot, api, "engine")
    press(bot, api, "openmm")
    press(bot, api, "▶️")
    spec = store.spec(1)
    assert (spec.params["ns"], spec.params["engine"]) == (10.0, "openmm")
    assert (store.work(1) / "1e0l.pdb").read_bytes() == b"ATOM"


def test_typing_key_value_edits_the_open_card(env):
    bot, api, store, _ = env
    bot.handle_update(message("/md 1E0L"))
    bot.handle_update(message("T=450 ff=charmm27"))
    assert "450" in api.last_text() and "charmm27" in api.last_text()
    bot.handle_update(message("T=hot"))
    assert "T=hot" in api.last_text()
    press(bot, api, "▶️")
    assert (store.spec(1).params["T"], store.spec(1).params["ff"]) == (450.0, "charmm27")


def test_cancel_drops_the_card_and_its_files(env, tmp_path):
    bot, api, store, _ = env
    upload(bot, api, "1e0l.pdb", b"ATOM")
    press(bot, api, "✖️")
    assert store.specs() == [] and not any((tmp_path / "staging").iterdir())


def test_scripts_skip_the_card(env):
    bot, api, store, _ = env
    upload(bot, api, "a.zip", zip_bytes({"run.sh": "true"}))
    assert len(store.specs()) == 1


def test_continue_a_finished_md_run(env):
    bot, api, store, _ = env
    bot.handle_update(message("/md 1E0L ns=1"))
    press(bot, api, "▶️")
    store.set_status(1, jobs.Status("done", attempt=1))
    bot.handle_update(message("/dolicz 1 5"))
    spec = store.spec(2)
    assert spec.params["extend_ns"] == 5.0 and spec.params["pdb"] == "1E0L"


def test_continue_needs_a_finished_md_job(env):
    bot, api, store, _ = env
    upload(bot, api, "a.zip", zip_bytes({"run.sh": "true"}))
    bot.handle_update(message("/dolicz 1 5"))
    assert "md" in api.last_text() and len(store.specs()) == 1


def test_namd_tarball_installs_without_questions(env):
    bot, api, store, _ = env
    name = "NAMD_3.0.1_Linux-x86_64-multicore.tar.gz"
    upload(bot, api, name, b"tar")
    (spec,) = store.specs()
    assert (spec.kind, spec.entry) == ("tool", "namd")


def test_retry_of_a_terminal_job_without_kept_files_says_so(env):
    bot, api, store, _ = env
    store.create("script", "run.sh", "1E0L", {}, {"run.sh": b"true"}, keep_input=False)
    store.set_status(1, jobs.Status("failed", attempt=1))
    bot.handle_update(message("/retry 1"))
    assert len(store.specs()) == 1 and "terminal" in api.last_text()


def test_a_card_started_from_the_panel_leaves_the_chat_too(env, tmp_path):
    bot, api, store, _ = env
    upload(bot, api, "1e0l.pdb", b"ATOM")
    (token,) = bot.state["drafts"]
    (tmp_path / "staging" / token / actions.CLOSED).touch()
    bot.tick()
    assert bot.state["drafts"] == {} and not (tmp_path / "staging" / token).exists()


# --- the panel (Mini App) -----------------------------------------------------------------------------

class FakeExposure:
    def __init__(self, public=False):
        self.public, self.calls = public, []

    def status(self):
        return self.public

    def set_public(self, on):
        self.calls.append(on)
        self.public = on


@pytest.fixture
def panel(tmp_path):
    api, clock = FakeApi(), Clock()
    store = jobs.Store(tmp_path)
    exposure = FakeExposure()
    bot = Bot(store, api, OWNER, tmp_path / "bot.json", clock=clock, system_info=lambda: "",
              web_url="https://vostro.example.ts.net/", exposure=exposure)
    return bot, api, store, exposure


def test_panel_button_opens_the_mini_app(panel):
    bot, api, _, _ = panel
    bot.handle_update(message("/panel"))
    (button,) = last_buttons(api)
    assert button["web_app"]["url"] == "https://vostro.example.ts.net/"


def test_cards_offer_the_panel_too(panel):
    bot, api, _, _ = panel
    bot.handle_update(message("/md 1E0L"))
    assert any(b.get("web_app") for b in last_buttons(api))


def test_without_tailscale_the_panel_explains_how(env):
    bot, api, _, _ = env
    bot.handle_update(message("/panel"))
    assert "remote.sh" in api.last_text()


def test_funnel_on_and_off(panel):
    bot, api, _, exposure = panel
    bot.handle_update(message("/funnel on"))
    assert exposure.calls == [True] and "publiczn" in api.last_text()
    bot.handle_update(message("/funnel"))
    assert "włączony" in api.last_text()
    bot.handle_update(message("/funnel off"))
    assert exposure.calls == [True, False]


# --- results too big for Telegram -----------------------------------------------------------------------

def big_job(store, size=60 * 1024 ** 2):
    job = finished_job(store, {"small.txt": b"ok"})
    with open(store.work(job) / "traj.xtc", "wb") as f:
        f.truncate(size)  # sparse: no disk used
    return job


def test_with_the_panel_a_big_file_gets_a_download_link_and_scp(panel):
    bot, api, store, _ = panel
    bot.ssh_target = "student@laptop.example.ts.net"
    job = big_job(store)
    bot.tick()
    text = next(t for t in api.texts() if "traj.xtc" in t)
    assert "scp student@laptop.example.ts.net:" in text and f"jobs/{job:04d}/work/traj.xtc" in text
    link = next(b["url"] for b in last_buttons(api) if "url" in b)
    assert link.startswith("https://vostro.example.ts.net/dl/")
    from compute_bot import links
    assert links.parse(bot.link_secret, link.rsplit("/", 1)[1], bot.clock()) == (job, "traj.xtc")


def test_without_the_panel_a_big_file_can_come_in_parts(env):
    bot, api, store, _ = env
    job = big_job(store, size=100 * 1024 ** 2)
    bot.tick()
    assert any(f"/get {job} traj.xtc czesci" in t for t in api.texts())
    bot.handle_update(message(f"/get {job} traj.xtc"))
    assert "czesci" in api.last_text()
    bot.handle_update(message(f"/get {job} traj.xtc czesci"))
    parts = [c[2]["document"][0] for c in api.sent("sendDocument") if c[2]["document"][0].startswith("traj.xtc.")]
    assert parts == ["traj.xtc.part01", "traj.xtc.part02", "traj.xtc.part03"]
    assert "cat traj.xtc.part* &gt; traj.xtc" in api.last_text()  # HTML: shown as >


def test_files_too_big_even_for_parts_point_to_remote_access(env):
    bot, api, store, _ = env
    job = big_job(store, size=3 * 1024 ** 3)
    bot.handle_update(message(f"/get {job} traj.xtc czesci"))
    assert "remote.sh" in api.last_text() and api.sent("sendDocument") == []
