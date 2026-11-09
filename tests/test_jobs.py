"""The job model as the bot and the runner use it: files on disk, states, what to run."""
import io
import zipfile

import pytest

from compute_bot import jobs
from compute_bot.recipes import RECIPES, ParamError


def zip_bytes(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return out.getvalue()


# --- what the owner typed ---------------------------------------------------------------------------

def test_caption_with_recipe_and_params():
    assert jobs.parse_caption("qsar target=LogKd max_desc=3") == ("qsar", {"target": "LogKd", "max_desc": "3"})


def test_caption_with_only_params_and_slash_command():
    assert jobs.parse_caption("T=600 ns=0.5") == (None, {"T": "600", "ns": "0.5"})
    assert jobs.parse_caption("/md 1E0L T=600") == ("md", {"pdb": "1E0L", "T": "600"})


def test_caption_quotes_keep_spaces():
    assert jobs.parse_caption('qsar target="EXP LogKda"') == ("qsar", {"target": "EXP LogKda"})


def test_caption_rejects_loose_words():
    with pytest.raises(ParamError):
        jobs.parse_caption("qsar please")


def test_empty_caption():
    assert jobs.parse_caption(None) == (None, {})


# --- what a file becomes ------------------------------------------------------------------------------

def test_zip_with_run_script_is_a_script_job():
    files = jobs.unpack("1E0L.zip", zip_bytes({"1E0L/run.sh": "echo hi", "1E0L/em.gro": "x"}))
    assert set(files) == {"run.sh", "em.gro"}  # the common top folder is gone
    assert jobs.detect("1E0L.zip", None, files) == ("script", "run.sh")


def test_single_files_by_extension():
    assert jobs.detect("md600.tpr", None, {"md600.tpr": b""}) == ("tpr", "md600.tpr")
    assert jobs.detect("a.ipynb", None, {"a.ipynb": b""}) == ("notebook", "a.ipynb")
    assert jobs.detect("a.py", None, {"a.py": b""}) == ("python", "a.py")
    assert jobs.detect("1ubq.pdb", None, {"1ubq.pdb": b""}) == ("recipe", "md")


def test_a_named_recipe_wins_and_a_table_without_one_needs_a_question():
    assert jobs.detect("dane.xlsx", "qsar", {"dane.xlsx": b""}) == ("recipe", "qsar")
    assert jobs.detect("dane.xlsx", None, {"dane.xlsx": b""}) is None


def test_unknown_word_is_an_error():
    with pytest.raises(ParamError):
        jobs.detect("dane.xlsx", "kawa", {"dane.xlsx": b""})


def test_zip_slip_is_refused():
    with pytest.raises(ParamError):
        jobs.unpack("x.zip", zip_bytes({"../evil.sh": "rm -rf /"}))


# --- commands ---------------------------------------------------------------------------------------

def test_commands_per_kind():
    spec = jobs.Spec(1, "script", "run.sh", "x.zip", {}, 0)
    assert jobs.command(spec, "/env/bin/python") == ["bash", "run.sh"]
    assert jobs.command(spec._replace(kind="tpr", entry="md.tpr"), "py") == ["md_step", "md"]
    assert jobs.command(spec._replace(kind="python", entry="a.py"), "/e/python") == ["/e/python", "a.py"]
    assert jobs.command(spec._replace(kind="recipe", entry="qsar"), "/e/python") == [
        "/e/python", "-m", "compute_bot.recipes.qsar"]
    notebook = jobs.command(spec._replace(kind="notebook", entry="a b.ipynb"), "/e/python")
    assert notebook[:3] == ["/e/python", "-m", "jupyter"] and notebook[-1] == "a b.ipynb"
    assert "a b_wynik.ipynb" in notebook


# --- the store ------------------------------------------------------------------------------------------

def test_created_job_is_queued_with_files_and_validated_params(tmp_path):
    store = jobs.Store(tmp_path)
    params = RECIPES["qsar"].validate({"max_desc": "3"})
    spec = store.create("recipe", "qsar", "dane.xlsx", params, {"dane.xlsx": b"xlsx"})
    assert spec.id == 1
    assert (store.work(1) / "dane.xlsx").read_bytes() == b"xlsx"
    assert jobs.json.loads((store.work(1) / "params.json").read_text())["max_desc"] == 3
    assert store.status(1).state == "queued"
    assert [s.id for s in store.specs()] == [1]
    assert store.create("script", "run.sh", "b.zip", {}, {"run.sh": b"true"}).id == 2


def test_half_written_job_is_invisible(tmp_path):
    store = jobs.Store(tmp_path)
    (store.jobs / "0007.tmp").mkdir(parents=True)
    assert store.specs() == []
    assert store.create("script", "run.sh", "b.zip", {}, {"run.sh": b"true"}).id == 8  # never reuses a crashed one


def test_status_round_trip_and_cancel_flag(tmp_path):
    store = jobs.Store(tmp_path)
    store.create("script", "run.sh", "b.zip", {}, {"run.sh": b"true"})
    store.set_status(1, jobs.Status(state="running", attempt=1, started=5.0))
    assert store.status(1) == jobs.Status(state="running", attempt=1, started=5.0)
    assert not store.cancel_requested(1)
    store.request_cancel(1)
    assert store.cancel_requested(1)


def test_files_outside_the_work_dir_are_refused(tmp_path):
    store = jobs.Store(tmp_path)
    store.create("script", "run.sh", "b.zip", {}, {"run.sh": b"true"})
    assert store.result_file(1, "run.sh").name == "run.sh"
    assert store.result_file(1, "../spec.json") is None
    assert store.result_file(1, "nie_ma.txt") is None


# --- decisions -------------------------------------------------------------------------------------------

def test_next_job_is_the_oldest_queued():
    statuses = {1: jobs.Status("done"), 2: jobs.Status("queued"), 3: jobs.Status("queued")}
    assert jobs.next_job(statuses) == 2
    assert jobs.next_job({1: jobs.Status("failed")}) is None


def test_after_a_crash_a_running_job_runs_again_until_the_attempts_run_out():
    again = jobs.recover(jobs.Status("running", attempt=1))
    assert again.state == "queued" and again.attempt == 1
    assert jobs.recover(jobs.Status("paused", attempt=2)).state == "queued"
    given_up = jobs.recover(jobs.Status("running", attempt=jobs.MAX_ATTEMPTS))
    assert given_up.state == "failed" and "restart" in given_up.progress
    assert jobs.recover(jobs.Status("done", attempt=1)) == jobs.Status("done", attempt=1)


def test_thermal_hysteresis():
    assert jobs.thermal_action(95.0, paused=False) == "stop"
    assert jobs.thermal_action(85.0, paused=True) is None  # still too warm to go on
    assert jobs.thermal_action(70.0, paused=True) == "cont"
    assert jobs.thermal_action(70.0, paused=False) is None
    assert jobs.thermal_action(None, paused=True) == "cont"  # no sensor: never stay stopped


def test_progress_from_gromacs_and_recipe_lines():
    log = ("step 100, will finish Tue Nov 10 03:12:45 2026\n"
           "imb F  1% step 4200, will finish Tue Nov 10 04:00:01 2026\r")
    assert jobs.progress_from_log(log) == "krok 4200, koniec ok. Tue Nov 10 04:00:01 2026"
    assert jobs.progress_from_log("x\nPROGRESS: NPT 40%\nother\n") == "NPT 40%"
    assert jobs.progress_from_log("nothing") == ""


# --- quality of life ------------------------------------------------------------------------------------

def test_priority_jumps_the_queue_but_ties_keep_the_order():
    statuses = {1: jobs.Status("queued"), 2: jobs.Status("queued"), 3: jobs.Status("queued")}
    assert jobs.next_job(statuses, {3: 1}) == 3
    assert jobs.next_job(statuses, {2: 1, 3: 1}) == 2


def test_url_job_needs_a_kind_it_can_tell_from_the_link_or_a_word():
    assert jobs.detect_url("https://x.org/a/md600.tpr", None) == ("tpr", "md600.tpr", "md600.tpr")
    assert jobs.detect_url("https://x.org/dane.zip?dl=1", "skrypt") == ("script", "run.sh", "dane.zip")
    assert jobs.detect_url("https://drive.google.com/file/d/ABC/view", "omics")[:2] == ("recipe", "omics")
    with pytest.raises(ParamError):
        jobs.detect_url("https://drive.google.com/file/d/ABC/view", None)
    with pytest.raises(ParamError):
        jobs.detect_url("file:///etc/passwd", "qsar")


def test_store_keeps_priority_and_fetch_list(tmp_path):
    store = jobs.Store(tmp_path)
    link = [{"url": "https://x.org/r.tsv", "name": "r.tsv"}]
    spec = store.create("recipe", "omics", "report.tsv", {}, {}, fetch=link)
    assert store.spec(spec.id).fetch == [{"url": "https://x.org/r.tsv", "name": "r.tsv"}]
    store.set_priority(spec.id, 1)
    assert store.priorities() == {spec.id: 1}


def test_a_job_with_files_to_fetch_starts_with_the_fetcher():
    spec = jobs.Spec(1, "recipe", "qsar", "x", {}, 0, fetch=[{"url": "https://x.org/a.xlsx", "name": "a.xlsx"}])
    assert jobs.command(spec, "/e/python") == ["/e/python", "-m", "compute_bot.fetch", "--",
                                               "/e/python", "-m", "compute_bot.recipes.qsar"]


def test_finished_job_can_be_removed_a_running_one_cannot(tmp_path):
    store = jobs.Store(tmp_path)
    store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    store.set_status(1, jobs.Status("running", attempt=1))
    with pytest.raises(ParamError):
        store.remove(1)
    store.set_status(1, jobs.Status("done", attempt=1))
    store.remove(1)
    assert store.ids() == []


# --- engines and packages --------------------------------------------------------------------------------

def tgz_bytes(files, extra=None):
    import tarfile
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if extra:
            archive.addfile(extra)
    return out.getvalue()


def test_charmm_gui_tgz_is_unpacked_and_recognised():
    files = jobs.unpack("charmm-gui.tgz", tgz_bytes({"charmm-gui-123/gromacs/README": b"gmx grompp",
                                                     "charmm-gui-123/gromacs/step5_production.mdp": b"",
                                                     "charmm-gui-123/toppar/x.rtf": b""}))
    assert "gromacs/README" in files
    assert jobs.detect("charmm-gui.tgz", None, files) == ("recipe", "charmm")


def test_tar_with_links_or_escapes_is_refused():
    import tarfile
    link = tarfile.TarInfo("a/evil")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    with pytest.raises(ParamError):
        jobs.unpack("x.tgz", tgz_bytes({"a/ok.txt": b"x"}, extra=link))
    with pytest.raises(ParamError):
        jobs.unpack("x.tar.gz", tgz_bytes({"../up.txt": b"x"}))


def test_namd_download_becomes_an_install_job():
    name = "NAMD_3.0.1_Linux-x86_64-multicore.tar.gz"
    assert jobs.detect(name, None, {name: b""}) == ("tool", "namd")
    spec = jobs.Spec(1, "tool", "namd", name, {}, 0)
    assert jobs.command(spec, "/e/python") == ["/e/python", "-m", "compute_bot.tools", "namd", name]


def test_a_continuation_shares_the_parents_files_without_copying(tmp_path):
    store = jobs.Store(tmp_path)
    parent = store.create("recipe", "md", "1E0L", {"pdb": "1E0L"}, {}).id
    (store.work(parent) / "gromacs").mkdir()
    (store.work(parent) / "gromacs/md.xtc").write_bytes(b"traj")
    child = store.create_from(parent, {"pdb": "1E0L", "extend_ns": 5.0}).id
    linked = store.work(child) / "gromacs/md.xtc"
    assert linked.read_bytes() == b"traj"
    assert linked.stat().st_ino == (store.work(parent) / "gromacs/md.xtc").stat().st_ino
    assert jobs.json.loads((store.work(child) / "params.json").read_text())["extend_ns"] == 5.0
    assert store.spec(child).name.startswith("1E0L")
