"""The same queue from a terminal over SSH: what the owner types and what it prints."""
import pytest

from compute_bot import cli, jobs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPUTE_HOME", str(tmp_path / "home"))
    return jobs.Store(tmp_path / "home")


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out + out.err


def test_submit_a_folder_with_run_sh(home, tmp_path, capsys):
    folder = tmp_path / "1E0L"
    (folder / "sub").mkdir(parents=True)
    (folder / "run.sh").write_text("md_step md")
    (folder / "sub" / "topol.top").write_text("top")
    code, out = run(capsys, "submit", str(folder))
    assert code == 0 and "#1" in out
    (spec,) = home.specs()
    assert (spec.kind, spec.entry) == ("script", "run.sh")
    assert (home.work(1) / "sub/topol.top").read_text() == "top"


def test_submit_a_table_needs_a_recipe_word_and_checks_params(home, tmp_path, capsys):
    table = tmp_path / "dane.xlsx"
    table.write_bytes(b"x")
    code, out = run(capsys, "submit", str(table))
    assert code == 2 and "qsar" in out
    code, out = run(capsys, "submit", str(table), "qsar", "max_desc=lots")
    assert code == 2 and "max_desc" in out
    code, _ = run(capsys, "submit", str(table), "qsar", "max_desc=3")
    assert code == 0 and home.spec(1).params["max_desc"] == 3


def test_md_from_a_pdb_code(home, capsys):
    code, _ = run(capsys, "md", "1E0L", "T=600")
    assert code == 0 and home.spec(1).params["pdb"] == "1E0L"


def test_jobs_log_cancel_pause_and_where(home, tmp_path, capsys):
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "run.sh").write_text("sleep 100")
    run(capsys, "submit", str(folder))
    home.log_path(1).write_text("step 10, will finish Tue Nov 10 03:12:45 2026\n")
    code, out = run(capsys, "jobs")
    assert "#1" in out and "queued" in out
    code, out = run(capsys, "log", "1")
    assert "will finish" in out
    run(capsys, "cancel", "1")
    assert home.cancel_requested(1)
    run(capsys, "pause")
    assert home.hold
    run(capsys, "resume")
    assert not home.hold
    code, out = run(capsys, "where", "1")
    assert out.strip() == str(home.work(1))


def test_unknown_job_is_a_clear_error(home, capsys):
    code, out = run(capsys, "log", "7")
    assert code == 2 and "#7" in out
