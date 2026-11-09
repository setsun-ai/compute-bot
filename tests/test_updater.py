"""Updating from a release tarball, with a fake GitHub that serves this very repository."""
import io
import json
import tarfile
from pathlib import Path

import pytest

from compute_bot import VERSION, updater
from compute_bot.updater import UpdateError, Updater, chosen_app

REPO = Path(__file__).resolve().parent.parent


def tarball(version, break_code=False, env_extra=""):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as archive:
        for path in sorted(REPO.rglob("*")):
            rel = path.relative_to(REPO)
            if not path.is_file() or rel.parts[0] in (".git", "tests") or "__pycache__" in rel.parts:
                continue
            data = path.read_bytes()
            if rel == Path("compute_bot/__init__.py"):
                data = data.replace(f'VERSION = "{VERSION}"'.encode(), f'VERSION = "{version}"'.encode())
            if rel == Path("compute_bot/jobs.py") and break_code:
                data = b"raise SystemExit('broken')\n"
            if rel == Path("deploy/environment.yml"):
                data += env_extra.encode()
            info = tarfile.TarInfo(f"compute-bot-{version}/{rel}")
            info.size, info.mode = len(data), path.stat().st_mode & 0o777
            archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def github(tag, body):
    def fetch(url):
        if url.endswith("/releases/latest"):
            return json.dumps({"tag_name": tag}).encode()
        assert url.endswith(f"/refs/tags/{tag}")
        return body
    return fetch


def test_a_good_release_is_installed_and_launch_picks_it(tmp_path):
    up = Updater(tmp_path / "update", github("v9.0.0", tarball("9.0.0")))
    assert up.latest() == "v9.0.0"
    assert up.install("v9.0.0") is False  # same environment
    assert updater.version_of(tmp_path / "update/app") == "9.0.0"
    assert (tmp_path / "update/app/compute_bot/bin/md_step").stat().st_mode & 0o100
    assert chosen_app(REPO, tmp_path / "update") == tmp_path / "update/app"


def test_new_packages_are_reported(tmp_path):
    up = Updater(tmp_path / "update", github("v9.0.0", tarball("9.0.0", env_extra="  - biopython\n")))
    assert up.install("v9.0.0") is True


def test_wrong_version_or_failing_selftest_changes_nothing(tmp_path):
    up = Updater(tmp_path / "update", github("v9.0.0", tarball("8.0.0")))
    with pytest.raises(UpdateError):
        up.install("v9.0.0")
    up = Updater(tmp_path / "update", github("v9.0.0", tarball("9.0.0", break_code=True)))
    with pytest.raises(UpdateError):
        up.install("v9.0.0")
    assert not (tmp_path / "update/app").exists()


def test_rollback_goes_one_step_back_then_to_the_installed_copy(tmp_path):
    Updater(tmp_path / "update", github("v9.0.0", tarball("9.0.0"))).install("v9.0.0")
    up = Updater(tmp_path / "update", github("v9.1.0", tarball("9.1.0")))
    up.install("v9.1.0")
    up.rollback()
    assert updater.version_of(tmp_path / "update/app") == "9.0.0"
    up.rollback()
    assert chosen_app(REPO, tmp_path / "update") == REPO
    with pytest.raises(UpdateError):
        up.rollback()


def test_an_older_download_never_wins_over_a_fresh_install(tmp_path):
    Updater(tmp_path / "update", github("v0.0.1", tarball("0.0.1"))).install("v0.0.1")
    assert chosen_app(REPO, tmp_path / "update") == REPO


def test_runner_restarts_only_when_launch_would_start_other_code(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPUTE_INSTALLED", str(REPO))
    assert updater.restart_wanted(tmp_path) is False
    Updater(tmp_path / "update", github("v9.0.0", tarball("9.0.0"))).install("v9.0.0")
    assert updater.restart_wanted(tmp_path) is True
