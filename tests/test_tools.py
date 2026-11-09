"""Installing NAMD from the tarball the owner downloaded, the way the runner starts it."""
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAME = "NAMD_3.0.1_Linux-x86_64-multicore.tar.gz"


def namd_tarball(path):
    with tarfile.open(path, "w:gz") as archive:
        data = b"#!/bin/sh\necho NAMD 3.0.1 fake\n"
        info = tarfile.TarInfo("NAMD_3.0.1_Linux-x86_64-multicore/namd3")
        info.size, info.mode = len(data), 0o755
        archive.addfile(info, io.BytesIO(data))


def install(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    namd_tarball(work / NAME)
    env = {**os.environ, "COMPUTE_HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, "-m", "compute_bot.tools", "namd", NAME], cwd=work, env=env,
                          capture_output=True, text=True), work


def test_namd_lands_on_the_jobs_path_and_runs(tmp_path):
    result, work = install(tmp_path)
    assert result.returncode == 0, result.stderr
    binary = tmp_path / "home/tools/bin/namd3"
    assert subprocess.run([str(binary)], capture_output=True, text=True).stdout.strip() == "NAMD 3.0.1 fake"
    assert "NAMD zainstalowany" in (work / "summary.txt").read_text()


def test_installing_twice_changes_nothing(tmp_path):
    install(tmp_path)
    result, _ = install(tmp_path)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (tmp_path / "home/tools").iterdir()) == ["NAMD_3.0.1_Linux-x86_64-multicore", "bin"]


def test_runner_puts_installed_tools_on_the_path(tmp_path, monkeypatch):
    from compute_bot import jobs
    from compute_bot.runner import Runner
    monkeypatch.setenv("COMPUTE_HOME", str(tmp_path))
    env = Runner(jobs.Store(tmp_path), sys.executable, 1).environment()
    assert str(tmp_path / "tools" / "bin") in env["PATH"].split(os.pathsep)
