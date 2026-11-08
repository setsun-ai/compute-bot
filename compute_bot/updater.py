"""
Updating from GitHub releases. The service can't write to /opt, so a downloaded version lives in
COMPUTE_HOME/update/app (the one before it in app.bak); launch.py starts whichever of that and the installed copy
is newer. Only a release whose version matches its tag and whose selftest passes replaces what's there.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from . import REPO, VERSION

HERE = Path(__file__).resolve().parent.parent


class UpdateError(Exception):
    """Why an update can't go ahead, in words for the owner."""


def version_tuple(text):
    return tuple(int(p) for p in text.split("."))


def version_of(app):
    try:
        match = re.search(r'^VERSION = "(\d+\.\d+\.\d+)"', (Path(app) / "compute_bot/__init__.py").read_text(), re.M)
    except OSError:
        return None
    return match[1] if match else None


def web_get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "compute-bot"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        raise UpdateError(f"GitHub odpowiedział błędem {e.code}") from None
    except OSError:
        raise UpdateError("brak połączenia z GitHubem") from None


def chosen_app(installed, folder):
    """What launch.py starts: the downloaded app when it is newer than the installed one."""
    downloaded = Path(folder) / "app"
    mine, theirs = version_of(installed), version_of(downloaded)
    if theirs and (not mine or version_tuple(theirs) > version_tuple(mine)):
        return downloaded
    return Path(installed)


def restart_wanted(home):
    """The code launch.py would start now is not the code that is running (after an update or a rollback)."""
    installed = os.environ.get("COMPUTE_INSTALLED")
    if not installed:
        return False
    return version_of(chosen_app(installed, Path(home) / "update")) != VERSION


class Updater:
    def __init__(self, folder, fetch=web_get):
        self.folder, self.fetch = Path(folder), fetch
        self.app, self.previous, self.new = self.folder / "app", self.folder / "app.bak", self.folder / "app.new"

    def latest(self):
        try:
            tag = json.loads(self.fetch(f"https://api.github.com/repos/{REPO}/releases/latest")).get("tag_name")
        except ValueError:
            raise UpdateError("dziwna odpowiedź GitHuba") from None
        if not isinstance(tag, str) or not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise UpdateError("dziwna odpowiedź GitHuba")
        return tag

    def install(self, tag):
        """Returns True when the new version needs other packages (deploy/environment.yml changed)."""
        data = self.fetch(f"https://codeload.github.com/{REPO}/tar.gz/refs/tags/{tag}")
        shutil.rmtree(self.new, ignore_errors=True)
        self.new.mkdir(parents=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                for member in archive.getmembers():
                    parts = Path(member.name).parts[1:]  # GitHub puts everything in repo-tag/
                    if not parts or ".." in parts or member.issym() or member.islnk():
                        continue
                    member.name = str(Path(*parts))
                    archive.extract(member, self.new, filter="data") if sys.version_info >= (3, 12) \
                        else archive.extract(member, self.new)
        except (tarfile.TarError, OSError):
            shutil.rmtree(self.new, ignore_errors=True)
            raise UpdateError("pobrane archiwum jest uszkodzone") from None
        if version_of(self.new) != tag[1:]:
            shutil.rmtree(self.new, ignore_errors=True)
            raise UpdateError("pobrany kod ma inną wersję niż wydanie")
        check = subprocess.run([sys.executable, "-m", "compute_bot", "--selftest"], cwd=self.new,
                               capture_output=True, timeout=120, env={**os.environ, "PYTHONPATH": str(self.new)})
        if check.returncode != 0:
            shutil.rmtree(self.new, ignore_errors=True)
            raise UpdateError("nowa wersja nie przeszła selftestu")
        env_file = "deploy/environment.yml"
        current = self.app if self.app.exists() else HERE
        changed = _read(self.new / env_file) != _read(current / env_file)
        shutil.rmtree(self.previous, ignore_errors=True)
        if self.app.exists():
            os.replace(self.app, self.previous)
        os.replace(self.new, self.app)
        return changed

    def rollback(self):
        if not self.app.exists():
            raise UpdateError("nie ma wersji, do której można wrócić")
        shutil.rmtree(self.new, ignore_errors=True)
        if self.previous.exists():
            shutil.rmtree(self.app)
            os.replace(self.previous, self.app)
        else:
            shutil.rmtree(self.app)


def _read(path):
    try:
        return Path(path).read_bytes()
    except OSError:
        return None
