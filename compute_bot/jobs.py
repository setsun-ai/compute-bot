"""
Jobs on disk. One writer per file: the bot writes a job's spec once (the job directory appears with an atomic
rename), the runner alone writes its status, the bot alone touches the cancel flag. Standard library only.

  jobs/0001/spec.json     what to run (bot)
  jobs/0001/status.json   state, attempt, times, progress (runner)
  jobs/0001/log.txt       stdout + stderr of the job (runner)
  jobs/0001/cancel        the owner asked to stop it (bot)
  jobs/0001/priority      the owner moved it up the queue (bot)
  jobs/0001/input/        the files as sent, for /retry (bot)
  jobs/0001/work/         the owner's files, params.json, and everything the job writes
  hold                    the owner paused everything (bot)
  runner.json             heartbeat: time, current job, why nothing starts (runner)
"""
from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import tarfile
import time
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from .recipes import RECIPES, ParamError

STATES = ("queued", "running", "paused", "done", "failed", "cancelled")
FINAL = frozenset(("done", "failed", "cancelled"))
MAX_ATTEMPTS = 3  # starts after a crash or reboot, before the job is given up
HOT, COOL = 92.0, 75.0  # °C: stop the job at HOT, let it go on below COOL
MAX_UNPACKED = 2 * 1024 ** 3


class Spec(NamedTuple):
    id: int
    kind: str  # script | tpr | notebook | python | recipe
    entry: str  # run.sh, the .tpr/.ipynb/.py file, or the recipe name
    name: str  # what the owner sent, for messages
    params: dict
    created: float
    fetch: list = []  # [{"url", "name"}]: downloaded into work/ by the job itself before it starts


class Status(NamedTuple):
    state: str = "queued"
    attempt: int = 0
    started: float | None = None
    finished: float | None = None
    exit_code: int | None = None
    progress: str = ""
    pause: str = ""  # while paused: "heat" or "hold"


# --- what the owner typed and sent -------------------------------------------------------------------

def parse_caption(text):
    """'qsar target=LogKd' → ('qsar', {'target': 'LogKd'}); '/md 1E0L' → ('md', {'pdb': '1E0L'})."""
    if not text or not text.strip():
        return None, {}
    try:
        words = shlex.split(text.strip())
    except ValueError:
        raise ParamError("Niedomknięty cudzysłów w podpisie.") from None
    recipe = None
    if words and "=" not in words[0]:
        recipe = words.pop(0).lstrip("/").split("@")[0].lower()
    params = {}
    for word in words:
        key, sep, value = word.partition("=")
        if sep:
            params[key] = value
        elif recipe == "md" and "pdb" not in params and re.fullmatch(r"[0-9][A-Za-z0-9]{3}", word):
            params["pdb"] = word.upper()
        elif recipe == "dock" and "receptor" not in params and re.fullmatch(r"[0-9][A-Za-z0-9]{3}", word):
            params["receptor"] = word.upper()
        else:
            raise ParamError(f"Nie rozumiem „{word}”. Parametry piszę jako klucz=wartość.")
    return recipe, params


TARBALLS = (".tgz", ".tar.gz", ".tar")


def unpack(filename, data):
    """The files of an upload: a zip or tarball is unpacked (without a folder common to all), the rest is itself."""
    lowered = filename.lower()
    if lowered.endswith(TARBALLS) and not is_namd(filename):
        return unpack_tar(data)
    if not lowered.endswith(".zip"):
        return {safe_name(filename): data}
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ParamError("To nie jest poprawny plik zip.") from None
    return {target: archive.read(member) for member, target in zip_layout(archive).items()}


def unpack_tar(data):
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data))
        members = archive.getmembers()
    except tarfile.TarError:
        raise ParamError("To nie jest poprawne archiwum tar/tgz.") from None
    files = [m for m in members if m.isfile()]
    if any(not (m.isfile() or m.isdir()) for m in members):
        raise ParamError("W archiwum są dowiązania albo pliki specjalne; spakuj same pliki.")
    if sum(m.size for m in files) > MAX_UNPACKED:
        raise ParamError("Archiwum po rozpakowaniu ma ponad 2 GB.")
    targets = strip_common_top([PurePosixPath(m.name) for m in files])
    return {target: archive.extractfile(m).read() for m, target in zip(files, targets, strict=True)}


def strip_common_top(paths):
    """Safe relative targets for archive paths, without a folder that holds everything."""
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise ParamError(f"Niebezpieczna ścieżka w archiwum: {path}")
    if not paths:
        raise ParamError("Archiwum jest puste.")
    top = {p.parts[0] for p in paths}
    strip = len(top) == 1 and all(len(p.parts) > 1 for p in paths)
    return [str(PurePosixPath(*p.parts[1:])) if strip else str(p) for p in paths]


def is_namd(filename):
    return bool(re.match(r"NAMD_[\d.]+.*Linux-x86_64.*\.tar\.gz$", PurePosixPath(filename).name))


def zip_layout(archive):
    """Member → where it goes: no absolute or ../ paths, no macOS junk, no folder common to all files."""
    members = [m for m in archive.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX/")
               and not PurePosixPath(m.filename).name.startswith("._")]
    if sum(m.file_size for m in members) > MAX_UNPACKED:
        raise ParamError("Zip po rozpakowaniu ma ponad 2 GB.")
    targets = strip_common_top([PurePosixPath(m.filename) for m in members])
    return dict(zip(members, targets, strict=True))


def safe_name(filename):
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or name in (".", ".."):
        raise ParamError("Dziwna nazwa pliku.")
    return name


SINGLE = {".tpr": "tpr", ".ipynb": "notebook", ".py": "python"}


CHARMM_MARKERS = ("gromacs/README", "openmm/openmm_run.py", "namd/README")


def detect(filename, recipe_word, files):
    """(kind, entry) for an upload, or None when only the owner can say what a table is for."""
    if is_namd(filename) and recipe_word is None:
        return "tool", "namd"
    if recipe_word is not None:
        if recipe_word not in RECIPES:
            raise ParamError(f"Nie znam „{recipe_word}”. Są: {', '.join(RECIPES)}.")
        return "recipe", recipe_word
    if "run.sh" in files:
        return "script", "run.sh"
    if any(marker in files for marker in CHARMM_MARKERS):
        return "recipe", "charmm"
    if len(files) == 1:
        (name,) = files
        suffix = PurePosixPath(name).suffix.lower()
        if suffix in SINGLE:
            return SINGLE[suffix], name
        if suffix in (".pdb", ".cif"):
            return "recipe", "md"
    return None


URL_WORDS = {"skrypt": ("script", "run.sh"), "script": ("script", "run.sh")}


def detect_url(url, word):
    """(kind, entry, file name) for a job whose input is a link; the file arrives only when the job runs."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ParamError("Link musi zaczynać się od http:// albo https://.")
    name = PurePosixPath(urllib.parse.unquote(parsed.path)).name
    named = bool(re.search(r"\.[A-Za-z0-9]{1,5}$", name)) and "drive.google.com" not in parsed.netloc
    if not named:
        name = "pobrany.zip" if word in URL_WORDS else "pobrany"
    if word in URL_WORDS:
        return (*URL_WORDS[word], name)
    if word is not None:
        if word not in RECIPES:
            raise ParamError(f"Nie znam „{word}”. Są: {', '.join(RECIPES)}, skrypt.")
        return "recipe", word, name
    if named:
        found = detect(name, None, {name: b""})
        if found:
            return (*found, name)
    raise ParamError("Z linku nie widać, co to za plik. Dopisz słowo: qsar, omics, dock, md albo skrypt "
                     "(zip z run.sh).")


def command(spec, python):
    """What the runner starts, in the job's work directory; files to fetch come first, in the same process."""
    run = _command(spec, python)
    return [python, "-m", "compute_bot.fetch", "--", *run] if spec.fetch else run


def _command(spec, python):
    if spec.kind == "script":
        return ["bash", spec.entry]
    if spec.kind == "tpr":
        return ["md_step", PurePosixPath(spec.entry).stem]
    if spec.kind == "python":
        return [python, spec.entry]
    if spec.kind == "notebook":
        out = PurePosixPath(spec.entry).stem + "_wynik.ipynb"
        return [python, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=-1", "--output", out, spec.entry]
    if spec.kind == "recipe":
        return [python, "-m", f"compute_bot.recipes.{spec.entry}"]
    if spec.kind == "tool":
        return [python, "-m", "compute_bot.tools", spec.entry, spec.name]
    raise ValueError(spec.kind)


# --- decisions (pure) ---------------------------------------------------------------------------------

def queue_order(statuses, priorities=None):
    """Queued jobs in the order they will start: higher priority first, among equals the oldest."""
    priorities = priorities or {}
    queued = [job for job, status in statuses.items() if status.state == "queued"]
    return sorted(queued, key=lambda job: (-priorities.get(job, 0), job))


def next_job(statuses, priorities=None):
    order = queue_order(statuses, priorities)
    return order[0] if order else None


def recover(status):
    """A job that was running when the runner stopped: run it again, unless it already had all its chances."""
    if status.state not in ("running", "paused"):
        return status
    if status.attempt >= MAX_ATTEMPTS:
        return status._replace(state="failed", finished=time.time(),
                               progress=f"przerwane {status.attempt} razy (restart?), nie próbuję dalej")
    return status._replace(state="queued", pause="")


def thermal_action(temp, paused, hot=HOT, cool=COOL):
    if paused and (temp is None or temp <= cool):
        return "cont"
    if not paused and temp is not None and temp >= hot:
        return "stop"
    return None


GMX_ETA = re.compile(r"step (\d+), will finish (.+?)\s*$")


def progress_from_log(text):
    """The last progress a job printed: mdrun's ETA or a recipe's PROGRESS line."""
    best = ""
    for line in re.split(r"[\r\n]+", text):
        if line.startswith("PROGRESS:"):
            best = line[len("PROGRESS:"):].strip()
        elif match := GMX_ETA.search(line):
            best = f"krok {match[1]}, koniec ok. {match[2]}"
    return best


# --- the store ---------------------------------------------------------------------------------------

def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs = self.root / "jobs"

    def dir(self, job):
        return self.jobs / f"{job:04d}"

    def work(self, job):
        return self.dir(job) / "work"

    def log_path(self, job):
        return self.dir(job) / "log.txt"

    def ids(self):
        if not self.jobs.exists():
            return []
        return sorted(int(p.name) for p in self.jobs.iterdir() if p.name.isdigit() and (p / "spec.json").exists())

    def new_slot(self):
        """The next number and its temporary directory; mkdir fails on a collision, so two creators never share one."""
        self.jobs.mkdir(parents=True, exist_ok=True)
        while True:
            taken = [int(p.name.split(".")[0]) for p in self.jobs.iterdir() if p.name.split(".")[0].isdigit()]
            job = max(taken, default=0) + 1
            tmp = self.jobs / f"{job:04d}.tmp"
            try:
                tmp.mkdir()
                return job, tmp
            except FileExistsError:
                continue

    def create(self, kind, entry, name, params, files, fetch=(), keep_input=True):
        """The directory gets its final name only when everything is in it: the runner never sees half a job."""
        job, tmp = self.new_slot()
        work = tmp / "work"
        work.mkdir()
        (tmp / "input").mkdir()
        folders = (tmp / "input", work) if keep_input else (work,)  # input/ stays as sent, for /retry
        for folder in folders:
            for rel, data in files.items():  # bytes from Telegram, or a Path from the terminal (copied)
                target = folder / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(data, Path):
                    shutil.copyfile(data, target)
                else:
                    target.write_bytes(data)
                if rel.endswith(".sh"):
                    target.chmod(0o755)
        if kind == "recipe":
            write_json(work / "params.json", params)
        spec = Spec(job, kind, entry, name, params, time.time(), list(fetch))
        write_json(tmp / "spec.json", spec._asdict())
        os.replace(tmp, self.dir(job))
        return spec

    def create_from(self, parent, params):
        """A continuation of a finished job: its work/ hard-linked (no copies), new parameters, same recipe.
        The job must write new files instead of changing linked ones."""
        spec = self.spec(parent)
        job, tmp = self.new_slot()
        source = self.work(parent)
        (tmp / "work").mkdir()
        for path in source.rglob("*"):
            target = tmp / "work" / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.name != "params.json":
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
        (tmp / "input").mkdir(parents=True, exist_ok=True)
        write_json(tmp / "work" / "params.json", params)
        child = Spec(job, spec.kind, spec.entry, f"{spec.name} (ciąg #{parent})", params, time.time())
        write_json(tmp / "spec.json", child._asdict())
        os.replace(tmp, self.dir(job))
        return child

    def inputs(self, job):
        folder = self.dir(job) / "input"
        return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def spec(self, job):
        return Spec(**json.loads((self.dir(job) / "spec.json").read_text()))

    def specs(self):
        return [self.spec(job) for job in self.ids()]

    def status(self, job):
        try:
            return Status(**json.loads((self.dir(job) / "status.json").read_text()))
        except FileNotFoundError:
            return Status()

    def statuses(self):
        return {job: self.status(job) for job in self.ids()}

    def set_status(self, job, status):
        write_json(self.dir(job) / "status.json", status._asdict())

    def set_priority(self, job, value):
        (self.dir(job) / "priority").write_text(str(int(value)))

    def priorities(self):
        found = {}
        for job in self.ids():
            try:
                found[job] = int((self.dir(job) / "priority").read_text())
            except (FileNotFoundError, ValueError):
                pass
        return found

    def remove(self, job):
        if self.status(job).state not in FINAL:
            raise ParamError(f"#{job} jeszcze się nie skończyło: najpierw /cancel {job}.")
        shutil.rmtree(self.dir(job))

    def set_hold(self, on):
        flag = self.root / "hold"
        if on:
            self.root.mkdir(parents=True, exist_ok=True)
            flag.touch()
        else:
            flag.unlink(missing_ok=True)

    @property
    def hold(self):
        return (self.root / "hold").exists()

    def runner_state(self):
        try:
            return json.loads((self.root / "runner.json").read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def set_runner_state(self, value):
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "runner.json", value)

    def request_cancel(self, job):
        (self.dir(job) / "cancel").touch()

    def cancel_requested(self, job):
        return (self.dir(job) / "cancel").exists()

    def log_tail(self, job, size=4000):
        try:
            with open(self.log_path(job), "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - size))
                return f.read().decode(errors="replace")
        except FileNotFoundError:
            return ""

    def result_file(self, job, rel):
        """A file the job made, by its path in work/; nothing outside work/."""
        work = self.work(job).resolve()
        path = (work / rel).resolve()
        if work not in path.parents or not path.is_file():
            return None
        return path

    def outputs(self, job):
        """Files in work/ with their sizes, smallest first; params.json is ours, not a result."""
        work = self.work(job)
        found = [(p, p.stat().st_size) for p in work.rglob("*") if p.is_file() and p.name != "params.json"]
        return sorted(found, key=lambda item: (item[1], str(item[0])))
