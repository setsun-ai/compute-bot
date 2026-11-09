"""
The queue from a terminal (SSH over Tailscale): the same jobs as in Telegram, and the bot still reports them.

  compute submit FOLDER|FILE [qsar|omics|dock|md|charmm] [key=value...]
  compute md 1E0L T=600 ns=0.5          compute dock 1STP ...
  compute jobs | status N | log N [-f] | cancel N | where N | pause | resume

No parameter card here: what you type is what runs (checked against the registry first). A submitted folder is
copied into the job; it is not kept a second time for /retry. Standard library only.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from . import jobs
from .recipes import RECIPES, ParamError


def store():
    return jobs.Store(os.environ.get("COMPUTE_HOME", "/var/lib/compute-bot"))


def collect(path):
    path = Path(path).expanduser()
    if path.is_dir():
        return {str(p.relative_to(path)): p for p in sorted(path.rglob("*")) if p.is_file()}
    if path.is_file():
        if path.name.lower().endswith((".zip", *jobs.TARBALLS)):
            return jobs.unpack(path.name, path.read_bytes())
        return {path.name: path}
    raise ParamError(f"Nie ma {path}.")


def submit(s, args):
    if not args:
        raise ParamError("compute submit FOLDER|PLIK [przepis] [klucz=wartość...]")
    target, rest = args[0], args[1:]
    word, params = jobs.parse_caption(" ".join(rest))
    files = collect(target)
    found = jobs.detect(Path(target).name, word, {k: b"" for k in files})
    if found is None:
        raise ParamError(f"Nie wiem, co zrobić z {Path(target).name}: dopisz {', '.join(RECIPES)}.")
    kind, entry = found
    return create(s, kind, entry, Path(target).name, params, files)


def create(s, kind, entry, name, params, files):
    if kind == "recipe":
        params = RECIPES[entry].validate(params)
    elif params:
        raise ParamError("Parametry klucz=wartość są tylko dla przepisów.")
    spec = s.create(kind, entry, name, params, files, keep_input=False)
    print(f"#{spec.id} {name} w kolejce. Wiadomości przyjdą na Telegram; tutaj: compute log {spec.id} -f")
    return 0


def recipe(s, word, args):
    _, params = jobs.parse_caption(" ".join([word, *args]))
    return create(s, "recipe", word, f"{word} {' '.join(args)}".strip(), params, {})


def job_id(s, args):
    if not args or not args[0].isdigit() or int(args[0]) not in s.ids():
        raise ParamError(f"Nie ma zadania #{args[0] if args else '?'}. Lista: compute jobs")
    return int(args[0])


def show_jobs(s):
    for spec in s.specs()[-20:]:
        status = s.status(spec.id)
        extra = f"  {status.progress}" if status.progress and status.state in ("running", "paused") else ""
        print(f"#{spec.id:<4} {status.state:<9} {spec.name}{extra}")
    if s.hold:
        print("(pauza: compute resume)")
    return 0


def follow(s, job):
    shown = 0
    while True:
        text = s.log_tail(job, 10 ** 7)
        print(text[shown:], end="", flush=True)
        shown = len(text)
        if s.status(job).state in jobs.FINAL:
            return 0
        time.sleep(2)


def main(argv):
    os.umask(0o007)  # the runner (user compute, same group) must be able to write into what we create
    s = store()
    word, args = (argv[0], argv[1:]) if argv else ("jobs", [])
    try:
        if word == "submit":
            return submit(s, args)
        if word in RECIPES and word != "env":
            return recipe(s, word, args)
        if word == "jobs":
            return show_jobs(s)
        if word in ("pause", "resume"):
            s.set_hold(word == "pause")
            print("Pauza." if word == "pause" else "Kolejka rusza.")
            return 0
        job = job_id(s, args)
        if word == "log":
            return follow(s, job) if "-f" in args else print(s.log_tail(job, 20000)) or 0
        if word == "status":
            print(s.status(job)._asdict())
            return 0
        if word == "cancel":
            s.request_cancel(job)
            print(f"Zatrzymuję #{job}.")
            return 0
        if word == "where":
            print(s.work(job))
            return 0
        raise ParamError(f"Nie znam „{word}”. Zobacz: head -12 {__file__}")
    except ParamError as e:
        print(f"compute: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
