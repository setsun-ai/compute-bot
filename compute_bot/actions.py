"""
What the owner can do to jobs, in one place for every front end (Telegram chat, Mini App, terminal).
Each function checks its inputs and raises ParamError with words for the owner; it returns the new Spec.
"""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

from . import jobs, results
from .recipes import RECIPES, ParamError


def as_typed(value):
    """A stored parameter back as the owner would type it, so it can be validated again with changes."""
    if isinstance(value, bool):
        return "tak" if value else "nie"
    if isinstance(value, list):
        return ",".join(map(str, value))
    return str(value)


def typed_params(spec):
    return {k: as_typed(v) for k, v in spec.params.items() if v not in (None, "", [])}


def retry(store, job, changes=None):
    """The same files again; changes (key → text) replace those parameters of a recipe."""
    spec = store.spec(job)
    files = store.inputs(job)
    if not files and not spec.fetch and spec.kind != "recipe":
        raise ParamError(f"#{job} przyszło z terminala i jego pliki nie są trzymane drugi raz: "
                         "compute submit … jeszcze raz.")
    params = spec.params
    if changes:
        if spec.kind != "recipe":
            raise ParamError("Zmiana parametrów działa tylko dla przepisów.")
        params = RECIPES[spec.entry].validate({**typed_params(spec), **changes})
    return store.create(spec.kind, spec.entry, spec.name, params, files, spec.fetch)


def extend(store, job, ns):
    spec, status = store.spec(job), store.status(job)
    if spec.kind != "recipe" or spec.entry != "md" or status.state != "done":
        raise ParamError(f"Doliczać można skończone zadania md; #{job} to {spec.name} ({status.state}).")
    params = RECIPES["md"].validate({**typed_params(spec), "extend_ns": str(ns)})
    return store.create_from(job, params)


def top(store, job):
    if store.status(job).state != "queued":
        raise ParamError(f"#{job} nie czeka w kolejce.")
    store.set_priority(job, max(store.priorities().values(), default=0) + 1)


CLOSED = ".closed"  # a card started or discarded in the Mini App; the bot drops it from bot.json on its next tick


def disk(store):
    """Every job by size, biggest first: [{"id", "name", "state", "finished", "size", "junk"}] (junk: what slim
    would free)."""
    found = []
    for spec in store.specs():
        work = store.work(spec.id)
        files = [(str(path.relative_to(work)), size) for path, size in store.outputs(spec.id)]
        status = store.status(spec.id)
        found.append({"id": spec.id, "name": spec.name, "state": status.state, "finished": status.finished,
                      "size": sum(size for _, size in files),
                      "junk": sum(size for rel, size in files if results.is_junk(rel))})
    return sorted(found, key=lambda job: (-job["size"], job["id"]))


def slim(store, job):
    """Remove a finished job's checkpoints, backups and crash dumps; returns the bytes freed."""
    if store.status(job).state not in jobs.FINAL:
        raise ParamError(f"#{job} jeszcze się liczy: odchudzić można dopiero skończone zadanie.")
    work = store.work(job)
    freed = 0
    for path, size in store.outputs(job):
        if results.is_junk(str(path.relative_to(work))):
            path.unlink()
            freed += size
    return freed


def old_finished(store, days, now):
    return [job for job, s in store.statuses().items()
            if s.state in jobs.FINAL and s.finished and now - s.finished > days * 86400]


def close_staged(folder):
    folder = Path(folder)
    if not folder.is_dir() or (folder / CLOSED).exists():
        raise ParamError("Ta karta jest już uruchomiona albo nieaktualna.")
    (folder / CLOSED).touch()


def start_staged(store, folder, recipe, name, typed, fetch=()):
    """A recipe job from staged files (a card); a folder can be started once."""
    folder = Path(folder)
    marker = folder / CLOSED
    if not folder.is_dir() or marker.exists():
        raise ParamError("Ta karta jest już uruchomiona albo nieaktualna.")
    params = RECIPES[recipe].validate(typed)
    if recipe == "md" and not params["pdb"] and not fetch and not any(folder.iterdir()):
        raise ParamError("Podaj kod PDB albo wyślij plik .pdb.")
    files = {str(p.relative_to(folder)): p for p in folder.rglob("*") if p.is_file() and p.name != "draft.json"}
    marker.touch()
    spec = store.create("recipe", recipe, name, params, files, fetch)
    return spec


def discard_staged(folder):
    shutil.rmtree(folder, ignore_errors=True)


def script_zip(store, job):
    """(file name, zip bytes) of a GROMACS md job as an editable script: inputs, .mdp files, run.sh."""
    spec = store.spec(job)
    if spec.kind != "recipe" or spec.entry != "md":
        raise ParamError(f"Skrypt robię z zadań md; #{job} to {spec.name}.")
    gromacs = store.work(job) / "gromacs"
    if not (gromacs / "run.sh").exists():
        raise ParamError(f"#{job} nie ma skryptu GROMACS (silnik OpenMM, zadanie sprzed wersji 0.2.1 albo układ "
                         "jeszcze się nie zbudował).")
    from .recipes.md.export import script_files
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in script_files(gromacs):
            archive.write(gromacs / rel, rel)
    return f"skrypt_{job:04d}.zip", out.getvalue()
