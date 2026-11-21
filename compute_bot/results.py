"""What a finished job's files are for: the results zip, slimming the disk, copying everything to a computer.

Three kinds of files, in this order of importance:
  results  – summary.txt, highlights, anything in a results folder (wyniki/, results/, …);
  tables   – tables, plots and text (.csv, .png, .xvg, …) anywhere;
  the rest – trajectories, structures, intermediates: they go in the zip only when everything fits.
Junk (checkpoints, GROMACS backups, crash dumps) never goes in the zip and is what slimming removes.
"""
from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath

RESULT_DIRS = {"wyniki", "results", "wykresy", "figures", "plots", "raport", "report"}
RESULT_FILES = {"summary.txt", "highlights.txt"}
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".md", ".png", ".svg", ".pdf", ".jpg", ".jpeg", ".json", ".xlsx",
                  ".xvg", ".dat", ".html", ".ipynb"}
JUNK = re.compile(r"(\.cpt$|^#.*#$|^core(\.\d+)?$|^step\d+[bc]\.pdb$)")  # by file name
COPY_EXCLUDE = ("*.cpt", "#*#")
COPY_TO = "~/compute-bot"  # where copies land on the owner's computer, until /kopia says otherwise


def is_junk(rel):
    return bool(JUNK.search(PurePosixPath(rel).name))


def rank(rel, highlights=()):
    """0 results, 1 tables and plots, 2 the rest."""
    path = PurePosixPath(rel)
    if rel in highlights or path.name in RESULT_FILES or RESULT_DIRS & set(path.parts[:-1]):
        return 0
    return 1 if path.suffix.lower() in TABLE_SUFFIXES else 2


def pack(files, budget, highlights=()):
    """(in the zip, left on the laptop) from [(rel, size)]: everything when it fits; otherwise results first,
    then tables and plots, smallest first, and the rest stays. Junk is neither."""
    files = [(rel, size) for rel, size in files if not is_junk(rel)]
    if sum(size for _, size in files) <= budget:
        return sorted(files, key=lambda f: (f[1], f[0])), []
    included, left, used = [], [], 0
    for rel, size in sorted(files, key=lambda f: (rank(f[0], highlights), f[1], f[0])):
        if rank(rel, highlights) < 2 and used + size <= budget:
            included.append((rel, size))
            used += size
        else:
            left.append((rel, size))
    return included, left


def slimmable(files):
    """The files slimming removes: checkpoints, backups, crash dumps. Results and trajectories stay."""
    return [rel for rel in files if is_junk(rel)]


def copy_command(host, remote_dir, local_base, job):
    """One line for a terminal: the job's whole folder to LOCAL_BASE/bot_NNNN, without checkpoints."""
    local = f"{local_base.rstrip('/')}/bot_{job:04d}"
    local_shell = local if local.startswith("~/") and " " not in local else shlex.quote(local)
    excludes = " ".join(f"--exclude {shlex.quote(pattern)}" for pattern in COPY_EXCLUDE)
    return (f"mkdir -p {local_shell} && rsync -av {excludes} "
            f"{host}:{shlex.quote(str(remote_dir).rstrip('/') + '/')} {local_shell}/")
