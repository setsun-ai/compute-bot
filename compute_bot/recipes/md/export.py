"""gromacs/run.sh – the MD stages of a job as an editable script – and the input files that go with it.
Standard library only: the bot (system Python, no numpy) imports script_files() to pack the folder for the owner."""
from __future__ import annotations

import shlex
from pathlib import Path

SCRIPT = "run.sh"
INPUT_OPTIONS = ("-f", "-c", "-r", "-t", "-p", "-n")  # grompp options that read a file

HEADER = """\
#!/bin/bash
# Etapy MD tego zadania (przepis md), dokładnie tak, jak policzył je bot.
# Chcesz coś zmienić? Edytuj pliki .mdp obok tego skryptu (albo linie md_step poniżej),
# potem wyślij ten folder jako zip do bota – policzy go jak własny skrypt (run.sh).
# Struktura startowa: {start} (białko w wodzie z jonami, po pdb2gmx/editconf/solvate/genion);
# topologia: topol.top + *.itp.
# md_step NAZWA opcje-grompp = gmx grompp -o NAZWA.tpr, potem gmx mdrun -deffnm NAZWA; po restarcie
# skończony etap jest pomijany, przerwany wznawia się z NAZWA.cpt. MD_MAXWARN = dozwolone ostrzeżenia grompp.
set -euo pipefail
cd "$(dirname "$0")"
export GMX_MAXBACKUP=-1
"""


def run_script(stages, maxwarn: str = "0") -> str:
    """The script for [(name, grompp_args), ...] in run order (pure)."""
    stages = list(stages)
    start = next((a[i + 1] for _, a in stages for i, opt in enumerate(a[:-1]) if opt == "-c"), "?")
    lines = [f"MD_MAXWARN={shlex.quote(str(maxwarn))} md_step {shlex.join([name, *args])}\n" for name, args in stages]
    return HEADER.format(start=start) + "\n" + "".join(lines)


def parse_script(text: str) -> list[tuple[str, list[str]]]:
    """The md_step lines of a run.sh: [(name, grompp_args), ...]; other lines are ignored."""
    out = []
    for line in text.splitlines():
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        while words and "=" in words[0] and not words[0].startswith("-"):
            words = words[1:]  # VAR=value prefixes
        if len(words) >= 2 and words[0] == "md_step":
            out.append((words[1], words[2:]))
    return out


def script_inputs(stages) -> list[str]:
    """Files the stages read that no earlier stage writes (every output of stage NAME is NAME.*)."""
    made, need = set(), []
    for name, args in stages:
        for i in range(len(args) - 1):  # no zip(strict=...): the bot's Python may be older than 3.10
            opt, value = args[i], args[i + 1]
            if opt in INPUT_OPTIONS and value.split(".", 1)[0] not in made and value not in need:
                need.append(value)
        made.add(name)
    return need


def script_files(gromacs_dir) -> list[str]:
    """What to pack so run.sh runs elsewhere: run.sh, the start structure, topol.top, every *.itp and the .mdp files
    (names relative to gromacs_dir, sorted). No trajectories, checkpoints or other stage outputs. [] without run.sh."""
    d = Path(gromacs_dir)
    script = d / SCRIPT
    if not script.is_file():
        return []
    names = {SCRIPT, *script_inputs(parse_script(script.read_text(errors="replace")))}
    names |= {p.name for p in d.glob("*.itp")}
    return sorted(n for n in names if (d / n).is_file())
