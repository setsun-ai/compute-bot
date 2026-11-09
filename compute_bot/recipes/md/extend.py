"""extend_ns: continue the production of a finished job. The bot hard-links the parent's whole work/ tree into the new
job, so nothing that exists may be written in place: every extension is a new segment md_ext<k> made of new files only
(GROMACS: convert-tpr + mdrun -cpi -noappend; OpenMM: new DCD parts from the last state)."""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from . import mdp as M
from .gmxtools import Progress, RecipeError, parse_performance, run_gmx, run_streaming, write_text


def segment_done(d: Path, name: str) -> bool:
    if name == "md":
        log = d / "md.log"
        return (d / "md.done.json").exists() or (
            (d / "md.gro").exists() and log.exists() and "Finished mdrun" in log.read_text(errors="replace"))
    return (d / f"{name}.done.json").exists()


def production_segments(d: Path) -> list[str]:
    """md, then every finished extension in order (inherited from parent jobs or made here)."""
    out, k = ["md"], 1
    while (d / f"md_ext{k}.done.json").exists():
        out.append(f"md_ext{k}")
        k += 1
    return out


def claim_segment(d: Path, work: Path) -> tuple[str, int]:
    """The segment this job extends into. Recorded with this job's path, so a rerun picks the same one and a child
    job (which inherits the record through the hard links) takes the next one."""
    record = d / "extend.json"
    me = str(Path(work).resolve())
    if record.exists():
        claim = json.loads(record.read_text())
        if claim.get("work") == me:
            return claim["segment"], claim["index"]
    k = len(production_segments(d))
    seg = f"md_ext{k}"
    for left in [*d.glob(f"{seg}.*"), *d.glob(f"{seg}_*")]:  # an unfinished attempt inherited from the parent
        left.unlink()
    write_text(record, json.dumps({"work": me, "segment": seg, "index": k}))
    return seg, k


def segment_end_step(d: Path, name: str) -> int:
    done = d / f"{name}.done.json"
    if done.exists():
        info = json.loads(done.read_text())
        if "end_step" in info:
            return int(info["end_step"])
    return int(M.parse_mdp((d / f"{name}.mdp").read_text())["nsteps"])


def production_settings(d: Path) -> tuple[float, bool]:
    """(temperature, pressure coupling) of the production as it was run."""
    mdp = M.parse_mdp((d / "md.mdp").read_text())
    return float(mdp["ref_t"].split()[0]), mdp.get("pcoupl", "no").lower() not in ("no", "")


def segment_files(d: Path, seg: str, ext: str) -> list[Path]:
    if seg == "md":
        return [d / f"md.{ext}"]
    return sorted(d.glob(f"{seg}.part*.{ext}"))


def extend_gromacs(work: Path, gdir: Path, p: dict, env: dict) -> str:
    if not segment_done(gdir, "md") or not (gdir / "md.cpt").exists():
        raise RecipeError("extend_ns: w zadaniu nie ma skończonej produkcji GROMACS (md.gro, md.cpt) – przedłużyć "
                          "można tylko skończone zadanie")
    seg, k = claim_segment(gdir, work)
    if segment_done(gdir, seg):
        print(f"{seg} już policzony, pomijam", flush=True)
        return seg
    prev = "md" if k == 1 else f"md_ext{k - 1}"
    ext_ps = p["extend_ns"] * 1000
    first = segment_end_step(gdir, prev)
    temp, pcoupl = production_settings(gdir)
    stage = M.Stage(seg, f"przedłużenie {k}", M.nsteps(ext_ps), temp, pcoupl=pcoupl, first_step=first)
    log = gdir / "extend.log"
    if not (gdir / f"{seg}.tpr").exists():
        run_gmx(["convert-tpr", "-s", f"{prev}.tpr", "-extend", f"{ext_ps:g}", "-o", f"{seg}_tmp.tpr"], gdir, env,
                log=log)
        os.replace(gdir / f"{seg}_tmp.tpr", gdir / f"{seg}.tpr")
    if (gdir / f"{seg}.cpt").exists():
        cpi = f"{seg}.cpt"
        print(f"wznawiam {seg} z checkpointu", flush=True)
    else:
        cpi = f"{prev}.cpt"
    threads = env.get("COMPUTE_THREADS") or str(os.cpu_count() or 1)
    # -noappend: the outputs of the run being extended (maybe the parent's files) stay untouched
    cmd = ["gmx", "mdrun", "-v", "-s", f"{seg}.tpr", "-deffnm", seg, "-cpi", cpi, "-noappend", "-nt", threads,
           "-cpt", "5", *shlex.split(env.get("MD_MDRUN_ARGS", ""))]
    prog = Progress([stage])
    prev_log = sorted(gdir.glob(f"{prev}.part*.log")) or [gdir / f"{prev}.log"]
    if prev_log[-1].exists():
        prog.set_rate_from_nsday(parse_performance(prev_log[-1].read_text(errors="replace")))
    run_streaming(cmd, stage, gdir, env, prog)
    parts = segment_files(gdir, seg, "log")
    if not parts or "Finished mdrun" not in parts[-1].read_text(errors="replace"):
        raise RecipeError(f"przedłużenie {seg} nie skończyło się – zobacz gromacs/{seg}.out")
    write_text(gdir / f"{seg}.done.json", json.dumps({"segment": seg, "prev": prev, "first_step": first,
                                                       "end_step": first + stage.nsteps, "ps": ext_ps}))
    return seg
