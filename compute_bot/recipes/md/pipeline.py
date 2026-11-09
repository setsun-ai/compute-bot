"""The `md` recipe end to end. Every step is idempotent: a step whose final output exists is skipped, outputs are
written under a temporary name and renamed when complete, and every mdrun goes through md_step (resume from .cpt)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import mdp as M
from .gmxtools import (
    Progress,
    RecipeError,
    gmx_env,
    parse_gro,
    parse_molecules,
    parse_net_charge,
    parse_performance,
    run_gmx,
    run_md_step,
)

GDIR = "gromacs"
OUT = "wyniki"


def progress(text: str):
    print(f"PROGRESS: {text}", flush=True)


def finished(gdir: Path, name: str) -> bool:
    log = gdir / f"{name}.log"
    return (gdir / f"{name}.gro").exists() and log.exists() and "Finished mdrun" in log.read_text(errors="replace")


# --- structure ------------------------------------------------------------------------------------------------------

def prepare_input(work: Path, gdir: Path, p: dict) -> dict:
    """Check the structure and write gromacs/clean.pdb (once); the report is kept in gromacs/structure.json."""
    from . import structure as S

    cache = gdir / "structure.json"
    if cache.exists() and (gdir / "clean.pdb").exists():
        return json.loads(cache.read_text())
    progress("sprawdzam strukturę")
    src = S.find_input(work, p["pdb"])
    info = S.inspect_structure(src, p["model"])
    tmp = gdir / "clean_tmp.pdb"
    info["notes"] = S.prepare_structure(src, tmp, p["model"], p["chains"])
    info["report"] = S.describe_structure(info, p["model"])
    info["code"] = p["pdb"].upper() or src.stem
    tmp.rename(gdir / "clean.pdb")
    cache.write_text(json.dumps(info, ensure_ascii=False, indent=1))
    return info


# --- topology, box, water, ions -------------------------------------------------------------------------------------

def build_system(gdir: Path, p: dict, env: dict) -> dict:
    """pdb2gmx → editconf → solvate → genion; returns the system table (cached in system.json)."""
    log = gdir / "build.log"
    water = M.resolve_water(p["ff"], p["water"])
    if not (gdir / "processed.gro").exists():
        progress("przygotowanie 1/4: topologia (pdb2gmx)")
        out = run_gmx(["pdb2gmx", "-f", "clean.pdb", "-o", "processed_tmp.gro", "-p", "topol_pdb2gmx.top",
                       "-i", "posre.itp", "-ff", p["ff"], "-water", water, "-ignh"], gdir, env, log=log)
        (gdir / "pdb2gmx.out").write_text(out.stdout + out.stderr)
        os.replace(gdir / "processed_tmp.gro", gdir / "processed.gro")
    if not (gdir / "box.gro").exists():
        progress("przygotowanie 2/4: pudełko (editconf)")
        run_gmx(["editconf", "-f", "processed.gro", "-o", "box_tmp.gro", "-c", "-d", f"{p['d']:g}", "-bt", p["box"]],
                gdir, env, log=log)
        os.replace(gdir / "box_tmp.gro", gdir / "box.gro")
    if not (gdir / "solv.gro").exists():
        progress("przygotowanie 3/4: woda (solvate)")
        shutil.copyfile(gdir / "topol_pdb2gmx.top", gdir / "topol_solv.top")  # solvate edits the topology in place
        run_gmx(["solvate", "-cp", "box.gro", "-cs", M.solvent_box(water), "-o", "solv_tmp.gro",
                 "-p", "topol_solv.top"], gdir, env, log=log)
        os.replace(gdir / "solv_tmp.gro", gdir / "solv.gro")
    if not (gdir / "ions.gro").exists() or not (gdir / "system.json").exists():
        progress("przygotowanie 4/4: jony (genion)")
        shutil.copyfile(gdir / "topol_solv.top", gdir / "topol.top")
        (gdir / "ions.mdp").write_text(M.ions_mdp(p["ff"]))
        g = run_gmx(["grompp", "-f", "ions.mdp", "-c", "solv.gro", "-p", "topol.top", "-o", "ions.tpr",
                     "-maxwarn", "1"], gdir, env, log=log)  # the one warning: net charge before neutralising
        charge = parse_net_charge(g.stdout + g.stderr)
        args = ["genion", "-s", "ions.tpr", "-o", "ions_tmp.gro", "-p", "topol.top", "-pname", "NA", "-nname", "CL",
                "-neutral", "-conc", f"{p['salt']:g}"]
        if p["seed"] != -1:
            args += ["-seed", str(p["seed"])]
        run_gmx(args, gdir, env, stdin="SOL\n", log=log)
        system = system_table(gdir, p, water, charge, "ions_tmp.gro")
        (gdir / "system.json").write_text(json.dumps(system, indent=1))
        os.replace(gdir / "ions_tmp.gro", gdir / "ions.gro")
    return json.loads((gdir / "system.json").read_text())


def system_table(gdir: Path, p: dict, water: str, charge: float, gro: str) -> dict:
    mols = parse_molecules((gdir / "topol.top").read_text())
    count = {}
    for name, n in mols:
        count[name] = count.get(name, 0) + n
    box = parse_gro((gdir / gro).read_text())
    na, cl = count.get("NA", 0), count.get("CL", 0)
    protein = sum(n for name, n in mols if name not in ("SOL", "NA", "CL"))
    return {"ff": p["ff"], "water": water, "box_type": p["box"], "d": p["d"], "box_vectors": box["vectors"],
            "box_lengths": box["lengths"], "volume": box["volume"], "atoms": box["atoms"],
            "waters": count.get("SOL", 0), "na": na, "cl": cl, "protein_molecules": protein,
            "charge_protein": round(charge, 3), "charge_total": round(charge + na - cl, 3), "salt": p["salt"]}


# --- MD stages ------------------------------------------------------------------------------------------------------

def write_mdps(gdir: Path, plan, p: dict):
    for s in plan:
        path = gdir / f"{s.name}.mdp"
        if path.exists() and (gdir / f"{s.name}.tpr").exists():
            continue  # a stage already prepared keeps its .mdp (it must match the .tpr)
        path.write_text(M.make_mdp(s, p["ff"], p["seed"]))


def run_stages(gdir: Path, plan, env: dict):
    prog = Progress(plan)
    for s in plan:
        if s.dynamic:
            for prev in plan:
                if prev.dynamic and (gdir / f"{prev.name}.log").exists():
                    prog.set_rate_from_nsday(parse_performance((gdir / f"{prev.name}.log").read_text(errors="replace")))
        run_md_step(s, gdir, env, prog)
        if not finished(gdir, s.name):
            raise RecipeError(f"etap {s.title} ({s.name}) nie skończył się (brak {s.name}.gro) – zobacz {s.name}.log")


# --- main -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m compute_bot.recipes.md")
    ap.add_argument("--analiza", "--analysis", action="store_true", dest="analysis_only",
                    help="tylko analiza istniejących wyników (bez liczenia)")
    args = ap.parse_args(argv)
    work = Path.cwd()
    try:
        p = json.loads((work / "params.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"ERROR: nie mogę przeczytać params.json ({exc})", file=sys.stderr)
        return 2
    threads = int(os.environ.get("COMPUTE_THREADS") or os.cpu_count() or 1)
    env = gmx_env(threads)
    gdir, out = work / GDIR, work / OUT
    gdir.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    plan = M.plan_stages(p)
    try:
        if not shutil.which("gmx", path=env.get("PATH")):
            raise RecipeError("brak programu gmx (GROMACS) w środowisku – uruchom przepis env")
        if args.analysis_only:
            if not (gdir / "system.json").exists():
                raise RecipeError("nie ma jeszcze zbudowanego układu do analizy")
            struct = json.loads((gdir / "structure.json").read_text())
            system = json.loads((gdir / "system.json").read_text())
        else:
            struct = prepare_input(work, gdir, p)
            system = build_system(gdir, p, env)
            write_mdps(gdir, plan, p)
            run_stages(gdir, plan, env)
        from . import analysis

        progress("analiza")
        analysis.run(work, gdir, out, plan, p, struct, system, env)
    except RecipeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    progress("gotowe")
    return 0
