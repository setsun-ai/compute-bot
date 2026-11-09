"""The `charmm` recipe end to end: find the package, read its README into steps, write the protocol table, run the
steps (resumably), analyse whatever is finished, write summary.txt and highlights.txt."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from ..md import analysis as MA
from ..md.gmxtools import RecipeError, gmx_env
from . import analysis as A
from . import engines as E
from . import package as K
from . import report as R

OUT = "wyniki"


def progress(text: str):
    print(f"PROGRESS: {text}", flush=True)


def available_engines(env: dict) -> dict:
    return {"gromacs": shutil.which("gmx", path=env.get("PATH")) is not None,
            "openmm": importlib.util.find_spec("openmm") is not None,
            "namd": K.namd_binary() is not None}


def prepare(work: Path, p: dict, env: dict, analysis_only: bool = False):
    found = K.find_engines(work)
    available = {e: True for e in K.ENGINES} if analysis_only else available_engines(env)
    engine = K.choose_engine(found, p.get("engine", "auto"), available)
    edir = found[engine]
    root = K.package_root(edir, work)
    readme = K._read(edir / "README")
    steps, notes = K.plan(engine, readme, lambda f: K._read(edir / f), p,
                          K.ref_constants(edir) if engine == "namd" else None)
    info = K.describe_system(root, edir, engine, readme or "")
    others = [f"{e}/" for e in found if e != engine]
    return engine, edir, steps, notes, info, others


def write_outputs(work: Path, out: Path, engine: str, steps, data: dict, info: dict, notes, others, p: dict):
    import matplotlib

    matplotlib.use("Agg")
    plots = MA.make_plots(out, data["plan"], data["stages"], data["offsets"], data["energies"], data["rmsd"],
                          data["res"], data["em"])
    for g in data["stages"]:
        e = data["energies"][g.name]
        terms = [k for k in ("Temperature", "Potential", "Pressure", "Density", "Volume") if k in e]
        MA.write_csv(out / f"energia_{g.name}.csv", ["czas_ps"] + terms,
                     [[f"{t + g.start:.3f}"] + [f"{e[k][i]:.4f}" for k in terms] for i, t in enumerate(e["time"])])
    if data["rmsd"]:
        MA.write_csv(out / "rmsd.csv", ["etap", "czas_ps", "rmsd_nm"],
                     [[n, f"{x + data['offsets'][n]:.3f}", f"{y:.4f}"]
                      for n, (xs, ys) in data["rmsd"].items() for x, y in zip(xs, ys, strict=False)])
    res = data["res"]
    if "rg" in res:
        MA.write_csv(out / "rg.csv", ["czas_ps", "rg_nm"],
                     [[f"{a + data['offsets']['md']:.3f}", f"{b:.4f}"] for a, b in zip(*res["rg"], strict=False)])
    if "rmsf" in res:
        MA.write_csv(out / "rmsf.csv", ["reszta", "rmsf_nm"],
                     [[lab, f"{v:.4f}"] for lab, v in zip(*res["rmsf"], strict=False)])
    if "dssp" in res:
        h, b = MA.dssp_fractions(res["dssp"][1])
        MA.write_csv(out / "dssp.csv", ["czas_ps", "dssp", "helisa", "beta"],
                     [[f"{t + data['offsets']['md']:.3f}", s, f"{x:.3f}", f"{y:.3f}"]
                      for t, s, x, y in zip(res["dssp"][0], res["dssp"][1], h, b, strict=False)])
    if data["em"].get("series") is not None:
        MA.write_csv(out / "em_energia.csv", ["krok", "Epot_kJ_mol"],
                     [[int(a), f"{b:.2f}"] for a, b in zip(*data["em"]["series"], strict=False)])
    by_name = {s.name: s for s in steps}
    MA.write_csv(out / "wydajnosc.csv", ["krok", "ps", "ns_na_dobe"],
                 [[n, f"{by_name[n].ps:g}", "" if x is None else f"{x:.2f}"] for n, x in data["perf"].items()])
    text = R.summary(engine, info, steps, data, notes, others, p)
    (work / "summary.txt").write_text(text)
    (work / "highlights.txt").write_text("".join(f"{OUT}/{name}\n" for name in plots[:4]))
    (out / "analiza.json").write_text(json.dumps({"engine": engine, "finished": data["finished"],
                                                  "warnings": data["warnings"], "notes": notes,
                                                  "performance_ns_day": data["perf"]}, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m compute_bot.recipes.charmm")
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
    env["OPENMM_CPU_THREADS"] = str(threads)
    out = work / OUT
    out.mkdir(exist_ok=True)
    try:
        engine, edir, steps, notes, info, others = prepare(work, p, env, args.analysis_only)
        MA.write_csv(out / "protokol_charmm.csv", ["etap", "parametr", "wartosc", "zrodlo"],
                     R.protocol_table(engine, edir.name, info, steps, others))
        prod = [s for s in steps if s.kind == "prod"]
        progress(f"paczka CHARMM-GUI {info.get('version', '')}: {K.ENGINE_TITLES[engine]}, {info['ff']}, "
                 f"{len(steps)} kroków (produkcja {sum(s.ps for s in prod) / 1000:g} ns)")
        if not args.analysis_only:
            E.run_all(steps, edir, env, threads, K.namd_binary())
        progress("analiza")
        data = A.collect(engine, edir, steps, env, info)
        write_outputs(work, out, engine, steps, data, info, notes, others, p)
    except RecipeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    progress("gotowe")
    return 0
