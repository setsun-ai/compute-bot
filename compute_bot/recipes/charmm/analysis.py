"""Data of the finished steps, the same for all three engines: energies (gmx energy / OpenMM StateDataReporter /
NAMD ENERGY lines), trajectories with MDAnalysis (PSF + xtc/dcd: RMSD of the backbone, Rg, RMSF of Cα, DSSP), in the
shapes md's plotting code takes, so the plots look exactly like the manual `md` run."""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..md import analysis as MA
from ..md.gmxtools import RecipeError, fresh, parse_em_log, parse_performance, read_xvg
from . import package as K
from .engines import done

KCAL = 4.184
AMU_KG = 1.66053907e-27
MAX_FRAMES = 2000


# --- energies -------------------------------------------------------------------------------------------------------

def _float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return math.nan


def parse_openmm_log(text: str) -> dict:
    """StateDataReporter columns (Step, Potential, Temperature, Volume, Density, Speed) + energies around the
    minimisation that openmm_run.py prints."""
    out: dict = {}
    header, sep, rows = None, "\t", []
    for line in text.splitlines():
        if line.startswith('#"'):
            sep = "\t" if "\t" in line else ","
            header = [h.strip().strip('"') for h in line[1:].split(sep)]
            continue
        if header is None:
            continue
        vals = [v.strip() for v in line.split(sep)]
        if len(vals) != len(header):
            continue
        rows.append([_float(v) for v in vals])
    names = {"Step": "Step", "Time (ps)": "time_abs", "Potential Energy (kJ/mole)": "Potential",
             "Temperature (K)": "Temperature", "Box Volume (nm^3)": "Volume", "Density (g/mL)": "Density",
             "Speed (ns/day)": "speed"}
    if header and rows:
        data = np.array(rows, dtype=float)
        for i, h in enumerate(header):
            if h in names:
                out[names[h]] = data[:, i] * (1000.0 if h == "Density (g/mL)" else 1.0)
    energies = re.findall(r"^\s*([-+\d.eE]+)\s*kJ/mol\s*$", text, re.M)
    if energies and "Initial system energy" in text:
        out["e_initial"] = float(energies[0])
        if "Energy minimization" in text and len(energies) > 1:
            out["e_min"] = float(energies[1])
    return out


def parse_namd_log(text: str) -> dict:
    """ENERGY lines under the last ETITLE, as arrays keyed by the ETITLE names (TS, POTENTIAL, TEMP, ...)."""
    header, rows = None, []
    for line in text.splitlines():
        if line.startswith("ETITLE:"):
            header = line.split()[1:]
        elif line.startswith("ENERGY:") and header:
            vals = line.split()[1:]
            if len(vals) == len(header):
                try:
                    rows.append([float(v) for v in vals])
                except ValueError:
                    continue
    if not rows:
        return {}
    data = np.array(rows)
    return {h: data[:, i] for i, h in enumerate(header)}


def namd_nsday(text: str, dt_ps: float) -> float | None:
    found = re.findall(r"^TIMING:.*?Wall:\s+[\d.]+,\s+([\d.eE+-]+)/step", text, re.M)
    if not found or not dt_ps or float(found[-1]) <= 0:
        return None
    return dt_ps * 1e-3 * 86400 / float(found[-1])


def step_energy(engine: str, edir: Path, s: K.Step, env: dict, mass_amu: float | None, warn: list) -> dict:
    """Series of one dynamic step (time from the start of the step, ps), terms named as md's plots expect."""
    if engine == "gromacs":
        if not (edir / f"{s.name}.edr").exists():
            return {}
        try:
            return MA.stage_energy(edir, edir, MA.Gmx(edir, env), s.name)
        except (RecipeError, OSError, ValueError, IndexError) as exc:
            warn.append(f"energia {s.name}: {str(exc)[:120]}")
            return {}
    text = K._read(edir / (s.stdout or f"{s.name}.out")) or ""
    if engine == "openmm":
        d = parse_openmm_log(text)
        if "Step" not in d:
            return {}
        out = {"time": d["Step"] * s.dt}
        for k in ("Potential", "Temperature", "Volume", "Density"):
            if k in d:
                out[k] = d[k]
        return out
    d = parse_namd_log(text)
    if "TS" not in d:
        return {}
    mini = int(s.settings.get("min_steps") or 0)
    first = d["TS"][0] + mini  # NAMD numbers MD steps after the minimisation steps
    keep = d["TS"] > first if mini else np.ones(len(d["TS"]), dtype=bool)
    out = {"time": (d["TS"][keep] - first) * s.dt}
    if "POTENTIAL" in d:
        out["Potential"] = d["POTENTIAL"][keep] * KCAL
    if "TEMP" in d:
        out["Temperature"] = d["TEMP"][keep]
    if s.pcoupl and "PRESSURE" in d:
        out["Pressure"] = d["PRESSURE"][keep]
    if s.pcoupl and "VOLUME" in d:
        out["Volume"] = d["VOLUME"][keep] / 1000.0
        if mass_amu:
            out["Density"] = mass_amu * AMU_KG / (d["VOLUME"][keep] * 1e-30)
    return out


def minimisation(engine: str, edir: Path, steps: list, env: dict, warn: list) -> dict:
    """md's em dict (series for the plot, converged/steps/epot/fmax) or energies before/after (OpenMM)."""
    if engine == "gromacs":
        s = next((x for x in steps if x.kind == "min" and done(edir, x)), None)
        if s is None:
            return {}
        em = parse_em_log(K._read(edir / f"{s.name}.log") or "")
        em["name"] = s.name
        if not (edir / f"{s.name}.edr").exists():
            return em
        try:
            gmx = MA.Gmx(edir, env)
            xvg_path = edir / f"{s.name}_potential.xvg"
            if not fresh(xvg_path, edir / f"{s.name}.edr"):
                gmx(["energy", "-f", f"{s.name}.edr", "-o", xvg_path.name], stdin="Potential\n\n")
            xvg = read_xvg(xvg_path)
            em["series"] = (xvg.x, xvg.data[:, 1])
        except (RecipeError, OSError, ValueError, IndexError) as exc:
            warn.append(f"wykres minimalizacji: {str(exc)[:120]}")
        return em
    s = next((x for x in steps if x.settings.get("min_steps") and done(edir, x)), None)
    if s is None:
        return {}
    text = K._read(edir / (s.stdout or f"{s.name}.out")) or ""
    em = {"name": s.name, "steps": s.settings.get("min_steps"), "converged": None, "fmax": None}
    if engine == "openmm":
        d = parse_openmm_log(text)
        em["epot"], em["epot_start"] = d.get("e_min"), d.get("e_initial")
        return em
    d = parse_namd_log(text)
    if "TS" in d and "POTENTIAL" in d:
        mini = int(s.settings["min_steps"])
        sel = d["TS"] <= d["TS"][0] + mini
        em["series"] = (d["TS"][sel] - d["TS"][0], d["POTENTIAL"][sel] * KCAL)
        em["epot"], em["epot_start"] = float(em["series"][1][-1]), float(em["series"][1][0])
    return em


def performance(engine: str, edir: Path, s: K.Step) -> float | None:
    if engine == "gromacs":
        return parse_performance(K._read(edir / f"{s.name}.log") or "")
    text = K._read(edir / (s.stdout or f"{s.name}.out")) or ""
    if engine == "openmm":
        sp = parse_openmm_log(text).get("speed")
        sp = sp[np.isfinite(sp)] if sp is not None else []
        return float(sp[-1]) if len(sp) else None
    return namd_nsday(text, s.dt)


# --- stages as md's plots see them ----------------------------------------------------------------------------------

def group_name(s: K.Step) -> str:
    if s.kind == "prod":
        return "md"
    return "npt" if s.pcoupl else "nvt"


def groups(steps: list) -> tuple[list, dict, dict]:
    """md-style stages (nvt/npt/md) made of consecutive package steps; start time of every dynamic step and group."""
    t, step_start, out = 0.0, {}, []
    for s in steps:
        if not s.dynamic:
            continue
        step_start[s.name] = t
        g = group_name(s)
        if not out or out[-1]["name"] != g:
            out.append({"name": g, "steps": [], "start": t})
        out[-1]["steps"].append(s)
        t += s.ps
    # md's plots know three colours: a second NVT/NPT block joins the first one with that name
    merged = {}
    for g in out:
        if g["name"] in merged:
            merged[g["name"]]["steps"] += g["steps"]
        else:
            merged[g["name"]] = g
    plan = []
    for g in merged.values():
        names = [s.name for s in g["steps"]]
        label = names[0] if len(names) == 1 else f"{names[0]}…{names[-1]}"
        title = {"md": "produkcja", "nvt": "równowagowanie NVT", "npt": "równowagowanie NPT"}[g["name"]]
        st = g["steps"]
        # the attributes md's Stage has, so md's plotting code can take these
        plan.append(SimpleNamespace(name=g["name"], title=f"{title} ({label})", steps=st, start=g["start"],
                                    pcoupl=st[-1].pcoupl, posres=st[0].posres, temp=st[-1].temp, dynamic=True,
                                    nsteps=sum(s.nsteps for s in st), ps=sum(s.ps for s in st)))
    return plan, step_start, {g.name: g.start for g in plan}


def concat(series: list[tuple[float, dict]]) -> dict:
    """Join step series into one stage series; time shifted by each step's start within the stage."""
    keys = set.intersection(*(set(d) for _, d in series)) if series else set()
    out = {}
    for k in keys:
        if k == "time":
            out[k] = np.concatenate([d["time"] + off for off, d in series])
        else:
            out[k] = np.concatenate([d[k] for _, d in series])
    return out


# --- trajectories ---------------------------------------------------------------------------------------------------

def topology_files(edir: Path, engine: str) -> tuple[Path | None, Path | None]:
    """(topology with bonds if possible, reference coordinates)."""
    psf = next(iter(sorted(edir.glob("step*_input.psf"))), None)
    ref = None
    if engine == "gromacs":
        ref = next((edir / f for f in ("step4.0_minimization.gro",) if (edir / f).exists()), None)
        ref = ref or next(iter(sorted(edir.glob("step*_input.gro"))), None)
    ref = ref or next(iter(sorted(edir.glob("step*_input.pdb"))), None)
    top = psf or (next(iter(sorted(edir.glob("step*_input.gro"))), None) if engine == "gromacs" else ref)
    return top, ref


def trajectory_analysis(edir: Path, engine: str, steps: list, step_start: dict, plan: list, warn: list) -> dict:
    """RMSD (per stage), Rg, RMSF and DSSP (production) with MDAnalysis; {} with a warning when not possible."""
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import align, rms
        from MDAnalysis.transformations import unwrap
    except ImportError:
        warn.append("brak MDAnalysis – bez RMSD/RMSF/Rg/DSSP")
        return {}
    import warnings

    warnings.filterwarnings("ignore", module="MDAnalysis")
    top, ref_path = topology_files(edir, engine)
    trajs = [(s, edir / s.traj) for s in steps if s.dynamic and s.traj and done(edir, s)
             and (edir / s.traj).exists() and (edir / s.traj).stat().st_size > 0]
    if top is None or not trajs:
        if trajs:
            warn.append("brak pliku topologii (step3_input.psf) – bez RMSD/RMSF")
        return {}
    try:
        u = mda.Universe(str(top), str(trajs[0][1]))
        ref = mda.Universe(str(top), str(ref_path)) if ref_path and ref_path != top else mda.Universe(str(top))
    except Exception as exc:  # MDAnalysis raises many kinds of errors for unreadable files
        warn.append(f"MDAnalysis nie czyta {top.name}/{trajs[0][1].name}: {str(exc)[:120]}")
        return {}
    protein = u.select_atoms("protein")
    if len(protein) == 0:
        warn.append("w układzie nie ma białka – bez RMSD/RMSF/Rg/DSSP")
        return {}
    bb, ca = u.select_atoms("protein and backbone"), u.select_atoms("protein and name CA")
    ref_bb = ref.select_atoms("protein and backbone")
    if len(ref_bb) != len(bb):
        warn.append("struktura odniesienia nie pasuje do trajektorii – bez RMSD")
        return {}
    has_bonds = hasattr(u.atoms, "bonds") and len(u.atoms.bonds) > 0
    if not has_bonds:
        try:
            protein.guess_bonds()
            has_bonds = True
        except Exception:  # noqa: BLE001 – guessing needs element radii; without them no unwrapping
            warn.append("bez wiązań w topologii – białko nie jest scalane przez granice pudełka")
    ref_c = ref_bb.positions - ref_bb.center_of_mass()
    total = 0
    for _, path in trajs:
        try:
            u.load_new(str(path))
            total += len(u.trajectory)
        except Exception as exc:  # noqa: BLE001
            warn.append(f"{path.name}: {str(exc)[:100]}")
    stride = max(1, math.ceil(total / MAX_FRAMES))
    group_of = {s.name: g.name for g in plan for s in g.steps}
    rmsd: dict = {}
    rg_t, rg_y, ca_xyz, prod_t, prod_files = [], [], [], [], []
    for s, path in trajs:
        try:
            u.load_new(str(path))
        except Exception:  # noqa: BLE001 – already reported above
            continue
        if has_bonds:
            u.trajectory.add_transformations(unwrap(protein))
        n = len(u.trajectory)
        every = (s.settings.get("traj_every") or 0) * s.dt
        if not every and n:
            every = s.ps / n
        g = group_of[s.name]
        g_start = next(x.start for x in plan if x.name == g)
        xs, ys = rmsd.setdefault(g, ([], []))
        for i, ts in enumerate(u.trajectory[::stride]):
            k = i * stride
            t_local = s.ps - (n - 1 - k) * every  # the last frame is the end of the step
            t = step_start[s.name] - g_start + t_local
            mob = bb.positions
            com = bb.center_of_mass()
            xs.append(t)
            ys.append(rms.rmsd(mob, ref_bb.positions, center=True, superposition=True) / 10.0)
            if s.kind == "prod":
                R, _ = align.rotation_matrix(mob - com, ref_c)
                ca_xyz.append((ca.positions - com) @ np.asarray(R).T)
                rg_t.append(t)
                rg_y.append(protein.radius_of_gyration() / 10.0)
                prod_t.append(t)
            _ = ts
        if s.kind == "prod":
            prod_files.append((s, path))
    res: dict = {"rmsd": {k: (np.array(x), np.array(y)) for k, (x, y) in rmsd.items() if x}}
    if rg_t:
        res["rg"] = (np.array(rg_t), np.array(rg_y))
        xyz = np.array(ca_xyz)
        rmsf = np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0)) / 10.0
        res["rmsf"] = ([f"{r.resname}{r.resid}" for r in ca.residues], rmsf)
        res["dssp"] = dssp(u, protein, prod_files, stride, np.array(prod_t), warn, has_bonds)
        if res["dssp"] is None:
            del res["dssp"]
    res["reference"] = ref_path.name if ref_path else top.name
    return res


def dssp(u, protein, prod_files, stride, times, warn, has_bonds):
    try:
        from MDAnalysis.analysis.dssp import DSSP
        from MDAnalysis.transformations import unwrap
    except ImportError:
        warn.append("DSSP pominięte (MDAnalysis < 2.8)")
        return None
    lines = []
    try:
        for _, path in prod_files:
            u.load_new(str(path))
            if has_bonds:
                u.trajectory.add_transformations(unwrap(protein))
            run = DSSP(protein).run(step=stride)
            lines += ["".join("~" if c == "-" else c for c in row) for row in run.results.dssp]
    except Exception as exc:  # noqa: BLE001 – e.g. fewer than 6 residues, missing backbone atoms
        warn.append(f"DSSP pominięte ({str(exc)[:120]})")
        return None
    if not lines:
        return None
    t = times if len(times) == len(lines) else np.arange(len(lines), dtype=float)
    return t, lines


# --- everything -----------------------------------------------------------------------------------------------------

def collect(engine: str, edir: Path, steps: list, env: dict, info: dict) -> dict:
    warn: list = []
    finished = [s for s in steps if done(edir, s)]
    plan, step_start, offsets = groups(steps)
    energies, stages = {}, []
    for g in plan:
        series = []
        for s in g.steps:
            if s in finished:
                e = step_energy(engine, edir, s, env, info.get("mass_amu"), warn)
                if e and len(e.get("time", [])):
                    series.append((step_start[s.name] - g.start, e))
        if series:
            energies[g.name] = concat(series)
            stages.append(g)
    em = minimisation(engine, edir, steps, env, warn)
    traj = trajectory_analysis(edir, engine, steps, step_start, plan, warn)
    perf = {s.name: performance(engine, edir, s) for s in finished if s.dynamic}
    return {"plan": plan, "stages": stages, "offsets": offsets, "energies": energies, "em": em,
            "rmsd": traj.pop("rmsd", {}), "res": traj, "perf": perf, "warnings": warn,
            "finished": [s.name for s in finished]}
