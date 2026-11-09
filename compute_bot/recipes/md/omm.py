"""engine=openmm: the same protocol as the GROMACS path (EM, NVT and NPT with heavy-atom restraints, production) with
OpenMM on the CPU, and the same analysis data read with MDAnalysis.

Resuming: each stage writes a state checkpoint (XML, atomic) every MD_CHECKPOINT_S seconds of wall time (300) and on
SIGTERM. A restart loads it and writes a new DCD/CSV part; frames a crashed part wrote after its last checkpoint are
dropped when the parts are read back (kept_frames). A finished stage has NAME.done.json and is skipped."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import mdp as M
from .extend import claim_segment, production_segments, segment_done, segment_end_step
from .gmxtools import Progress, RecipeError, fresh_output, write_text

SOLVENT = {"HOH", "NA", "CL"}
# force field -> (protein xml, family for the water files)
FF_XML = {
    "amber99sb-ildn": ("amber99sbildn.xml", "amber"),
    "amber99sb": ("amber99sb.xml", "amber"),
    "amber03": ("amber03.xml", "amber"),
    "amber14sb": ("amber14/protein.ff14SB.xml", "amber14"),
    "charmm27": ("charmm36.xml", "charmm36"),
}
WATER_XML = {
    "amber": {"tip3p": "tip3p.xml", "tip4p": "tip4pew.xml", "spce": "spce.xml"},
    "amber14": {"tip3p": "amber14/tip3p.xml", "tip4p": "amber14/tip4pew.xml", "spce": "amber14/spce.xml"},
    "charmm36": {"tip3p": "charmm36/water.xml", "tip4p": "charmm36/tip4pew.xml", "spce": "charmm36/spce.xml"},
}
SOLVENT_MODEL = {"tip3p": "tip3p", "tip4p": "tip4pew", "spce": "spce"}

POSRES_K = 1000.0  # kJ/mol/nm^2, as pdb2gmx's posre.itp


class Interrupted(RuntimeError):
    """SIGTERM/SIGINT during a stage: the state was saved, the next run resumes."""


def openmm_forcefield(ff: str, water: str) -> tuple[list[str], str, list[str]]:
    """OpenMM xml files, the addSolvent water model and notes for the owner."""
    if ff not in FF_XML:
        raise RecipeError(f"ff={ff}: tego pola nie ma w OpenMM – wybierz amber99sb-ildn, amber99sb, amber03, "
                          "charmm27 (→ CHARMM36) albo engine=gromacs")
    protein, family = FF_XML[ff]
    w = M.resolve_water(ff, water)
    if w not in WATER_XML[family]:
        raise RecipeError(f"water={w}: tej wody nie ma w OpenMM dla {ff} – możliwe tip3p, tip4p (TIP4P-Ew), spce")
    notes = []
    if ff == "charmm27":
        notes.append("OpenMM nie ma CHARMM27 – użyto CHARMM36 (charmm36.xml + jego TIP3P)")
    if w == "tip4p":
        notes.append("TIP4P w OpenMM: TIP4P-Ew")
    return [protein, WATER_XML[family][w]], SOLVENT_MODEL[w], notes


def box_vectors(shape: str, a: float) -> list[list[float]]:
    """Box vectors [nm] with image distance `a`, in GROMACS' orientation (editconf -bt)."""
    if shape == "cubic":
        return [[a, 0, 0], [0, a, 0], [0, 0, a]]
    if shape == "dodecahedron":  # rhombic dodecahedron, xy-square
        return [[a, 0, 0], [0, a, 0], [a / 2, a / 2, a * math.sqrt(2) / 2]]
    if shape == "octahedron":  # truncated octahedron
        return [[a, 0, 0], [a / 3, a * 2 * math.sqrt(2) / 3, 0], [-a / 3, a * math.sqrt(2) / 3, a * math.sqrt(6) / 3]]
    raise RecipeError(f"box={shape}?")


def diameter(xyz: np.ndarray) -> float:
    """Largest distance between two atoms (what editconf -d adds the margin to)."""
    pts = np.asarray(xyz, dtype=float)
    if len(pts) > 50:
        try:
            from scipy.spatial import ConvexHull

            pts = pts[ConvexHull(pts).vertices]
        except Exception:  # noqa: BLE001 - degenerate hull: fall back to all points
            pass
    best = 0.0
    for i in range(0, len(pts), 512):
        diff = pts[i:i + 512, None, :] - pts[None, :, :]
        best = max(best, float(np.sqrt((diff ** 2).sum(-1)).max()))
    return best


def kept_frames(starts: list[int], counts: list[int], nst: int, end: int) -> list[list[int]]:
    """Steps of the DCD frames to keep, per part. Frames come at absolute multiples of `nst`; a part's frames after
    the next part's start (written after the checkpoint it resumed from) are dropped, and nothing after `end`."""
    out = []
    for i, (start, count) in enumerate(zip(starts, counts, strict=True)):
        limit = starts[i + 1] if i + 1 < len(starts) else end
        first = (start // nst + 1) * nst
        out.append([s for s in (first + j * nst for j in range(count)) if s <= limit])
    return out


CSV_NAMES = {"Step": "Step", "Time (ps)": "Time", "Potential Energy (kJ/mole)": "Potential",
             "Temperature (K)": "Temperature", "Box Volume (nm^3)": "Volume", "Density (g/mL)": "Density"}


def parse_state_csv(text: str) -> dict:
    """StateDataReporter output → columns named like gmx energy terms (density in kg/m³)."""
    rows = list(csv.reader(io.StringIO(text.lstrip("#"))))
    if not rows:
        return {}
    names = [CSV_NAMES.get(h.strip().strip('"'), h.strip().strip('"')) for h in rows[0]]
    data = [[float(v) for v in r] for r in rows[1:] if len(r) == len(names)]
    arr = np.array(data, dtype=float).reshape(-1, len(names))
    cols = {n: arr[:, i] for i, n in enumerate(names)}
    if "Step" in cols:
        cols["Step"] = cols["Step"].astype(int)
    if "Density" in cols:
        cols["Density"] = cols["Density"] * 1000.0
    return cols


# --- build ----------------------------------------------------------------------------------------------------------

def _nonbonded(ff: str) -> dict:
    from openmm import app, unit

    kw = dict(nonbondedMethod=app.PME, constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)
    if ff.startswith("charmm"):
        kw.update(nonbondedCutoff=1.2 * unit.nanometer, switchDistance=1.0 * unit.nanometer)
    else:
        kw.update(nonbondedCutoff=1.0 * unit.nanometer)
    return kw


def build_openmm(odir: Path, p: dict) -> dict:
    """clean.pdb → hydrogens (pH 7) → water box (padding d) → NaCl + neutralise → system.xml/system.pdb."""
    cache = odir / "system.json"
    if cache.exists() and (odir / "system.xml").exists() and (odir / "system.pdb").exists():
        return json.loads(cache.read_text())
    import openmm
    from openmm import XmlSerializer, app, unit
    from pdbfixer import PDBFixer

    print("PROGRESS: przygotowanie (OpenMM): wodory, woda, jony", flush=True)
    files, model, _ = openmm_forcefield(p["ff"], p["water"])
    fixer = PDBFixer(filename=str(odir / "clean.pdb"))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()  # terminal OXT and anything still missing
    fixer.addMissingAtoms()
    ff = app.ForceField(*files)
    mod = app.Modeller(fixer.topology, fixer.positions)
    mod.addHydrogens(ff, pH=7.0)
    residues = [f"{r.name}{r.id}" for r in mod.topology.residues() if r.name not in SOLVENT]
    # box like editconf -d: image distance = solute diameter + 2 d (OpenMM's padding uses the axis extent instead)
    xyz = np.array(mod.positions.value_in_unit(unit.nanometer))
    vecs = [openmm.Vec3(*v) * unit.nanometer for v in box_vectors(p["box"], diameter(xyz) + 2 * p["d"])]
    mod.addSolvent(ff, model=model, boxVectors=vecs, ionicStrength=p["salt"] * unit.molar, neutralize=True,
                   positiveIon="Na+", negativeIon="Cl-")
    system = ff.createSystem(mod.topology, **_nonbonded(p["ff"]))
    nb = next(f for f in system.getForces() if f.__class__.__name__ == "NonbondedForce")
    charges = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                        for i in range(nb.getNumParticles())])
    solvent = np.array([a.residue.name in SOLVENT for a in mod.topology.atoms()])
    names = [r.name for r in mod.topology.residues()]
    vecs = np.array([[v.value_in_unit(unit.nanometer) for v in vec] for vec in mod.topology.getPeriodicBoxVectors()])
    with open(fresh_output(odir / ".system.pdb.tmp"), "w") as fh:
        app.PDBFile.writeFile(mod.topology, mod.positions, fh)
    os.replace(odir / ".system.pdb.tmp", odir / "system.pdb")
    write_text(odir / "system.xml", XmlSerializer.serialize(system))
    info = {"engine": "openmm", "ff": p["ff"], "water": M.resolve_water(p["ff"], p["water"]), "xml": files,
            "box_type": p["box"], "d": p["d"], "box_vectors": vecs.round(4).tolist(),
            "box_lengths": [float(np.linalg.norm(v)) for v in vecs], "volume": float(abs(np.linalg.det(vecs))),
            "atoms": int(system.getNumParticles()), "waters": names.count("HOH"), "na": names.count("NA"),
            "cl": names.count("CL"), "salt": p["salt"], "charge_protein": round(float(charges[~solvent].sum()), 3),
            "charge_total": round(float(charges.sum()), 3), "residues": residues}
    write_text(cache, json.dumps(info, indent=1))
    return info


# --- running --------------------------------------------------------------------------------------------------------

@contextmanager
def _stop_on_signal():
    flag = {"stop": False}

    def handler(signum, frame):
        flag["stop"] = True

    old = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield flag
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _threads() -> int:
    return int(os.environ.get("COMPUTE_THREADS") or os.cpu_count() or 1)


def _simulation(odir: Path, stage, seed: int, ref=None):
    from openmm import (
        CustomExternalForce,
        LangevinMiddleIntegrator,
        MonteCarloBarostat,
        Platform,
        XmlSerializer,
        app,
        unit,
    )

    system = XmlSerializer.deserialize((odir / "system.xml").read_text())
    pdb = app.PDBFile(str(odir / "system.pdb"))
    if ref is not None:
        force = CustomExternalForce("0.5*posres_k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("posres_k", POSRES_K)
        for name in ("x0", "y0", "z0"):
            force.addPerParticleParameter(name)
        for atom in pdb.topology.atoms():
            if atom.residue.name not in SOLVENT and atom.element is not None and atom.element.symbol != "H":
                force.addParticle(atom.index, [float(x) for x in ref[atom.index]])
        system.addForce(force)
    if stage.dynamic and stage.pcoupl:
        barostat = MonteCarloBarostat(1.0 * unit.bar, stage.temp * unit.kelvin, 25)
        if seed != -1:
            barostat.setRandomNumberSeed(seed)
        system.addForce(barostat)
    integrator = LangevinMiddleIntegrator(stage.temp * unit.kelvin, 1.0 / unit.picosecond, M.DT_PS * unit.picosecond)
    if seed != -1:
        integrator.setRandomNumberSeed(seed)
    platform = Platform.getPlatformByName("CPU")
    sim = app.Simulation(pdb.topology, system, integrator, platform, {"Threads": str(_threads())})
    return sim, pdb


def run_em(odir: Path, stage, prog: Progress):
    import openmm
    from openmm import XmlSerializer, app, unit

    print(prog.start(stage), flush=True)
    sim, pdb = _simulation(odir, stage, -1)
    sim.context.setPositions(pdb.positions)
    energies, last = [], {"fmax": float("nan")}

    class Reporter(openmm.MinimizationReporter):
        def report(self, iteration, x, grad, args):
            g = np.asarray(grad).reshape(-1, 3)
            # the minimiser's own gradient: constraint terms are inside its objective, unlike State forces
            last["fmax"] = float(np.sqrt((g * g).sum(axis=1)).max())
            energies.append((iteration, args["system energy"]))
            msg = prog.update({"step": iteration, "epot": args["system energy"], "fmax": last["fmax"]})
            if msg:
                print(msg, flush=True)
            return False

    sim.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=0, reporter=Reporter())
    state = sim.context.getState(getPositions=True, getEnergy=True)
    fmax = last["fmax"]
    epot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    write_text(odir / "em_energy.csv", "iteracja,Epot\n" + "".join(f"{i},{e:.3f}\n" for i, e in energies))
    write_text(odir / "em.state.xml", XmlSerializer.serialize(state))
    with open(fresh_output(odir / "em.pdb"), "w") as fh:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), fh)
    write_text(odir / "em.done.json", json.dumps({"converged": fmax < 1000, "steps": len(energies), "epot": epot,
                                                  "fmax": fmax, "emtol": 1000.0, "end_step": 0}))


def run_dynamic(odir: Path, stage, start_state: Path, seed: int, prog: Progress, intervals=None):
    """One MD stage from `start_state` (positions, velocities, box of the previous stage), resumable."""
    from openmm import XmlSerializer, app, unit

    name = stage.name
    print(prog.start(stage), flush=True)
    start = XmlSerializer.deserialize(start_state.read_text())
    ref = start.getPositions(asNumpy=True).value_in_unit(unit.nanometer) if stage.posres else None
    sim, _ = _simulation(odir, stage, seed, ref)
    chk = odir / f"{name}.chk.xml"
    if chk.exists():
        print(f"wznawiam {name} z checkpointu", flush=True)
        sim.context.setState(XmlSerializer.deserialize(chk.read_text()))
    else:
        sim.context.setPeriodicBoxVectors(*start.getPeriodicBoxVectors())
        sim.context.setPositions(start.getPositions())
        sim.context.setStepCount(stage.first_step)
        sim.context.setTime(stage.first_step * M.DT_PS)
        if stage.gen_vel:
            sim.context.setVelocitiesToTemperature(stage.temp * unit.kelvin,
                                                   seed if seed != -1 else int.from_bytes(os.urandom(3), "big"))
        else:
            sim.context.setVelocities(start.getVelocities())
    parts_path = odir / f"{name}.parts.json"
    parts = json.loads(parts_path.read_text()) if parts_path.exists() else []
    step0 = sim.currentStep
    parts.append({"part": len(parts) + 1, "start": step0})
    write_text(parts_path, json.dumps(parts))
    nst_x, nst_e = intervals or M.output_intervals(stage)
    part = parts[-1]["part"]
    dcd = app.DCDReporter(str(fresh_output(odir / f"{name}.part{part}.dcd")), nst_x)
    data = app.StateDataReporter(str(fresh_output(odir / f"{name}.part{part}.csv")), nst_e, step=True, time=True,
                                 potentialEnergy=True, temperature=True, volume=True, density=True)
    sim.reporters += [dcd, data]
    end = stage.first_step + stage.nsteps
    every = float(os.environ.get("MD_CHECKPOINT_S", "300"))
    crash_after = os.environ.get("MD_TEST_CRASH_AFTER_CHECKPOINT") == "1"
    t0 = last = time.time()

    def checkpoint():
        state = sim.context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                     enforcePeriodicBox=False)
        write_text(chk, XmlSerializer.serialize(state))

    with _stop_on_signal() as flag:
        while sim.currentStep < end:
            sim.step(min(500, end - sim.currentStep))
            msg = prog.update({"step": sim.currentStep})
            if msg:
                print(msg, flush=True)
            if flag["stop"] or time.time() - last >= every:
                checkpoint()
                last = time.time()
                if crash_after and name == "md":  # tests only: a power cut right after a checkpoint
                    os.kill(os.getpid(), signal.SIGKILL)
                if flag["stop"]:
                    raise Interrupted(f"{stage.title} przerwany na kroku {sim.currentStep - stage.first_step}"
                                      f"/{stage.nsteps}, stan zapisany")
    seconds = max(time.time() - t0, 1e-6)
    ns_day = (end - step0) * M.DT_PS * 1e-3 / seconds * 86400
    prog.set_rate_from_nsday(ns_day, force=True)
    for reporter in (dcd, data):
        out = getattr(reporter, "_out", None)
        if out is not None:
            out.close()
    final = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    write_text(odir / f"{name}.state.xml", XmlSerializer.serialize(final))
    wrapped = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(fresh_output(odir / f"{name}.pdb"), "w") as fh:
        app.PDBFile.writeFile(sim.topology, wrapped.getPositions(), fh)
    write_text(odir / f"{name}.done.json", json.dumps({
        "first_step": stage.first_step, "end_step": end, "ns_day": ns_day, "nst_x": nst_x, "nst_e": nst_e,
        "temp": stage.temp, "pcoupl": stage.pcoupl}))


def run_stages(odir: Path, plan, p: dict):
    prog = Progress(plan)
    prev = None
    for s in plan:
        if segment_done(odir, s.name):
            print(f"{s.name} już policzony, pomijam", flush=True)
            info = json.loads((odir / f"{s.name}.done.json").read_text())
            prog.set_rate_from_nsday(info.get("ns_day"))
        elif s.name == "em":
            run_em(odir, s, prog)
        else:
            run_dynamic(odir, s, odir / f"{prev}.state.xml", p["seed"], prog)
        prev = s.name


def extend_openmm(work: Path, odir: Path, p: dict) -> str:
    if not segment_done(odir, "md"):
        raise RecipeError("extend_ns: w zadaniu nie ma skończonej produkcji OpenMM (md.done.json) – przedłużyć "
                          "można tylko skończone zadanie")
    seg, k = claim_segment(odir, work)
    if segment_done(odir, seg):
        print(f"{seg} już policzony, pomijam", flush=True)
        return seg
    prev = "md" if k == 1 else f"md_ext{k - 1}"
    md = json.loads((odir / "md.done.json").read_text())
    stage = M.Stage(seg, f"przedłużenie {k}", M.nsteps(p["extend_ns"] * 1000), md["temp"], pcoupl=md["pcoupl"],
                    first_step=segment_end_step(odir, prev))
    prog = Progress([stage])
    prog.set_rate_from_nsday(md.get("ns_day"))
    run_dynamic(odir, stage, odir / f"{prev}.state.xml", p["seed"], prog, intervals=(md["nst_x"], md["nst_e"]))
    return seg


# --- reading back for the analysis ----------------------------------------------------------------------------------

def _segment_parts(odir: Path, seg: str):
    """[(dcd, kept steps, csv columns filtered)] of one stage/segment."""
    from MDAnalysis.coordinates.DCD import DCDReader

    parts = json.loads((odir / f"{seg}.parts.json").read_text())
    done = json.loads((odir / f"{seg}.done.json").read_text())
    starts = [q["start"] for q in parts]
    counts, files = [], []
    for q in parts:
        dcd = odir / f"{seg}.part{q['part']}.dcd"
        try:
            n = DCDReader(str(dcd)).n_frames if dcd.exists() and dcd.stat().st_size > 0 else 0
        except (OSError, ValueError, RuntimeError):
            n = 0
        counts.append(n)
        files.append((dcd, odir / f"{seg}.part{q['part']}.csv"))
    kept = kept_frames(starts, counts, done["nst_x"], done["end_step"])
    out = []
    for i, ((dcd, csv_path), steps) in enumerate(zip(files, kept, strict=True)):
        limit = starts[i + 1] if i + 1 < len(starts) else done["end_step"]
        cols = parse_state_csv(csv_path.read_text()) if csv_path.exists() else {}
        if cols:
            keep = cols["Step"] <= limit
            cols = {k: v[keep] for k, v in cols.items()}
        out.append((dcd, steps, cols))
    return out


def _energy(parts_list) -> dict:
    cols = [c for parts in parts_list for _, _, c in parts if c]
    if not cols:
        return {}
    keys = [k for k in cols[0] if all(k in c for c in cols)]
    merged = {k: np.concatenate([c[k] for c in cols]) for k in keys}
    merged["time"] = merged.pop("Step") * M.DT_PS
    merged.pop("Time", None)
    return merged


def _protein_trajectory(u, protein_idx, parts_list):
    coords, steps = [], []
    for parts in parts_list:
        for dcd, kept, _ in parts:
            if not kept:
                continue
            u.load_new(str(dcd))
            for i in range(len(kept)):
                u.trajectory[i]
                coords.append(u.atoms[protein_idx].positions.copy())
            steps += kept
    return np.array(coords, dtype=np.float32), np.array(steps) * M.DT_PS


def collect(odir: Path, out: Path, plan, system: dict) -> dict:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MDAnalysis chatter about DCD timesteps, CRYST1, formal charges
        return _collect(odir, out, plan, system)


def _collect(odir: Path, out: Path, plan, system: dict) -> dict:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
    from MDAnalysis.coordinates.memory import MemoryReader

    from .analysis import write_csv

    res = {"warnings": []}
    em = {}
    if segment_done(odir, "em"):
        em = json.loads((odir / "em.done.json").read_text())
        rows = [ln.split(",") for ln in (odir / "em_energy.csv").read_text().splitlines()[1:] if ln]
        em["series"] = (np.array([float(r[0]) for r in rows]), np.array([float(r[1]) for r in rows]))
        write_csv(out / "em_energia.csv", ["iteracja", "Epot_kJ_mol"], [[r[0], r[1]] for r in rows])
        if not em["converged"]:
            res["warnings"].append(f"minimalizacja: Fmax {em['fmax']:.3g} > 1000 kJ/mol/nm")
    stages = [s for s in plan if s.dynamic and segment_done(odir, s.name)]
    u = mda.Universe(str(odir / "system.pdb"))
    protein = u.select_atoms("not resname HOH NA CL")
    ref_u = mda.Universe(str(odir / "em.pdb"))
    ref = mda.Merge(ref_u.atoms[protein.indices])
    energies, rmsd, perf, pu = {}, {}, {}, None
    for s in stages:
        segs = production_segments(odir) if s.name == "md" else [s.name]
        parts_list = [_segment_parts(odir, seg) for seg in segs]
        energies[s.name] = _energy(parts_list)
        coords, times = _protein_trajectory(u, protein.indices, parts_list)
        perf[s.name] = json.loads((odir / f"{segs[-1]}.done.json").read_text()).get("ns_day")
        if not len(coords):
            continue
        stage_u = mda.Merge(protein)
        stage_u.load_new(coords, format=MemoryReader)
        r = rms.RMSD(stage_u, ref, select="backbone").run()
        rmsd[s.name] = (times, r.results.rmsd[:, 2] / 10.0)
        if s.name == "md":
            pu, md_times = stage_u, times
    if pu is not None:
        _production(pu, md_times, ref, out, res, system)
    return {"em": em, "stages": stages, "energies": energies, "rmsd": rmsd, "res": res, "perf": perf}


def _production(pu, times, ref, out: Path, res: dict, system: dict):
    from MDAnalysis.analysis import align, rms

    from .analysis import write_csv

    align.AlignTraj(pu, ref, select="backbone", in_memory=True).run()
    ca = pu.select_atoms("name CA")
    rmsf = rms.RMSF(ca).run().results.rmsf / 10.0
    labels = system.get("residues") or []
    if len(labels) != len(ca):
        labels = [f"{a.resname}{a.resid}" for a in ca]
    res["rmsf"] = (labels, rmsf)
    write_csv(out / "rmsf.csv", ["reszta", "rmsf_nm"], [[lab, f"{v:.4f}"] for lab, v in zip(labels, rmsf, strict=True)])
    rg = np.array([pu.atoms.radius_of_gyration() for _ in pu.trajectory]) / 10.0
    res["rg"] = (times, rg)
    write_csv(out / "rg.csv", ["czas_ps", "rg_nm"], [[f"{a:.3f}", f"{b:.4f}"] for a, b in zip(times, rg, strict=True)])
    try:
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

        hba = HydrogenBondAnalysis(pu, donors_sel="(name N* or name O*)", hydrogens_sel="name H*",
                                   acceptors_sel="(name O* or name N*)", d_a_cutoff=3.5, d_h_a_angle_cutoff=150,
                                   update_selections=False)
        hba.run()
        counts = hba.count_by_time()
        res["hbonds"] = (times, counts)
        write_csv(out / "wiazania_h.csv", ["czas_ps", "liczba"],
                  [[f"{a:.3f}", int(b)] for a, b in zip(times, counts, strict=True)])
    except Exception as exc:  # noqa: BLE001 - optional analysis, reported in the summary
        res["warnings"].append(f"mostki H: {str(exc)[:120]}")
    try:
        from MDAnalysis.analysis.dssp import DSSP

        codes = DSSP(pu).run().results.dssp
        lines = ["".join(c if c != "-" else "~" for c in row) for row in codes]
        res["dssp"] = (times, lines)
        res["dssp_note"] = "DSSP (MDAnalysis): 3 stany H/E/pętla"
        from .analysis import dssp_fractions

        helix, beta = dssp_fractions(lines)
        write_csv(out / "dssp.csv", ["czas_ps", "dssp", "helisa", "beta"],
                  [[f"{a:.3f}", s, f"{h:.3f}", f"{b:.3f}"] for a, s, h, b in zip(times, lines, helix, beta,
                                                                                 strict=True)])
    except Exception as exc:  # noqa: BLE001
        res["warnings"].append(f"DSSP pominięte ({str(exc)[:120]})")
    _viewer(pu, ref, out, res)


def _viewer(pu, ref, out: Path, res: dict):
    import MDAnalysis as mda

    ref.atoms.write(str(fresh_output(out / "bialko_em.pdb")))
    with mda.Writer(str(fresh_output(out / "bialko.xtc")), pu.atoms.n_atoms) as w:
        for _ in pu.trajectory:
            w.write(pu.atoms)
    frames = len(pu.trajectory)
    max_frames = max(1, min(200, int(19e6 / max(pu.atoms.n_atoms * 81, 1))))
    skip = max(1, math.ceil(frames / max_frames))
    with mda.Writer(str(fresh_output(out / "bialko_klatki.pdb")), multiframe=True) as w:
        for _ in pu.trajectory[::skip]:
            w.write(pu.atoms)
    res["viewer_skip"] = skip
