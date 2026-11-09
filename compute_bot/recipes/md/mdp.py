"""Stage plan and .mdp files generated from the recipe parameters (pure functions, no I/O)."""
from __future__ import annotations

from dataclasses import dataclass, field

DT_PS = 0.002  # 2 fs, possible thanks to h-bond constraints
EQUIL_T_MAX = 300.0  # equilibrate at room temperature, heat (if asked) only in production


def nsteps(ps: float) -> int:
    return int(round(ps / DT_PS))


def default_water(ff: str) -> str:
    """The water model each force field was parametrised with (GROMACS' 'recommended' entry)."""
    if ff.startswith("oplsaa"):
        return "tip4p"
    if ff.startswith("gromos"):
        return "spc"
    return "tip3p"  # amber*, charmm27


def resolve_water(ff: str, water: str) -> str:
    return default_water(ff) if water == "auto" else water


def solvent_box(water: str) -> str:
    """Pre-equilibrated water box for gmx solvate (3-site models share spc216.gro)."""
    return "tip4p.gro" if water == "tip4p" else "spc216.gro"


def nonbonded(ff: str) -> list[tuple[str, str]]:
    """Cut-off settings the force field was developed with (PME for electrostatics everywhere)."""
    common = [("cutoff-scheme", "Verlet"), ("coulombtype", "PME")]
    if ff.startswith("charmm"):
        return common + [("rcoulomb", "1.2"), ("vdwtype", "cut-off"), ("vdw-modifier", "force-switch"),
                         ("rvdw-switch", "1.0"), ("rvdw", "1.2"), ("DispCorr", "no")]
    if ff.startswith("gromos"):
        return common + [("rcoulomb", "1.4"), ("rvdw", "1.4"), ("DispCorr", "EnerPres")]
    return common + [("rcoulomb", "1.0"), ("rvdw", "1.0"), ("DispCorr", "EnerPres")]


def nice_interval(n: int, frames: int) -> int:
    """Largest of 10, 20, 50, 100, 200, 500, ... steps that still gives about `frames` outputs over n steps."""
    raw = max(n / max(frames, 1), 1)
    best = 10
    for k in range(1, 10):
        for m in (1, 2, 5):
            if m * 10 ** k <= raw:
                best = m * 10 ** k
    return best


@dataclass
class Stage:
    name: str  # deffnm: em, nvt, npt, md
    title: str  # for the owner
    nsteps: int
    temp: float = 300.0
    posres: bool = False
    gen_vel: bool = False
    pcoupl: bool = False
    grompp: list = field(default_factory=list)

    @property
    def ps(self) -> float:
        return self.nsteps * DT_PS

    @property
    def dynamic(self) -> bool:
        return self.name != "em"


def plan_stages(p: dict) -> list[Stage]:
    """EM, then NVT and NPT with heavy-atom position restraints (skipped when 0 ps), then free production.
    Each stage starts from the previous coordinates (and checkpoint: velocities, thermostat state); velocities are
    generated once, in the first dynamic stage."""
    t_prod = float(p["T"])
    t_eq = min(t_prod, EQUIL_T_MAX)
    npt_prod = p.get("ensemble", "npt") != "nvt"
    stages = [Stage("em", "minimalizacja", 50000)]
    if p["nvt_ps"] > 0:
        stages.append(Stage("nvt", "NVT", nsteps(p["nvt_ps"]), t_eq, posres=True))
    if p["npt_ps"] > 0:
        stages.append(Stage("npt", "NPT", nsteps(p["npt_ps"]), t_eq, posres=True, pcoupl=True))
    stages.append(Stage("md", "produkcja", nsteps(p["ns"] * 1000), t_prod, pcoupl=npt_prod))
    prev = "ions"
    for s in stages:
        args = ["-f", f"{s.name}.mdp", "-c", f"{prev}.gro"]
        if s.posres:
            args += ["-r", f"{prev}.gro"]
        if s.dynamic and prev not in ("ions", "em"):
            args += ["-t", f"{prev}.cpt"]
        s.grompp = args + ["-p", "topol.top"]
        prev = s.name
    first_dynamic = next(s for s in stages if s.dynamic)
    first_dynamic.gen_vel = True
    return stages


def _fmt(x: float) -> str:
    return f"{x:g}"


def make_mdp(stage: Stage, ff: str, seed: int) -> str:
    """The .mdp text for one stage (as in a hand-made course run: 2 fs, h-bonds, PME, V-rescale, C-rescale)."""
    if stage.name == "em":
        rows = [("integrator", "steep"), ("emtol", "1000.0"), ("emstep", "0.01"), ("nsteps", str(stage.nsteps)),
                ("nstenergy", "10"), ("nstlist", "10")] + nonbonded(ff) + [("pbc", "xyz")]
        return render(rows, "Energy minimisation (steepest descent)")
    nstenergy = nice_interval(stage.nsteps, 500)
    nstxout = nice_interval(stage.nsteps, 1000 if stage.name == "md" else 100)
    nstlog = max(nice_interval(stage.nsteps, 100), nstenergy)
    rows = []
    if stage.posres:
        rows.append(("define", "-DPOSRES"))
    t = _fmt(stage.temp)
    rows += [
        ("integrator", "md"), ("dt", _fmt(DT_PS)), ("nsteps", str(stage.nsteps)),
        ("nstxout-compressed", str(nstxout)), ("compressed-x-grps", "System"),
        ("nstenergy", str(nstenergy)), ("nstcalcenergy", str(min(100, nstenergy))), ("nstlog", str(nstlog)),
        ("continuation", "no" if stage.gen_vel else "yes"),
        ("constraints", "h-bonds"), ("constraint-algorithm", "lincs"),
        ("nstlist", "10"),
    ] + nonbonded(ff) + [
        ("tcoupl", "V-rescale"), ("tc-grps", "Protein Non-Protein"), ("tau_t", "0.1 0.1"), ("ref_t", f"{t} {t}"),
    ]
    if stage.pcoupl:
        rows += [("pcoupl", "C-rescale"), ("pcoupltype", "isotropic"), ("tau_p", "1.0"), ("ref_p", "1.0"),
                 ("compressibility", "4.5e-5")]
        if stage.posres:
            rows.append(("refcoord_scaling", "com"))
    else:
        rows.append(("pcoupl", "no"))
    if stage.gen_vel:
        rows += [("gen_vel", "yes"), ("gen_temp", t), ("gen_seed", str(seed))]
    else:
        rows.append(("gen_vel", "no"))
    rows.append(("pbc", "xyz"))
    what = {"nvt": "NVT with heavy-atom position restraints", "npt": "NPT with heavy-atom position restraints",
            "md": "production, no restraints"}[stage.name]
    return render(rows, f"{what}, {stage.ps:g} ps at {t} K")


def ions_mdp(ff: str) -> str:
    return render([("integrator", "steep"), ("nsteps", "0")] + nonbonded(ff) + [("pbc", "xyz")],
                  "Only to build ions.tpr for genion")


def render(rows: list[tuple[str, str]], title: str) -> str:
    width = max(len(k) for k, _ in rows)
    return f"; {title} (generated by compute_bot md recipe)\n" + "".join(f"{k:<{width}} = {v}\n" for k, v in rows)


def parse_mdp(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if "=" in line:
            k, v = (part.strip() for part in line.split("=", 1))
            out[k] = v
    return out
