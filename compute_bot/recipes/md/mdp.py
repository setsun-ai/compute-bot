"""Stage plan and .mdp files generated from the recipe parameters (pure functions, no I/O)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .gmxtools import RecipeError

DT_PS = 0.002  # 2 fs, possible thanks to h-bond constraints
EQUIL_T_MAX = 300.0  # equilibrate at room temperature, heat (if asked) only in production
STAGE_NAMES = ("em", "nvt", "npt", "md")  # the prefixes params.mdp accepts; no prefix = production


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
    first_step: int = 0  # an extension continues the step count of the run it extends
    dt: float = DT_PS

    @property
    def ps(self) -> float:
        return self.nsteps * self.dt

    @property
    def dynamic(self) -> bool:
        return self.name != "em"


def equilibration_temp(p: dict) -> float:
    """T_eq when given, else room temperature (or T when T is lower)."""
    t_eq = float(p.get("T_eq") or 0)
    return t_eq if t_eq > 0 else min(float(p["T"]), EQUIL_T_MAX)


def plan_stages(p: dict) -> list[Stage]:
    """EM, then NVT and NPT with heavy-atom position restraints (skipped when 0 ps), then free production.
    Each stage starts from the previous coordinates (and checkpoint: velocities, thermostat state); velocities are
    generated once, in the first dynamic stage. Stage overrides from params.mdp that change the length, time step or
    temperature are reflected in the plan (progress and summary follow what is really run)."""
    t_prod = float(p["T"])
    t_eq = equilibration_temp(p)
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
    overrides = parse_stage_overrides(p.get("mdp", "")) if p.get("engine", "gromacs") == "gromacs" else {}
    for s in stages:
        adjust_stage(s, overrides.get(s.name, []))
    return stages


def _override(overrides, key: str) -> str | None:
    """The value the overrides give `key` (the last one wins, as in apply_overrides), else None."""
    found = None
    for k, v in overrides or ():
        if _norm(k) == _norm(key):
            found = v
    return found


def _number(key: str, value: str) -> float:
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        raise RecipeError(f"mdp: {key}={value} – oczekuję liczby") from None


def adjust_stage(stage: Stage, overrides) -> Stage:
    """Apply the overrides that change the plan itself: dt (the stage keeps its length in ps unless nsteps is given
    too), nsteps, ref_t (first group; it beats T / T_eq, gen_temp follows)."""
    ps = stage.ps
    dt, n, ref_t = (_override(overrides, k) for k in ("dt", "nsteps", "ref_t"))
    if stage.dynamic and dt is not None:
        stage.dt = _number("dt", dt)
        if stage.dt <= 0:
            raise RecipeError(f"mdp: dt={dt} – krok czasowy musi być dodatni")
        stage.nsteps = int(round(ps / stage.dt))
    if n is not None:
        stage.nsteps = int(_number("nsteps", n))
    if stage.dynamic and ref_t is not None:
        stage.temp = _number("ref_t", ref_t)
    return stage


def _fmt(x: float) -> str:
    return f"{x:g}"


def output_intervals(stage: Stage) -> tuple[int, int]:
    """(trajectory, energy) output intervals in steps: ~1000 production frames, ~100 per equilibration stage."""
    return nice_interval(stage.nsteps, 1000 if stage.name == "md" else 100), nice_interval(stage.nsteps, 500)


_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def parse_stage_overrides(text: str) -> dict[str, list[tuple[str, str]]]:
    """`[stage:]key=value;...` from params.mdp → {stage: [(key, value), ...]} in the given order; no prefix means
    the production (md). Only the syntax is checked, grompp judges the content."""
    out: dict[str, list[tuple[str, str]]] = {}
    for item in re.split(r"[;\n]", text or ""):
        item = item.strip()
        if not item:
            continue
        key, sep, value = (part.strip() for part in item.partition("="))
        stage = "md"
        if ":" in key:
            stage, key = (part.strip() for part in key.split(":", 1))
            if stage.lower() not in STAGE_NAMES:
                raise RecipeError(f"mdp: „{item}” – nieznany etap „{stage}”; dozwolone przedrostki: "
                                  f"{', '.join(STAGE_NAMES)} (bez przedrostka = produkcja md), np. nvt:ref_t=310 310")
            stage = stage.lower()
        if not sep or not _KEY.fullmatch(key) or not value or "=" in value:
            raise RecipeError(f"mdp: „{item}” – oczekuję [etap:]klucz=wartość rozdzielone średnikami, "
                              "np. mdp=nstxout-compressed=500;npt:tau_p=5")
        out.setdefault(stage, []).append((key, value))
    return out


def parse_overrides(text: str, stage: str = "md") -> list[tuple[str, str]]:
    """The overrides of one stage (default: the production, i.e. entries without a prefix)."""
    return parse_stage_overrides(text).get(stage, [])


def _norm(key: str) -> str:
    return key.lower().replace("_", "-")


def apply_overrides(rows: list[tuple[str, str]], overrides) -> list[tuple[str, str]]:
    """Replace keys (GROMACS treats - and _ alike, any case) or append new ones."""
    rows = list(rows)
    for key, value in overrides or ():
        hits = [i for i, (k, _) in enumerate(rows) if _norm(k) == _norm(key)]
        if hits:
            rows[hits[0]] = (rows[hits[0]][0], value)
            rows = [r for i, r in enumerate(rows) if i not in hits[1:]]
        else:
            rows.append((key, value))
    return rows


def make_mdp(stage: Stage, ff: str, seed: int, overrides=None) -> str:
    """The .mdp text for one stage (as in a hand-made course run: 2 fs, h-bonds, PME, V-rescale, C-rescale)."""
    if stage.name == "em":
        rows = [("integrator", "steep"), ("emtol", "1000.0"), ("emstep", "0.01"), ("nsteps", str(stage.nsteps)),
                ("nstenergy", "10"), ("nstlist", "10")] + nonbonded(ff) + [("pbc", "xyz")]
        return render(apply_overrides(rows, overrides), "Energy minimisation (steepest descent)")
    nstxout, nstenergy = output_intervals(stage)
    nstlog = max(nice_interval(stage.nsteps, 100), nstenergy)
    rows = []
    if stage.posres:
        rows.append(("define", "-DPOSRES"))
    t = _fmt(stage.temp)
    rows += [
        ("integrator", "md"), ("dt", _fmt(stage.dt)), ("nsteps", str(stage.nsteps)),
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
    if overrides:
        rows = apply_overrides(rows, overrides)
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
