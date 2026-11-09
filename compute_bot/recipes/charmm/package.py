"""A CHARMM-GUI package: where the engine folders are, which steps its README runs, what each step's input file says
(protocol), and the owner's ns/T overrides written into copies of those inputs. Pure functions except the few
`*_dir`/`read_*` helpers that look at the unpacked files."""
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..md.gmxtools import RecipeError
from ..md.mdp import nice_interval, parse_mdp
from . import csh

ENGINES = ("gromacs", "openmm", "namd")
ENGINE_TITLES = {"gromacs": "GROMACS", "openmm": "OpenMM", "namd": "NAMD"}
MIN_PROD_FRAMES = 100  # fewer production frames than this and the trajectory interval is made shorter
NAMD_HELP = ("NAMD nie jest zainstalowany (brak namd3/namd2 w PATH). NAMD ma licencję, której conda nie rozprowadza: "
             "pobierz „Linux-x86_64-multicore” (NAMD 3, wersja CPU) z https://www.ks.uiuc.edu/Research/namd/ "
             "(darmowa rejestracja), wyślij plik .tar.gz botowi albo na laptopie uruchom "
             "`sudo bash deploy/install-namd.sh PLIK.tar.gz`. Możesz też policzyć tę paczkę w GROMACS albo OpenMM: "
             "engine=gromacs / engine=openmm (gdy paczka ma te foldery).")


# --- where things are -----------------------------------------------------------------------------------------------

def _looks_like(engine: str, d: Path) -> bool:
    if engine == "gromacs":
        return any(d.glob("*.mdp")) and (any(d.glob("*.top")) or (d / "README").exists())
    if engine == "openmm":
        return (d / "openmm_run.py").exists()
    return any(d.glob("step*.inp")) and not (d / "openmm_run.py").exists() and (
        (d / "README").exists() or any("structure" in p.read_text(errors="replace")[:4000].lower()
                                       for p in d.glob("step*.inp")))


def find_engines(work: Path) -> dict[str, Path]:
    """Engine folders of the package: work/gromacs etc., or one level down (an unstripped charmm-gui-XXXX/ folder);
    the work dir itself when only one engine folder was zipped."""
    skip = {"wyniki", "__pycache__", "toppar", "restraints"}
    levels = [[work], sorted(p for p in work.iterdir() if p.is_dir() and p.name not in skip)]
    levels.append(sorted(q for p in levels[1] for q in p.iterdir() if q.is_dir() and q.name not in skip))
    found: dict[str, Path] = {}
    for engine in ENGINES:
        for level in levels[1:]:
            hit = next((d for d in level if d.name.lower() == engine and _looks_like(engine, d)), None)
            if hit is not None:
                found[engine] = hit
                break
    if not found:
        for engine in ENGINES:
            if _looks_like(engine, work):
                found[engine] = work
                break
    return found


def package_root(edir: Path, work: Path) -> Path:
    """Folder with input.config.dat, step1-3 files and toppar/: the parent of the engine folder."""
    return edir.parent if edir != work else work


def ref_constants(edir: Path):
    """For NAMD: distinct non-zero force constants in a restraint reference PDB column (B or O)."""
    def read(path: str, col: str):
        text = _read(edir / path)
        if not text:
            return []
        lo, hi = (60, 66) if col.upper() == "B" else (54, 60)
        vals = set()
        for ln in text.splitlines():
            if ln.startswith(("ATOM", "HETATM")):
                v = _num(ln[lo:hi])
                if v:
                    vals.add(round(v, 4))
        return sorted(vals, reverse=True)[:4]
    return read


def namd_binary() -> str | None:
    return shutil.which("namd3") or shutil.which("namd2")


def choose_engine(found: dict[str, Path], wanted: str, available: dict[str, bool]) -> str:
    """engine=auto → gromacs, then openmm, then namd (only if installed). Readable errors otherwise."""
    have = ", ".join(f"{e}/" for e in ENGINES if e in found) or "żadnego"
    if not found:
        raise RecipeError("nie widzę paczki CHARMM-GUI: brak folderu gromacs/, openmm/ ani namd/ (wyślij cały "
                          "charmm-gui.tgz albo zip z tymi folderami)")
    if wanted != "auto":
        if wanted not in found:
            raise RecipeError(f"paczka nie ma folderu {wanted}/ (są: {have}); pobierz z CHARMM-GUI pliki dla "
                              f"{ENGINE_TITLES[wanted]} albo wybierz engine=auto")
        if not available.get(wanted, False):
            if wanted == "namd":
                raise RecipeError(NAMD_HELP)
            raise RecipeError(f"{ENGINE_TITLES[wanted]} nie jest dostępny w środowisku – uruchom przepis env")
        return wanted
    for engine in ENGINES:
        if engine in found and available.get(engine, False):
            return engine
    if "namd" in found:
        raise RecipeError(NAMD_HELP)
    raise RecipeError(f"brak programu dla folderów {have} w środowisku – uruchom przepis env")


# --- input-file settings --------------------------------------------------------------------------------------------

def _num(text, default=None):
    if text is None:
        return default
    m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(text))
    return float(m.group(1)) if m else default


def _norm(key: str) -> str:
    return key.lower().replace("_", "-")


def mdp_dict(text: str) -> dict:
    return {_norm(k): v for k, v in parse_mdp(text).items()}


def gromacs_settings(text: str) -> dict:
    m = mdp_dict(text)
    integ = m.get("integrator", "md").lower()
    minim = integ in ("steep", "cg", "l-bfgs")
    nsteps = int(_num(m.get("nsteps"), 0))
    dt = None if minim else _num(m.get("dt"), 0.001)
    defines = m.get("define", "")
    rest = []
    if "-DPOSRES" in defines:
        bb = re.search(r"-DPOSRES_FC_BB=([\d.eE+-]+)", defines)
        sc = re.search(r"-DPOSRES_FC_SC=([\d.eE+-]+)", defines)
        part = "położenia białka"
        if bb or sc:
            part += f" (szkielet {bb.group(1) if bb else '?'}, łańcuchy boczne {sc.group(1) if sc else '?'} kJ/mol/nm²)"
        rest.append(part)
    for name, what in (("DIHRES", "dwuścienne"), ("POSRES_LIPID", "lipidy"), ("POSRES_LIG", "ligand")):
        if f"-D{name}" in defines:
            fc = re.search(rf"-D{name}_FC=([\d.eE+-]+)", defines)
            rest.append(what + (f" ({fc.group(1)})" if fc else ""))
    tc = m.get("tcoupl", "no")
    temps = [float(x) for x in m.get("ref-t", m.get("ref_t", "")).split() if _num(x) is not None]
    thermostat = ""
    if not minim:
        if integ in ("sd", "bd"):
            thermostat = f"Langevin (integrator {integ}, tau-t {m.get('tau-t', '?')} ps)"
        elif tc.lower() not in ("no", ""):
            thermostat = f"{tc} (tau-t {m.get('tau-t', '?')} ps, grupy {m.get('tc-grps', '?')})"
        else:
            thermostat = "brak"
    pc = m.get("pcoupl", "no")
    barostat = ""
    if not minim and pc.lower() not in ("no", ""):
        barostat = (f"{pc} {m.get('pcoupltype', 'isotropic')} (tau-p {m.get('tau-p', '?')} ps, "
                    f"ściśliwość {m.get('compressibility', '?')} 1/bar)")
    traj = int(_num(m.get("nstxout-compressed", m.get("nstxtcout")), 0) or 0)
    vdw = f"{m.get('vdwtype', 'cut-off')}"
    if m.get("vdw-modifier"):
        vdw += f" {m['vdw-modifier']}"
    if m.get("rvdw-switch"):
        vdw += f" {m['rvdw-switch']}–{m.get('rvdw', '?')} nm"
    else:
        vdw += f" {m.get('rvdw', '?')} nm"
    return {
        "integrator": integ, "minimize": minim, "dt_ps": dt, "nsteps": 0 if minim else nsteps,
        "min_steps": nsteps if minim else 0, "emtol": _num(m.get("emtol")) if minim else None,
        "temp": temps[0] if temps and not minim else None, "thermostat": thermostat,
        "barostat": barostat, "pressure": _num(m.get("ref-p")) if barostat else None,
        "restraints": "; ".join(rest), "constraints": m.get("constraints", "none"),
        "vdw": vdw, "coulomb": f"{m.get('coulombtype', '?')} {m.get('rcoulomb', '?')} nm",
        "gen_vel": m.get("gen-vel", "no").lower() == "yes", "traj_every": traj,
        "energy_every": int(_num(m.get("nstenergy"), 0) or 0), "pcoupl": bool(barostat),
    }


def inp_dict(text: str) -> dict:
    """openmm_run.py input: `key = value  # comment`, keys case-insensitive (as omm_readinputs.py)."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = (p.strip() for p in line.split("=", 1))
            out[k.lower()] = v
    return out


def openmm_settings(text: str) -> dict:
    m = inp_dict(text)
    yes = lambda k: m.get(k, "no").lower() == "yes"  # noqa: E731
    nsteps = int(_num(m.get("nstep"), 0))
    mini = int(_num(m.get("mini_nstep"), 0))
    rest = []
    if yes("rest"):
        rest.append(f"położenia białka (szkielet {m.get('fc_bb', '?')}, łańcuchy boczne {m.get('fc_sc', '?')} "
                    "kJ/mol/nm²)")
        for k, what in (("fc_lpos", "lipidy"), ("fc_ldih", "dwuścienne lipidów"), ("fc_cdih", "dwuścienne cukrów")):
            if _num(m.get(k), 0):
                rest.append(f"{what} ({m[k]})")
    barostat = ""
    if yes("pcouple"):
        barostat = f"Monte Carlo {m.get('p_type', 'isotropic')} (co {m.get('p_freq', '15')} kroków)"
    vdw = m.get("vdw", "Force-switch")
    return {
        "integrator": "Langevin (LangevinIntegrator)" if nsteps else "", "minimize": mini > 0 and nsteps == 0,
        "dt_ps": _num(m.get("dt"), 0.002), "nsteps": nsteps, "min_steps": mini,
        "emtol": _num(m.get("mini_tol")) if mini else None,
        "temp": _num(m.get("temp"), 300.0) if nsteps else None,
        "thermostat": f"Langevin (tarcie {m.get('fric_coeff', '1')} 1/ps)" if nsteps else "",
        "barostat": barostat, "pressure": _num(m.get("p_ref")) if barostat else None,
        "restraints": "; ".join(rest), "constraints": m.get("cons", "HBonds"),
        "vdw": f"{vdw} {m.get('r_on', '1.0')}–{m.get('r_off', '1.2')} nm",
        "coulomb": f"{m.get('coulomb', 'PME')} {m.get('r_off', '1.2')} nm",
        "gen_vel": yes("gen_vel"), "traj_every": int(_num(m.get("nstdcd"), 0) or 0),
        "energy_every": int(_num(m.get("nstout"), 0) or 0), "pcoupl": bool(barostat),
    }


def namd_lines(text: str) -> list[tuple[str, str]]:
    """(keyword lower-case, value) of a NAMD config, comments and trailing `;` removed, `set` variables expanded."""
    variables, out = {}, []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(";").strip()
        if not line:
            continue
        bits = line.split(None, 1)
        key, value = bits[0], bits[1].strip() if len(bits) > 1 else ""
        value = re.sub(r"\$\{?(\w+)\}?", lambda mm: variables.get(mm.group(1).lower(), mm.group(0)), value)
        if key.lower() == "set":
            name, _, val = value.partition(" ")
            variables[name.lower()] = val.strip().rstrip(";").strip()
            out.append(("set " + name.lower(), val.strip()))
            continue
        out.append((key.lower(), value))
    return out


def namd_settings(text: str, read_ref=None) -> dict:
    kv, totals = {}, {"minimize": 0, "run": 0}
    for k, v in namd_lines(text):
        if k in totals:
            totals[k] += int(_num(v, 0) or 0)
        else:
            kv.setdefault(k, v)
    kv.update({k: str(v) for k, v in totals.items() if v})
    on = lambda k: kv.get(k, "off").lower() in ("on", "yes", "true")  # noqa: E731
    run = int(_num(kv.get("run"), 0) or 0)
    if not run and "numsteps" in kv and "minimize" not in kv:
        run = int(_num(kv["numsteps"], 0))
    mini = int(_num(kv.get("minimize"), 0) or 0)
    temp = _num(kv.get("langevintemp") or kv.get("reassigntemp") or kv.get("temperature"))
    thermostat = []
    if on("langevin"):
        thermostat.append(f"Langevin (tłumienie {kv.get('langevindamping', '?')} 1/ps)")
    if _num(kv.get("reassignfreq"), 0):
        thermostat.append(f"ponowne losowanie prędkości co {kv['reassignfreq']} kroków")
    if on("tcouple"):
        thermostat.append("Berendsen (tCouple)")
    barostat = ""
    if on("langevinpiston"):
        barostat = (f"Nosé–Hoover–Langevin piston (okres {kv.get('langevinpistonperiod', '?')} fs, zanik "
                    f"{kv.get('langevinpistondecay', '?')} fs"
                    f"{', elastyczna komórka' if on('useflexiblecell') else ''})")
    elif on("berendsenpressure"):
        barostat = "Berendsen"
    rest = ""
    if on("constraints"):
        rest = f"harmoniczne wg {kv.get('consref', '?')} (kolumna {kv.get('conskcol', 'B')}"
        scale = kv.get("constraintscaling")
        rest += f", skala {scale})" if scale else ")"
        if read_ref is not None and kv.get("conskfile"):
            ks = read_ref(kv["conskfile"], kv.get("conskcol", "B"))
            if ks:
                f = _num(scale, 1.0)
                rest += ": k = " + ", ".join(f"{k * f:g}" for k in ks) + " kcal/mol/Å²"
    if on("colvars") or kv.get("colvarsconfig"):
        rest += ("; " if rest else "") + "colvars (np. dwuścienne)"
    dt_fs = _num(kv.get("timestep"), 1.0)
    rigid = kv.get("rigidbonds", "none")
    return {
        "integrator": "Verlet + Langevin" if on("langevin") else "Verlet", "minimize": mini > 0 and run == 0,
        "dt_ps": dt_fs / 1000.0, "nsteps": run, "min_steps": mini, "emtol": None,
        "temp": temp if run else None, "thermostat": "; ".join(thermostat) if run else "",
        "barostat": barostat, "pressure": _num(kv.get("langevinpistontarget")) if barostat else None,
        "restraints": rest, "constraints": f"rigidBonds {rigid}" + (" (wiązania z H)" if rigid == "all" else ""),
        "vdw": (f"{'force-switch' if on('vdwforceswitching') else 'switch' if on('switching') else 'cut-off'} "
                f"{_num(kv.get('switchdist'), 0) / 10:g}–{_num(kv.get('cutoff'), 0) / 10:g} nm"),
        "coulomb": f"PME {_num(kv.get('cutoff'), 0) / 10:g} nm" if on("pme") else f"cut-off {kv.get('cutoff')} Å",
        "gen_vel": "temperature" in kv and "binvelocities" not in kv,
        "traj_every": int(_num(kv.get("dcdfreq"), 0) or 0), "energy_every": int(_num(kv.get("outputenergies"), 0) or 0),
        "pcoupl": bool(barostat), "output": kv.get("outputname", ""),
    }


SETTINGS = {"gromacs": gromacs_settings, "openmm": openmm_settings, "namd": namd_settings}


# --- the README as a list of steps ----------------------------------------------------------------------------------

@dataclass
class Step:
    name: str  # base name of the outputs: deffnm, restart stem, NAMD outputName
    engine: str
    kind: str  # min, equil, prod
    config: str  # input file the README gives the engine (mdp / inp), relative to the engine folder
    text: str  # its contents (for NAMD production: after the README's sed)
    virtual: bool = False  # made by sed/cp in the README, not a file of the package
    args: list = field(default_factory=list)  # gromacs: grompp options; openmm: openmm_run.py options
    maxwarn: str = "0"
    stdout: str | None = None
    script: str = ""  # openmm: path of openmm_run.py
    done_file: str = ""  # openmm: restart file written at the very end
    traj: str = ""  # trajectory file name
    settings: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)  # field -> (new, old) from params
    selected: bool = True
    title: str = ""
    chunk: int = 0  # production part number (1..n)
    finished: bool = False
    package_prod_ps: float = 0.0  # production length of the package as downloaded (before ns=)

    # the attributes md's progress/analysis code reads
    @property
    def nsteps(self) -> int:
        return int(self.settings.get("nsteps") or 0)

    @property
    def dt(self) -> float:
        return float(self.settings.get("dt_ps") or 0.0)

    @property
    def ps(self) -> float:
        return self.nsteps * self.dt

    @property
    def temp(self) -> float:
        return float(self.settings.get("temp") or 0.0)

    @property
    def dynamic(self) -> bool:
        return self.nsteps > 0

    @property
    def pcoupl(self) -> bool:
        return bool(self.settings.get("pcoupl"))

    @property
    def posres(self) -> bool:
        return bool(self.settings.get("restraints"))


def classify(name: str, config: str, s: dict) -> str:
    low = f"{name} {config}".lower()
    if s.get("minimize") or "minimi" in low:
        return "min"
    if "equil" in low:
        return "equil"
    if "prod" in low:
        return "prod"
    return "equil" if s.get("restraints") else "prod"


def _opts(tokens: list[str]) -> dict:
    out, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", t):
            val = tokens[i + 1] if i + 1 < len(tokens) and not (tokens[i + 1].startswith("-") and
                                                                 not re.fullmatch(r"-\d+", tokens[i + 1])) else ""
            out[t] = val
            i += 2 if val else 1
        else:
            i += 1
    return out


def _stem(path: str, exts=(".tpr", ".rst", ".chk", ".out", ".log", ".inp", ".mdp", ".dcd")) -> str:
    base = Path(path).name
    for e in exts:
        if base.endswith(e):
            return base[: -len(e)]
    return base


def gromacs_steps(commands, read) -> tuple[list[Step], list[str]]:
    steps, notes, pending = [], [], {}
    for cmd in commands:
        argv = cmd.argv
        k = next((i for i, a in enumerate(argv) if Path(a).name.startswith("gmx")), None)
        if k is None or k + 1 >= len(argv):
            notes.append(f"README linia {cmd.line}: pomijam „{cmd.text[:80]}”")
            continue
        sub, rest = argv[k + 1], argv[k + 2:]
        if sub == "grompp":
            opts = _opts(rest)
            tpr = opts.get("-o", "topol.tpr")
            keep, i = [], 0
            while i < len(rest):
                if rest[i] in ("-o", "-maxwarn", "-po"):
                    i += 2
                    continue
                keep.append(rest[i])
                i += 1
            pending[_stem(tpr)] = (keep, opts.get("-maxwarn", "0") or "0")
        elif sub == "mdrun":
            opts = _opts(rest)
            name = opts.get("-deffnm") or _stem(opts.get("-s", "topol.tpr"))
            args, maxwarn = pending.pop(name, ([], "0"))
            config = _opts(args).get("-f", "grompp.mdp")
            text = cmd.files.get(config) or read(config) or ""
            if not args and read(f"{name}.tpr") is None:
                notes.append(f"README linia {cmd.line}: mdrun {name} bez grompp i bez {name}.tpr – pomijam")
                continue
            s = gromacs_settings(text) if text else {}
            steps.append(Step(name, "gromacs", classify(name, config, s), config, text, config in cmd.files, args,
                              maxwarn, traj=f"{name}.xtc", settings=s))
        else:
            notes.append(f"README linia {cmd.line}: pomijam „gmx {sub}”")
    return steps, notes


def openmm_steps(commands, read) -> tuple[list[Step], list[str]]:
    steps, notes = [], []
    for cmd in commands:
        k = next((i for i, a in enumerate(cmd.argv) if a.endswith("openmm_run.py")), None)
        if k is None:
            notes.append(f"README linia {cmd.line}: pomijam „{cmd.text[:80]}”")
            continue
        args = cmd.argv[k + 1:]
        opts = _opts(args)
        config = opts.get("-i", "")
        text = cmd.files.get(config) or read(config) or ""
        done = opts.get("-orst") or opts.get("-ochk") or "output.rst"
        name = _stem(cmd.stdout) if cmd.stdout else _stem(done)
        s = openmm_settings(text)
        steps.append(Step(name, "openmm", classify(name, config, s), config, text, config in cmd.files, args,
                          stdout=cmd.stdout, script=cmd.argv[k], done_file=done,
                          traj=opts.get("-odcd") or ("output.dcd" if s.get("traj_every") else ""), settings=s))
    return steps, notes


def namd_steps(commands, read, read_ref=None) -> tuple[list[Step], list[str]]:
    steps, notes = [], []
    for cmd in commands:
        if not any(Path(a).name.startswith("namd") for a in cmd.argv):
            notes.append(f"README linia {cmd.line}: pomijam „{cmd.text[:80]}”")
            continue
        files = [a for a in cmd.argv[1:] if not a.startswith(("+", "-")) and not Path(a).name.startswith("namd")]
        if not files:
            notes.append(f"README linia {cmd.line}: namd bez pliku wejściowego – pomijam")
            continue
        config = files[-1]
        text = cmd.files.get(config) or read(config) or ""
        s = namd_settings(text, read_ref)
        name = s.get("output") or _stem(cmd.stdout or config)
        steps.append(Step(name, "namd", classify(name, config, s), config, text, config in cmd.files,
                          stdout=cmd.stdout or f"{name}.out", traj=f"{name}.dcd" if s.get("traj_every") else "",
                          settings=s))
    return steps, notes


# Solution Builder's README for each engine, used when the package came without one
STANDARD_README = {
    "gromacs": """
set init = step3_input
set mini_prefix = step4.0_minimization
set equi_prefix = step4.1_equilibration
set prod_prefix = step5_production
set prod_step   = step5
gmx grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -n index.ndx -maxwarn -1
gmx mdrun -v -deffnm ${mini_prefix}
gmx grompp -f ${equi_prefix}.mdp -o ${equi_prefix}.tpr -c ${mini_prefix}.gro -r ${init}.gro -p topol.top -n index.ndx
gmx mdrun -v -deffnm ${equi_prefix}
set cnt    = 1
set cntmax = 10
while ( ${cnt} <= ${cntmax} )
    @ pcnt = ${cnt} - 1
    set istep = ${prod_step}_${cnt}
    set pstep = ${prod_step}_${pcnt}
    if ( ${cnt} == 1 ) then
        set pstep = ${equi_prefix}
        gmx grompp -f ${prod_prefix}.mdp -o ${istep}.tpr -c ${pstep}.gro -p topol.top -n index.ndx
    else
        gmx grompp -f ${prod_prefix}.mdp -o ${istep}.tpr -c ${pstep}.gro -t ${pstep}.cpt -p topol.top -n index.ndx
    endif
    gmx mdrun -v -deffnm ${istep}
    @ cnt += 1
end
""",  # noqa: E501
    "openmm": """
set init = step3_input
set equi_prefix = step4_equilibration
set prod_prefix = step5_production
set prod_step   = step5
python -u openmm_run.py -i ${equi_prefix}.inp -t toppar.str -p ${init}.psf -c ${init}.crd -b sysinfo.dat -orst ${equi_prefix}.rst -odcd ${equi_prefix}.dcd > ${equi_prefix}.out
set cnt = 1
set cntmax = 10
while ( ${cnt} <= ${cntmax} )
    @ pcnt = ${cnt} - 1
    set istep = ${prod_step}_${cnt}
    set pstep = ${prod_step}_${pcnt}
    if ( ${cnt} == 1 ) set pstep = ${equi_prefix}
    python -u openmm_run.py -i ${prod_prefix}.inp -t toppar.str -p ${init}.psf -c ${init}.crd -irst ${pstep}.rst -orst ${istep}.rst -odcd ${istep}.dcd > ${istep}.out
    @ cnt += 1
end
""",  # noqa: E501
    "namd": """
set equi_prefix = step4_equilibration
set prod_prefix = step5_production
set prod_step   = step5
namd2 ${equi_prefix}.inp > ${equi_prefix}.out
set cnt    = 1
set cntmax = 10
while ( ${cnt} <= ${cntmax} )
    if ( ${cnt} == 1 ) then
        set outputname = "${prod_step}_${cnt}"
        sed "s/${prod_prefix}/${outputname}/" ${prod_prefix}.inp > ${prod_step}_run.inp
    else
        @ cntprev = ${cnt} - 1
        set inputname  = "${prod_step}_${cntprev}"
        set outputname = "${prod_step}_${cnt}"
        sed "s/${equi_prefix}/${inputname}/" ${prod_prefix}.inp | sed "s/${prod_prefix}/${outputname}/" > ${prod_step}_run.inp
    endif
    namd2 ${prod_step}_run.inp > ${outputname}.out
    @ cnt += 1
end
""",  # noqa: E501
}


def readme_version(text: str) -> str:
    m = re.search(r"CHARMM-GUI \(?[^)\n]*\)?\s*(v[\d.]+)", text)
    return m.group(1) if m else ""


def interpret_steps(engine: str, readme: str, read, force=None, read_ref=None):
    res = csh.interpret(readme, read, force)
    if engine == "gromacs":
        steps, notes = gromacs_steps(res.commands, read)
    elif engine == "openmm":
        steps, notes = openmm_steps(res.commands, read)
    else:
        steps, notes = namd_steps(res.commands, read, read_ref)
    return steps, res.notes + notes, res.variables


def parse_steps_param(text: str) -> list[str]:
    return [t.strip().lower().removeprefix("step") for t in re.split(r"[,; ]+", text or "") if t.strip()]


def step_selected(name: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    n = name.lower().removeprefix("step")
    return any(n == t or (n.startswith(t) and not n[len(t)].isdigit()) for t in tokens if len(n) >= len(t))


def plan(engine: str, readme: str | None, read, p: dict, read_ref=None) -> tuple[list[Step], list[str]]:
    """Steps of the README with the owner's ns/T applied; production is split into as many README chunks as
    needed (cntmax), each at most as long as the package's own chunk."""
    notes = []
    if not readme:
        readme = STANDARD_README[engine]
        notes.append("brak README w folderze – używam standardowego protokołu Solution Builder")
    try:
        steps, n2, variables = interpret_steps(engine, readme, read, read_ref=read_ref)
    except csh.CshError as exc:
        raise RecipeError(f"nie rozumiem README paczki ({exc})") from None
    if not steps:
        raise RecipeError(f"README folderu {engine}/ nie uruchamia żadnego kroku {ENGINE_TITLES[engine]}")
    prod = [s for s in steps if s.kind == "prod"]
    package_prod_ps = sum(s.ps for s in prod)
    ns = float(p.get("ns") or 0)
    if ns > 0 and prod and prod[0].ps > 0:
        want = max(1, math.ceil(ns * 1000 / prod[0].ps - 1e-9))
        if want != len(prod) and "cntmax" in variables:
            steps2, n3, _ = interpret_steps(engine, readme, read, force={"cntmax": want}, read_ref=read_ref)
            if sum(s.kind == "prod" for s in steps2) == want:
                steps, n2 = steps2, n3
        prod = [s for s in steps if s.kind == "prod"]
    notes += n2
    for i, s in enumerate(prod, 1):
        s.chunk = i
        s.package_prod_ps = package_prod_ps
    apply_params(steps, p)
    tokens = parse_steps_param(p.get("steps", ""))
    for s in steps:
        s.selected = step_selected(s.name, tokens)
        s.title = step_title(s, len(prod))
    if tokens and not any(s.selected for s in steps):
        raise RecipeError(f"steps={p.get('steps')}: README nie ma takich kroków (są: "
                          f"{', '.join(s.name for s in steps)})")
    return steps, notes


def step_title(s: Step, n_prod: int) -> str:
    if s.kind == "min":
        return "minimalizacja"
    if s.kind == "equil":
        ens = "NPT" if s.pcoupl else "NVT"
        return f"równowagowanie {ens}" + (" z więzami" if s.posres else "")
    return f"produkcja {s.chunk}/{n_prod}" if n_prod > 1 else "produkcja"


def apply_params(steps: list[Step], p: dict):
    """Fill Step.changes (and the settings the run will use) from params ns and T."""
    ns, T = float(p.get("ns") or 0), float(p.get("T") or 0)
    prod = [s for s in steps if s.kind == "prod" and s.dt]
    for s in steps:
        s.changes = {}
        if T > 0 and s.dynamic and s.temp and abs(s.temp - T) > 1e-6:
            s.changes["temp"] = (T, s.temp)
    if ns > 0 and prod:
        per = int(round(ns * 1000 / len(prod) / prod[0].dt))
        for s in prod:
            if per != s.nsteps:
                s.changes["nsteps"] = (per, s.nsteps)
    if prod:
        nst = [s.changes.get("nsteps", (s.nsteps,))[0] for s in prod]
        frames = sum(n // s.settings["traj_every"] for n, s in zip(nst, prod, strict=True)
                     if s.settings.get("traj_every"))
        if frames < MIN_PROD_FRAMES:
            want = math.ceil(MIN_PROD_FRAMES / len(prod))
            for n, s in zip(nst, prod, strict=True):
                old = s.settings.get("traj_every") or 0
                new = nice_interval(n, want) if n >= 10 * want else max(1, n // want)
                if old and new < old:
                    s.changes["traj_every"] = (new, old)
    for s in steps:
        for key, (new, _) in s.changes.items():
            s.settings[key] = new


# --- writing the overrides into copies ------------------------------------------------------------------------------

def _fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def edit_mdp(text: str, changes: dict) -> str:
    """Replace values of mdp keys (- and _ alike, any case), append missing ones; comments are kept."""
    lines = text.splitlines()
    done = set()
    for i, line in enumerate(lines):
        body, sep, comment = line.partition(";")
        if "=" not in body:
            continue
        key, _, value = body.partition("=")
        k = _norm(key.strip())
        if k in changes:
            new = changes[k](value.strip()) if callable(changes[k]) else changes[k]
            lines[i] = f"{key.rstrip():<23} = {new}" + (f" ;{comment}" if sep else "")
            done.add(k)
    for k, v in changes.items():
        if k not in done and not callable(v):
            lines.append(f"{k:<23} = {v}")
    return "\n".join(lines) + "\n"


def gromacs_edit(text: str, changes: dict) -> str:
    edits = {}
    if "temp" in changes:
        t = _fmt(changes["temp"][0])
        edits["ref-t"] = lambda v: " ".join(t for _ in v.split()) or t
        if "gen-temp" in mdp_dict(text):
            edits["gen-temp"] = t
        # annealing schedules keep their own temperatures; they are listed in the protocol table
    if "nsteps" in changes:
        edits["nsteps"] = str(changes["nsteps"][0])
    if "traj_every" in changes:
        key = "nstxtcout" if "nstxtcout" in mdp_dict(text) else "nstxout-compressed"
        edits[key] = str(changes["traj_every"][0])
    return edit_mdp(text, edits)


def openmm_edit(text: str, changes: dict) -> str:
    keys = {}
    if "temp" in changes:
        t = _fmt(changes["temp"][0])
        keys["temp"] = t
        if "gen_temp" in inp_dict(text):
            keys["gen_temp"] = t
    if "nsteps" in changes:
        keys["nstep"] = str(changes["nsteps"][0])
    if "traj_every" in changes:
        keys["nstdcd"] = str(changes["traj_every"][0])
    lines, done = text.splitlines(), set()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(\w+)(\s*=\s*)(\S+)(.*)$", line)
        if m and m.group(2).lower() in keys:
            k = m.group(2).lower()
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{keys[k]:<{len(m.group(4))}}{m.group(5)}"
            done.add(k)
    lines += [f"{k:<12}= {v}" for k, v in keys.items() if k not in done]
    return "\n".join(lines) + "\n"


_NAMD_TEMP_KEYS = ("langevintemp", "reassigntemp", "temperature", "langevinpistontemp", "rescaletemp",
                   "loweandersentemp")


def namd_edit(text: str, changes: dict) -> str:
    lines = text.splitlines()
    t = _fmt(changes["temp"][0]) if "temp" in changes else None
    has_set_temp = t is not None and any(re.match(r"\s*set\s+temp\s", ln, re.I) for ln in lines)
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(\S+)(\s+)([^;#\s]+)(.*)$", line)
        if not m:
            continue
        key, value = m.group(2).lower(), m.group(4)
        new = None
        if t is not None and key == "set":
            mm = re.match(r"^(\s*set\s+temp\s+)([^;#\s]+)(.*)$", line, re.I)
            if mm:
                lines[i] = mm.group(1) + t + mm.group(3)
            continue
        if t is not None and not has_set_temp and key in _NAMD_TEMP_KEYS and _num(value) is not None:
            new = t
        if "nsteps" in changes and key in ("run", "numsteps") and _num(value) is not None:
            n = changes["nsteps"][0]
            new = str(n if key == "run" or int(_num(value)) < 10 * n else int(_num(value)))
        if "traj_every" in changes and key == "dcdfreq":
            new = str(changes["traj_every"][0])
        if new is not None:
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{new}{m.group(5)}"
    return "\n".join(lines) + "\n"


EDITORS = {"gromacs": gromacs_edit, "openmm": openmm_edit, "namd": namd_edit}


def bot_config_name(s: Step) -> str:
    """Copy written for a step: NAMD gets one per step (the README rewrites one file per production part)."""
    if s.engine == "namd":
        return f"bot_{s.name}.inp"
    return "bot_" + Path(s.config).name


def needs_copy(s: Step) -> bool:
    return bool(s.changes) or (s.engine == "namd" and s.virtual)


def config_for_run(s: Step) -> tuple[str, str]:
    """(file name the engine reads, its text)."""
    if not needs_copy(s):
        return s.config, s.text
    return bot_config_name(s), EDITORS[s.engine](s.text, s.changes) if s.changes else s.text


# --- the system the package describes -------------------------------------------------------------------------------

AMINO = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HSD", "HSE", "HSP", "ILE", "LEU", "LYS",
         "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN"}
WATERS = {"TIP3": "TIP3P w wersji CHARMM (z LJ na H)", "TIP4": "TIP4P", "SPC": "SPC", "SOL": "woda (SOL)",
          "WAT": "woda (WAT)", "TP3M": "TIP3P (CHARMM)", "OPC": "OPC", "TIP5": "TIP5P"}
IONS = {"SOD": "Na⁺", "POT": "K⁺", "CLA": "Cl⁻", "CAL": "Ca²⁺", "MG": "Mg²⁺", "CES": "Cs⁺", "LIT": "Li⁺",
        "RUB": "Rb⁺", "ZN2": "Zn²⁺", "BAR": "Ba²⁺", "CD2": "Cd²⁺", "NA": "Na⁺", "CL": "Cl⁻", "K": "K⁺",
        "Na+": "Na⁺", "Cl-": "Cl⁻", "K+": "K⁺"}
FORCE_FIELDS = {"c36m": "CHARMM36m", "c36": "CHARMM36", "c27": "CHARMM22/CMAP (c27)", "ff14sb": "AMBER ff14SB",
                "ff19sb": "AMBER ff19SB", "amber": "AMBER", "opls": "OPLS-AA/M", "drude": "Drude polarowalne"}


def parse_psf(text: str) -> dict:
    """Atoms of a CHARMM PSF: count, residues per name, mass, charge of the protein and of everything."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "!NATOM" in ln), None)
    if start is None:
        return {}
    n = int(lines[start].split()[0])
    residues, mass, charge, qprot, protein_res, segs = {}, 0.0, 0.0, 0.0, set(), {}
    last = None
    for ln in lines[start + 1:start + 1 + n]:
        b = ln.split()
        if len(b) < 8:
            continue
        seg, resid, resn = b[1], b[2], b[3]
        q, m = float(b[6]), float(b[7])
        mass += m
        charge += q
        key = (seg, resid, resn)
        if key != last:
            residues[resn] = residues.get(resn, 0) + 1
            segs.setdefault(seg, set()).add(resid)
            if resn in AMINO:
                protein_res.add(key)
            last = key
        if resn in AMINO:
            qprot += q
    return {"atoms": n, "residues": residues, "mass_amu": mass, "charge": round(charge, 3),
            "protein_charge": round(qprot, 3), "protein_residues": len(protein_res),
            "segments": {k: len(v) for k, v in segs.items()}}


def gro_info(text: str) -> dict:
    lines = text.rstrip("\n").splitlines()
    try:
        box = [float(x) * 10 for x in lines[-1].split()[:3]]
        return {"atoms": int(lines[1].split()[0]), "box_A": box}
    except (IndexError, ValueError):
        return {}


def top_molecules(text: str) -> dict:
    parts = re.split(r"^\s*\[\s*molecules\s*\]\s*$", text, flags=re.M)
    out = {}
    if len(parts) < 2:
        return out
    for line in parts[-1].splitlines():
        line = line.split(";", 1)[0].strip()
        if line.startswith("["):
            break
        b = line.split()
        if len(b) == 2 and b[1].isdigit():
            out[b[0]] = out.get(b[0], 0) + int(b[1])
    return out


def charmm_sets(text: str) -> dict:
    """`SET A = 71` / ` set a 77` lines of CHARMM stream files, keys lower-case."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(?:if .* )?set\s+(\w+)\s*=?\s*(\S+)", line, re.I)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def describe_system(root: Path, edir: Path, engine: str, readme: str) -> dict:
    """What CHARMM-GUI built: force field, water, ions, box, atoms (from the package files)."""
    info: dict = {"version": readme_version(readme or ""), "notes": []}
    cfg = {}
    if (root / "input.config.dat").exists():
        try:
            cfg = json.loads((root / "input.config.dat").read_text())
        except ValueError:
            cfg = {}
    ff_type = ((cfg.get("forcefield") or {}).get("type") or "") if isinstance(cfg.get("forcefield"), dict) else ""
    toppar_text = " ".join(t for t in (_read(edir / "toppar.str"), _read(root / "toppar.str")) if t)
    for inp in sorted(edir.glob("step*.inp"))[:1]:
        toppar_text += " " + (_read(inp) or "")
    ff, source = "", ""
    if ff_type:
        ff, source = FORCE_FIELDS.get(ff_type.lower(), ff_type), "input.config.dat"
    elif "all36m_prot" in toppar_text:
        ff, source = "CHARMM36m", "toppar (par_all36m_prot.prm)"
    elif "all36_prot" in toppar_text:
        ff, source = "CHARMM36", "toppar (par_all36_prot.prm)"
    elif "all22_prot" in toppar_text:
        ff, source = "CHARMM22/CMAP", "toppar (par_all22_prot.prm)"
    elif re.search(r"CHARMM36m?", readme or ""):
        ff, source = re.search(r"CHARMM36m?", readme).group(0), "README"
    info["ff"], info["ff_source"] = ff or "nieznane", source
    if cfg.get("hmr"):
        info["notes"].append("HMR (przeniesienie masy na wodory) włączone")
    psf_path = next((p for p in sorted(edir.glob("step*_input.psf"))), None) or next(
        (p for p in sorted(root.glob("step*_pbcsetup.psf"))), None)
    psf = parse_psf(_read(psf_path) or "") if psf_path else {}
    info["psf"] = psf_path.name if psf_path else ""
    counts = dict(psf.get("residues", {}))
    if engine == "gromacs" and (edir / "topol.top").exists():
        mols = top_molecules((edir / "topol.top").read_text(errors="replace"))
        info["molecules"] = mols
        for k, v in mols.items():
            if k in WATERS or k in IONS:
                counts[k] = v
    water = {k: v for k, v in counts.items() if k in WATERS}
    ions = {k: v for k, v in counts.items() if k in IONS}
    info["water_model"] = ", ".join(WATERS[k] for k in water) or "brak"
    info["waters"] = sum(water.values())
    info["ions"] = ions
    info["atoms"] = psf.get("atoms")
    info["mass_amu"] = psf.get("mass_amu")
    info["charge_total"] = psf.get("charge")
    info["protein_charge"] = psf.get("protein_charge")
    info["protein_residues"] = psf.get("protein_residues")
    info["segments"] = psf.get("segments", {})
    conc = None
    for name in ("step2.2_ions_count.str", "step2_ions_count.str"):
        t = _read(root / name)
        if t:
            sets = charmm_sets(t)
            if "conc" in sets:
                conc = _num(sets["conc"])
                info["conc_source"] = name
                break
    if conc is None and ions and info["waters"]:
        pos = sum(v for k, v in ions.items() if IONS[k].endswith("⁺"))
        neg = sum(v for k, v in ions.items() if IONS[k].endswith("⁻"))
        conc = round(min(pos, neg) * 55.5 / info["waters"], 3)
        info["conc_source"] = "szacunek z liczby jonów i wód"
    info["conc"] = conc
    box = {}
    for t, src in ((_read(edir / "step3_input.str"), "step3_input.str"),
                   (_read(root / "step3_pbcsetup.str"), "step3_pbcsetup.str"),
                   (_read(root / "step2.1_waterbox.prm"), "step2.1_waterbox.prm")):
        if t:
            sets = charmm_sets(t)
            if "a" in sets:
                box = {"shape": sets.get("xtltype", sets.get("boxtype", "")).lower(),
                       "abc": [_num(sets.get(x)) for x in "abc"],
                       "angles": [_num(sets.get(x), 90.0) for x in ("alpha", "beta", "gamma")], "source": src}
                break
    if not box:
        for p in (edir / "sysinfo.dat",):
            t = _read(p)
            try:
                dims = json.loads(t).get("dimensions") if t else None
            except ValueError:
                dims = None
            if dims:
                box = {"shape": "", "abc": [float(x) for x in dims[:3]], "angles": [float(x) for x in dims[3:6]],
                       "source": "sysinfo.dat"}
    if not box and cfg.get("dimensions"):
        d = cfg["dimensions"]
        box = {"shape": "", "abc": [float(x) for x in d[:3]], "angles": [float(x) for x in d[3:6]],
               "source": "input.config.dat"}
    if engine == "gromacs":
        gro = next((p for p in sorted(edir.glob("step*_input.gro"))), None)
        g = gro_info(_read(gro) or "") if gro else {}
        if g:
            info["atoms"] = info["atoms"] or g["atoms"]
            if not box:
                box = {"shape": "", "abc": g["box_A"], "angles": [90.0] * 3, "source": gro.name}
    info["box"] = box
    return info


SHAPES = {"cubic": "kostka (cubic)", "rect": "prostopadłościan", "tetragonal": "prostopadłościan (tetragonal)",
          "orthorhombic": "prostopadłościan", "octa": "ośmiościan ścięty", "hexa": "graniastosłup sześciokątny"}


def box_text(box: dict) -> str:
    if not box:
        return "nieznane"
    a = box["abc"]
    shape = SHAPES.get(box.get("shape", ""), box.get("shape", ""))
    dims = " × ".join(f"{x / 10:.2f}" for x in a if x is not None) + " nm"
    angles = box.get("angles") or []
    if angles and any(abs(x - 90) > 0.01 for x in angles if x is not None):
        dims += " (kąty " + ", ".join(f"{x:g}" for x in angles) + "°)"
    return (shape + ", " if shape else "") + dims


def ions_text(info: dict) -> str:
    ions = info.get("ions") or {}
    if not ions:
        return "brak"
    text = ", ".join(f"{n} {IONS[k]} ({k})" for k, n in ions.items())
    if info.get("conc") is not None:
        text += f"; stężenie {info['conc']:g} M"
        if info.get("conc_source", "").startswith("szacunek"):
            text += " (szacunek)"
    return text
