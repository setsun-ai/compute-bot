"""Running the package's steps: GROMACS through md_step (skip finished, resume from .cpt), OpenMM through the
package's own openmm_run.py (skip a step whose restart file is complete, start the next one from it), NAMD with
namd3/namd2 +pN (skip a step that ended). PROGRESS lines with the end of the step and of the whole run."""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from ..md.gmxtools import RecipeError, fatal_error, fmt_duration, md_step_path, parse_progress
from . import package as K

# --- is a step done -------------------------------------------------------------------------------------------------

def _text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def gromacs_done(edir: Path, s: K.Step) -> bool:
    return (edir / f"{s.name}.gro").exists() and "Finished mdrun" in _text(edir / f"{s.name}.log")


def openmm_done(edir: Path, s: K.Step) -> bool:
    """openmm_run.py writes the restart file only after the last step; an XML state must be complete."""
    f = edir / s.done_file
    if not f.exists() or f.stat().st_size == 0:
        return False
    if f.suffix == ".chk":
        return True
    with open(f, "rb") as fh:
        fh.seek(max(0, f.stat().st_size - 200))
        return b"</State>" in fh.read()


def namd_done(edir: Path, s: K.Step) -> bool:
    out = _text(edir / (s.stdout or f"{s.name}.out"))
    return (edir / f"{s.name}.coor").exists() and ("End of program" in out or "WallClock:" in out)


DONE = {"gromacs": gromacs_done, "openmm": openmm_done, "namd": namd_done}


def done(edir: Path, s: K.Step) -> bool:
    return DONE[s.engine](edir, s)


# --- progress -------------------------------------------------------------------------------------------------------

class Eta:
    """PROGRESS lines: step i/N, % of the step, end of the step and of everything left (rate in steps/s carried over
    from step to step, since the cost of one step barely depends on dt)."""

    def __init__(self, steps: list, every: float = 30.0, now=time.time):
        self.steps = [s for s in steps if s.selected]
        self.every, self.now = every, now
        self.rate = None
        self.current = None
        self.first = None
        self.last_emit = -1e18

    def start(self, s) -> str:
        self.current, self.first, self.last_emit = s, None, -1e18
        i = self.steps.index(s) + 1
        what = f"{s.title} ({s.name}"
        what += f", {s.ps:g} ps, {s.temp:g} K)" if s.dynamic else ")"
        return f"PROGRESS: krok {i}/{len(self.steps)}: {what} – start" + self._total(0)

    def _later(self) -> int:
        i = self.steps.index(self.current)
        return sum(s.nsteps for s in self.steps[i + 1:] if s.dynamic and not getattr(s, "finished", False))

    def _total(self, step: int) -> str:
        if not self.rate or self.current is None or not self.current.dynamic:
            return ""
        left = (max(self.current.nsteps - step, 0) + self._later()) / self.rate
        end = datetime.now() + timedelta(seconds=left)
        return f"; do końca całości ~{fmt_duration(left)} (ok. {end:%d.%m %H:%M})"

    def update(self, step: int, remaining: float | None = None, rate: float | None = None,
               extra: str = "") -> str | None:
        """step = MD steps done in the current step (minimisation: pass extra only)."""
        t = self.now()
        if rate:
            self.rate = rate
        elif self.first is None:
            self.first = (t, step)
        elif t - self.first[0] >= 5 and step > self.first[1]:
            self.rate = (step - self.first[1]) / (t - self.first[0])
        if t - self.last_emit < self.every:
            return None
        self.last_emit = t
        s = self.current
        i = self.steps.index(s) + 1
        head = f"PROGRESS: krok {i}/{len(self.steps)}: {s.title}"
        if not s.dynamic or step < 0:
            return f"{head}{extra}"
        pct = 100.0 * step / max(s.nsteps, 1)
        msg = f"{head} {math.floor(min(pct, 100))}% (krok {step}/{s.nsteps})"
        if remaining is None and self.rate:
            remaining = max(s.nsteps - step, 0) / self.rate
        if remaining is not None:
            end = datetime.now() + timedelta(seconds=remaining)
            msg += f", koniec kroku ~{fmt_duration(remaining)} (ok. {end:%d.%m %H:%M})"
        return msg + extra + self._total(step)


# --- line parsers ---------------------------------------------------------------------------------------------------

def parse_clock(text: str) -> float | None:
    """OpenMM's 'Time Remaining': '1:02:03', '5:07', '2 days, 1:00:00' or '--'."""
    text = text.strip().strip('"')
    days = 0
    m = re.match(r"(\d+)\s+days?,\s*(.*)$", text)
    if m:
        days, text = int(m.group(1)), m.group(2)
    parts = text.split(":")
    if not all(p.strip().isdigit() for p in parts) or not parts:
        return None
    secs = 0
    for p in parts:
        secs = secs * 60 + int(p)
    return days * 86400 + secs


class OpenMMLines:
    """StateDataReporter rows on stdout (tab- or comma-separated, header line starting with #")."""

    def __init__(self):
        self.header = None

    def __call__(self, line: str) -> dict | None:
        if line.startswith('#"'):
            sep = "\t" if "\t" in line else ","
            self.header = [h.strip().strip('"') for h in line[1:].split(sep)]
            self.sep = sep
            return None
        if self.header is None:
            return None
        vals = line.split(self.sep)
        if len(vals) != len(self.header):
            return None
        row = dict(zip(self.header, (v.strip() for v in vals), strict=True))
        try:
            step = int(float(row.get("Step", "nan")))
        except ValueError:
            return None
        out = {"step": step}
        if "Time Remaining" in row:
            out["remaining"] = parse_clock(row["Time Remaining"])
        if "Speed (ns/day)" in row:
            try:
                out["nsday"] = float(row["Speed (ns/day)"])
            except ValueError:
                pass
        return out


_NAMD_TIMING = re.compile(r"^TIMING:\s+(\d+)\s+CPU:.*?Wall:\s+[\d.]+,\s+([\d.eE+-]+)/step,"
                          r"\s+([\d.eE+-]+)\s+hours remaining")


def parse_namd_timing(line: str) -> dict | None:
    m = _NAMD_TIMING.match(line)
    if not m:
        return None
    return {"step": int(m.group(1)), "s_per_step": float(m.group(2)), "remaining": float(m.group(3)) * 3600}


# --- running a command with live output -----------------------------------------------------------------------------

def stream(cmd: list[str], cwd: Path, env: dict, out_path: Path, on_line, keep=lambda line: True) -> tuple[int, str]:
    """Run cmd, feed every output line to on_line, write kept lines to out_path. Returns (exit code, kept text)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    kept, buf = [], b""
    with open(out_path, "wb") as log:
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"\r\n|[\r\n]", buf)
            for raw in lines:
                text = raw.decode(errors="replace").rstrip()
                if not text.strip():
                    continue
                msg = on_line(text)
                if msg:
                    print(msg, flush=True)
                if keep(text):
                    log.write(raw + b"\n")
                    log.flush()
                    kept.append(text)
        if buf.strip():
            log.write(buf + b"\n")
            kept.append(buf.decode(errors="replace"))
    return proc.wait(), "\n".join(kept)


def _need(edir: Path, files, s: K.Step):
    for f in files:
        if f and not (edir / f).exists():
            raise RecipeError(f"krok {s.name} potrzebuje {f}, a go nie ma – policz najpierw wcześniejsze kroki "
                              "(steps=) albo wyślij całą paczkę z CHARMM-GUI")


def write_config(edir: Path, s: K.Step) -> str:
    """Write the step's input copy (when ns/T changed it, or NAMD's sed-made file); returns the file to use."""
    name, text = K.config_for_run(s)
    if name != s.config:
        path = edir / name
        if not path.exists() or path.read_text(errors="replace") != text:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text)
            os.replace(tmp, path)
    return name


def _replace_opt(args: list[str], opt: str, value: str) -> list[str]:
    args = list(args)
    if opt in args:
        args[args.index(opt) + 1] = value
    return args


# --- GROMACS --------------------------------------------------------------------------------------------------------

def grompp_maxwarn(value: str) -> str:
    """CHARMM-GUI writes -maxwarn -1 (= ignore all warnings); GROMACS ≥ 2023 wants a positive number."""
    try:
        n = int(value)
    except ValueError:
        return "0"
    return "1000" if n < 0 else str(n)


def run_gromacs(s: K.Step, edir: Path, env: dict, eta: Eta):
    opts = K._opts(s.args)
    if not (edir / f"{s.name}.tpr").exists():
        _need(edir, [opts.get(o) for o in ("-c", "-r", "-t", "-p", "-n")], s)
    cfg = write_config(edir, s)
    args = _replace_opt(s.args, "-f", cfg)
    env = dict(env, MD_MAXWARN=grompp_maxwarn(s.maxwarn))
    print(eta.start(s), flush=True)

    def on_line(text):
        info = parse_progress(text)
        if info is None:
            return None
        if "fmax" in info:
            return eta.update(-1, extra=f", krok {info['step']}, Fmax {info['fmax']:.3g}, Epot {info['epot']:.4g} "
                                        "kJ/mol")
        return eta.update(info["step"], remaining=info.get("remaining"))

    rc, text = stream([md_step_path(), s.name, *args], edir, env, edir / f"{s.name}.out", on_line,
                      keep=lambda t: parse_progress(t) is None)
    for line in text.splitlines():
        if line.startswith("md_step:"):
            print(line, flush=True)
    if rc != 0:
        raise RecipeError(f"krok {s.name} przerwany błędem: {fatal_error(text)} (log: {edir.name}/{s.name}.out, "
                          f"{edir.name}/{s.name}.log)")
    if not gromacs_done(edir, s):
        raise RecipeError(f"krok {s.name} nie skończył się (brak {s.name}.gro) – zobacz {edir.name}/{s.name}.log")


# --- OpenMM ---------------------------------------------------------------------------------------------------------

_ALLOWED = re.compile(r"""^DEFAULT_PLATFORMS\s*=\s*[^\n]*$""", re.M)
_PLATFORM = re.compile(r"""Platform\.getPlatformByName\(\s*['"](?:CUDA|OpenCL|HIP|Metal)['"]\s*\)""")


def _allow_cpu(match):
    line = match[0].rstrip()
    if "CPU" in line:
        return line
    if line[-1] in ")]":  # ('CUDA', 'OpenCL') or ['CUDA', 'OpenCL']
        return line[:-1].rstrip().rstrip(",") + ", 'CPU'" + line[-1]
    return line + ", 'CPU'"


def openmm_script(edir: Path, script: str) -> tuple[str, list[str]]:
    """The package's openmm_run.py on the CPU: newer versions take --platform, older ones hard-code CUDA and get a
    patched copy bot_openmm_run.py (only the platform lines differ)."""
    src = _text(edir / script)
    if not src:
        raise RecipeError(f"brak {script} w folderze openmm/ paczki")
    patched = _PLATFORM.sub("Platform.getPlatformByName('CPU')", src)
    patched = re.sub(r"prop\s*=\s*dict\(\s*Cuda\w*\s*=[^)]*\)", "prop = dict()", patched)
    # CHARMM-GUI 3.5 takes --platform but lets through only the GPU names it lists here
    patched = _ALLOWED.sub(_allow_cpu, patched)
    # OpenMM 8: the minimiser tolerance is a force; older scripts give it in energy units (same number)
    patched = re.sub(r"(tolerance\s*=\s*[^,)]*?\*\s*kilojoule)\s*/\s*mole\b(?!\s*/)", r"\1/(mole*nanometer)", patched)
    extra = ["--platform", "CPU"] if "--platform" in src else []
    if patched == src:
        return script, extra
    name = "bot_" + Path(script).name
    path = edir / name
    if _text(path) != patched:
        path.write_text(patched)
    return name, extra


def run_openmm(s: K.Step, edir: Path, env: dict, eta: Eta, python: str = sys.executable):
    opts = K._opts(s.args)
    _need(edir, [opts.get(o) for o in ("-irst", "-ichk", "-icrst", "-p", "-c", "-t", "-b")], s)
    cfg = write_config(edir, s)
    script, extra = openmm_script(edir, s.script)
    args = _replace_opt(s.args, "-i", cfg)
    print(eta.start(s), flush=True)
    parse = OpenMMLines()
    state = {"mini": False}

    def on_line(text):
        if text.startswith("Energy minimization"):
            state["mini"] = True
            return eta.update(-1, extra=f", minimalizacja ({s.settings.get('min_steps', 0)} kroków)")
        info = parse(text)
        if info is None:
            return None
        rate = info["nsday"] / (s.dt * 1e-3) / 86400 if info.get("nsday") and s.dt else None
        return eta.update(info["step"], remaining=info.get("remaining"), rate=rate)

    out = edir / (s.stdout or f"{s.name}.out")
    rc, text = stream([python, "-u", script, *args, *extra], edir, env, out, on_line)
    if rc != 0:
        tail = [ln for ln in text.splitlines() if ln.strip()][-3:]
        raise RecipeError(f"krok {s.name} (OpenMM) przerwany błędem: {' | '.join(tail)[:500]} "
                          f"(log: {edir.name}/{out.name})")
    if not openmm_done(edir, s):
        raise RecipeError(f"krok {s.name} nie zapisał {s.done_file} – zobacz {edir.name}/{out.name}")


# --- NAMD -----------------------------------------------------------------------------------------------------------

def namd_inputs(text: str) -> list[str]:
    files = []
    for key, value in K.namd_lines(text):
        if key in ("structure", "coordinates", "bincoordinates", "binvelocities", "extendedsystem", "consref",
                   "conskfile") and value and "$" not in value:
            files.append(value.split()[0])
    return files


def run_namd(s: K.Step, edir: Path, env: dict, eta: Eta, binary: str, threads: int):
    cfg = write_config(edir, s)
    _need(edir, namd_inputs(K.config_for_run(s)[1]), s)
    print(eta.start(s), flush=True)
    min_steps = int(s.settings.get("min_steps") or 0)

    def on_line(text):
        info = parse_namd_timing(text)
        if info is None:
            if text.startswith("MINIMIZER") or text.startswith("LINE MINIMIZER"):
                return eta.update(-1, extra=f", minimalizacja ({min_steps} kroków)")
            return None
        rate = 1.0 / info["s_per_step"] if info["s_per_step"] > 0 else None
        return eta.update(max(info["step"] - min_steps, 0), remaining=info["remaining"], rate=rate)

    out = edir / (s.stdout or f"{s.name}.out")
    rc, text = stream([binary, f"+p{threads}", cfg], edir, env, out, on_line)
    if rc != 0 or not namd_done(edir, s):
        err = [ln for ln in text.splitlines() if "ERROR" in ln or "FATAL" in ln.upper()][-2:]
        tail = err or [ln for ln in text.splitlines() if ln.strip()][-3:]
        raise RecipeError(f"krok {s.name} (NAMD) przerwany błędem: {' | '.join(tail)[:500]} "
                          f"(log: {edir.name}/{out.name})")


# --- all steps ------------------------------------------------------------------------------------------------------

def run_all(steps: list[K.Step], edir: Path, env: dict, threads: int, namd: str | None = None, eta: Eta | None = None):
    eta = eta or Eta(steps)
    for s in steps:
        s.finished = done(edir, s)
    for s in steps:
        if not s.selected:
            continue
        if s.finished:
            print(f"PROGRESS: {s.name} już policzony, pomijam", flush=True)
            continue
        if s.engine == "gromacs":
            run_gromacs(s, edir, env, eta)
        elif s.engine == "openmm":
            run_openmm(s, edir, env, eta)
        else:
            run_namd(s, edir, env, eta, namd, threads)
        s.finished = True
