"""Running GROMACS tools non-interactively and reading what they write (parsers are pure functions)."""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


class RecipeError(RuntimeError):
    """Something the owner can fix, in Polish."""


def gmx_env(threads: int) -> dict:
    env = dict(os.environ)
    env["GMX_MAXBACKUP"] = "-1"  # no #file.1# backups when a step is redone
    env["OMP_NUM_THREADS"] = str(threads)
    env["COMPUTE_THREADS"] = str(threads)
    return env


def fatal_error(text: str) -> str:
    """The 'Fatal error:' paragraph GROMACS prints, else the last non-empty lines."""
    m = re.search(r"(?:Fatal error|Error in user input):?\s*\n(.*?)(?:\n\s*\n|For more information)", text, re.S)
    if m:
        return " ".join(m.group(1).split())[:600]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "GROMACS reminds you" not in ln]
    return " | ".join(lines[-4:])[:600]


def run_gmx(args: list[str], cwd: Path, env: dict, stdin: str = "", log: Path | None = None,
            check: bool = True) -> subprocess.CompletedProcess:
    """gmx <args> with `stdin` piped in (group choices), output appended to `log`."""
    proc = subprocess.run(["gmx", *args], cwd=cwd, input=stdin, text=True, capture_output=True, env=env)
    if log is not None:
        with open(log, "a") as fh:
            fh.write(f"$ gmx {' '.join(args)}" + (f"   <<< {stdin!r}" if stdin else "") + "\n")
            fh.write(proc.stdout + proc.stderr + "\n")
    if check and proc.returncode != 0:
        raise RecipeError(f"gmx {args[0]} nie powiódł się: {fatal_error(proc.stdout + proc.stderr)}")
    return proc


def fresh(out: Path, *inputs: Path) -> bool:
    """True when `out` exists and is not older than any existing input (analysis outputs are reused)."""
    if not out.exists() or out.stat().st_size == 0:
        return False
    t = out.stat().st_mtime
    return all(not i.exists() or i.stat().st_mtime <= t for i in inputs)


# --- parsers --------------------------------------------------------------------------------------------------------

@dataclass
class Xvg:
    legends: list
    xlabel: str
    ylabel: str
    data: np.ndarray  # first column = x

    @property
    def x(self) -> np.ndarray:
        return self.data[:, 0]

    def column(self, legend: str) -> np.ndarray:
        return self.data[:, 1 + self.legends.index(legend)]

    def has(self, legend: str) -> bool:
        return legend in self.legends


def parse_xvg(text: str) -> Xvg:
    legends, xlabel, ylabel, rows = {}, "", "", []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s == "&":
            continue
        if s.startswith("@"):
            m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', s)
            if m:
                legends[int(m.group(1))] = m.group(2)
            m = re.match(r'@\s+([xy])axis\s+label\s+"(.*)"', s)
            if m:
                xlabel, ylabel = (m.group(2), ylabel) if m.group(1) == "x" else (xlabel, m.group(2))
            continue
        try:
            rows.append([float(v) for v in s.split()])
        except ValueError:
            continue
    width = max((len(r) for r in rows), default=1)
    data = np.array([r for r in rows if len(r) == width], dtype=float).reshape(-1, width)
    names = [legends.get(i, f"y{i}") for i in range(width - 1)]
    return Xvg(names, xlabel, ylabel, data)


def read_xvg(path: Path) -> Xvg:
    return parse_xvg(Path(path).read_text())


def parse_performance(log: str) -> float | None:
    """ns/day from the last 'Performance:' line of an mdrun log (a resumed run appends one per part)."""
    found = re.findall(r"^Performance:\s+([\d.]+)", log, re.M)
    return float(found[-1]) if found else None


def parse_em_log(log: str) -> dict:
    out = {"converged": False, "steps": None, "epot": None, "fmax": None, "emtol": None}
    m = re.search(r"converged to Fmax < ([\d.eE+-]+) in (\d+) steps", log)
    if m:
        out.update(converged=True, emtol=float(m.group(1)), steps=int(m.group(2)))
    else:
        m = re.search(r"(?:converged to machine precision|did not converge to Fmax < [\d.eE+-]+) in (\d+) steps", log)
        if m:
            out["steps"] = int(m.group(1))
        m = re.search(r"Fmax < ([\d.eE+-]+)", log)
        if m:
            out["emtol"] = float(m.group(1))
    m = re.search(r"Potential Energy\s*=\s*([-\d.eE+]+)", log)
    if m:
        out["epot"] = float(m.group(1))
    m = re.search(r"Maximum force\s*=\s*([-\d.eE+]+)", log)
    if m:
        out["fmax"] = float(m.group(1))
    if out["converged"] and out["fmax"] is not None and out["emtol"] is not None:
        out["converged"] = out["fmax"] < out["emtol"]
    return out


_DATE = r"(\w{3} \w{3}\s+\d+ \d\d:\d\d:\d\d \d{4})"


def parse_progress(line: str) -> dict | None:
    """One line of `mdrun -v` chatter: MD step + finish time / remaining seconds, or EM step + Epot + Fmax."""
    m = re.search(r"Step=\s*(\d+),.*?Epot=\s*([-\d.eE+]+)\s+Fmax=\s*([-\d.eE+]+?),?\s", line + " ")
    if m:
        return {"step": int(m.group(1)), "epot": float(m.group(2)), "fmax": float(m.group(3))}
    m = re.search(r"step\s+(\d+),\s+will finish\s+" + _DATE, line)
    if m:
        return {"step": int(m.group(1)), "finish": m.group(2)}
    m = re.search(r"step\s+(\d+),\s+remaining wall clock time:\s+(\d+)\s*s", line)
    if m:
        return {"step": int(m.group(1)), "remaining": int(m.group(2))}
    return None


def parse_molecules(top: str) -> list[tuple[str, int]]:
    """[ molecules ] of a topology (the last such section)."""
    parts = re.split(r"^\s*\[\s*molecules\s*\]\s*$", top, flags=re.M)
    if len(parts) < 2:
        return []
    out = []
    for line in parts[-1].splitlines():
        line = line.split(";", 1)[0].strip()
        if line.startswith("["):
            break
        bits = line.split()
        if len(bits) == 2 and bits[1].isdigit():
            out.append((bits[0], int(bits[1])))
    return out


def parse_gro(text: str) -> dict:
    """Atom count and box of a .gro file; box vectors from the 3 or 9 numbers of the last line."""
    lines = text.rstrip("\n").splitlines()
    natoms = int(lines[1].split()[0])
    v = [float(x) for x in lines[-1].split()]
    v += [0.0] * (9 - len(v))
    a = np.array([v[0], v[3], v[4]])
    b = np.array([v[5], v[1], v[6]])
    c = np.array([v[7], v[8], v[2]])
    return {"atoms": natoms, "vectors": [a.round(4).tolist(), b.round(4).tolist(), c.round(4).tolist()],
            "lengths": [float(np.linalg.norm(x)) for x in (a, b, c)], "volume": float(abs(np.dot(a, np.cross(b, c))))}


def parse_net_charge(grompp_output: str) -> float:
    m = re.search(r"System has non-zero total charge:\s*([-\d.eE+]+)", grompp_output)
    return float(m.group(1)) if m else 0.0


def gro_residues(text: str) -> list[tuple[int, str]]:
    """(resnr, resname) in order of appearance, from a protein-only .gro."""
    out, last = [], None
    for line in text.splitlines()[2:-1]:
        if len(line) < 20:
            continue
        key = (line[0:5], line[5:10])
        if key != last:
            out.append((int(line[0:5]), line[5:10].strip()))
            last = key
    return out


# --- md_step with progress ------------------------------------------------------------------------------------------

def md_step_path() -> str:
    found = shutil.which("md_step")
    if found:
        return found
    local = Path(__file__).resolve().parents[2] / "bin" / "md_step"
    if local.exists():
        return str(local)
    raise RecipeError("nie widzę md_step (compute_bot/bin) na PATH")


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "<1 min"
    h, m = divmod(round(seconds / 60), 60)
    if h >= 48:
        return f"{h / 24:.1f} dni"
    return f"{h} h {m:02d} min" if h else f"{m} min"


class Progress:
    """Turns mdrun -v lines into PROGRESS lines: stage n/N, % of steps, end of stage and of the whole run."""

    def __init__(self, plan, every: float = 30.0, now=time.time):
        self.plan = plan
        self.every = every
        self.now = now
        self.rate = None  # steps per second, measured, carried to later stages
        self._reset(None)

    def _reset(self, stage):
        self.stage = stage
        self.first = None
        self.last_emit = -1e18

    def set_rate_from_nsday(self, nsday: float | None):
        if nsday and self.rate is None:
            self.rate = nsday / (0.002e-3 * 86400)  # 2 fs steps

    def start(self, stage) -> str:
        self._reset(stage)
        i = self.plan.index(stage) + 1
        what = f"{stage.title} ({stage.ps:g} ps, {stage.temp:g} K)" if stage.dynamic else stage.title
        return f"PROGRESS: etap {i}/{len(self.plan)}: {what} – start" + self._total_eta(0)

    def _later_steps(self) -> int:
        i = self.plan.index(self.stage)
        return sum(s.nsteps for s in self.plan[i + 1:] if s.dynamic)

    def _total_eta(self, step: int) -> str:
        if not self.rate or not self.stage.dynamic:
            return ""
        left = (self.stage.nsteps - step + self._later_steps()) / self.rate
        end = datetime.now() + timedelta(seconds=left)
        return f"; do końca całości ~{fmt_duration(left)} (ok. {end:%d.%m %H:%M})"

    def line(self, text: str) -> str | None:
        info = parse_progress(text)
        if info is None or self.stage is None:
            return None
        t = self.now()
        step = info["step"]
        if self.first is None:
            self.first = (t, step)
        elif t - self.first[0] >= 5 and step > self.first[1]:
            self.rate = (step - self.first[1]) / (t - self.first[0])
        if t - self.last_emit < self.every:
            return None
        self.last_emit = t
        i = self.plan.index(self.stage) + 1
        head = f"PROGRESS: etap {i}/{len(self.plan)}: {self.stage.title}"
        if "fmax" in info:
            return f"{head}, krok {step}, Fmax {info['fmax']:.3g}, Epot {info['epot']:.4g} kJ/mol"
        pct = 100.0 * step / max(self.stage.nsteps, 1)
        msg = f"{head} {math.floor(pct)}% (krok {step}/{self.stage.nsteps})"
        if "finish" in info:
            try:
                msg += f", koniec etapu ok. {datetime.strptime(info['finish'], '%a %b %d %H:%M:%S %Y'):%d.%m %H:%M}"
            except ValueError:
                msg += f", koniec etapu {info['finish']}"
        elif "remaining" in info:
            msg += f", etap jeszcze ~{fmt_duration(info['remaining'])}"
        return msg + self._total_eta(step)


def run_md_step(stage, gdir: Path, env: dict, progress: Progress) -> str:
    """`md_step NAME grompp-args` with live progress; returns its (non-chatter) output. Raises RecipeError."""
    print(progress.start(stage), flush=True)
    out_path = gdir / f"{stage.name}.out"
    proc = subprocess.Popen([md_step_path(), stage.name, *stage.grompp], cwd=gdir, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    kept, buf = [], b""
    with open(out_path, "ab") as log:
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)
            for raw in lines:
                text = raw.decode(errors="replace").rstrip()
                if not text.strip():
                    continue
                msg = progress.line(text)
                if msg:
                    print(msg, flush=True)
                if parse_progress(text) is None:
                    log.write(raw + b"\n")
                    kept.append(text)
                    if text.startswith("md_step:"):
                        print(text, flush=True)
    rc = proc.wait()
    text = "\n".join(kept)
    if rc != 0:
        raise RecipeError(f"etap {stage.title} ({stage.name}) przerwany błędem: {fatal_error(text)} "
                          f"(cały log: gromacs/{stage.name}.out, gromacs/{stage.name}.log)")
    return text
