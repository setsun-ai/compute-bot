"""Analysis of finished stages: energies, PBC fix, RMSD/RMSF/Rg/H-bonds/DSSP, plots, CSVs, summary.txt.
GROMACS outputs are reused when newer than their inputs, so re-running the recipe only redraws."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .gmxtools import RecipeError, fresh, gro_residues, parse_em_log, parse_performance, read_xvg, run_gmx

ENERGY_TERMS = ["Temperature", "Pressure", "Density", "Volume", "Potential"]
STAGE_COLORS = {"nvt": "#4C72B0", "npt": "#55A868", "md": "#C44E52"}
DSSP_CODES = [("H", "α-helisa", "#d62728"), ("G", "helisa 3₁₀", "#ff9896"), ("I", "helisa π", "#9467bd"),
              ("E", "β-nić", "#1f77b4"), ("B", "β-mostek", "#aec7e8"), ("T", "zwrot", "#2ca02c"),
              ("S", "zgięcie", "#98df8a"), ("P", "PPII", "#bcbd22"), ("~", "pętla", "#f2f2f2"),
              ("=", "przerwa", "#7f7f7f")]


def finished(gdir: Path, name: str) -> bool:
    log = gdir / f"{name}.log"
    return (gdir / f"{name}.gro").exists() and log.exists() and "Finished mdrun" in log.read_text(errors="replace")


def write_csv(path: Path, header: list, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def tail_mean(y: np.ndarray, frac: float = 0.1) -> float:
    n = max(1, int(math.ceil(len(y) * frac)))
    return float(np.mean(y[-n:]))


def head_mean(y: np.ndarray, frac: float = 0.1) -> float:
    n = max(1, int(math.ceil(len(y) * frac)))
    return float(np.mean(y[:n]))


def _charge(q: float) -> str:
    return "0" if abs(q) < 5e-4 else f"{q:+g}"


def mean_sd(y) -> str:
    return f"{np.mean(y):.1f}±{np.std(y):.1f}"


def dssp_fractions(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: fraction of residues in helix (H, G, I) and in β (E, B)."""
    helix = np.array([sum(c in "HGI" for c in s) / max(len(s), 1) for s in lines])
    beta = np.array([sum(c in "EB" for c in s) / max(len(s), 1) for s in lines])
    return helix, beta


class Gmx:
    def __init__(self, gdir: Path, env: dict):
        self.gdir, self.env, self.log = gdir, env, gdir / "analysis.log"

    def __call__(self, args, stdin="", check=True):
        return run_gmx(args, self.gdir, self.env, stdin=stdin, log=self.log, check=check)


def run(work: Path, gdir: Path, out: Path, plan, p: dict, struct: dict, system: dict, env: dict):
    import matplotlib

    matplotlib.use("Agg")
    gmx = Gmx(gdir, env)
    res = {"warnings": []}
    em = analyse_em(gdir, out, gmx, res)
    stages = [s for s in plan if s.dynamic and finished(gdir, s.name)]
    offsets, t = {}, 0.0
    for s in plan:
        if s.dynamic:
            offsets[s.name] = t
            t += s.ps
    energies, rmsd = {}, {}
    for s in stages:
        energies[s.name] = stage_energy(gdir, out, gmx, s.name)
        rmsd[s.name] = stage_rmsd(gdir, gmx, s.name)
    prod = next((s for s in stages if s.name == "md"), None)
    if prod is not None:
        production_analysis(gdir, out, gmx, res, system)
    plots = make_plots(out, plan, stages, offsets, energies, rmsd, res, em)
    perf = {s.name: parse_performance((gdir / f"{s.name}.log").read_text(errors="replace")) for s in stages}
    write_csv(out / "wydajnosc.csv", ["etap", "ps", "ns_na_dobe"], [[s.name, s.ps, perf[s.name]] for s in stages])
    write_csv(out / "uklad.csv", ["wielkosc", "wartosc"], system_rows(system))
    write_csv(out / "rmsd.csv", ["etap", "czas_ps", "rmsd_nm"],
              [[n, f"{x + offsets[n]:.3f}", f"{y:.4f}"]
               for n, (xs, ys) in rmsd.items() for x, y in zip(xs, ys, strict=False)])
    (out / "struktura.txt").write_text(struct["report"] + "\n\n" + "\n".join(struct.get("notes", [])) + "\n")
    summary = make_summary(p, plan, stages, struct, system, em, energies, rmsd, res, perf)
    (work / "summary.txt").write_text(summary)
    (work / "highlights.txt").write_text("".join(f"{OUT_REL}/{name}\n" for name in plots[:4]))
    (out / "analiza.json").write_text(json.dumps({"warnings": res["warnings"], "performance_ns_day": perf,
                                                  "em": {k: v for k, v in em.items() if k != "series"}},
                                                 ensure_ascii=False, indent=1))


OUT_REL = "wyniki"


# --- pieces ---------------------------------------------------------------------------------------------------------

def analyse_em(gdir, out, gmx, res) -> dict:
    if not finished(gdir, "em"):
        return {}
    em = parse_em_log((gdir / "em.log").read_text(errors="replace"))
    xvg_path = gdir / "em_potential.xvg"
    if not fresh(xvg_path, gdir / "em.edr"):
        gmx(["energy", "-f", "em.edr", "-o", "em_potential.xvg"], stdin="Potential\n\n")
    xvg = read_xvg(xvg_path)
    em["series"] = (xvg.x, xvg.data[:, 1])
    write_csv(out / "em_energia.csv", ["krok", "Epot_kJ_mol"],
              [[int(a), f"{b:.2f}"] for a, b in zip(*em["series"], strict=False)])
    if not em["converged"]:
        res["warnings"].append(f"minimalizacja nie osiągnęła Fmax < {em['emtol'] or 1000:g} (Fmax {em['fmax']:.3g})")
    if em["epot"] is not None and em["epot"] >= 0:
        res["warnings"].append("Epot po minimalizacji jest dodatnia – sprawdź strukturę")
    return em


def stage_energy(gdir, out, gmx, name) -> dict:
    xvg_path = gdir / f"{name}_energy.xvg"
    if not fresh(xvg_path, gdir / f"{name}.edr"):
        gmx(["energy", "-f", f"{name}.edr", "-o", xvg_path.name], stdin="\n".join(ENERGY_TERMS) + "\n\n")
    xvg = read_xvg(xvg_path)
    series = {leg: xvg.column(leg) for leg in xvg.legends}
    series["time"] = xvg.x
    write_csv(out / f"energia_{name}.csv", ["czas_ps"] + xvg.legends,
              [[f"{row[0]:.3f}"] + [f"{v:.4f}" for v in row[1:]] for row in xvg.data])
    return series


def stage_rmsd(gdir, gmx, name):
    nopbc = gdir / f"{name}_noPBC.xtc"
    if not fresh(nopbc, gdir / f"{name}.xtc"):
        gmx(["trjconv", "-s", f"{name}.tpr", "-f", f"{name}.xtc", "-o", nopbc.name, "-pbc", "mol", "-center"],
            stdin="Protein\nSystem\n")
    xvg_path = gdir / f"{name}_rmsd.xvg"
    if not fresh(xvg_path, nopbc):
        gmx(["rms", "-s", "em.tpr", "-f", nopbc.name, "-o", xvg_path.name, "-tu", "ps"], stdin="Backbone\nBackbone\n")
    xvg = read_xvg(xvg_path)
    return xvg.x, xvg.data[:, 1]


def _gmx_selection_tool(gmx, new_args, old_args, stdin_old):
    """GROMACS ≥2024 tools take -sel/-r/-t; older ones ask for groups on stdin."""
    try:
        return gmx(new_args)
    except RecipeError:
        return gmx(old_args, stdin=stdin_old)


def production_analysis(gdir, out, gmx, res, system):
    nopbc = gdir / "md_noPBC.xtc"
    names = gro_residues((gdir / "processed.gro").read_text())
    # RMSF per residue (C-alpha, fitted)
    if not fresh(gdir / "rmsf.xvg", nopbc):
        gmx(["rmsf", "-s", "md.tpr", "-f", nopbc.name, "-res", "-o", "rmsf.xvg"], stdin="C-alpha\n")
    rmsf = read_xvg(gdir / "rmsf.xvg").data
    labels = [f"{n}{r}" for r, n in names] if len(names) == len(rmsf) else [str(int(r)) for r in rmsf[:, 0]]
    res["rmsf"] = (labels, rmsf[:, 1])
    write_csv(out / "rmsf.csv", ["reszta", "rmsf_nm"],
              [[lab, f"{v:.4f}"] for lab, v in zip(labels, rmsf[:, 1], strict=False)])
    # radius of gyration
    if not fresh(gdir / "gyrate.xvg", nopbc):
        _gmx_selection_tool(gmx, ["gyrate", "-s", "md.tpr", "-f", nopbc.name, "-sel", "Protein", "-o", "gyrate.xvg"],
                            ["gyrate", "-s", "md.tpr", "-f", nopbc.name, "-o", "gyrate.xvg"], "Protein\n")
    rg = read_xvg(gdir / "gyrate.xvg")
    res["rg"] = (rg.x, rg.data[:, 1])
    write_csv(out / "rg.csv", ["czas_ps", "rg_nm"],
              [[f"{a:.3f}", f"{b:.4f}"] for a, b in zip(*res["rg"], strict=False)])
    # protein-protein hydrogen bonds
    try:
        if not fresh(gdir / "hbnum.xvg", nopbc):
            _gmx_selection_tool(gmx, ["hbond", "-s", "md.tpr", "-f", nopbc.name, "-r", "Protein", "-t", "Protein",
                                      "-num", "hbnum.xvg"],
                                ["hbond", "-s", "md.tpr", "-f", nopbc.name, "-num", "hbnum.xvg"], "Protein\nProtein\n")
        hb = read_xvg(gdir / "hbnum.xvg")
        res["hbonds"] = (hb.x, hb.data[:, 1])
        write_csv(out / "wiazania_h.csv", ["czas_ps", "liczba"],
                  [[f"{a:.3f}", int(b)] for a, b in zip(*res["hbonds"], strict=False)])
    except RecipeError as exc:
        res["warnings"].append(f"mostki H: {exc}")
    # secondary structure (gmx dssp exists from GROMACS 2023)
    try:
        if not fresh(gdir / "dssp.dat", nopbc):
            gmx(["dssp", "-s", "md.tpr", "-f", nopbc.name, "-o", "dssp.dat"])
        lines = [ln.strip() for ln in (gdir / "dssp.dat").read_text().splitlines() if ln.strip()]
        t = res["rg"][0] if len(res["rg"][0]) == len(lines) else np.arange(len(lines), dtype=float)
        res["dssp"] = (t, lines)
        helix, beta = dssp_fractions(lines)
        write_csv(out / "dssp.csv", ["czas_ps", "dssp", "helisa", "beta"],
                  [[f"{a:.3f}", s, f"{h:.3f}", f"{b:.3f}"] for a, s, h, b in zip(t, lines, helix, beta, strict=False)])
    except (RecipeError, OSError) as exc:
        res["warnings"].append(f"DSSP pominięte ({str(exc)[:120]})")
    viewer_files(gdir, out, gmx, nopbc, res)


def viewer_files(gdir, out, gmx, nopbc, res):
    """Protein only, fitted on the backbone: full xtc + multi-model PDB with a stride (≤ ~20 MB) + EM structure."""
    if not fresh(out / "bialko_em.pdb", gdir / "em.gro"):
        gmx(["trjconv", "-s", "em.tpr", "-f", "em.gro", "-o", str(out / "bialko_em.pdb")], stdin="Protein\n")
    if not fresh(out / "bialko.xtc", nopbc):
        gmx(["trjconv", "-s", "md.tpr", "-f", nopbc.name, "-fit", "rot+trans", "-o", str(out / "bialko.xtc")],
            stdin="Backbone\nProtein\n")
    frames = len(res["rg"][0])
    natoms = sum(1 for ln in (out / "bialko_em.pdb").read_text().splitlines() if ln.startswith(("ATOM", "HETATM")))
    max_frames = max(1, min(200, int(19e6 / max(natoms * 81, 1))))
    skip = max(1, math.ceil(frames / max_frames))
    if not fresh(out / "bialko_klatki.pdb", nopbc):
        gmx(["trjconv", "-s", "md.tpr", "-f", nopbc.name, "-fit", "rot+trans", "-skip", str(skip),
             "-o", str(out / "bialko_klatki.pdb")], stdin="Backbone\nProtein\n")
    res["viewer_skip"] = skip


def system_rows(s: dict) -> list:
    v = s["box_vectors"]
    return [["pole_silowe", s["ff"]], ["woda", s["water"]], ["pudelko", s["box_type"]],
            ["odleglosc_od_sciany_nm", s["d"]],
            ["wektor_a_nm", " ".join(f"{x:.4f}" for x in v[0])], ["wektor_b_nm", " ".join(f"{x:.4f}" for x in v[1])],
            ["wektor_c_nm", " ".join(f"{x:.4f}" for x in v[2])],
            ["dlugosci_abc_nm", " ".join(f"{x:.4f}" for x in s["box_lengths"])], ["objetosc_nm3", f"{s['volume']:.3f}"],
            ["atomy", s["atoms"]], ["wody", s["waters"]], ["Na", s["na"]], ["Cl", s["cl"]],
            ["stezenie_NaCl_M", s["salt"]], ["ladunek_bialka_e", s["charge_protein"]],
            ["ladunek_ukladu_e", s["charge_total"]]]


# --- plots ----------------------------------------------------------------------------------------------------------

def _running(y, n):
    if len(y) < 3 * n or n < 2:
        return None
    return np.convolve(y, np.ones(n) / n, mode="valid")


def make_plots(out, plan, stages, offsets, energies, rmsd, res, em) -> list[str]:
    import matplotlib.pyplot as plt

    made = []
    titles = {s.name: s.title for s in plan}
    if stages:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        panels = [("Temperature", "T [K]"), ("Potential", "Epot [kJ/mol]"), ("Pressure", "p [bar]"),
                  ("Density", "gęstość [kg/m³]")]
        for ax, (term, label) in zip(axes.flat, panels, strict=False):
            for s in stages:
                e = energies[s.name]
                if term not in e:
                    continue
                x = e["time"] + offsets[s.name]
                ax.plot(x, e[term], color=STAGE_COLORS[s.name], lw=0.8, alpha=0.6, label=titles[s.name])
                n = max(2, len(x) // 50)
                smooth = _running(e[term], n)
                if smooth is not None:
                    ax.plot(x[n - 1:], smooth, color=STAGE_COLORS[s.name], lw=1.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("czas [ps]")
        fig.suptitle("Równowagowanie i produkcja: temperatura, energia potencjalna, ciśnienie, gęstość")
        fig.tight_layout()
        fig.savefig(out / "rownowagowanie.png", dpi=150)
        plt.close(fig)
        made.append("rownowagowanie.png")
    if rmsd:
        has_rg = "rg" in res
        fig, axes = plt.subplots(2 if has_rg else 1, 1, figsize=(9, 6.5 if has_rg else 3.8), squeeze=False)
        ax = axes[0, 0]
        for name, (x, y) in rmsd.items():
            ax.plot(x + offsets[name], y, color=STAGE_COLORS[name], lw=1, label=titles[name])
        ax.set_ylabel("RMSD szkieletu [nm]")
        ax.set_xlabel("czas [ps]")
        ax.set_title("RMSD szkieletu względem struktury po minimalizacji (po dopasowaniu)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if has_rg:
            ax = axes[1, 0]
            x, y = res["rg"]
            ax.plot(x + offsets["md"], y, color="#8172B2", lw=1)
            ax.set_ylabel("Rg [nm]")
            ax.set_xlabel("czas [ps]")
            ax.set_title("Promień bezwładności białka (produkcja)")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rmsd_rg.png", dpi=150)
        plt.close(fig)
        made.append("rmsd_rg.png")
    if "rmsf" in res:
        labels, y = res["rmsf"]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.18), 3.8))
        ax.bar(range(len(y)), y, color="#4C72B0")
        step = max(1, len(labels) // 40)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=90, fontsize=7)
        ax.set_ylabel("RMSF Cα [nm]")
        ax.set_title("RMSF reszt w produkcji")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rmsf.png", dpi=150)
        plt.close(fig)
        made.append("rmsf.png")
    if "dssp" in res:
        made.append(plot_dssp(out, res["dssp"], offsets["md"]))
    if "hbonds" in res:
        x, y = res["hbonds"]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(x + offsets["md"], y, color="#937860", lw=0.8)
        ax.set_xlabel("czas [ps]")
        ax.set_ylabel("liczba mostków H")
        ax.set_title("Wiązania wodorowe białko–białko (produkcja)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "wiazania_h.png", dpi=150)
        plt.close(fig)
        made.append("wiazania_h.png")
    if em.get("series") is not None:
        x, y = em["series"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(x, y, color="#333333")
        ax.set_xlabel("krok minimalizacji")
        ax.set_ylabel("Epot [kJ/mol]")
        ax.set_title("Minimalizacja energii")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "minimalizacja.png", dpi=150)
        plt.close(fig)
    return made


def plot_dssp(out, dssp, offset) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    t, lines = dssp
    width = max(len(s) for s in lines)
    index = {c: i for i, (c, _, _) in enumerate(DSSP_CODES)}
    grid = np.array([[index.get(c, index["~"]) for c in s.ljust(width, "~")] for s in lines]).T
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    extent = [t[0] + offset, t[-1] + offset if len(t) > 1 else t[0] + offset + 1, 0.5, width + 0.5]
    ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest", extent=extent,
              cmap=ListedColormap([c for _, _, c in DSSP_CODES]), vmin=-0.5, vmax=len(DSSP_CODES) - 0.5)
    present = {c for s in lines for c in s}
    ax.legend(handles=[Patch(color=col, label=name) for c, name, col in DSSP_CODES if c in present],
              fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    ax.set_ylabel("reszta")
    ax.set_title("Struktura drugorzędowa w czasie (gmx dssp)")
    helix, beta = dssp_fractions(lines)
    ax2.plot(t + offset, 100 * helix, color="#d62728", label="helisa (H+G+I)")
    ax2.plot(t + offset, 100 * beta, color="#1f77b4", label="β (E+B)")
    ax2.set_ylabel("% reszt")
    ax2.set_xlabel("czas [ps]")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "dssp.png", dpi=150)
    plt.close(fig)
    return "dssp.png"


# --- summary --------------------------------------------------------------------------------------------------------

def make_summary(p, plan, stages, struct, system, em, energies, rmsd, res, perf) -> str:
    done = {s.name for s in stages}
    prod = next(s for s in plan if s.name == "md")
    ens = "NPT" if prod.pcoupl else "NVT"
    lines = [f"MD {struct.get('code', '')}: {system['ff']} + {system['water']}, produkcja {p['ns']:g} ns "
             f"w {p['T']:g} K ({ens})"]
    chains = ", ".join(f"{c['name']} ({c['residues']} reszt)" for c in struct["chains"])
    model = f"model {p['model']} z {struct['models']}" if struct["models"] > 1 else "1 model"
    lines.append(f"Struktura: {struct['file']}, {model}; łańcuchy: {chains}.")
    lines += [f"- {n}" for n in struct.get("notes", [])]
    L = system["box_lengths"]
    lines.append(f"Układ: {system['box_type']} (d={system['d']:g} nm), a,b,c = {L[0]:.2f}, {L[1]:.2f}, {L[2]:.2f} nm, "
                 f"V = {system['volume']:.1f} nm³; {system['atoms']} atomów, {system['waters']} wód, "
                 f"{system['na']} Na⁺, {system['cl']} Cl⁻ ({system['salt']:g} M); ładunek białka "
                 f"{_charge(system['charge_protein'])} e, układu {_charge(system['charge_total'])} e.")
    lines.append("Protonacja i końce: pdb2gmx (Lys/Arg +, Asp/Glu −, His wg sieci wiązań H), końce NH₃⁺/COO⁻.")
    if em:
        state = "zbieżna" if em["converged"] else "NIEZBIEŻNA"
        lines.append(f"Minimalizacja: {state}, {em['steps']} kroków, Fmax {em['fmax']:.0f} kJ/mol/nm "
                     f"(próg {em['emtol'] or 1000:g}), Epot {em['epot']:.4g} kJ/mol.")
    for s in stages:
        e = energies[s.name]
        bits = []
        if "Temperature" in e:
            bits.append(f"T {mean_sd(e['Temperature'])} K")
        if "Pressure" in e:
            bits.append(f"p {mean_sd(e['Pressure'])} bar")
        if s.pcoupl and "Density" in e:
            bits.append(f"gęstość {mean_sd(e['Density'])} kg/m³ (koniec {tail_mean(e['Density']):.0f})")
        if "Potential" in e:
            bits.append(f"Epot {np.mean(e['Potential']):.4g} kJ/mol")
        if s.name in rmsd:
            y = rmsd[s.name][1]
            bits.append(f"RMSD szkieletu śr. {np.mean(y):.3f}, koniec {tail_mean(y):.3f}, max {np.max(y):.3f} nm")
        restr = ", więzy na atomach ciężkich" if s.posres else ""
        lines.append(f"{s.title[0].upper() + s.title[1:]} {s.ps:g} ps ({s.temp:g} K{restr}): " + "; ".join(bits) + ".")
        if s.pcoupl and "Density" in e and tail_mean(e["Density"]) < 900:
            res["warnings"].append(f"{s.title}: gęstość spadła do {tail_mean(e['Density']):.0f} kg/m³ – woda paruje "
                                   "(przy wysokiej T i 1 bar); do rozwijania białka lepsze NVT")
    if "md" in done:
        extra = []
        if "rg" in res:
            y = res["rg"][1]
            extra.append(f"Rg {head_mean(y):.3f} → {tail_mean(y):.3f} nm")
        if "hbonds" in res:
            y = res["hbonds"][1]
            extra.append(f"mostki H białko–białko {head_mean(y):.0f} → {tail_mean(y):.0f}")
        if "dssp" in res:
            h, b = dssp_fractions(res["dssp"][1])
            extra.append(f"DSSP helisa {100 * head_mean(h):.0f}% → {100 * tail_mean(h):.0f}%, "
                         f"β {100 * head_mean(b):.0f}% → {100 * tail_mean(b):.0f}%")
        if "rmsf" in res:
            labels, y = res["rmsf"]
            i = int(np.argmax(y))
            extra.append(f"najruchliwsza reszta {labels[i]} (RMSF {y[i]:.3f} nm)")
        if extra:
            lines.append("Produkcja (początek → koniec): " + "; ".join(extra) + ".")
    else:
        lines.append("Produkcja jeszcze nie skończona – analiza tylko gotowych etapów.")
    speed = [f"{s.title} {perf[s.name]:.1f}" for s in stages if perf.get(s.name)]
    if speed:
        lines.append("Wydajność [ns/dzień]: " + ", ".join(speed) + ".")
    if res["warnings"]:
        lines.append("Uwaga: " + "; ".join(res["warnings"]) + ".")
    lines.append("Pliki: wyniki/ (CSV, PNG, struktura.txt, uklad.csv, bialko.xtc + bialko_em.pdb, "
                 f"bialko_klatki.pdb co {res.get('viewer_skip', 1)}. klatkę), pełne pliki GROMACS w gromacs/.")
    text = "\n".join(lines) + "\n"
    if len(text) > 2500:  # keep the message short: drop structure notes first, then cut
        keep = [ln for ln in lines if not ln.startswith("- ")]
        text = "\n".join(keep) + "\n"
        text = text[:2490] + ("…\n" if len(text) > 2490 else "")
    return text
