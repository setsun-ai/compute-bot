"""What the owner reads: the protocol table (wyniki/protokol_charmm.csv) to compare CHARMM-GUI's choices with the
manual setup, and summary.txt with the protocol, the bot's changes and the results. Pure text building."""
from __future__ import annotations

import numpy as np

from ..md.analysis import dssp_fractions, head_mean, mean_sd, tail_mean
from . import package as K

KIND = {"min": "minimalizacja", "equil": "równowagowanie", "prod": "produkcja"}


def _g(x) -> str:
    return "" if x is None else f"{x:g}"


def blocks(steps: list) -> list[list]:
    """Consecutive steps with the same settings (the production parts) as one block."""
    out = []
    for s in steps:
        key = (s.kind, s.config, tuple(sorted((k, str(v)) for k, v in s.settings.items() if k != "output")),
               tuple(sorted(s.changes)))
        if out and out[-1][0] == key:
            out[-1][1].append(s)
        else:
            out.append((key, [s]))
    return [b for _, b in out]


def block_label(b: list) -> str:
    return b[0].name if len(b) == 1 else f"{b[0].name}…{b[-1].name} ({len(b)}×)"


def _changed(s, key, text, fmt=_g):
    if key in s.changes:
        new, old = s.changes[key]
        return f"{text} (w paczce: {fmt(old)})", "zmienione przez bota (params)"
    return text, s.config


def step_rows(s) -> list[tuple[str, str, str]]:
    st = s.settings
    rows = [("rodzaj", KIND[s.kind], "README"), ("plik", s.config, "README")]
    if st.get("min_steps"):
        tol = f", próg {_g(st.get('emtol'))} kJ/mol/nm" if st.get("emtol") else ""
        how = st.get("integrator") if s.kind == "min" and s.engine == "gromacs" else "minimalizacja przed MD"
        rows.append(("minimalizacja", f"{how}, {st['min_steps']} kroków{tol}", s.config))
    if s.dynamic:
        rows.append(("integrator", st.get("integrator", ""), s.config))
        rows.append(("krok_czasowy_fs", _g(round(s.dt * 1000, 4)), s.config))
        v, src = _changed(s, "nsteps", str(s.nsteps))
        rows.append(("liczba_krokow", v, src))
        rows.append(("czas_ps", _g(round(s.ps, 4)), "liczba kroków × dt"))
        v, src = _changed(s, "temp", _g(s.temp))
        rows.append(("temperatura_K", v, src))
        rows.append(("termostat", st.get("thermostat", ""), s.config))
        baro = st.get("barostat") or "brak (NVT)"
        if st.get("pressure") is not None:
            baro += f", {_g(st['pressure'])} bar"
        rows.append(("barostat", baro, s.config))
        rows.append(("predkosci_poczatkowe", "losowane" if st.get("gen_vel") else "z poprzedniego kroku", s.config))
        every = (st.get("traj_every") or 0) * s.dt
        v, src = _changed(s, "traj_every", f"{_g(round(every, 4))} ps" if every else "brak",
                          fmt=lambda old: f"{_g(round(old * s.dt, 4))} ps")
        rows.append(("zapis_klatek", v, src))
    rows.append(("wiezy_polozen", st.get("restraints") or "brak", s.config))
    rows.append(("wiezy_wiazan", st.get("constraints", ""), s.config))
    rows.append(("vdw", st.get("vdw", ""), s.config))
    rows.append(("elektrostatyka", st.get("coulomb", ""), s.config))
    return rows


def system_rows(engine: str, edir_name: str, info: dict, others: list) -> list[tuple[str, str, str]]:
    rows = [("silnik", f"{K.ENGINE_TITLES[engine]} (folder {edir_name}/)", "wybór bota (engine=)"),
            ("wersja_charmm_gui", info.get("version") or "?", "README")]
    rows.append(("pole_silowe", info["ff"], info.get("ff_source", "")))
    rows.append(("woda", f"{info['water_model']}, {info['waters']} cząsteczek", info.get("psf") or "topol.top"))
    rows.append(("jony", K.ions_text(info), info.get("conc_source") or info.get("psf", "")))
    rows.append(("pudelko", K.box_text(info.get("box") or {}), (info.get("box") or {}).get("source", "")))
    if info.get("atoms"):
        rows.append(("atomy", str(info["atoms"]), info.get("psf", "")))
    if info.get("protein_residues"):
        rows.append(("reszty_bialka", str(info["protein_residues"]), info.get("psf", "")))
        rows.append(("ladunek_bialka_e", _g(info.get("protein_charge")), info.get("psf", "")))
    if info.get("charge_total") is not None:
        rows.append(("ladunek_ukladu_e", _g(info["charge_total"]), info.get("psf", "")))
    for note in info.get("notes", []):
        rows.append(("uwaga", note, "input.config.dat"))
    if others:
        rows.append(("inne_silniki_w_paczce", ", ".join(others), "paczka"))
    return rows


def protocol_table(engine: str, edir_name: str, info: dict, steps: list, others: list) -> list[list[str]]:
    rows = [["układ", k, v, src] for k, v, src in system_rows(engine, edir_name, info, others)]
    for b in blocks(steps):
        label = block_label(b)
        rows += [[label, k, v, src] for k, v, src in step_rows(b[0])]
        if not b[0].selected:
            rows.append([label, "uruchomiony", "nie (steps=)", "params"])
    return rows


# --- summary --------------------------------------------------------------------------------------------------------

def _ens(s) -> str:
    return "NPT" if s.pcoupl else "NVT"


def block_line(b: list) -> str:
    s = b[0]
    st = s.settings
    head = (b[0].name if len(b) == 1 else f"{b[0].name}…{b[-1].name}") + ": "
    if s.kind == "min":
        tol = f" (Fmax < {_g(st.get('emtol'))})" if st.get("emtol") else ""
        text = f"minimalizacja {st.get('integrator', '')} {st.get('min_steps')} kroków{tol}"
    else:
        n = len(b)
        dur = f"{n} × {s.ps / 1000:g} ns" if n > 1 else (f"{s.ps:g} ps" if s.ps < 1000 else f"{s.ps / 1000:g} ns")
        text = ""
        if st.get("min_steps"):
            text = f"minimalizacja {st['min_steps']} kroków, potem "
        text += f"{dur} {_ens(s)}, dt {s.dt * 1000:g} fs, {s.temp:g} K, {st.get('thermostat', '')}"
        if st.get("barostat"):
            text += f" + {st['barostat']}" + (f" {_g(st.get('pressure'))} bar" if st.get("pressure") else "")
    text += "; więzy: " + (st.get("restraints") or "brak")
    return head + text


def changes_line(steps: list) -> str:
    bits = []
    prod = [s for s in steps if s.kind == "prod"]
    if prod and any("nsteps" in s.changes for s in prod):
        old = prod[0].package_prod_ps / 1000
        bits.append(f"produkcja {sum(s.ps for s in prod) / 1000:g} ns ({len(prod)} cz.) zamiast {old:g} ns")
    temps = {s.changes["temp"] for s in steps if "temp" in s.changes}
    for new, old in sorted(temps):
        bits.append(f"T {new:g} K zamiast {old:g} K")
    traj = next((s.changes["traj_every"] for s in prod if "traj_every" in s.changes), None)
    if traj:
        bits.append(f"klatki produkcji co {traj[0] * prod[0].dt:g} ps zamiast {traj[1] * prod[0].dt:g} ps "
                    "(żeby było ≥100 klatek do analizy)")
    return "; ".join(bits)


def common_line(steps: list) -> str:
    dyn = [s for s in steps if s.dynamic] or steps
    keys = [("constraints", "wiązania"), ("vdw", "vdW"), ("coulomb", "elektrostatyka")]
    out = []
    for key, label in keys:
        vals = {s.settings.get(key, "") for s in dyn}
        if len(vals) == 1:
            out.append(f"{label} {vals.pop()}")
    return ", ".join(out)


def summary(engine: str, info: dict, steps: list, data: dict, notes: list, others: list, p: dict) -> str:
    lines = []
    v = info.get("version") or ""
    lines.append(f"CHARMM-GUI {v} → {K.ENGINE_TITLES[engine]}: {info['ff']}, woda {info['water_model']}: "
                 f"{info['waters']} cząsteczek; jony: {K.ions_text(info)}.")
    atoms = f", {info['atoms']} atomów" if info.get("atoms") else ""
    prot = ""
    if info.get("protein_residues"):
        prot = f"; białko {info['protein_residues']} reszt, ładunek {_g(info.get('protein_charge'))} e"
    lines.append(f"Pudełko: {K.box_text(info.get('box') or {})}{atoms}{prot}.")
    lines.append("Protokół z paczki:")
    lines += ["- " + block_line(b) for b in blocks(steps)]
    common = common_line(steps)
    if common:
        lines.append("Wspólne: " + common + ".")
    ch = changes_line(steps)
    if ch:
        lines.append(f"Zmiany bota (w kopiach bot_*, oryginały nietknięte): {ch}.")
    skipped = [s.name for s in steps if not s.selected]
    if skipped:
        lines.append(f"Pominięte (steps={p.get('steps')}): {', '.join(skipped)}.")
    lines += results_lines(data)
    speed = [f"{n} {x:.1f}" for n, x in data["perf"].items() if x]
    if speed:
        lines.append("Wydajność [ns/dzień]: " + ", ".join(speed[:3]) + (" …" if len(speed) > 3 else "") + ".")
    warn = list(data["warnings"]) + [n for n in notes if not n.startswith("README linia")]
    if engine == "gromacs" and any(s.kind == "min" for s in steps):
        warn.append("minimalizacja w pojedynczej precyzji (README proponuje gmx_d)")
    if warn:
        lines.append("Uwaga: " + "; ".join(warn) + ".")
    if others:
        lines.append(f"Paczka ma też: {', '.join(others)} (engine=… żeby policzyć tam).")
    lines.append("Do porównania z ręcznym md: wyniki/protokol_charmm.csv (pole, woda, jony, pudełko, termostat, "
                 "barostat, więzy każdego kroku).")
    text = "\n".join(lines) + "\n"
    if len(text) > 2500:
        drop = [ln for ln in lines if not ln.startswith(("Uwaga", "Paczka ma też", "Wydajność"))]
        text = "\n".join(drop) + "\n"
        text = text[:2490] + ("…\n" if len(text) > 2490 else "")
    return text


def results_lines(data: dict) -> list[str]:
    lines = []
    em = data.get("em") or {}
    if em.get("epot") is not None:
        bits = [f"Epot {em['epot']:.4g} kJ/mol"]
        if em.get("epot_start") is not None:
            bits = [f"Epot {em['epot_start']:.4g} → {em['epot']:.4g} kJ/mol"]
        if em.get("fmax") is not None:
            state = "zbieżna" if em.get("converged") else "NIEZBIEŻNA"
            bits.append(f"{state}, Fmax {em['fmax']:.0f} kJ/mol/nm po {em.get('steps')} krokach")
        lines.append("Minimalizacja: " + ", ".join(bits) + ".")
    for g in data["stages"]:
        e = data["energies"][g.name]
        bits = []
        if "Temperature" in e:
            bits.append(f"T {mean_sd(e['Temperature'])} K")
        if "Pressure" in e:
            bits.append(f"p {mean_sd(e['Pressure'])} bar")
        if "Density" in e:
            bits.append(f"gęstość {mean_sd(e['Density'])} kg/m³")
        if "Potential" in e:
            bits.append(f"Epot {np.mean(e['Potential']):.4g} kJ/mol")
        if g.name in data["rmsd"]:
            y = data["rmsd"][g.name][1]
            bits.append(f"RMSD szkieletu śr. {np.mean(y):.3f}, koniec {tail_mean(y):.3f} nm")
        if bits:
            lines.append(f"{g.title[0].upper() + g.title[1:]}: " + "; ".join(bits) + ".")
    res = data["res"]
    extra = []
    if "rg" in res:
        y = res["rg"][1]
        extra.append(f"Rg {head_mean(y):.3f} → {tail_mean(y):.3f} nm")
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
    if res.get("reference"):
        lines.append(f"RMSD względem {res['reference']}.")
    if not any(g.name == "md" for g in data["stages"]):
        lines.append("Produkcja jeszcze nie skończona – analiza tylko gotowych kroków.")
    return lines
