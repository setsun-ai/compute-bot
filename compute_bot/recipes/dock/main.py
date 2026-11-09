"""File input/output and the order of steps; the computations live in the sibling modules."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from compute_bot.recipes.dock.common import UserError, log, progress, warn

PREP_VERSION = 1  # bump when receptor preparation changes, so cached receptors are rebuilt
TOP_COMPLEXES = 3
TOP_CONTACTS = 5


def _threads() -> int:
    try:
        return max(1, int(os.environ.get("COMPUTE_THREADS") or os.cpu_count() or 1))
    except ValueError:
        return 1


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _sha1(data) -> str:
    return hashlib.sha1(data if isinstance(data, bytes) else data.encode()).hexdigest()


def load_params(work: Path) -> dict:
    path = work / "params.json"
    if not path.exists():
        raise UserError("Brak params.json w katalogu zadania.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- steps

def locate_inputs(work: Path, params: dict, out: Path):
    from compute_bot.recipes.dock import inputs

    receptors, ligand_files = inputs.find_inputs(work)
    if not receptors and not ligand_files and inputs.unpack_zips(work):
        receptors, ligand_files = inputs.find_inputs(work)
    notes = []
    code = (params.get("receptor") or "").strip()
    if receptors:
        receptor = inputs.pick_receptor(receptors)
        if code:
            notes.append(f"podano receptor={code}, ale użyto wysłanego pliku {receptor.name}.")
        source = receptor.name
    elif code:
        code = inputs.check_pdb_code(code)
        progress(f"pobieranie {code} z RCSB")
        receptor = inputs.download_entry(code, out / "pobrane")
        source = f"{code} (RCSB)"
    else:
        raise UserError("Brak receptora: wyślij plik .pdb/.cif (może być w zipie z ligandami) "
                        "albo podaj receptor=KOD, np. receptor=1STP.")
    rows = []
    for path in ligand_files:
        rows.extend(inputs.read_ligand_file(path))
    return receptor, source, rows, notes


def read_structure(path: Path):
    import gemmi

    try:
        st = gemmi.read_structure(str(path))
    except (RuntimeError, ValueError) as exc:
        raise UserError(f"Nie mogę odczytać struktury {path.name}: {exc}") from None
    if len(st) == 0 or not any(len(ch) for ch in st[0]):
        raise UserError(f"{path.name}: plik nie zawiera atomów.")
    st.setup_entities()
    return st


def reference_entry(box, out: Path, warnings: list):
    """The co-crystallised ligand as RDKit molecule + its redock row, or (None, None)."""
    from rdkit import Chem

    from compute_bot.recipes.dock import inputs, ligands

    if box.ligand is None:
        return None, None
    lig = box.ligand
    template = inputs.download_ideal_sdf(lig.resname, out / "pobrane")
    if template is None:
        warnings.append(f"brak szablonu {lig.resname} z RCSB (sieć?) – rzędy wiązań z Open Babel.")
    mol, how = ligands.reference_mol(lig.pdb_block(), template)
    if mol is None:
        warnings.append(f"nie udało się odczytać chemii ligandu {lig.label} – bez redockingu.")
        return None, None
    Chem.AssignStereochemistryFrom3D(mol)
    Chem.MolToMolFile(mol, str(out / f"ligand_krystaliczny_{lig.resname}.sdf"))
    row = {"name": f"ref_{lig.resname}", "smiles": Chem.MolToSmiles(mol), "reference": True,
           "source": f"kryształ ({how})", "error": ""}
    return mol, row


def prepare_receptor(st, structure_bytes: bytes, params: dict, out: Path):
    """Cached: a restart reuses the same receptor (hydrogen placement need not be bit-identical between runs)."""
    from compute_bot.recipes.dock import receptor

    meta_path, pdbqt_path, pdb_path = out / "receptor.json", out / "receptor.pdbqt", out / "receptor.pdb"
    key = _sha1(structure_bytes + json.dumps([params["ph"], params["keep_waters"], PREP_VERSION]).encode())
    if meta_path.exists() and pdbqt_path.exists() and pdb_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("key") == key:
            log("Receptor: używam przygotowanego wcześniej (wyniki/receptor.pdbqt).")
            return meta
    progress("przygotowanie receptora (pdbfixer, Meeko)")
    clean, info = receptor.clean_structure(st, params["keep_waters"])
    clean_pdb = clean.make_pdb_string()
    try:
        pdbqt, with_h, method, dropped = receptor.prepare(clean_pdb, params["ph"])
    except UserError:
        raise
    except Exception as exc:  # pdbfixer raises many types on broken files; the owner gets the reason
        raise UserError(f"Nie udało się przygotować receptora (pdbfixer): {str(exc)[:200]}") from exc
    _write(pdb_path, with_h)
    _write(pdbqt_path, pdbqt)
    meta = {"key": key, "method": method, "dropped": dropped, "info": info, "clean_pdb": clean_pdb,
            "pdbqt_sha1": _sha1(pdbqt)}
    _write(meta_path, json.dumps(meta, indent=1))
    return meta


def dock_all(todo, n_total, n_skipped, v, params, settings, refmol, dirs, rec_records):
    from rdkit import Chem

    from compute_bot.recipes.dock import docking, ligands

    poses_dir, lig_dir = dirs["pozy"], dirs["ligandy"]
    times = [r["seconds"] for r in rec_records if r.get("seconds")]
    for i, lig in enumerate(todo):
        name = lig["name"]
        mean = sum(times) / len(times) if times else None
        eta = None if mean is None else mean * (len(todo) - i)
        progress(f"ligand {n_skipped + i + 1}/{n_total} {name} (ETA {format_eta(eta)})")
        record = {"name": name, "smiles": lig["smiles"], "reference": bool(lig.get("reference")),
                  "signature": docking.signature(lig["smiles"], settings), "source": lig.get("source", "")}
        attempt = docking.mark_running(poses_dir, name)
        t0 = time.time()
        if attempt > docking.MAX_ATTEMPTS:
            record.update(status="failed", error=f"proces przerwał się {attempt - 1}× przy tym ligandzie")
        else:
            try:
                clean, notes = ligands.clean_smiles(lig["smiles"])
                record["props"] = ligands.properties(Chem.MolFromSmiles(clean))
                charged = ligands.protonate_smiles(clean, params["ph"])
                mol3d = ligands.embed(charged, params["seed"])
                lig_pdbqt = ligands.to_pdbqt(mol3d)
                _write(lig_dir / f"{name}.pdbqt", lig_pdbqt)
                poses, energies = docking.dock(v, lig_pdbqt, params["exhaustiveness"], params["poses"])
                _write(poses_dir / f"{name}.pdbqt", poses)
                pose_mol, sdf = ligands.poses_to_mols(poses)
                _write(poses_dir / f"{name}.sdf", sdf)
                affinity = energies[0][0]
                record.update(status="done", docked_smiles=charged, notes=notes, energies=energies,
                              affinity=affinity,
                              ligand_efficiency=ligands.ligand_efficiency(affinity, record["props"]["heavy_atoms"]),
                              pose_sdf=f"wyniki/pozy/{name}.sdf", pose_pdbqt=f"wyniki/pozy/{name}.pdbqt")
                if record["reference"] and refmol is not None:
                    rmsds = [ligands.symmetric_rmsd(pose_mol, refmol, probe_conf=c.GetId())
                             for c in pose_mol.GetConformers()]
                    record["rmsd"] = [None if r is None else round(r, 2) for r in rmsds]
            except Exception as exc:  # per-ligand failure is logged in the results, the screen goes on
                record.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        record["seconds"] = round(time.time() - t0, 1)
        if record["status"] == "done":
            times.append(record["seconds"])
        else:
            warn(f"{name}: {record['error']}")
        docking.write_record(poses_dir, record)
        docking.clear_running(poses_dir, name)


def format_eta(seconds):
    from compute_bot.recipes.dock.results import format_eta as fmt

    return fmt(seconds)


def analyse(ranked, receptor_pdb: str, dirs):
    """Complexes for the best few and contacts for the top hits (+ the reference ligand)."""
    from rdkit import Chem

    from compute_bot.recipes.dock import receptor, results

    rec_atoms = receptor.pdb_atoms(receptor_pdb)
    picked = ranked[:TOP_CONTACTS] + [r for r in ranked[TOP_CONTACTS:] if r.get("reference")]
    contact_rows = []
    for k, r in enumerate(picked):
        supplier = Chem.SDMolSupplier(str(dirs["work"] / r["pose_sdf"]), removeHs=False)
        mol = next(iter(supplier), None)
        if mol is None:
            continue
        res, hb = results.contacts(results.mol_atoms(mol), rec_atoms)
        r["contacts"], r["hbonds"] = res, hb
        contact_rows.append({"name": r["name"], "affinity_kcal_mol": r["affinity"], "residues_4A": " ".join(res),
                             "hbond_candidates": " ".join(f"{h['ligand_atom']}-{h['residue']}:{h['atom']}:"
                                                          f"{h['distance']:.2f}" for h in hb)})
        rank_no = ranked.index(r) + 1
        if k < TOP_COMPLEXES or r.get("reference"):
            _write(dirs["kompleksy"] / f"{rank_no:02d}_{r['name']}.pdb", results.complex_pdb(receptor_pdb, mol))
    import csv

    with open(dirs["out"] / "kontakty.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "affinity_kcal_mol", "residues_4A", "hbond_candidates"])
        w.writeheader()
        w.writerows(contact_rows)


def build_summary(ctx) -> str:
    from compute_bot.recipes.dock.box import fmt_vec

    p, box, meta = ctx["params"], ctx["box"], ctx["receptor"]
    info = meta["info"]
    removed = ", ".join(f"{k}×{v}" for k, v in sorted(info["removed_hetero"].items())) or "nic"
    chains = ",".join(info["chains"])
    lines = [f"Dokowanie AutoDock Vina ({p['scoring']}), receptor {ctx['source']}, łańcuchy {chains}.",
             f"Receptor: pdbfixer (pH {p['ph']:g}) + {meta['method'].split(' (')[0]}; usunięto wody: "
             f"{info['removed_waters'] if not p['keep_waters'] else 'nie (keep_waters)'}, HETATM: {removed}."]
    if meta["dropped"]:
        lines.append(f"Reszty pominięte przy PDBQT: {', '.join(meta['dropped'][:8])}.")
    lines.append(f"Pudełko {box.note}: środek {fmt_vec(box.center)}, rozmiar {fmt_vec(box.size)} Å.")
    ref = ctx["ref_record"]
    if ref and ref.get("rmsd") and ref["rmsd"][0] is not None:
        top, best = ref["rmsd"][0], min(r for r in ref["rmsd"] if r is not None)
        verdict = ("protokół odtwarza pozę krystaliczną (< 2 Å)" if top < 2 else
                   "najlepsza poza NIE odtwarza kryształu (≥ 2 Å) – wyniki ostrożnie; spróbuj większego "
                   "exhaustiveness, innej protonacji albo vinardo")
        lines.append(f"Redocking {box.ligand.resname}: RMSD najlepszej pozy {top:.2f} Å (najmniejsze z "
                     f"{len(ref['rmsd'])} póz: {best:.2f} Å), {ref['affinity']:.1f} kcal/mol – {verdict}.")
    elif ref:
        lines.append(f"Redocking {box.ligand.resname}: nieudany ({ref.get('error', 'brak RMSD')[:100]}).")
    else:
        lines.append("Redocking: brak ligandu krystalicznego – podaj ligand_ref, by sprawdzić protokół.")
    n_in = ctx["n_input"]
    lines.append(f"Zadokowano {ctx['n_done']} z {n_in} ligandów (bez kontroli), nieudane: {ctx['n_failed']}"
                 f"{' (wyniki/nieudane.csv)' if ctx['n_failed'] else ''}. exhaustiveness={p['exhaustiveness']}, "
                 f"pozy={p['poses']}, seed={p['seed']}, średnio {ctx['mean_s']:.1f} s/ligand.")
    ranked = ctx["ranked"]
    if ranked:
        lines.append("Top 10 [kcal/mol, LE = −ΔG/atom ciężki]:")
        for i, r in enumerate(ranked[:10], start=1):
            flag = " (kontrola)" if r.get("reference") else ""
            lines.append(f"{i:>2}. {r['name'][:28]}{flag}  {r['affinity']:.1f}  LE {r['ligand_efficiency'] or 0:.2f}"
                         f"  Lipinski {r['props']['lipinski_violations']}")
        best = ranked[0]
        if best.get("contacts"):
            closest = {}
            for h in best.get("hbonds", []):  # sorted by distance: keep the closest pair per residue
                closest.setdefault(h["residue"], h["distance"])
            hb = ", ".join(f"{res} {d:.1f} Å" for res, d in list(closest.items())[:5]) or "brak"
            lines.append(f"Kontakty {best['name'][:28]} (≤4 Å): {' '.join(best['contacts'][:12])}; "
                         f"kandydaci na mostki H: {hb}.")
    for w in ctx["warnings"]:
        lines.append(f"Uwaga: {w}")
    lines.append("Pamiętaj: wynik Vina/Vinardo to przybliżona ocena (błąd ~2 kcal/mol), nie zmierzone powinowactwo – "
                 "dobry do odsiewu, słaby do szeregowania podobnych związków. Receptor sztywny, protonacja "
                 "regułowa (Open Babel), duże cząsteczki mają z natury lepsze wyniki (patrz LE). Pozy obejrzyj w "
                 "PyMOL: wyniki/kompleksy/*.pdb, wyniki/box.pdb; tabela: wyniki/ranking.csv.")
    from compute_bot.recipes.dock.results import fit

    return fit("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main

def run(work: Path) -> None:
    from compute_bot.recipes.dock import box as boxmod
    from compute_bot.recipes.dock import inputs

    params = load_params(work)
    out = work / "wyniki"
    dirs = {"work": work, "out": out, "pozy": out / "pozy", "ligandy": out / "ligandy_pdbqt",
            "kompleksy": out / "kompleksy"}
    for d in list(dirs.values())[1:]:
        d.mkdir(parents=True, exist_ok=True)
    threads = _threads()

    receptor_path, source, rows, warnings = locate_inputs(work, params, out)
    st = read_structure(receptor_path)
    box = boxmod.choose_box(st, params)
    warnings.extend(box.warnings)
    _write(out / "box.txt", boxmod.vina_box_text(box.center, box.size, f"pudełko {box.note}"))
    _write(out / "box.pdb", boxmod.box_corners_pdb(box.center, box.size))
    log(f"Pudełko {box.note}: środek {box.center}, rozmiar {box.size}")

    refmol, ref_row = reference_entry(box, out, warnings)
    reserved = {ref_row["name"]} if ref_row else set()
    names = inputs.unique_names([r["name"] for r in rows], reserved)
    for r, n in zip(rows, names, strict=True):
        r["name"] = n
    input_failed = []
    from rdkit import Chem

    valid = []
    for r in rows:
        if not r["error"] and Chem.MolFromSmiles(r["smiles"]) is None:
            r["error"] = "nieprawidłowy SMILES"
        (input_failed if r["error"] else valid).append(r)
    if not valid and ref_row is None:
        raise UserError("Brak ligandów do dokowania: wyślij .smi/.csv (kolumna smiles), .sdf albo .mol2 "
                        "(może być w jednym zipie z receptorem)" +
                        (f"; żaden z {len(rows)} wierszy nie jest poprawny." if rows else "."))
    ligands_all = ([ref_row] if ref_row else []) + valid  # the control goes first

    meta = prepare_receptor(st, receptor_path.read_bytes(), params, out)
    from compute_bot.recipes.dock import receptor as recmod

    near = recmod.residues_near(meta["clean_pdb"], meta["dropped"], box.center, box.size)
    if near:
        warnings.append(f"w pobliżu pudełka usunięto reszty {', '.join(near)} (Meeko ich nie przyjął).")
    metals = [f"{name}" for name, el, xyz in meta["info"]["removed_atoms"]
              if el not in ("Cl", "Na", "K", "Br", "I", "F") and boxmod.inside(box, xyz)]
    if metals:
        warnings.append(f"z kieszeni usunięto jony {', '.join(sorted(set(metals)))} – Vina nie widzi metalu.")
    if not meta["method"].startswith("Meeko"):
        warnings.append(f"receptor przez {meta['method']}.")

    settings = {"box": [box.center, box.size], "scoring": params["scoring"], "exhaustiveness": params["exhaustiveness"],
                "poses": params["poses"], "seed": params["seed"], "ph": params["ph"], "receptor": meta["pdbqt_sha1"]}
    from compute_bot.recipes.dock import docking

    todo = docking.pending(ligands_all, dirs["pozy"], settings)
    skipped = len(ligands_all) - len(todo)
    if skipped:
        log(f"Wznowienie: pominięto {skipped} ligandów z gotowym wynikiem.")
    previous = [docking.read_record(dirs["pozy"], lig["name"]) for lig in ligands_all if lig not in todo]
    if todo:
        progress(f"mapy Vina ({params['scoring']}, {threads} wątki)")
        v = docking.make_vina(out / "receptor.pdbqt", box.center, box.size, params["scoring"], threads, params["seed"])
        dock_all(todo, len(ligands_all), skipped, v, params, settings, refmol, dirs, previous)

    progress("ranking i analiza")
    records = [docking.read_record(dirs["pozy"], lig["name"]) for lig in ligands_all]
    records = [r for r in records if r]
    from compute_bot.recipes.dock import results

    ranked = results.rank([r for r in records if r["status"] == "done"])
    failed = input_failed + [r for r in records if r["status"] == "failed"]
    results.write_ranking(out / "ranking.csv", ranked)
    results.write_failed(out / "nieudane.csv", failed)
    analyse(ranked, (out / "receptor.pdb").read_text(), dirs)
    pngs = results.plots(out, ranked)
    _write(work / "highlights.txt", "".join(f"{p.relative_to(work)}\n" for p in pngs[:4]))

    ref_record = next((r for r in records if r.get("reference")), None)
    done_times = [r["seconds"] for r in records if r["status"] == "done" and r.get("seconds")]
    ctx = {"params": params, "box": box, "receptor": meta, "source": source, "ref_record": ref_record,
           "n_input": len(rows), "n_done": sum(1 for r in ranked if not r.get("reference")),
           "n_failed": len(failed), "ranked": ranked, "warnings": warnings,
           "mean_s": sum(done_times) / len(done_times) if done_times else 0.0}
    _write(work / "summary.txt", build_summary(ctx))
    progress(f"gotowe: {len(ranked)} zadokowanych, {len(failed)} nieudanych")


def main() -> int:
    try:
        run(Path.cwd())
    except UserError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0
