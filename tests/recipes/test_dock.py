"""Tests of the docking recipe: pure functions on data with a known answer, and a slow end-to-end run on 1STP."""
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("gemmi")
np = pytest.importorskip("numpy")

from compute_bot.recipes import RECIPES  # noqa: E402
from compute_bot.recipes.dock import box as boxmod  # noqa: E402
from compute_bot.recipes.dock import inputs, ligands, results  # noqa: E402
from compute_bot.recipes.dock.common import UserError  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


# ---------------------------------------------------------------- ligand tables

def test_csv_with_smiles_and_name_columns():
    text = "Name,SMILES,pIC50\nasp,CC(=O)Oc1ccccc1C(=O)O,5\ncaf,Cn1cnc2c1c(=O)n(C)c(=O)n2C,4\n"
    rows = inputs.parse_csv_text(text)
    assert [r["name"] for r in rows] == ["asp", "caf"]
    assert rows[0]["smiles"] == ASPIRIN


def test_csv_semicolon_and_unusual_headers():
    text = "compound_id;canonical_smiles\nX1;CCO\nX2;c1ccccc1\n"
    rows = inputs.parse_csv_text(text)
    assert [(r["name"], r["smiles"]) for r in rows] == [("X1", "CCO"), ("X2", "c1ccccc1")]


def test_csv_without_names_and_without_smiles_header():
    # No column called smiles: find it by content; no name column: generate names
    text = "structure,activity\nCCO,1\nCCN,2\nc1ccccc1O,3\n"
    rows = inputs.parse_csv_text(text)
    assert [r["smiles"] for r in rows] == ["CCO", "CCN", "c1ccccc1O"]
    assert len({r["name"] for r in rows}) == 3


def test_csv_without_any_smiles_is_a_polish_error():
    with pytest.raises(UserError, match="SMILES"):
        inputs.parse_csv_text("a,b\n1,2\n3,4\n")


def test_smi_with_header_comments_and_missing_names():
    text = "smiles name\n# comment\nCCO ethanol\nc1ccccc1\tbenzene\nCCN\n\n"
    rows = inputs.parse_smi_text(text)
    assert [r["smiles"] for r in rows] == ["CCO", "c1ccccc1", "CCN"]
    assert rows[0]["name"] == "ethanol" and rows[1]["name"] == "benzene"
    assert rows[2]["name"]  # generated


def test_sdf_names_from_title_or_property():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    blocks = []
    for i, (smi, title) in enumerate([(ASPIRIN, "aspirin"), (CAFFEINE, "")]):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        mol.SetProp("_Name", title)
        extra = "" if title else "> <ID>\nCHEMBL113\n\n"
        blocks.append(Chem.MolToMolBlock(mol) + f"> <idx>\n{i}\n\n{extra}$$$$\n")
    rows = inputs.parse_sdf_text("".join(blocks))
    assert [r["name"] for r in rows] == ["aspirin", "CHEMBL113"]
    canon = Chem.CanonSmiles
    assert canon(rows[0]["smiles"]) == canon(ASPIRIN)
    assert canon(rows[1]["smiles"]) == canon(CAFFEINE)


def test_sdf_bad_record_is_kept_as_failed_row():
    good = "aspirin\n  RDKit\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n" \
           "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n"
    bad = "broken\n  RDKit\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n" \
          "    0.0000    0.0000    0.0000 Xx  0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n"
    rows = inputs.parse_sdf_text(good + bad)
    assert len(rows) == 2
    assert rows[0]["smiles"] == "C"
    assert rows[1]["smiles"] == "" and rows[1]["error"]


def test_unique_safe_names():
    names = inputs.unique_names(["a b", "a b", "x/y", "", "ref_BTN"], reserved={"ref_BTN"})
    assert names[0] == "a_b" and names[1] == "a_b_2"
    assert "/" not in names[2]
    assert names[3]
    assert names[4] != "ref_BTN"
    assert len(set(names)) == 5


def test_pdb_code_validation():
    assert inputs.check_pdb_code(" 1stp ") == "1STP"
    with pytest.raises(UserError):
        inputs.check_pdb_code("stp")


# ---------------------------------------------------------------- box

def test_box_from_coords_adds_padding_on_each_side():
    coords = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]])
    center, size = boxmod.box_from_coords(coords, padding=5.0, min_size=0)
    assert center == pytest.approx([5.0, 2.0, 1.0])
    assert size == pytest.approx([20.0, 14.0, 12.0])


def test_box_from_coords_minimum_size():
    coords = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    _, size = boxmod.box_from_coords(coords, padding=1.0, min_size=10.0)
    assert size == pytest.approx([10.0, 10.0, 10.0])


def _pdb_line(rec, serial, name, resname, chain, resnum, xyz, element):
    return (f"{rec:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resnum:>4}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2}")


def _toy_structure():
    import gemmi

    lines = []
    s = 1
    for name, xyz in (("N", (0, 0, 0)), ("CA", (1, 0, 0)), ("C", (2, 0, 0)), ("O", (2, 1, 0))):
        lines.append(_pdb_line("ATOM", s, name, "GLY", "A", 45, xyz, name[0]))
        s += 1
    for name, xyz in (("N", (10, 0, 0)), ("CA", (11, 0, 0)), ("C", (12, 0, 0)), ("O", (12, 2, 0))):
        lines.append(_pdb_line("ATOM", s, name, "GLY", "A", 88, xyz, name[0]))
        s += 1
    for name, xyz in (("N", (50, 0, 0)), ("CA", (51, 0, 0)), ("C", (52, 0, 0)), ("O", (52, 1, 0))):
        lines.append(_pdb_line("ATOM", s, name, "GLY", "B", 45, xyz, name[0]))
        s += 1
    # a drug-like ligand (8 heavy atoms), a sulfate, two waters
    for i in range(8):
        lines.append(_pdb_line("HETATM", s, f"C{i + 1}", "LIG", "A", 301, (5 + i * 0.5, 5, 5), "C"))
        s += 1
    lines.append(_pdb_line("HETATM", s, "S", "SO4", "A", 302, (20, 20, 20), "S"))
    s += 1
    for k, o in enumerate("OOOO"):
        lines.append(_pdb_line("HETATM", s, f"O{k + 1}", "SO4", "A", 302, (20 + k, 21, 20), o))
        s += 1
    lines.append(_pdb_line("HETATM", s, "O", "HOH", "A", 401, (30, 30, 30), "O"))
    lines.append(_pdb_line("HETATM", s + 1, "O", "HOH", "A", 402, (31, 30, 30), "O"))
    st = gemmi.read_pdb_string("\n".join(lines) + "\nEND\n")
    st.setup_entities()
    return st


def test_parse_residue_spec():
    assert boxmod.parse_residues("A:45, A:88,B:12A, 7") == [("A", 45, ""), ("A", 88, ""), ("B", 12, "A"), (None, 7, "")]
    with pytest.raises(UserError):
        boxmod.parse_residues("A:xx")


def test_box_from_residues():
    st = _toy_structure()
    center, size, note = boxmod.box_from_residues(st, boxmod.parse_residues("A:45,A:88"), padding=4.0, min_size=0)
    assert center == pytest.approx([6.0, 1.0, 0.0])
    assert size == pytest.approx([20.0, 10.0, 8.0])
    assert "A:45" in note


def test_box_from_residues_missing_or_ambiguous():
    st = _toy_structure()
    with pytest.raises(UserError, match="A:99"):
        boxmod.box_from_residues(st, boxmod.parse_residues("A:99"), padding=4.0)
    with pytest.raises(UserError, match="45"):
        boxmod.box_from_residues(st, boxmod.parse_residues("45"), padding=4.0)  # in A and B


def test_find_reference_ligand_by_name_and_automatically():
    st = _toy_structure()
    ref = boxmod.find_ligand(st, "lig")
    assert ref.resname == "LIG" and ref.chain == "A" and ref.coords.shape == (8, 3)
    auto = boxmod.druglike_ligands(st)
    assert [r.resname for r in auto] == ["LIG"]  # sulfate and water are not ligands
    with pytest.raises(UserError, match="XYZ"):
        boxmod.find_ligand(st, "XYZ")


def test_choose_box_priority_and_error():
    st = _toy_structure()
    params = {"center": [1.0, 2.0, 3.0], "size": [20.0, 20.0, 20.0], "ligand_ref": "LIG", "residues": "",
              "padding": 5.0}
    box = boxmod.choose_box(st, params)
    assert box.center == [1.0, 2.0, 3.0] and box.source == "center"
    params["center"] = None
    box = boxmod.choose_box(st, params)
    assert box.source == "ligand_ref" and box.ligand.resname == "LIG"
    params["ligand_ref"] = ""
    box = boxmod.choose_box(st, params)
    assert box.source == "auto" and box.ligand.resname == "LIG"
    st.remove_ligands_and_waters()
    with pytest.raises(UserError, match="ligand_ref"):
        boxmod.choose_box(st, params)


def test_box_files_text():
    txt = boxmod.vina_box_text([1.0, 2.0, 3.0], [10.0, 12.0, 14.0])
    assert "center_x = 1.000" in txt and "size_z = 14.000" in txt
    pdb = boxmod.box_corners_pdb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    assert pdb.count("HETATM") == 8 and pdb.count("CONECT") >= 8
    assert "   1.000   1.000   1.000" in pdb


# ---------------------------------------------------------------- ligand properties and ranking

def test_properties_lipinski_and_le():
    from rdkit import Chem

    props = ligands.properties(Chem.MolFromSmiles(ASPIRIN))
    assert props["heavy_atoms"] == 13
    assert props["mw"] == pytest.approx(180.16, abs=0.05)
    assert props["hbd"] == 1 and props["hba"] == 3
    assert props["lipinski_violations"] == 0
    assert ligands.ligand_efficiency(-6.5, 13) == pytest.approx(0.5)
    assert ligands.ligand_efficiency(-6.5, 0) is None
    big = ligands.properties(Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(O)(O)O"))
    assert big["lipinski_violations"] == 2  # MW > 500, logP > 5


def test_lipinski_count_from_numbers():
    assert ligands.lipinski_count(mw=600, logp=6, hbd=6, hba=11) == 4
    assert ligands.lipinski_count(mw=500, logp=5, hbd=5, hba=10) == 0


def test_ranking_order():
    recs = [
        {"name": "b", "affinity": -7.0, "ligand_efficiency": 0.3},
        {"name": "a", "affinity": -9.1, "ligand_efficiency": 0.2},
        {"name": "c", "affinity": -7.0, "ligand_efficiency": 0.4},
        {"name": "d", "affinity": None, "ligand_efficiency": None},
    ]
    assert [r["name"] for r in results.rank(recs)] == ["a", "c", "b"]


def test_largest_fragment_and_neutralise():
    smi, notes = ligands.clean_smiles("[Na+].CC(=O)[O-]")
    assert smi == "CC(=O)O"
    assert notes


def test_protonation_at_ph():
    pytest.importorskip("openbabel")
    assert "[O-]" in ligands.protonate_smiles("CC(=O)O", 7.4)
    assert "[O-]" not in ligands.protonate_smiles("CC(=O)O", 1.0)


def test_symmetric_rmsd():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ref = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
    AllChem.EmbedMolecule(ref, randomSeed=3)
    ref = Chem.RemoveHs(ref)
    # same coordinates, but carboxylate and swapped atom order: RMSD must be 0
    probe = Chem.RenumberAtoms(ref, list(reversed(range(ref.GetNumAtoms()))))
    probe = Chem.RWMol(probe)
    assert ligands.symmetric_rmsd(probe, ref) == pytest.approx(0.0, abs=1e-6)
    conf = probe.GetConformer()
    for i in range(probe.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 1.0, p.y, p.z))
    assert ligands.symmetric_rmsd(probe, ref) == pytest.approx(1.0, abs=1e-6)


def test_contacts_by_distance():
    lig = [("O1", "O", (0.0, 0.0, 0.0)), ("C1", "C", (1.5, 0.0, 0.0))]
    rec = [
        ("A", "SER", "27", "OG", "O", (2.8, 0.0, 0.0)),  # H-bond candidate to O1 (2.8) and contact
        ("A", "LEU", "30", "CD1", "C", (5.0, 0.0, 0.0)),  # 3.5 from C1: contact, not an H-bond
        ("A", "GLY", "99", "N", "N", (9.0, 0.0, 0.0)),  # too far
    ]
    res, hb = results.contacts(lig, rec)
    assert res == ["A:SER27", "A:LEU30"]
    assert len(hb) == 1 and hb[0]["residue"] == "A:SER27" and hb[0]["distance"] == pytest.approx(2.8)


def test_format_eta():
    assert results.format_eta(None) == "?"
    assert results.format_eta(42) == "1 min"
    assert results.format_eta(3 * 3600 + 20 * 60) == "3 h 20 min"


# ---------------------------------------------------------------- resume

def test_resume_skips_done_and_failed_redoes_changed(tmp_path):
    from compute_bot.recipes.dock import docking

    settings = {"box": [[0, 0, 0], [20, 20, 20]], "scoring": "vina", "exhaustiveness": 8, "poses": 9, "seed": 42,
                "ph": 7.4}
    ligs = [{"name": n, "smiles": s} for n, s in (("a", "CCO"), ("b", "CCN"), ("c", "CCC"), ("d", "CCCl"))]
    out = tmp_path / "poses"
    out.mkdir()
    docking.write_record(out, {"name": "a", "status": "done", "signature": docking.signature("CCO", settings)})
    docking.write_record(out, {"name": "b", "status": "done", "signature": docking.signature("CCF", settings)})
    docking.write_record(out, {"name": "d", "status": "failed", "signature": docking.signature("CCCl", settings)})
    todo = docking.pending(ligs, out, settings)
    assert [lig["name"] for lig in todo] == ["b", "c"]  # b's SMILES changed; d failed for good
    # another box makes everything pending again
    settings2 = dict(settings, exhaustiveness=16)
    assert [lig["name"] for lig in docking.pending(ligs, out, settings2)] == ["a", "b", "c", "d"]


def test_crash_marker_counts_attempts(tmp_path):
    from compute_bot.recipes.dock import docking

    assert docking.mark_running(tmp_path, "x") == 1
    assert docking.mark_running(tmp_path, "x") == 2
    docking.clear_running(tmp_path, "x")
    assert docking.mark_running(tmp_path, "x") == 1


# ---------------------------------------------------------------- end to end

def _run(work, threads="4"):
    env = dict(os.environ, PYTHONPATH=str(REPO), COMPUTE_THREADS=threads)
    return subprocess.run([sys.executable, "-m", "compute_bot.recipes.dock"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=1800)


def test_user_error_without_ligands_or_box(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps(RECIPES["dock"].validate({})))
    proc = _run(tmp_path)
    assert proc.returncode != 0
    assert proc.stderr.strip().splitlines()[-1].startswith("ERROR: ")


@pytest.mark.slow
def test_end_to_end_1stp_redock_and_screen(tmp_path):
    pytest.importorskip("vina")
    pytest.importorskip("meeko")
    pytest.importorskip("pdbfixer")
    params = RECIPES["dock"].validate({"receptor": "1STP", "ligand_ref": "BTN", "exhaustiveness": "4", "poses": "5"})
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "ligandy.csv").write_text(
        "id;smiles\n"
        "biotyna;OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12\n"
        "benzamidyna;NC(=N)c1ccccc1\n"
        "sol_biotyny;[Na+].[O-]C(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12\n"
        "zepsuty;C1CC(\n")
    t0 = time.time()
    proc = _run(tmp_path)
    first = time.time() - t0
    print(f"first run {first:.0f} s")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "PROGRESS: ligand" in proc.stdout

    summary = (tmp_path / "summary.txt").read_text()
    print(summary)
    assert len(summary) <= 2500
    m = re.search(r"RMSD[^\n]*?(\d+[.,]\d+) Å", summary)
    assert m, summary
    rmsd = float(m.group(1).replace(",", "."))
    assert rmsd < 2.0

    with open(tmp_path / "wyniki" / "ranking.csv") as fh:
        rows = list(csv.DictReader(fh))
    names = [r["name"] for r in rows]
    assert set(names) == {"ref_BTN", "biotyna", "benzamidyna", "sol_biotyny"}
    affinities = [float(r["affinity_kcal_mol"]) for r in rows]
    assert affinities == sorted(affinities)
    assert all((tmp_path / r["pose_sdf"]).exists() for r in rows)
    failed = (tmp_path / "wyniki" / "nieudane.csv").read_text()
    assert "zepsuty" in failed
    for f in ("box.txt", "box.pdb", "receptor.pdbqt", "kontakty.csv"):
        assert (tmp_path / "wyniki" / f).exists(), f
    assert list((tmp_path / "wyniki" / "kompleksy").glob("*.pdb"))
    for line in (tmp_path / "highlights.txt").read_text().split():
        assert (tmp_path / line).exists()

    poses = sorted((tmp_path / "wyniki" / "pozy").glob("*.sdf"))
    stamps = {p.name: p.stat().st_mtime_ns for p in poses}
    t0 = time.time()
    proc2 = _run(tmp_path)
    second = time.time() - t0
    print(f"second run {second:.0f} s")
    assert proc2.returncode == 0, proc2.stderr[-3000:]
    assert "pominięto 4" in proc2.stdout
    assert {p.name: p.stat().st_mtime_ns for p in poses} == stamps
    assert (tmp_path / "summary.txt").read_text() == summary
