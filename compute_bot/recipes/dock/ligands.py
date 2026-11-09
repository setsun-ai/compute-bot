"""Ligand chemistry: cleaning, protonation at pH, 3D, Meeko PDBQT, drug-likeness numbers, symmetry-aware RMSD."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


def clean_smiles(smiles: str):
    """Largest organic fragment, neutralised. Returns (canonical SMILES, notes); ValueError if unreadable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("nieprawidłowy SMILES")
    notes = []
    if len(Chem.GetMolFrags(mol)) > 1:
        mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
        notes.append("usunięto sole/przeciwjony")
    before = Chem.MolToSmiles(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    after = Chem.MolToSmiles(mol)
    if after != before:
        notes.append("zobojętniono ładunki")
    return after, notes


def protonate_smiles(smiles: str, ph: float) -> str:
    """Open Babel's pH model (as `obabel -p <pH>`): carboxylic acids, amines, amidines... charged as at the pH.

    It is a rule-based model, not a pKa predictor; when Open Babel is missing or changes the skeleton, the input
    SMILES is kept.
    """
    try:
        from openbabel import openbabel as ob
    except ImportError:
        return smiles
    ob.obErrorLog.SetOutputLevel(0)
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "smi")
    mol = ob.OBMol()
    if not conv.ReadString(mol, smiles):
        return smiles
    mol.AddHydrogens(False, True, float(ph))
    out = conv.WriteString(mol).split()
    if not out:
        return smiles
    new, old = Chem.MolFromSmiles(out[0]), Chem.MolFromSmiles(smiles)
    if new is None or old is None or new.GetNumHeavyAtoms() != old.GetNumHeavyAtoms():
        return smiles
    return Chem.MolToSmiles(new)


def embed(smiles: str, seed: int):
    """3D with all hydrogens: ETKDGv3, then MMFF94 (UFF if MMFF lacks parameters)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("nie udało się wygenerować struktury 3D (ETKDG)")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def to_pdbqt(mol) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"Meeko: {err.strip() or 'nie udało się zapisać PDBQT'}")
    return text


def poses_to_mols(pdbqt_text: str):
    """Docked poses (Vina PDBQT) back to one RDKit molecule with a conformer per pose, plus the SDF text."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    pmol = PDBQTMolecule(pdbqt_text, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pmol)
    sdf, failures = RDKitMolCreate.write_sd_string(pmol)
    if not mols or mols[0] is None:
        raise ValueError("Meeko nie odtworzył cząsteczki z pozy")
    return mols[0], sdf


# ---------------------------------------------------------------- numbers for the ranking

def lipinski_count(mw, logp, hbd, hba) -> int:
    return int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)


def properties(mol) -> dict:
    """Descriptors of the neutral parent molecule (as the owner wrote it, before pH protonation)."""
    mol = Chem.RemoveHs(mol)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "lipinski_violations": lipinski_count(mw, logp, hbd, hba),
    }


def ligand_efficiency(affinity, heavy_atoms):
    """LE = -ΔG / heavy atoms [kcal/mol per heavy atom]."""
    if affinity is None or not heavy_atoms:
        return None
    return round(-float(affinity) / heavy_atoms, 3)


# ---------------------------------------------------------------- RMSD against the crystal pose

def _skeleton(mol, conf_id=-1):
    """Heavy-atom graph with element-only atoms and plain single bonds, and its coordinates.

    Matching on the bare graph makes the RMSD independent of protonation, charges, bond orders and aromaticity
    that differ between the crystal ligand and the docked (protonated) one; symmetry is handled by trying every
    graph automorphism.
    """
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    index = {old: new for new, old in enumerate(heavy)}
    skel = Chem.RWMol()
    for old in heavy:
        atom = Chem.Atom(mol.GetAtomWithIdx(old).GetAtomicNum())
        atom.SetNoImplicit(True)
        skel.AddAtom(atom)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in index and j in index:
            skel.AddBond(index[i], index[j], Chem.BondType.SINGLE)
    skel.UpdatePropertyCache(strict=False)
    pos = mol.GetConformer(conf_id).GetPositions()[heavy]
    return skel.GetMol(), np.asarray(pos)


def symmetric_rmsd(probe, ref, probe_conf=-1, ref_conf=-1, max_matches=20000):
    """In-place (no superposition) heavy-atom RMSD over all symmetry-equivalent atom mappings; None if no match."""
    p_skel, p_xyz = _skeleton(probe, probe_conf)
    r_skel, r_xyz = _skeleton(ref, ref_conf)
    if p_skel.GetNumAtoms() != r_skel.GetNumAtoms():
        return None
    matches = p_skel.GetSubstructMatches(r_skel, uniquify=False, useChirality=False, maxMatches=max_matches)
    if not matches:
        return None
    best = min(float(np.sum((p_xyz[list(m)] - r_xyz) ** 2)) for m in matches)
    return float(np.sqrt(best / len(r_xyz)))


def reference_mol(pdb_block: str, template_sdf: str | None):
    """The crystal ligand as RDKit molecule with bond orders: from the CCD template, else Open Babel perception."""
    pdbmol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False, proximityBonding=True)
    if pdbmol is not None and template_sdf:
        template = Chem.MolFromMolBlock(template_sdf, removeHs=False)
        if template is not None:
            try:
                return AllChem.AssignBondOrdersFromTemplate(Chem.RemoveHs(template), pdbmol), "szablon CCD"
            except (ValueError, RuntimeError):
                pass  # atoms missing in the crystal: the template does not match, fall through
    try:
        from openbabel import pybel
    except ImportError:
        return None, ""
    try:
        sdf = pybel.readstring("pdb", pdb_block).write("sdf")
    except (OSError, ValueError):
        return None, ""
    mol = Chem.MolFromMolBlock(sdf, removeHs=True)
    return (mol, "wiązania z Open Babel") if mol is not None else (None, "")
