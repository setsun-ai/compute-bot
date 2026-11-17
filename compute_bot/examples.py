"""
Ready inputs behind the guide's "try it" buttons: small, made up here (no course or personal data), quick on the
laptop, and shaped like what the owner will send later. Standard library only.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import NamedTuple


class Example(NamedTuple):
    recipe: str
    name: str
    typed: dict  # parameters as the owner would type them
    files: Callable[[], dict]
    note: str  # one line for the chat: what this run shows


def qsar_table():
    """30 compounds whose LogKd really depends on logD and V: selection should find those two."""
    rng = random.Random(7)
    rows = ["name,logD,Mw,V,HBD,polar,LogKd"]
    for i in range(1, 31):
        log_d, mw, vol = rng.uniform(-1, 5), rng.uniform(1.0, 4.0), rng.uniform(0.8, 3.0)
        hbd, polar = rng.randint(0, 4), rng.uniform(0, 1)
        y = 0.8 * log_d + 1.1 * vol - 0.5 + rng.gauss(0, 0.15)
        rows.append(f"zwiazek_{i:02d},{log_d:.3f},{mw:.3f},{vol:.3f},{hbd},{polar:.3f},{y:.3f}")
    return {"przyklad_qsar.csv": ("\n".join(rows) + "\n").encode()}


def omics_tables():
    """150 features, two groups of 8, QC every few injections, blanks; 12 features truly differ."""
    rng = random.Random(11)
    order = ["Blank_1", "QC_1"]
    for i in range(1, 9):
        order += [f"CONTROL_{i}", f"TREATED_{i}"]
        if i % 3 == 0:
            order.append(f"QC_{i // 3 + 1}")
    order += ["QC_4", "Blank_2"]
    sheet = ["sample,sample_type,group,run_order"]
    for n, name in enumerate(order, 1):
        kind = "blank" if name.startswith("Blank") else "qc" if name.startswith("QC") else "sample"
        group = name.split("_")[0] if kind == "sample" else ""
        sheet.append(f"{name},{kind},{group},{n}")
    table = ["feature_id," + ",".join(order)]
    for f in range(1, 151):
        base = 10 ** rng.uniform(4, 7)
        shift = 2.5 if f <= 6 else 0.4 if f <= 12 else 1.0
        cells = []
        for name in order:
            if name.startswith("Blank"):
                value = base * (0.9 if f > 140 else 0.01)  # the last ten are background
            else:
                value = base * (shift if name.startswith("TREATED") else 1.0) * rng.lognormvariate(0, 0.12)
            cells.append(f"{value:.1f}")
        table.append(f"F{f:03d}," + ",".join(cells))
    return {"intensywnosci.csv": ("\n".join(table) + "\n").encode(),
            "probki.csv": ("\n".join(sheet) + "\n").encode()}


def dock_ligands():
    return {"ligandy.smi": (b"OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12 biotyna\n"
                            b"NC(=N)c1ccccc1 benzamidyna\n"
                            b"OC(=O)CCCCC1CCSS1 kwas_liponowy\n"
                            b"CC(=O)Nc1ccc(O)cc1 paracetamol\n")}


EXAMPLES = {
    "md": Example("md", "przykład: Trp-cage 1L2Y", {"pdb": "1L2Y", "ns": "0.05"}, dict,
                  "Najmniejsze białko z zajęć (20 reszt), 50 ps produkcji: kilka–kilkanaście minut."),
    "qsar": Example("qsar", "przyklad_qsar.csv", {"max_desc": "2"}, qsar_table,
                    "30 wymyślonych związków, LogKd zależy od logD i V: model powinien wybrać właśnie te dwa."),
    "omics": Example("omics", "przykład omics", {"group": "group", "ref": "CONTROL", "annotate": "nie",
                                                 "enrich": "nie"}, omics_tables,
                     "150 cech, 2 grupy po 8, QC i blanki; 12 cech różni się naprawdę, 10 to tło z blanka."),
    "dock": Example("dock", "przykład: streptawidyna + biotyna", {"receptor": "1STP", "ligand_ref": "BTN",
                                                                  "exhaustiveness": "4"}, dock_ligands,
                    "Biotyna wraca na swoje miejsce (redocking) i wygrywa z trzema innymi cząsteczkami."),
}
