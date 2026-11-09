"""
Registry of recipes: what each one accepts, so the bot can check parameters before a job exists.

Standard library only: the bot runs on the system Python, the recipes themselves run in the conda environment.
A recipe module `compute_bot.recipes.<name>` is started with `python -m` in the job's work directory and reads
`params.json` (already validated, defaults filled in) and the files next to it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class ParamError(ValueError):
    """A parameter the owner typed that can't be used, in words for the owner."""


@dataclass(frozen=True)
class Param:
    kind: str  # int, float, str, bool, choice, vec3, list
    default: object
    help: str
    choices: tuple = ()
    low: float | None = None
    high: float | None = None

    def parse(self, name, text):
        text = str(text).strip()
        try:
            if self.kind == "int":
                value = int(text)
            elif self.kind == "float":
                value = float(text.replace(",", "."))
            elif self.kind == "bool":
                lowered = text.lower()
                if lowered not in ("1", "0", "tak", "nie", "yes", "no", "true", "false", "on", "off"):
                    raise ValueError
                value = lowered in ("1", "tak", "yes", "true", "on")
            elif self.kind == "vec3":
                parts = [float(p) for p in re.split(r"[;,/ ]+", text.replace(",", " ").strip()) if p]
                if len(parts) != 3:
                    raise ValueError
                value = parts
            elif self.kind == "list":
                value = [p.strip() for p in text.split(",") if p.strip()]
                bad = [p for p in value if self.choices and p not in self.choices]
                if bad:
                    raise ParamError(f"{name}: nie znam {', '.join(bad)} (możliwe: {', '.join(self.choices)})")
            elif self.kind == "choice":
                value = text.lower() if text.lower() in self.choices else text
                if value not in self.choices:
                    raise ParamError(f"{name}={text}? Możliwe: {', '.join(self.choices)}")
            else:
                if not text or len(text) > 200:
                    raise ValueError
                value = text
        except ParamError:
            raise
        except ValueError:
            raise ParamError(f"{name}={text}? Oczekuję: {self.describe()}") from None
        if self.low is not None and value < self.low or self.high is not None and value > self.high:
            raise ParamError(f"{name}={text} poza zakresem {self.low}–{self.high}")
        return value

    def describe(self):
        if self.kind == "choice":
            return " | ".join(self.choices)
        if self.kind == "list" and self.choices:
            return "lista z: " + ",".join(self.choices)
        return {"int": "liczba całkowita", "float": "liczba", "bool": "tak/nie", "vec3": "x,y,z",
                "str": "tekst", "list": "lista po przecinku"}[self.kind]


@dataclass(frozen=True)
class Recipe:
    name: str
    title: str
    help: str
    params: dict = field(default_factory=dict)
    inputs: tuple = ()  # file extensions that this recipe takes, for the "what to do with this file" buttons
    needs_file: bool = True

    def validate(self, given):
        """Typed parameters with defaults filled in; unknown names are an error, not ignored."""
        unknown = [k for k in given if k not in self.params]
        if unknown:
            raise ParamError(f"{self.name} nie ma parametru {', '.join(unknown)}. Są: {', '.join(self.params)}")
        values = {name: p.default for name, p in self.params.items()}
        for name, text in given.items():
            values[name] = self.params[name].parse(name, text)
        return values

    def usage(self):
        lines = [f"<b>{self.title}</b> (<code>{self.name}</code>)", self.help, ""]
        for name, p in self.params.items():
            shown = ",".join(map(str, p.default)) if isinstance(p.default, list) else p.default
            default = "" if p.default in (None, "") else f" [{shown}]"
            lines.append(f"<code>{name}=</code> {p.help} – {p.describe()}{default}")
        return "\n".join(lines)


FORCE_FIELDS = ("amber99sb-ildn", "amber14sb", "amber03", "amber99sb", "charmm27", "oplsaa", "gromos54a7")
WATERS = ("auto", "tip3p", "tip4p", "spc", "spce")
QSAR_MODELS = ("mlr", "pls", "ridge", "lasso", "enet", "knn", "dt", "rf", "svr", "xgb")

RECIPES = {r.name: r for r in (
    Recipe(
        "md", "Dynamika molekularna (GROMACS)",
        "PDB (plik albo kod) → topologia → pudełko z wodą i jonami → minimalizacja → NVT → NPT → produkcja "
        "→ analiza (T, Epot, gęstość, RMSD, RMSF, Rg, mostki H, DSSP) z wykresami. Po restarcie wznawia z checkpointu.",
        {
            "pdb": Param("str", "", "kod PDB, gdy nie wysyłasz pliku (np. 1E0L)"),
            "ff": Param("choice", "amber99sb-ildn", "pole siłowe", FORCE_FIELDS),
            "water": Param("choice", "auto", "model wody (auto = zgodny z polem)", WATERS),
            "T": Param("float", 300.0, "temperatura produkcji [K]", low=50, high=1000),
            "ns": Param("float", 1.0, "długość produkcji [ns]", low=0.001, high=1000),
            "nvt_ps": Param("float", 50.0, "NVT z więzami położeń [ps]", low=0, high=10000),
            "npt_ps": Param("float", 50.0, "NPT z więzami położeń [ps]", low=0, high=10000),
            "salt": Param("float", 0.15, "stężenie NaCl [M]", low=0, high=3),
            "box": Param("choice", "dodecahedron", "kształt pudełka", ("dodecahedron", "cubic", "octahedron")),
            "d": Param("float", 1.0, "odległość białka od ściany [nm]", low=0.5, high=3),
            "model": Param("int", 1, "który model z pliku NMR", low=1, high=1000),
            "chains": Param("str", "", "łańcuchy do zostawienia, np. A,B (puste = wszystkie)"),
            "seed": Param("int", -1, "ziarno prędkości (-1 = losowe)"),
            "ensemble": Param("choice", "npt", "produkcja w NPT czy NVT (NVT lepsze do rozwijania w wysokiej T)",
                              ("npt", "nvt")),
            "engine": Param("choice", "gromacs", "silnik", ("gromacs", "openmm")),
            "mdp": Param("str", "", "zmienione opcje produkcji, np. nstxout-compressed=500;tcoupl=nose-hoover"),
            "extend_ns": Param("float", 0.0, "dolicz tyle ns do skończonej produkcji (ustawia /dolicz)",
                               low=0, high=1000),
        },
        inputs=(".pdb", ".cif"), needs_file=False),
    Recipe(
        "qsar", "QSAR / QSPR",
        "Tabela (xlsx/csv) z odpowiedzią i deskryptorami albo SMILES. Liczy deskryptory (RDKit, Mordred, PM6 z MOPAC), "
        "odsiewa zmienne, dzieli dane, wybiera deskryptory, stroi modele na walidacji krzyżowej i ocenia je wg OECD "
        "(R², Q²LOO, Q²F1–F3, CCC, rm², Golbraikh–Tropsha, Y-scrambling), rysuje domenę stosowalności (Williams). "
        "Wiersze bez odpowiedzi dostają prognozę. Odpowiedź tekstowa = klasyfikacja.",
        {
            "target": Param("str", "", "kolumna z odpowiedzią (puste = zgadnij)"),
            "smiles": Param("str", "", "kolumna ze SMILES (puste = zgadnij)"),
            "name": Param("str", "", "kolumna z nazwą związku (puste = pierwsza tekstowa)"),
            "split": Param("choice", "sorted", "podział: co k-ty po posortowaniu y, Kennard–Stone, losowy, z kolumny",
                           ("sorted", "ks", "random", "column")),
            "split_col": Param("str", "", "kolumna z podziałem (T/V, train/test, uczący/testowy)"),
            "every": Param("int", 3, "przy split=sorted: co który związek do testu", low=2, high=10),
            "test": Param("float", 0.25, "udział testu przy ks/random", low=0.05, high=0.5),
            "descriptors": Param("list", ["given"], "skąd deskryptory",
                                 ("given", "rdkit", "mordred", "mopac", "morgan")),
            "max_desc": Param("int", 4, "najwięcej deskryptorów w modelu", low=1, high=20),
            "select": Param("choice", "auto", "wybór zmiennych",
                            ("auto", "exhaustive", "forward", "ga", "enet", "all")),
            "models": Param("list", list(QSAR_MODELS), "modele", QSAR_MODELS),
            "corr_max": Param("float", 0.95, "odrzuć deskryptor skorelowany z innym powyżej", low=0.5, high=1),
            "scramble": Param("int", 100, "liczba permutacji Y-scrambling", low=0, high=10000),
            "seed": Param("int", 42, "ziarno losowości"),
        },
        inputs=(".xlsx", ".xls", ".csv", ".tsv", ".txt", ".smi")),
    Recipe(
        "omics", "Proteomika / metabolomika",
        "Macierz intensywności (DIA-NN pg/gg_matrix, MS-DIAL alignment, albo cechy × próbki) + arkusz próbek. "
        "Kontrola jakości, odsiew blanków i braków, imputacja, normalizacja, korekta dryfu (QC-LOESS) i partii, "
        "CV w QC i D-ratio, PCA, PLS-DA z VIP i walidacją, testy na każdą cechę z FDR, volcano, heatmapa, "
        "adnotacja UniProt i wzbogacenie (g:Profiler).",
        {
            "layer": Param("choice", "auto", "rodzaj danych", ("auto", "proteomics", "metabolomics")),
            "group": Param("str", "group", "kolumna z grupą w arkuszu próbek"),
            "ref": Param("str", "", "grupa odniesienia (np. CONTROL)"),
            "covariates": Param("str", "", "kowariaty do modelu liniowego, np. age_years,sex"),
            "na_max": Param("float", 0.5, "odrzuć cechę z większym udziałem braków w każdej grupie", low=0, high=1),
            "blank_ratio": Param("float", 3.0, "cecha musi być x razy wyższa niż w blankach", low=0, high=100),
            "impute": Param("choice", "halfmin", "uzupełnianie braków", ("halfmin", "knn", "min", "none")),
            "normalize": Param("choice", "median", "normalizacja", ("median", "pqn", "is", "sum", "none")),
            "drift": Param("choice", "auto", "korekta dryfu po QC", ("auto", "loess", "none")),
            "batch": Param("choice", "auto", "korekta partii", ("auto", "median", "none")),
            "qc_cv": Param("float", 0.3, "najwyższe CV w próbkach QC", low=0, high=5),
            "test": Param("choice", "auto", "test statystyczny",
                          ("auto", "ttest", "welch", "mwu", "anova", "kruskal", "lm")),
            "fdr": Param("float", 0.05, "próg FDR (Benjamini–Hochberg)", low=0.0001, high=0.5),
            "fc": Param("float", 1.0, "próg |log2 FC| na volcano", low=0, high=10),
            "annotate": Param("bool", True, "adnotacja białek z UniProt (internet)"),
            "enrich": Param("bool", True, "wzbogacenie GO/KEGG/Reactome przez g:Profiler (internet)"),
            "organism": Param("str", "hsapiens", "organizm dla g:Profiler"),
        },
        inputs=(".tsv", ".csv", ".txt", ".xlsx", ".zip", ".parquet")),
    Recipe(
        "dock", "Dokowanie (AutoDock Vina)",
        "Receptor (PDB albo kod) + ligandy (SMILES w csv/smi albo sdf). Przygotowuje białko i ligandy (pH, 3D, "
        "Meeko), wyznacza pudełko z ligandu krystalicznego, reszt albo współrzędnych, dokuje (skrining wznawia się "
        "po przerwie) i zwraca ranking, pozy i kontakty.",
        {
            "receptor": Param("str", "", "kod PDB receptora, gdy nie ma pliku w zipie"),
            "ligand_ref": Param("str", "", "nazwa ligandu krystalicznego wyznaczająca pudełko (np. STI)"),
            "residues": Param("str", "", "reszty kieszeni, np. A:45,A:88"),
            "center": Param("vec3", None, "środek pudełka [Å]"),
            "size": Param("vec3", [22.0, 22.0, 22.0], "rozmiar pudełka [Å] (razem z center)"),
            "padding": Param("float", 6.0, "margines wokół ligandu/reszt [Å]", low=0, high=20),
            "exhaustiveness": Param("int", 8, "dokładność Vina", low=1, high=64),
            "poses": Param("int", 9, "liczba póz na ligand", low=1, high=30),
            "ph": Param("float", 7.4, "pH protonacji", low=0, high=14),
            "keep_waters": Param("bool", False, "zostaw wody krystaliczne"),
            "scoring": Param("choice", "vina", "funkcja oceny", ("vina", "vinardo")),
            "seed": Param("int", 42, "ziarno"),
        },
        inputs=(".zip", ".sdf", ".smi", ".csv", ".tsv", ".mol2", ".pdb", ".cif"), needs_file=False),
    Recipe(
        "charmm", "Paczka z CHARMM-GUI",
        "Zip pobrany z CHARMM-GUI (Solution Builder itp.): bot znajduje folder gromacs/, openmm/ albo namd/ i "
        "uruchamia protokół z README tej paczki (minimalizacja, równowagowanie, produkcja), wznawiając po "
        "przerwie, a potem tę samą analizę co md.",
        {
            "engine": Param("choice", "auto", "który folder paczki (auto = gromacs, potem openmm, potem namd)",
                            ("auto", "gromacs", "openmm", "namd")),
            "ns": Param("float", 0.0, "długość produkcji [ns] (0 = jak w paczce)", low=0, high=1000),
            "T": Param("float", 0.0, "temperatura [K] (0 = jak w paczce)", low=0, high=1000),
            "steps": Param("str", "", "które kroki, np. 4,5 (puste = wszystkie z README)"),
        },
        inputs=(".zip", ".tgz", ".gz")),
    Recipe(
        "env", "Aktualizacja środowiska",
        "Doinstalowuje pakiety z deploy/environment.yml do środowiska conda (po aktualizacji bota).",
        {}, needs_file=False),
)}

# Values offered as buttons on the parameter card (choice and bool parameters offer all their values anyway).
# Parameters not named here and not choice/bool are changed by typing key=value.
QUICK = {
    "md": {"engine": (), "ff": (), "T": (300, 310, 350, 400, 500, 600), "ns": (0.1, 0.5, 1, 5, 10, 50),
           "ensemble": (), "salt": (0, 0.1, 0.15, 0.5)},
    "qsar": {"split": (), "select": (), "max_desc": (2, 3, 4, 5, 6, 8), "every": (2, 3, 4, 5)},
    "omics": {"layer": (), "normalize": (), "impute": (), "drift": (), "test": (), "fdr": (0.01, 0.05, 0.1),
              "annotate": (), "enrich": ()},
    "dock": {"scoring": (), "exhaustiveness": (4, 8, 16, 32), "poses": (5, 9, 20), "padding": (4, 6, 8)},
    "charmm": {"engine": (), "ns": (0, 1, 5, 10, 50), "T": (0, 300, 310, 600)},
}


def quick_values(recipe, name):
    """The button values for one parameter of the card, as the owner would type them."""
    param = RECIPES[recipe].params[name]
    given = QUICK.get(recipe, {}).get(name, ())
    if given:
        return [f"{v:g}" if isinstance(v, float) else str(v) for v in given]
    if param.kind == "choice":
        return list(param.choices)
    if param.kind == "bool":
        return ["tak", "nie"]
    return []
