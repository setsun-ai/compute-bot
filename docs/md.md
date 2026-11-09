# Przepis `md` – dynamika molekularna białka (GROMACS)

Małe białko w wodzie od pliku PDB do wykresów: kontrola struktury → topologia → pudełko z wodą i jonami →
minimalizacja → NVT → NPT → produkcja → analiza. Liczy się na laptopie (CPU), więc każdy krok przetrwa restart:
gotowe etapy są pomijane, przerwany `mdrun` wznawia się z checkpointu (`md_step`).

## Co wysłać

- plik `.pdb` / `.cif` z podpisem, np. `md ff=amber99sb-ildn T=600 ns=0.5`, albo
- samą komendę z kodem PDB (plik zostanie pobrany z RCSB): `md pdb=1E0L T=600 ns=0.5`.

Przykład jak w projekcie 1 („Symulacje molekularne”): `md pdb=1E0L ff=amber99sb-ildn salt=0.15 nvt_ps=50 npt_ps=50 ns=0.5 T=600`.

## Parametry

| parametr | domyślnie | znaczenie |
|---|---|---|
| `pdb` | – | kod PDB, gdy nie ma pliku |
| `ff` | amber99sb-ildn | pole siłowe: amber99sb-ildn, amber03, amber99sb, charmm27, oplsaa, gromos54a7 |
| `water` | auto | woda; `auto` = zgodna z polem: AMBER/CHARMM → tip3p, OPLS-AA → tip4p, GROMOS → spc |
| `T` | 300 | temperatura produkcji [K] |
| `ns` | 1 | długość produkcji [ns] |
| `nvt_ps`, `npt_ps` | 50, 50 | równowagowanie z więzami położeń atomów ciężkich [ps] (0 = pomiń etap) |
| `salt` | 0.15 | NaCl [M]; układ jest zawsze zobojętniany |
| `box`, `d` | dodecahedron, 1.0 | kształt pudełka i odległość białka od ściany [nm] |
| `model` | 1 | który model z pliku NMR |
| `chains` | wszystkie | łańcuchy do zostawienia, np. `A,B` |
| `seed` | -1 | ziarno prędkości początkowych i genion (-1 = losowe) |

## Co się dzieje

1. **Struktura**: liczba modeli, łańcuchy z sekwencją, przerwy, brakujące atomy ciężkie i reszty (vs SEQRES), HETATM,
   woda, altloc. Zostaje jeden model i wybrane łańcuchy; woda krystaliczna, ligandy i jony są usuwane, z altloc zostaje
   pierwszy wariant, wodory z pliku są pomijane (`-ignh`). Brakujące atomy ciężkie, brakujące reszty wewnątrz łańcucha
   i niestandardowe aminokwasy (np. MSE) poprawia pdbfixer – wszystko to jest wypisane w podsumowaniu. Brakujące
   końce łańcucha nie są dobudowywane.
2. **Układ**: `pdb2gmx` (protonacja domyślna dla pH 7, His wg sieci wiązań H, naładowane końce), `editconf`,
   `solvate`, `genion -neutral -conc salt`. Zapisywane: wektory pudełka, objętość, liczba atomów, wód, Na⁺, Cl⁻,
   ładunek białka i układu.
3. **Etapy**: minimalizacja (steep, Fmax < 1000), NVT z `-DPOSRES` (prędkości z `gen_vel`, tylko tu), NPT z
   `-DPOSRES` i C-rescale, produkcja bez więzów. Krok 2 fs, więzy h-bonds (LINCS), PME, V-rescale (Protein /
   Non-Protein), ciągłość współrzędnych i prędkości przez `-t poprzedni.cpt`. Równowagowanie w 300 K (albo w T, gdy
   T < 300), produkcja w T – tak jak „podgrzanie do 600 K” w projekcie. Ustawienia oddziaływań zależą od pola
   (CHARMM: force-switch 1.0–1.2 nm, GROMOS: 1.4 nm, pozostałe: 1.0 nm + DispCorr).
4. **Postęp**: `etap 3/4: NPT 45% (krok 11250/25000), koniec etapu ok. 10.11 14:05; do końca całości ~2 h 10 min`.
   Dla ~13 tys. atomów laptop liczy rzędu 4–8 ns/dzień, więc 0.5 ns w 600 K to kilka godzin.
5. **Analiza** (po każdym uruchomieniu; same wykresy można odświeżyć `python -m compute_bot.recipes.md --analiza`):
   zbieżność minimalizacji, T / Epot / ciśnienie / gęstość, `trjconv -pbc mol -center`, RMSD szkieletu względem
   struktury po minimalizacji, RMSF Cα, Rg, mostki H białko–białko, `gmx dssp` (GROMACS ≥ 2023), wydajność ns/dzień.

## Co wraca

- `summary.txt` – najważniejsze liczby (po polsku), 4 wykresy jako zdjęcia: `rownowagowanie.png`, `rmsd_rg.png`,
  `rmsf.png`, `dssp.png`.
- `wyniki/` – CSV każdej serii (`energia_nvt.csv`, `energia_npt.csv`, `energia_md.csv`, `rmsd.csv`, `rmsf.csv`,
  `rg.csv`, `wiazania_h.csv`, `dssp.csv`, `em_energia.csv`), `uklad.csv`, `wydajnosc.csv`, `struktura.txt`,
  do obejrzenia: `bialko_em.pdb` + `bialko.xtc` (samo białko, dopasowane) i `bialko_klatki.pdb` (wielomodelowy,
  ≤ ~20 MB, np. do PyMOL).
- `gromacs/` – wszystkie pliki GROMACS (`topol.top`, `*.mdp`, `*.tpr`, `*.log`, `*.edr`, `*.xtc`, `*_noPBC.xtc`).

## Uwagi

- W 600 K i 1 bar (NPT) woda paruje i gęstość spada – podsumowanie to zaznacza. Do samego rozwijania białka
  wygodniejsza jest produkcja w NVT.
- Białka wielołańcuchowe: `-pbc mol` składa każdy łańcuch osobno, kompleks może wyglądać na rozerwany przez
  granicę pudełka (RMSD i tak jest liczone po dopasowaniu).
- Ligandy, jony z kryształu i kwasy nukleinowe nie są parametryzowane – są usuwane.
