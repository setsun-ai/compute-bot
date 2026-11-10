# Przepis `md` – dynamika molekularna białka (GROMACS albo OpenMM)

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
| `T_eq` | 0 | temperatura równowagowania NVT i NPT [K] (też `gen_temp`); 0 = 300 K, albo T, gdy T < 300 |
| `ns` | 1 | długość produkcji [ns] |
| `nvt_ps`, `npt_ps` | 50, 50 | równowagowanie z więzami położeń atomów ciężkich [ps] (0 = pomiń etap) |
| `salt` | 0.15 | NaCl [M]; układ jest zawsze zobojętniany |
| `box`, `d` | dodecahedron, 1.0 | kształt pudełka i odległość białka od ściany [nm] |
| `model` | 1 | który model z pliku NMR |
| `chains` | wszystkie | łańcuchy do zostawienia, np. `A,B` |
| `seed` | -1 | ziarno prędkości początkowych i genion (-1 = losowe) |
| `ensemble` | npt | produkcja w NPT (C-rescale / barostat MC) albo NVT |
| `engine` | gromacs | `gromacs` albo `openmm` (ten sam protokół i te same wyniki) |
| `mdp` | – | tylko GROMACS: zmiany w plikach mdp, `klucz=wartość;nvt:klucz=wartość` (bez przedrostka = produkcja) |
| `extend_ns` | 0 | dolicz tyle ns do skończonej produkcji (komenda /dolicz) |

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
   T < 300; inna temperatura: `T_eq=310`), produkcja w T – tak jak „podgrzanie do 600 K” w projekcie. Ustawienia oddziaływań zależą od pola
   (CHARMM: force-switch 1.0–1.2 nm, GROMOS: 1.4 nm, pozostałe: 1.0 nm + DispCorr).
4. **Postęp**: `etap 3/4: NPT 45% (krok 11250/25000), koniec etapu ok. 10.11 14:05; do końca całości ~2 h 10 min`.
   Dla ~13 tys. atomów laptop liczy rzędu 4–8 ns/dzień, więc 0.5 ns w 600 K to kilka godzin.
5. **Analiza** (po każdym uruchomieniu; same wykresy można odświeżyć `python -m compute_bot.recipes.md --analiza`):
   zbieżność minimalizacji, T / Epot / ciśnienie / gęstość, `trjconv -pbc mol -center`, RMSD szkieletu względem
   struktury po minimalizacji, RMSF Cα, Rg, mostki H białko–białko, `gmx dssp` (GROMACS ≥ 2023), wydajność ns/dzień.

## engine=openmm

Ten sam protokół w OpenMM (CPU, `COMPUTE_THREADS` wątków), pliki w `openmm/`:

- Struktura jak wyżej, potem pdbfixer (brakujące atomy, w tym końcowy OXT), `Modeller.addHydrogens` przy pH 7,
  pudełko o tym samym rozmiarze co w GROMACS (średnica białka + 2·d, ten sam kształt), woda, NaCl `salt` +
  zobojętnienie.
- Pola: amber99sb-ildn → `amber99sbildn.xml`, amber99sb, amber03, charmm27 → **CHARMM36** (`charmm36.xml` +
  jego TIP3P; CHARMM27 w OpenMM nie ma). Woda: tip3p, tip4p (→ TIP4P-Ew), spce. GROMOS i OPLS-AA tylko w GROMACS.
- Minimalizacja L-BFGS, potem NVT i NPT z więzami położeń atomów ciężkich (1000 kJ/mol/nm², jak `posre.itp`),
  barostat Monte Carlo (1 bar), produkcja; LangevinMiddle 2 fs (tarcie 1/ps), więzy HBonds, PME.
- Wznawianie: stan zapisywany co ~5 min (`MD_CHECKPOINT_S`) i przy SIGTERM; po restarcie nowa część trajektorii
  (`md.part2.dcd` …) od ostatniego stanu, klatki policzone po tym stanie są odrzucane. Gotowy etap ma `*.done.json`.
- Wyniki: DCD + CSV ze StateDataReporter, analiza MDAnalysis (RMSD, RMSF, Rg, mostki H, DSSP – 3 stany H/E/pętla).
  OpenMM nie zapisuje ciśnienia, więc tego panelu brak.

## mdp=… (tylko GROMACS)

Zmienia lub dopisuje opcje w plikach `.mdp`, np. `md pdb=1L2Y mdp=nstxout-compressed=100;tcoupl=nose-hoover;tau_t=1.0 1.0`.
Bez przedrostka – produkcja (`md.mdp`); inny etap z przedrostkiem `em:`, `nvt:`, `npt:` albo `md:`, np.
`mdp=nvt:ref_t=310 310;npt:tau_p=5;em:emtol=500;md:dt=0.001`. Nieznany przedrostek – błąd z listą dozwolonych.

- Sprawdzana jest tylko składnia (`[etap:]klucz=wartość`, `-` i `_` w kluczu znaczą to samo); resztę ocenia grompp –
  jego ostrzeżenie trafia do komunikatu błędu.
- `dt` zmienia krok, ale długość etapu w ps zostaje (np. `md:dt=0.001` przy `ns=1` → 1 000 000 kroków), chyba że
  podasz też `nsteps`. `ref_t` ma pierwszeństwo przed `T` / `T_eq` (a `gen_temp` idzie za nim) – podsumowanie mówi,
  w jakiej temperaturze naprawdę liczono.
- Podsumowanie wypisuje zmiany osobno dla każdego etapu. Zmiany etapu, który jest pominięty (`nvt_ps=0`), dają uwagę.
- Z `engine=openmm` – błąd; przy `extend_ns` – pomijane (przedłużenie liczy z ustawieniami produkcji).

## gromacs/run.sh – ten sam przebieg do własnej edycji

Po zbudowaniu układu (GROMACS) przepis zapisuje `gromacs/run.sh`: wszystkie etapy MD tego zadania jako linie
`md_step` dokładnie takie, jak je uruchomił (te same `-f/-c/-r/-t/-p`, to samo `MD_MAXWARN`). Startem jest
`ions.gro` (białko w wodzie z jonami). Żeby coś zmienić: zmień wybrane `.mdp` (albo linie w `run.sh`) i wyślij
folder jako zip do bota – policzy go jak własny skrypt (zob. `docs/skrypty.md`). Do zipa wystarczą: `run.sh`,
`ions.gro`, `topol.top`, `*.itp` i `em.mdp`, `nvt.mdp`, `npt.mdp`, `md.mdp` (bez trajektorii i checkpointów –
etapy policzą się od nowa). Skrypt da się uruchomić ponownie: gotowe etapy są pomijane. `md_step` jest w bocie;
u siebie zamiast niego: `gmx grompp -o NAZWA.tpr opcje…` i `gmx mdrun -deffnm NAZWA`. Przedłużeń (`extend_ns`)
i analizy skrypt nie zawiera.

## extend_ns – doliczanie

`/dolicz` tworzy nowe zadanie z całym katalogiem poprzedniego (pliki dowiązane twardo) i `extend_ns=X`:

- GROMACS: `gmx convert-tpr -extend` → `md_ext1.tpr`, potem `mdrun -cpi md.cpt -noappend` do nowych plików
  `md_ext1.part0002.*` (kolejne doliczenie: `md_ext2` z `md_ext1.cpt`). OpenMM: `md_ext1.part*.dcd` od stanu
  końcowego produkcji.
- Pliki poprzedniego zadania nigdy nie są zmieniane: wszystko, co trzeba zapisać, powstaje jako nowy plik.
- Analiza obejmuje całą produkcję (GROMACS: `md_full.xtc`/`md_full.edr` z `trjcat`/`eneconv`), podsumowanie
  podaje łączną długość. Przerwane doliczanie wznawia się z `md_ext1.cpt`; gotowe jest pomijane.

## Co wraca

- `summary.txt` – najważniejsze liczby (po polsku), 4 wykresy jako zdjęcia: `rownowagowanie.png`, `rmsd_rg.png`,
  `rmsf.png`, `dssp.png`.
- `wyniki/` – CSV każdej serii (`energia_nvt.csv`, `energia_npt.csv`, `energia_md.csv`, `rmsd.csv`, `rmsf.csv`,
  `rg.csv`, `wiazania_h.csv`, `dssp.csv`, `em_energia.csv`), `uklad.csv`, `wydajnosc.csv`, `struktura.txt`,
  do obejrzenia: `bialko_em.pdb` + `bialko.xtc` (samo białko, dopasowane) i `bialko_klatki.pdb` (wielomodelowy,
  ≤ ~20 MB, np. do PyMOL).
- `gromacs/` – wszystkie pliki GROMACS (`run.sh`, `topol.top`, `*.mdp`, `*.tpr`, `*.log`, `*.edr`, `*.xtc`,
  `*_noPBC.xtc`);
  przy `engine=openmm` – `openmm/` (`system.xml`, `system.pdb`, `*.state.xml`, `*.part*.dcd`, `*.part*.csv`).

## Uwagi

- W 600 K i 1 bar (NPT) woda paruje i gęstość spada – podsumowanie to zaznacza. Do samego rozwijania białka
  wygodniejsza jest produkcja w NVT (`ensemble=nvt`).
- Białka wielołańcuchowe: `-pbc mol` składa każdy łańcuch osobno, kompleks może wyglądać na rozerwany przez
  granicę pudełka (RMSD i tak jest liczone po dopasowaniu).
- Ligandy, jony z kryształu i kwasy nukleinowe nie są parametryzowane – są usuwane.
