# Przepis `charmm` – paczka z CHARMM-GUI (GROMACS / OpenMM / NAMD)

Projekt 2: to samo białko przygotowane w CHARMM-GUI (Solution Builder), policzone tym silnikiem, którego paczkę
pobrałeś, i porównane z ręcznym przygotowaniem (`md`). Bot czyta README z folderu silnika, uruchamia jego kroki
(minimalizacja → równowagowanie → produkcja w częściach), wypisuje, co wybrało CHARMM-GUI, i robi tę samą analizę co
`md`. Każdy krok przetrwa restart laptopa.

## Co wysłać

Plik `charmm-gui.tgz` (albo zip) prosto z CHARMM-GUI, z podpisem, np.:

- `charmm` – GROMACS (gdy jest w paczce), protokół dokładnie jak w paczce (zwykle 10 × 1 ns, 303.15 K),
- `charmm ns=1 T=300` – produkcja 1 ns w 300 K (jak w ręcznym `md`), reszta jak w paczce,
- `charmm engine=openmm ns=0.5`, `charmm engine=namd`,
- `charmm steps=4` – tylko minimalizacja i równowagowanie (szybkie sprawdzenie paczki przed długą produkcją).

Najlepiej zaznaczyć w CHARMM-GUI (krok „Input Generation”) silnik z zajęć: GROMACS, OpenMM albo NAMD. Paczka może mieć
kilka folderów; bot bierze ten z `engine=`. Wystarczy też zip samego folderu `gromacs/` (bez `toppar/` z góry paczki
OpenMM nie ruszy – wtedy wyślij całość).

## Parametry

| parametr | domyślnie | znaczenie |
|---|---|---|
| `engine` | auto | `gromacs`, `openmm`, `namd`; auto = gromacs, potem openmm, potem namd (gdy zainstalowany) |
| `ns` | 0 | długość produkcji [ns]; 0 = jak w paczce. Produkcja dalej idzie w częściach README (`step5_1`, `step5_2`, …), każda nie dłuższa niż część z paczki (np. `ns=2.5` → 3 × 0.833 ns) |
| `T` | 0 | temperatura [K] wszystkich kroków MD (termostat i losowanie prędkości); 0 = jak w paczce |
| `steps` | wszystkie | które kroki, np. `4`, `5`, `4.1,5_1` (nazwy plików bez „step”) |

`ns` i `T` są wpisywane do **kopii** plików (`bot_step5_production.mdp`, `bot_step5_production.inp`,
`bot_step5_1.inp` dla NAMD); oryginały z CHARMM-GUI zostają nietknięte. Gdy produkcja dałaby mniej niż 100 klatek do
analizy, bot zapisuje klatki częściej (tylko zapis, fizyka bez zmian) – to też jest w tabeli i podsumowaniu.

## Co się dzieje

1. **Paczka**: folder `gromacs/`, `openmm/` albo `namd/` (także o poziom niżej, np. `charmm-gui-1234567/gromacs/`).
2. **README**: to skrypt csh. Bot go nie uruchamia, tylko interpretuje (`set`, `@`, `if`, `while`, `printf`, `sed`)
   i z wynikowych komend robi listę kroków: GROMACS `gmx grompp` + `gmx mdrun`, OpenMM `python openmm_run.py …`,
   NAMD `namd2 plik.inp` (dla NAMD także pliki `step5_run.inp`, które README robi `sed`-em). Bez README – standardowy
   protokół Solution Builder. Kroki Membrane Buildera (`step6.1…6.6`, `step7_*`) też są czytane.
3. **Protokół** (`wyniki/protokol_charmm.csv`, kolumny: etap, parametr, wartość, źródło): pole siłowe
   (`input.config.dat`, nazwy plików toppar), woda i liczba cząsteczek, jony i stężenie (`step2.2_ions_count.str`),
   pudełko (`step3_pbcsetup.str` / `sysinfo.dat`), atomy, ładunek białka (PSF); dla każdego kroku: integrator, dt,
   liczba kroków, czas, T, termostat, barostat i ciśnienie, więzy położeń (stałe siłowe), więzy wiązań, vdW,
   elektrostatyka, prędkości, zapis klatek. Wartości zmienione przez bota mają dopisek „w paczce: …”.
4. **Liczenie**:
   - GROMACS: każdy krok przez `md_step` (gotowy pominięty, przerwany wznawia się z `.cpt`), z `-n index.ndx`, gdy
     README go używa. `-maxwarn -1` z README zamieniane na `-maxwarn 1000` (nowy GROMACS nie przyjmuje -1).
     Minimalizacja w pojedynczej precyzji (README proponuje `gmx_d`).
   - OpenMM: własny `openmm_run.py` paczki na CPU (`--platform CPU`; stare wersje z wpisanym na sztywno CUDA dostają
     kopię `bot_openmm_run.py` różniącą się tylko platformą). Krok z kompletnym plikiem restartu (`-orst`) jest
     pomijany, następny startuje z niego (`-irst`). Przerwany krok liczy się od początku (OpenMM pisze restart dopiero
     na końcu kroku).
   - NAMD: `namd3 +pN plik` (albo `namd2`) dla każdego kroku; krok z `NAZWA.coor` i „End of program” w logu jest
     pomijany, przerwany liczy się od początku.
5. **Postęp**: `krok 3/12: produkcja 1/10 45% (krok 225000/500000), koniec kroku ~2 h 10 min (ok. 10.11 14:05); do
   końca całości ~1.4 dni`. Szybkość mierzona w trakcie (OpenMM i NAMD podają ją same).
6. **Analiza** (jak w `md`, te same wykresy): T, Epot, ciśnienie, gęstość (GROMACS: `gmx energy`; NAMD: linie ENERGY,
   kcal → kJ, gęstość z objętości i masy z PSF; OpenMM: StateDataReporter paczki zapisuje tylko T i Epot), RMSD
   szkieletu (GROMACS: względem struktury po minimalizacji; OpenMM/NAMD: względem `step3_input.pdb`), Rg, RMSF Cα,
   DSSP (MDAnalysis, PSF + xtc/dcd, białko scalane przez granice pudełka), wydajność ns/dzień. Same wykresy można
   odświeżyć: `python -m compute_bot.recipes.charmm --analiza`.

## NAMD

NAMD ma licencję, więc nie ma go w condzie. Pobierz „Linux-x86_64-multicore” (NAMD 3, CPU) z
<https://www.ks.uiuc.edu/Research/namd/> (darmowa rejestracja) i wyślij plik `.tar.gz` botowi albo na laptopie:
`sudo bash deploy/install-namd.sh NAMD_3.x_Linux-x86_64-multicore.tar.gz`. Bez NAMD `engine=namd` kończy się
błędem z tą instrukcją, a `engine=auto` wybiera GROMACS albo OpenMM, jeśli paczka je ma.

## Co wraca

- `summary.txt` – co wybrało CHARMM-GUI (pole, woda, jony, pudełko), protokół krok po kroku (czas, ensemble, dt, T,
  termostat, barostat, więzy), zmiany bota, wyniki (T, p, gęstość, Epot, RMSD, Rg, DSSP, RMSF), wydajność;
  zdjęcia: `rownowagowanie.png`, `rmsd_rg.png`, `rmsf.png`, `dssp.png`.
- `wyniki/` – `protokol_charmm.csv` (do tabeli porównawczej z ręcznym `md`), `energia_nvt.csv` / `energia_npt.csv` /
  `energia_md.csv`, `rmsd.csv`, `rg.csv`, `rmsf.csv`, `dssp.csv`, `em_energia.csv`, `wydajnosc.csv`, `analiza.json`.
- folder silnika (`gromacs/`, `openmm/`, `namd/`) – wszystkie pliki wejściowe i wyniki silnika (`.log`, `.edr`, `.xtc`,
  `.dcd`, `.rst`, `.out`), kopie `bot_*`.

## Porównanie z ręcznym `md` (na co patrzeć)

| | ręcznie (`md`, domyślnie) | CHARMM-GUI (typowo) |
|---|---|---|
| pole siłowe | amber99sb-ildn (albo `ff=`) | CHARMM36m |
| woda | TIP3P | TIP3P w wersji CHARMM (LJ na wodorach) |
| jony | NaCl 0.15 M + zobojętnienie | KCl albo NaCl 0.15 M (Monte Carlo/odległość) |
| pudełko | dodekaedr, 1.0 nm od białka | kostka (zwykle ≥ 1.0 nm) |
| równowagowanie | NVT 50 ps + NPT 50 ps, 2 fs | 125 ps NVT, 1 fs |
| więzy położeń | ciężkie atomy (`-DPOSRES`, 1000 kJ/mol/nm²) | szkielet 400, łańcuchy boczne 40 kJ/mol/nm² |
| termostat / barostat | V-rescale / C-rescale | GROMACS: V-rescale lub Nosé–Hoover / C-rescale lub Parrinello–Rahman; OpenMM: Langevin / Monte Carlo; NAMD: Langevin / Langevin piston |
| oddziaływania | zależnie od pola | force-switch 1.0–1.2 nm, PME |

Rzeczywiste wartości Twojej paczki są w `wyniki/protokol_charmm.csv`.
