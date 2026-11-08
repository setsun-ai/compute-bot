# Własne skrypty: zip z `run.sh`, `.tpr`, `.py`, `.ipynb`

Gdy gotowy przepis (`md`, `qsar`, `omics`, `dock`) nie pasuje, wyślij własne obliczenie. Zadanie startuje w katalogu
z Twoimi plikami, z niskim priorytetem, w środowisku conda z GROMACS, OpenMM, RDKit, Vina, MOPAC, scikit-learn,
pandas, statsmodels i resztą (`deploy/environment.yml`). Wszystko, co skrypt zapisze, wraca jako `wyniki_NNNN.zip`
(duże pliki osobno przez `/get N plik`).

## Zip z `run.sh`

Spakuj folder (np. `1E0L/`) razem ze skryptem `run.sh`. Wspólny folder w zipie jest pomijany.

GROMACS najlepiej przez `md_step` – po restarcie laptopa skończone etapy są pomijane, a przerwany wznawia się
z checkpointu (`.cpt` co 5 minut):

```bash
#!/bin/bash
set -e
# md_step NAZWA opcje-grompp  (→ grompp -o NAZWA.tpr, potem mdrun -deffnm NAZWA)
md_step nvt   -f nvt.mdp   -c em.gro  -r em.gro  -p topol.top
md_step npt   -f npt.mdp   -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top
md_step md600 -f md600.mdp -c npt.gro -t npt.cpt -p topol.top

# analiza – szybka, może się powtórzyć
for s in nvt npt md600; do
  echo Temperature | gmx energy -f $s.edr -o ${s}_temp.xvg
  echo Potential   | gmx energy -f $s.edr -o ${s}_pot.xvg
done
echo "Protein System" | gmx trjconv -s md600.tpr -f md600.xtc -o md600_noPBC.xtc -pbc mol -center
echo "Backbone Backbone" | gmx rms -s em.tpr -f md600_noPBC.xtc -o md600_rmsd.xvg
```

Zmienne dostępne w skrypcie: `COMPUTE_THREADS` (ile wątków), `MD_MAXWARN` (ostrzeżenia grompp, domyślnie 0),
`MD_MDRUN_ARGS` (dodatkowe opcje mdrun). Skrypt powinien dawać się uruchomić drugi raz: po restarcie runner
startuje go jeszcze raz w tym samym katalogu (najwyżej 3 próby).

Pliki większe niż 20 MB (limit Telegrama dla bota) skrypt może pobrać sam: `curl -fLO https://…`.

## Pojedyncze pliki

- **`.tpr`** – od razu `md_step` (mdrun z wznawianiem), np. gdy grompp zrobiłeś na Macu.
- **`.py`** – `python plik.py` w środowisku conda.
- **`.ipynb`** – notebook wykonany od góry do dołu; wraca `plik_wynik.ipynb` z wynikami komórek. Kod z Colaba
  (`google.colab`, GPU) tu nie zadziała.

## Czego laptop nie zrobi

Brak GPU i 2 rdzenie: AlphaFold/ColabFold, ESMFold, trenowanie dużych sieci (PyTorch na CUDA) i MCX zostają na
Colabie. Surowe pliki MS (`.wiff`, `.raw`) przetwórz w DIA-NN / MS-DIAL u siebie i wyślij wyeksportowane tabele
do przepisu `omics`.
