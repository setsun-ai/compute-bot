# Słownik

- **Zadanie (job)** – jedno obliczenie w kolejce, katalog `jobs/NNNN/` pod `COMPUTE_HOME`.
- **Spec** – `spec.json`: co uruchomić. Zapisuje go tylko bot, raz, przy tworzeniu zadania. Katalog zadania pojawia się
  atomowo (rename z `NNNN.tmp`), więc runner nigdy nie widzi pół-zadania.
- **Status** – `status.json`: stan zadania. Zapisuje go tylko runner.
- **Stan** – `queued` (czeka), `running` (liczy), `paused` (wstrzymane, bo procesor za gorący), `done`, `failed`,
  `cancelled`. Ostatnie trzy są końcowe.
- **Rodzaj (kind)** – jak zadanie się uruchamia: `script` (zip z `run.sh`), `tpr` (sam plik .tpr → `md_step`),
  `notebook` (.ipynb), `python` (.py), `recipe` (gotowy przepis: `md`, `qsar`, `omics`, `dock`, `env`).
- **Przepis (recipe)** – gotowy pipeline w `compute_bot/recipes/`, uruchamiany pythonem ze środowiska conda.
  Czyta `params.json` i pliki z katalogu `work/`, pisze wyniki do `work/` i krótkie podsumowanie do `summary.txt`.
- **Parametry** – `klucz=wartość` z podpisu pliku albo z komendy; bot sprawdza je z rejestrem przepisów, zanim
  powstanie zadanie.
- **Próba (attempt)** – ile razy runner uruchomił zadanie. Po restarcie laptopa zadanie startuje znowu; `md_step`
  wznawia mdrun z checkpointu. Po `MAX_ATTEMPTS` próbach zadanie kończy się błędem.
- **md_step** – pomocnik dla skryptów GROMACS: krok, który już się skończył, jest pomijany, przerwany wznawia się z
  `.cpt`.
- **Zgłoszony stan (announced)** – ostatni stan, o którym bot napisał właścicielowi; należy do bota (`bot.json`).
