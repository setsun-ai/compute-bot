# compute-bot

Prywatny bot na Telegramie, który zamienia stary laptop w kolejkę obliczeń: wysyłasz plik, wracasz po kilku
godzinach, a wyniki (podsumowanie, wykresy, zip) czekają w czacie. Jedno zadanie naraz, z najniższym priorytetem,
więc inne programy na laptopie mają zawsze pierwszeństwo. Przy przegrzaniu (≥ 92 °C) zadanie jest wstrzymywane do
ostygnięcia. Po restarcie laptopa przerwane zadanie startuje znowu, a GROMACS wznawia się z checkpointu.

| Co wysyłasz | Co się dzieje | Opis |
|---|---|---|
| `.pdb` albo `/md 1E0L T=600 ns=0.5` | dynamika w GROMACS: topologia, woda, jony, EM, NVT, NPT, produkcja, analiza | [docs/md.md](docs/md.md) |
| xlsx/csv + `qsar` | deskryptory (RDKit, Mordred, MOPAC PM6), selekcja, MLR/PLS/kNN/drzewa/RF/SVR/XGB, walidacja OECD, domena | [docs/qsar.md](docs/qsar.md) |
| macierz DIA-NN / MS-DIAL + `omics` | QC, blanki, braki, normalizacja, dryf i partie, PCA, PLS-DA, testy z FDR, UniProt, g:Profiler | [docs/omics.md](docs/omics.md) |
| zip receptor + ligandy + `dock` | AutoDock Vina, redocking kontrolny, ranking, kontakty | [docs/dock.md](docs/dock.md) |
| zip z `run.sh`, `.tpr`, `.py`, `.ipynb` | dowolne obliczenie w środowisku conda | [docs/skrypty.md](docs/skrypty.md) |

Pliki większe niż 20 MB (limit Telegrama): `/url LINK omics …` – Google Drive, Dropbox albo zwykły link; plik
pobiera się w samym zadaniu i wznawia po przerwie.

Komendy: `/jobs`, `/job N`, `/log N`, `/get N [plik]`, `/cancel N`, `/retry N [klucz=wartość]`, `/top N`,
`/usun N`, `/usun stare`, `/pauza`, `/wznow`, `/co 3` (raport postępu co 3 h), `/cisza 23-7`, `/system`, `/env`,
`/update`, `/rollback`, `/help [przepis]`. Bot ostrzega, gdy runner przestaje odpowiadać albo kończy się dysk.

## Instalacja (Debian/Ubuntu, x86_64)

1. Załóż **nowego** bota u [@BotFather](https://t.me/BotFather) (osobny token: dwa programy na jednym się kłócą).
2. Na laptopie:

```bash
git clone https://github.com/setsun-ai/compute-bot ~/Documents/boty/compute-bot
```

```bash
sudo bash ~/Documents/boty/compute-bot/deploy/install.sh
```

Instalator pyta o token, zakłada użytkownika `compute`, środowisko conda w `/var/lib/compute-bot/env` (~3 GB,
pierwszy raz 10–30 minut) i dwie usługi: `compute-bot` (Telegram) i `compute-runner` (obliczenia). Można go
uruchomić jeszcze raz: zadania i ustawienia zostają. Liczbę wątków zmienisz w `/etc/compute-runner.env`.

Nowe wersje instaluje `/update` w bocie (wydania z GitHuba, sprawdzane selftestem); runner przełącza się na nową
wersję, gdy skończy bieżące zadanie. Jeśli nowa wersja potrzebuje nowych pakietów, bot poprosi o `/env`.

## Jak to działa

Dwie usługi i katalog `/var/lib/compute-bot/jobs/NNNN/`: bot zapisuje zadanie (`spec.json`, pliki w `work/`),
runner je liczy i tylko on zapisuje `status.json` i `log.txt`. Bot co kilkanaście sekund porównuje stany i pisze
o zmianach. Słownik pojęć: [GLOSSARY.md](GLOSSARY.md), kontrakt przepisów: [docs/RECIPE_CONTRACT.md](docs/RECIPE_CONTRACT.md).

## Rozwój

```bash
python3 -m pytest tests --ignore=tests/recipes   # rdzeń: sama biblioteka standardowa
```

Przepisy testuje się w środowisku z `deploy/environment.yml` (`pytest tests/recipes`, `-m slow` pobiera struktury
z RCSB i liczy GROMACS/Vina).
