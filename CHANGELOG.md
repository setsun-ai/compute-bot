# Changelog

## 0.2.1 (2026-11)

- **Nietypowe ustawienia MD:** `T_eq=` (temperatura równowagowania), `mdp=` dla każdego etapu z przedrostkiem
  (`nvt:tau_t=1.0 1.0;md:dt=0.001`), a `/skrypt N` (📝 przy wynikach) oddaje zadanie jako zip z plikami `.mdp` i
  `run.sh` – zmieniasz, odsyłasz, liczy się z Twoimi ustawieniami. Działa dla zadań md liczonych od tej wersji.

## 0.2.0 (2026-11)

- **Przewodnik** (`/guide`, 📖 w panelu, `docs/przewodnik.md`): pierwsze kroki, dynamika, QSAR, omika, dokowanie,
  CHARMM-GUI/NAMD, skrypty, wyniki, praca zdalna, rozwiązywanie problemów – z przykładami „Wypróbuj”.
- **Duże wyniki:** link do pobrania ważny 24 h (wznawialny), polecenie `scp`, a bez Tailscale kawałki po 45 MB
  (`/get N plik czesci`).
- **Lipidomika w `omics`:** tabele z Profindera/MFE i MZmine 3, projekt sparowany (`pair=`, Wilcoxon dla par),
  uśrednianie replikatów technicznych, klasy lipidów z nazw, KMD, Van Krevelen, RT vs m/z, Venn.
- Bot sam pisze o nowej wersji (raz, z przyciskiem instalacji); `/start` wita przewodnikiem i przykładem.
- Panel: przewodnik, przykłady, pobieranie dużych plików.

## 0.1.3 (2026-11)

- **Instalator działa z katalogu domowego.** Uruchomiony z `~/Documents/...` padał na
  `critical libmamba … cannot set current path: Permission denied`: micromamba jako użytkownik `compute` nie mogła
  wejść do zamkniętego katalogu, z którego wystartował instalator. Teraz instalator zaczyna od `/`.
- **Ponowne uruchomienie po nieudanej próbie** usuwa niedokończone środowisko (katalog bez `conda-meta`), zamiast
  utknąć na nim.

## 0.1.2 (2026-11)

- **Starsze paczki CHARMM-GUI z OpenMM 8:** minimalizacja w skryptach sprzed OpenMM 8 podawała tolerancję w
  kJ/mol zamiast kJ/mol/nm i kończyła się błędem jednostek; kopia skryptu poprawia jednostkę (ta sama wartość).
- Wydanie powstaje dopiero, gdy przejdą też testy przepisów (0.1.1 wyszło z jednym czerwonym testem).

## 0.1.1 (2026-11)

- **CHARMM-GUI + OpenMM na CPU:** skrypt `openmm_run.py` z paczek CHARMM-GUI 3.5 wpuszczał tylko platformy GPU
  (CUDA, OpenCL) i kończył się od razu na laptopie bez karty graficznej. Kopia skryptu dopisuje teraz CPU do listy.
- Pierwsze wydanie z kompletem testów w CI (0.1.0 zostało tylko tagiem).

## 0.1.0 (2026-11)

- **Kolejka obliczeń na Telegramie.** Plik albo komenda → zadanie w kolejce; jedno naraz, z najniższym
  priorytetem; wyniki (podsumowanie, wykresy, zip, duże pliki przez `/get`) wracają do czatu.
- **Przepisy:** `md` (GROMACS od PDB do analizy), `qsar` (deskryptory, selekcja, modele, walidacja OECD, domena),
  `omics` (DIA-NN / MS-DIAL → QC, normalizacja, statystyka, adnotacja), `dock` (AutoDock Vina z redockingiem).
  Do tego dowolne skrypty: zip z `run.sh`, `.tpr`, `.py`, `.ipynb`.
- **Odporność:** po restarcie laptopa zadanie startuje znowu, GROMACS wznawia się z checkpointu (`md_step`),
  dokowanie pomija gotowe ligandy, pobieranie z `/url` wznawia przerwany plik. Przy 92 °C zadanie czeka na
  ostygnięcie.
- **Wygoda:** `/pauza` i `/wznow`, `/top N`, `/retry N klucz=wartość`, `/usun N` i `/usun stare` z
  potwierdzeniem, cisza nocna, raport postępu co kilka godzin, ostrzeżenia o pełnym dysku i niedziałającym runnerze.
- **Aktualizacje** z wydań na GitHubie (`/update`, `/rollback`), nowe pakiety środowiska przez `/env`.
