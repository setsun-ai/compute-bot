# Changelog

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
