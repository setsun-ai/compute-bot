# Przewodnik

Ten sam tekst jest w bocie (`/guide`) i w panelu. Plik generuje `python3 -m compute_bot.guide`.

## 🚀 Pierwsze kroki

**Jak to działa**  
Wysyłasz plik albo komendę → otwiera się **karta** z ustawieniami → **▶️ Start** → laptop liczy (jedno zadanie naraz) → bot pisze, gdy ruszy, co kilka godzin i na końcu, z podsumowaniem, wykresami i zipem.  

**Pierwsze 10 minut**  
1. `/system` – czy laptop żyje: temperatura, RAM, dysk.  
2. Przycisk „Wypróbuj” niżej: krótka dynamika małego białka (kilka–kilkanaście minut).  
3. Gdy przyjdzie wynik, otwórz zip i zobacz `summary.txt` oraz folder `wyniki/`.  

**Karta**  
Przyciski zmieniają najczęstsze ustawienia; resztę wpisujesz jako `klucz=wartość` (np. `T=310 ns=2`). Zła wartość jest odrzucana od razu, więc nie stracisz nocy na literówkę.  

**W trakcie**  
`/jobs` – kolejka · `/job 3` – szczegóły · `/log 3` – co wypisuje program · `/cancel 3` – stop · `/pauza` i `/wznow` – wstrzymaj wszystko (np. gdy ktoś chce pograć).  

**Gdy coś pójdzie źle**  
Bot pokaże koniec logu i przycisk 🔁 Ponów. `/retry 3 T=300` uruchamia to samo ze zmienionym ustawieniem. Więcej: `/guide problemy`.  

## 🧬 Dynamika molekularna

**Co wysłać**  
Plik `.pdb` albo kod: `/md 1E0L`. Bot sam czyści strukturę (jeden model, bez wód i ligandów), dodaje wodę i jony, minimalizuje, równoważy (NVT, NPT z więzami) i liczy produkcję.  

**Ustawienia, które mają znaczenie**  
`ns=1` – długość produkcji; to ona decyduje o czasie liczenia.  
`T=300` – temperatura produkcji [K]; równowagowanie zawsze w 300 K.  
`ff=amber99sb-ildn` – pole siłowe; woda dobiera się sama.  
`engine=openmm` – drugi silnik, te same etapy (wolniejszy na CPU).  
`ensemble=nvt` – **użyj przy wysokiej T**: w NPT przy 600 K i 1 bar woda odparowuje (gęstość spada z ~1000 do kilkudziesięciu kg/m³) i wynik nie ma sensu.  

**Ile to trwa**  
Zmierzone na tym laptopie: ~50 ns na dobę dla 6 tys. atomów (Trp-cage w wodzie). Czas rośnie mniej więcej z liczbą atomów, więc ~13 tys. atomów (np. 1E0L) to około 20–25 ns na dobę: 0,5 ns w pół godziny, 10 ns w pół doby. Dokładną godzinę końca bot pokaże po pierwszych minutach produkcji.  

**Jak czytać wynik**  
• *Temperatura i gęstość*: mają falować wokół zadanej T i ~1000 kg/m³. Dryf = układ nie jest zrównoważony.  
• *RMSD szkieletu*: rośnie i się wypłaszcza (zwykle 0,1–0,3 nm) → struktura stabilna. Ciągły wzrost → białko się rozwija albo symulacja jest za krótka, by to ocenić.  
• *Rg* (promień żyracji): rośnie, gdy białko się rozluźnia.  
• *RMSF*: które reszty są ruchliwe (pętle, końce).  
• *DSSP*: ile helis i β zostaje w czasie.  

**Dalej**  
`/dolicz 4 5` – skończone zadanie #4 o 5 ns dłużej, bez liczenia od zera.  
`mdp=nstxout-compressed=500` – własne opcje GROMACS dla produkcji.  
Laptop się zrestartuje? Zadanie wznowi się samo z punktu sprzed kilku minut.  

## 📈 QSAR / QSPR

**Co wysłać**  
Tabelę xlsx/csv: wiersz = związek, kolumny = nazwa, odpowiedź (np. LogKd) i deskryptory *albo* SMILES. Podpis pliku: `qsar`. Wiersze bez odpowiedzi dostaną prognozę.  

**Ustawienia**  
`target=LogKd` – która kolumna jest odpowiedzią (zwykle zgadnie sam).  
`descriptors=rdkit` – policz deskryptory ze SMILES (`mordred` – dużo więcej, `mopac` – kwantowe PM6: HOMO, LUMO, dipol; trwa najdłużej).  
`max_desc=3` – ile deskryptorów w modelu. Zasada: co najmniej 5 związków uczących na deskryptor.  
`split=sorted` – co trzeci związek (po posortowaniu odpowiedzi) do testu.  

**Jak czytać wynik**  
• *R²* – dopasowanie do zbioru uczącego. Samo wysokie R² niczego nie dowodzi.  
• *Q²LOO* – walidacja krzyżowa: czy model przewiduje związki, których nie widział. Powinno być > 0,6 i niewiele niższe od R² (duża różnica = przeuczenie).  
• *Q²F1–F3, CCC, RMSE_P* – to samo na zbiorze testowym (walidacja zewnętrzna).  
• *Y-scrambling* – po wymieszaniu odpowiedzi model ma być zły; jeśli nadal jest dobry, to przypadek.  
• *Wykres Williamsa* – domena stosowalności: związki z dźwignią powyżej h* są poza nią i ich prognozom nie należy ufać.  

**Który model jest „najlepszy”**  
Bot wybiera po walidacji krzyżowej, nie po R² i nie po teście, a przy remisie bierze prostszy. W podsumowaniu jest uzasadnienie i równanie MLR do prezentacji.  

**Typowe potknięcia**  
Braki w deskryptorach (deskryptor z brakiem jest odrzucany – podsumowanie mówi, przez które związki); za mało związków na liczbę deskryptorów; SMILES z błędem (lista w wynikach).  

## 🔬 Proteomika, metabolomika, lipidomika

**Co wysłać**  
Tabelę z programu, który przetworzył surowe dane (DIA-NN, MS-DIAL, MZmine, Profinder), i najlepiej **arkusz próbek**: która kolumna to próbka, QC, blank, jaka grupa, kolejność iniekcji. Kilka plików spakuj w zip. Podpis: `omics`. Bez arkusza bot zgadnie typy z nazw (QC, Blank) i to napisze.  

**Ustawienia**  
`group=group` – kolumna z grupą; `ref=CONTROL` – grupa odniesienia.  
`pair=Donor_ID` – ta sama osoba w obu grupach (np. M1 i M12): testy dla par.  
`blank_ratio=10` – cecha musi być 10× wyższa niż w blanku (kryterium z zajęć z lipidomiki; domyślnie 3).  
`qc_cv=0.3` – odrzuć cechy z RSD w QC powyżej 30%.  
`covariates=age_years,sex` – uwzględnij w modelu liniowym.  

**Co robi po kolei**  
Audyt → złe iniekcje (wzorce wewnętrzne) → filtr blanka i braków → uzupełnienie braków → normalizacja → korekta dryfu i partii (po QC) → RSD w QC → PCA → test na każdą cechę z poprawką FDR → PLS-DA z VIP → adnotacja. W `wyniki/kroki.txt` jest, ile cech zostało po każdym kroku i dlaczego.  

**Jak czytać wynik**  
• *PCA*: próbki QC mają leżeć w ciasnej kupce. Rozrzucone QC = problem z pomiarem, nie z biologią.  
• *Mediana CV w QC* przed i po korekcie: po ma być niższa (dobrze poniżej 20–30%).  
• *Volcano*: w górze i na bokach są cechy istotne (FDR) i o dużej zmianie.  
• *FDR, nie p*: przy tysiącach cech samo p < 0,05 daje setki fałszywych trafień.  
• *PLS-DA*: patrz na dokładność z walidacji krzyżowej i test permutacyjny, nie na ładny wykres.  

**Surowe pliki** (.wiff, .raw, .mzML) przetwórz najpierw u siebie w DIA-NN / MS-DIAL / MZmine; bot bierze tabele. Duże tabele: `/url LINK omics`.  

## 💊 Dokowanie

**Co wysłać**  
Ligandy jako `.smi`, `.csv` (kolumny smiles, name) albo `.sdf`, z podpisem `dock receptor=1STP` (kod PDB), albo zip: receptor .pdb + ligandy.  

**Gdzie dokować (pudełko)**  
`ligand_ref=BTN` – wokół ligandu, który już siedzi w strukturze (najpewniejsze).  
`residues=A:45,A:88` – wokół wskazanych reszt.  
`center=10,5,-3` – ręcznie [Å].  
Bez niczego: jeśli w strukturze jest jeden ligand, bot weźmie jego miejsce.  

**Inne ustawienia**  
`exhaustiveness=8` – dokładność szukania; 16–32 dla ważnych wyników, czas rośnie liniowo.  
`ph=7.4` – protonacja białka i ligandów.  

**Jak czytać wynik**  
• *Redocking*: bot najpierw dokuje ligand z kryształu. RMSD < 2 Å = protokół odtwarza znaną pozę i reszcie wyników można ufać bardziej. Powyżej – zmień pudełko albo dokładność.  
• *Powinowactwo [kcal/mol]*: bardziej ujemne = lepiej, ale to **ranking, nie pomiar**. Różnice poniżej ~1 kcal/mol nic nie znaczą.  
• *LE* (efektywność liganda): powinowactwo na ciężki atom – nie faworyzuje dużych cząsteczek.  
• *kontakty.csv*: z którymi resztami ligand się styka i gdzie mogą być wiązania wodorowe.  

**Ile to trwa**  
Rząd minuty na ligand; setki ligandów to godziny. Po restarcie laptopa gotowe ligandy są pomijane.  

**Ograniczenia**  
Białko jest sztywne, wody i jony metali są usuwane, tautomery nie są wyliczane.  

## 📦 Paczki CHARMM-GUI i NAMD

**CHARMM-GUI**  
Na charmm-gui.org przygotuj układ (Solution Builder), zaznacz silnik (GROMACS / OpenMM / NAMD) i pobierz `charmm-gui.tgz`. Wyślij ten plik botowi – rozpozna paczkę sam.  

`engine=gromacs` – który folder paczki uruchomić (auto: GROMACS, potem OpenMM, potem NAMD).  
`ns=5` – długość produkcji, `T=310` – temperatura; 0 = jak w paczce.  

**Do sprawozdania**  
W `wyniki/protokol_charmm.csv` i w podsumowaniu jest tabela: pole siłowe, woda, jony, termostat, barostat, więzy, kroki – gotowe do porównania z ręcznym przygotowaniem z `/guide md`.  

**NAMD**  
Wymaga rejestracji, więc nie instaluje się sam. Pobierz z ks.uiuc.edu wersję *Linux-x86_64-multicore* i wyślij plik `NAMD_…tar.gz` botowi (jeśli ma ponad 20 MB: `/url LINK`). Od tej chwili `namd3` jest dostępny w każdym zadaniu.  

**Na co uważać**  
Starsze paczki zakładają kartę graficzną – bot przestawia je na procesor, ale OpenMM i NAMD są tu wyraźnie wolniejsze niż GROMACS. Produkcja z paczki bywa długa: ustaw `ns` świadomie.  

## 🛠 Własne skrypty i notebooki

**Kiedy**  
Gdy gotowy przepis nie pasuje: własny protokół GROMACS, skrypt Pythona, notebook z zajęć.  

**Co wysłać**  
• **zip z `run.sh`** – folder z plikami i skryptem; wszystko, co skrypt zapisze, wraca.  
• **.py** – uruchomi się w środowisku z GROMACS, OpenMM, RDKit, Vina, MOPAC, scikit-learn, pandas.  
• **.ipynb** – wykona się od góry do dołu, wraca notebook z wynikami (kod z Colaba z GPU nie zadziała).  
• **.tpr** – od razu mdrun.  

**GROMACS, który przeżyje restart**  
W `run.sh` zamiast grompp + mdrun pisz:  

```
md_step nvt -f nvt.mdp -c em.gro -r em.gro -p topol.top  
md_step npt -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top  
md_step md  -f md.mdp  -c npt.gro -t npt.cpt -p topol.top  
```

Skończony etap jest pomijany, przerwany wznawia się z checkpointu.  

**Zasady**  
Skrypt może zostać uruchomiony drugi raz (po restarcie laptopa), więc niech nie kasuje wyników. Liczba wątków jest w `$COMPUTE_THREADS`. Plik wejściowy ponad 20 MB: `/url LINK skrypt` albo `curl` w skrypcie.  

**Czego laptop nie zrobi**  
Brak karty graficznej: AlphaFold, ESMFold i trenowanie dużych sieci zostają na Colabie.  

## 📥 Wyniki i duże pliki

**Co przychodzi po zakończeniu**  
1. Podsumowanie (`summary.txt`) – najważniejsze liczby słowami.  
2. Do czterech najważniejszych wykresów.  
3. `wyniki_NNNN.zip` – tabele CSV, wszystkie wykresy, log.  

**Gdy plik jest za duży** (Telegram: 50 MB na plik)  
• Do 50 MB, ale poza zipem: `/get 3 gromacs/md.xtc`.  
• Większe, z włączonym panelem: bot daje przycisk **⬇️** z linkiem ważnym 24 h (urządzenie z Tailscale albo po `/funnel on`); przerwane pobieranie da się wznowić.  
• Z SSH: bot podaje gotowe polecenie `scp` dla pliku i całego katalogu.  
• Bez panelu i SSH: `/get 3 plik czesci` – kawałki po 45 MB (do 1 GB) i polecenie do sklejenia.  

**Trajektorie MD**  
Pełna trajektoria bywa duża, dlatego w `wyniki/` jest też lekka wersja samego białka (`bialko.xtc` + `bialko_em.pdb`) do obejrzenia w VMD/PyMOL/ChimeraX.  

**Wrócić do starego zadania**  
`/jobs` → `/job 3` → 📦 Wyniki. Pliki leżą na laptopie, dopóki ich nie usuniesz.  

**Miejsce na dysku**  
`/system` pokazuje wolne miejsce. Poniżej 2 GB nowe zadania czekają, a bot o tym pisze. `/usun 3` usuwa jedno zadanie z wynikami, `/usun stare` – skończone ponad 7 dni temu (zawsze z potwierdzeniem). Zanim usuniesz, pobierz to, co potrzebne.  

## 🖥 Panel, Tailscale i SSH

**Panel**  
Przycisk „Panel” obok pola wiadomości (albo `/panel`): kolejka na żywo z godziną końca, temperatura i dysk, wykresy i logi, formularze wszystkich ustawień, ten przewodnik.  
Działa, gdy telefon ma włączony Tailscale. Bez niego: `/funnel on` udostępnia panel publicznie (wejdzie tylko Twój Telegram), `/funnel off` – z powrotem prywatnie.  

**Terminal** (z Maca: `ssh vostro`)  

```
compute jobs                 # kolejka  
compute log 3 -f             # log na żywo  
compute submit folder/       # folder z run.sh  
compute submit dane.xlsx qsar max_desc=3  
compute md 1E0L T=310 ns=2  
cd "$(compute where 3)"      # katalog wyników  
compute-env                  # gmx, python, vina w powłoce  
```

To ta sama kolejka co w bocie: bot napisze o starcie i końcu także takich zadań.  

**VS Code i Jupyter**  
VS Code: Remote-SSH → `vostro`, interpreter `/var/lib/compute-bot/env/bin/python`.  
Jupyter: `ssh vostro-jupyter`, potem `compute-env; jupyter lab --no-browser` i link z tokenem otwierasz na Macu.  

**Uwaga na rdzenie**  
Ręczne liczenie w terminalu zabiera te same 2 rdzenie co kolejka: `compute pause` przed, `compute resume` po. Długie rzeczy w terminalu uruchamiaj w `tmux` albo wyślij do kolejki.  

## 🧯 Gdy coś nie działa

**❌ Zadanie skończyło się błędem**  
Bot pokazuje koniec logu; ostatnia linia `ERROR:` mówi po polsku, co poprawić (np. brak kolumny, zły kod PDB). Popraw i `/retry 3` albo `/retry 3 klucz=wartość`. Pełny log: `/log 3` i `log.txt` w zipie.  

**🌡️ „Za gorąco, wstrzymuję”**  
Normalne zabezpieczenie: powyżej 92 °C zadanie czeka, aż procesor ostygnie do 75 °C, i rusza samo. Jeśli dzieje się ciągle – laptop stoi na czymś miękkim albo ma zakurzony wentylator.  

**🔁 „Wznowione po przerwie”**  
Laptop się zrestartował albo zaktualizował. Dynamika traci najwyżej kilka minut. Po trzech takich przerwach zadanie kończy się błędem, żeby nie kręcić się w kółko.  

**⚠️ „Runner nie odpowiada”**  
Usługa licząca stoi albo laptop śpi. Przez SSH: `sudo systemctl restart compute-runner`. Gdy nie ma SSH – trzeba poprosić gospodarzy o restart laptopa; zadania wznowią się same.  

**💾 „Za mało miejsca”**  
`/usun stare` albo `/usun 3` – najpierw pobierz, co potrzebne (`/guide wyniki`).  

**⏳ Nic się nie dzieje**  
`/system`: czy runner działa i czy nie ma pauzy (`/wznow`). `/jobs`: może liczy się wcześniejsze zadanie – `/top 5` przesuwa #5 na początek.  

**Bot milczy**  
Telegram bywa niedostępny z laptopa przez chwilę; obliczenia idą dalej, a wiadomości dojdą później.  

**Nowa wersja**  
Bot sam napisze, gdy wyjdzie; `/update` instaluje, `/rollback` cofa. Jeśli po aktualizacji poprosi o `/env` – to doinstalowanie pakietów jako zwykłe zadanie.  
