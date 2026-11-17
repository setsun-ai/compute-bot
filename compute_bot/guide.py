"""
The guide: one text for the chat (/guide), the panel (/api/guide) and docs/przewodnik.md (python -m compute_bot.guide).
Written for someone who knows the biology and chemistry but doesn't run simulations every day: what to send,
which settings matter, how long it takes, how to read what comes back, what usually goes wrong.

Text uses only Telegram's HTML: <b>, <i>, <code>, <pre>. Standard library only.
"""
from __future__ import annotations

import re
import sys
from typing import NamedTuple


class Topic(NamedTuple):
    id: str
    title: str
    text: str
    recipe: str | None = None  # key=value examples in the text are parameters of this recipe
    example: str | None = None  # examples.EXAMPLES key behind the "try it" button


TOPICS = (
    Topic("start", "🚀 Pierwsze kroki", (
        "<b>Jak to działa</b>\n"
        "Wysyłasz plik albo komendę → otwiera się <b>karta</b> z ustawieniami → <b>▶️ Start</b> → laptop liczy "
        "(jedno zadanie naraz) → bot pisze, gdy ruszy, co kilka godzin i na końcu, z podsumowaniem, wykresami "
        "i zipem.\n\n"
        "<b>Pierwsze 10 minut</b>\n"
        "1. <code>/system</code> – czy laptop żyje: temperatura, RAM, dysk.\n"
        "2. Przycisk „Wypróbuj” niżej: krótka dynamika małego białka (kilka–kilkanaście minut).\n"
        "3. Gdy przyjdzie wynik, otwórz zip i zobacz <code>summary.txt</code> oraz folder <code>wyniki/</code>.\n\n"
        "<b>Karta</b>\n"
        "Przyciski zmieniają najczęstsze ustawienia; resztę wpisujesz jako <code>klucz=wartość</code> "
        "(np. <code>T=310 ns=2</code>). Zła wartość jest odrzucana od razu, więc nie stracisz nocy na literówkę.\n\n"
        "<b>W trakcie</b>\n"
        "<code>/jobs</code> – kolejka · <code>/job 3</code> – szczegóły · <code>/log 3</code> – co wypisuje "
        "program · <code>/cancel 3</code> – stop · <code>/pauza</code> i <code>/wznow</code> – wstrzymaj wszystko "
        "(np. gdy ktoś chce pograć).\n\n"
        "<b>Gdy coś pójdzie źle</b>\n"
        "Bot pokaże koniec logu i przycisk 🔁 Ponów. <code>/retry 3 T=300</code> uruchamia to samo ze zmienionym "
        "ustawieniem. Więcej: <code>/guide problemy</code>."), None, "md"),
    Topic("md", "🧬 Dynamika molekularna", (
        "<b>Co wysłać</b>\n"
        "Plik <code>.pdb</code> albo kod: <code>/md 1E0L</code>. Bot sam czyści strukturę (jeden model, bez wód "
        "i ligandów), dodaje wodę i jony, minimalizuje, równoważy (NVT, NPT z więzami) i liczy produkcję.\n\n"
        "<b>Ustawienia, które mają znaczenie</b>\n"
        "<code>ns=1</code> – długość produkcji; to ona decyduje o czasie liczenia.\n"
        "<code>T=300</code> – temperatura produkcji [K]; równowagowanie zawsze w 300 K.\n"
        "<code>ff=amber99sb-ildn</code> – pole siłowe; woda dobiera się sama.\n"
        "<code>engine=openmm</code> – drugi silnik, te same etapy (wolniejszy na CPU).\n"
        "<code>ensemble=nvt</code> – <b>użyj przy wysokiej T</b>: w NPT przy 600 K i 1 bar woda odparowuje "
        "(gęstość spada z ~1000 do kilkudziesięciu kg/m³) i wynik nie ma sensu.\n\n"
        "<b>Ile to trwa</b>\n"
        "Zmierzone na tym laptopie: ~50 ns na dobę dla 6 tys. atomów (Trp-cage w wodzie). Czas rośnie mniej "
        "więcej z liczbą atomów, więc ~13 tys. atomów (np. 1E0L) to około 20–25 ns na dobę: 0,5 ns w pół "
        "godziny, 10 ns w pół doby. Dokładną godzinę końca bot pokaże po pierwszych minutach produkcji.\n\n"
        "<b>Jak czytać wynik</b>\n"
        "• <i>Temperatura i gęstość</i>: mają falować wokół zadanej T i ~1000 kg/m³. Dryf = układ nie jest "
        "zrównoważony.\n"
        "• <i>RMSD szkieletu</i>: rośnie i się wypłaszcza (zwykle 0,1–0,3 nm) → struktura stabilna. Ciągły wzrost "
        "→ białko się rozwija albo symulacja jest za krótka, by to ocenić.\n"
        "• <i>Rg</i> (promień żyracji): rośnie, gdy białko się rozluźnia.\n"
        "• <i>RMSF</i>: które reszty są ruchliwe (pętle, końce).\n"
        "• <i>DSSP</i>: ile helis i β zostaje w czasie.\n\n"
        "<b>Dalej</b>\n"
        "<code>/dolicz 4 5</code> – skończone zadanie #4 o 5 ns dłużej, bez liczenia od zera.\n"
        "<code>mdp=nstxout-compressed=500</code> – dowolna opcja GROMACS dla produkcji; inny etap z "
        "przedrostkiem: <code>mdp=nvt:tau_t=1.0 1.0;md:dt=0.001</code>.\n"
        "<code>T_eq=350</code> – równowagowanie w innej temperaturze niż 300 K.\n"
        "<code>/skrypt 4</code> – zadanie jako zip do edycji (pliki .mdp + run.sh); zmieniasz, odsyłasz, "
        "liczy się z Twoimi ustawieniami.\n"
        "Laptop się zrestartuje? Zadanie wznowi się samo z punktu sprzed kilku minut."), "md", "md"),
    Topic("qsar", "📈 QSAR / QSPR", (
        "<b>Co wysłać</b>\n"
        "Tabelę xlsx/csv: wiersz = związek, kolumny = nazwa, odpowiedź (np. LogKd) i deskryptory <i>albo</i> "
        "SMILES. Podpis pliku: <code>qsar</code>. Wiersze bez odpowiedzi dostaną prognozę.\n\n"
        "<b>Ustawienia</b>\n"
        "<code>target=LogKd</code> – która kolumna jest odpowiedzią (zwykle zgadnie sam).\n"
        "<code>descriptors=rdkit</code> – policz deskryptory ze SMILES (<code>mordred</code> – dużo więcej, "
        "<code>mopac</code> – kwantowe PM6: HOMO, LUMO, dipol; trwa najdłużej).\n"
        "<code>max_desc=3</code> – ile deskryptorów w modelu. Zasada: co najmniej 5 związków uczących na "
        "deskryptor.\n"
        "<code>split=sorted</code> – co trzeci związek (po posortowaniu odpowiedzi) do testu.\n\n"
        "<b>Jak czytać wynik</b>\n"
        "• <i>R²</i> – dopasowanie do zbioru uczącego. Samo wysokie R² niczego nie dowodzi.\n"
        "• <i>Q²LOO</i> – walidacja krzyżowa: czy model przewiduje związki, których nie widział. Powinno być "
        "&gt; 0,6 i niewiele niższe od R² (duża różnica = przeuczenie).\n"
        "• <i>Q²F1–F3, CCC, RMSE_P</i> – to samo na zbiorze testowym (walidacja zewnętrzna).\n"
        "• <i>Y-scrambling</i> – po wymieszaniu odpowiedzi model ma być zły; jeśli nadal jest dobry, to "
        "przypadek.\n"
        "• <i>Wykres Williamsa</i> – domena stosowalności: związki z dźwignią powyżej h* są poza nią i ich "
        "prognozom nie należy ufać.\n\n"
        "<b>Który model jest „najlepszy”</b>\n"
        "Bot wybiera po walidacji krzyżowej, nie po R² i nie po teście, a przy remisie bierze prostszy. "
        "W podsumowaniu jest uzasadnienie i równanie MLR do prezentacji.\n\n"
        "<b>Typowe potknięcia</b>\n"
        "Braki w deskryptorach (deskryptor z brakiem jest odrzucany – podsumowanie mówi, przez które związki); "
        "za mało związków na liczbę deskryptorów; SMILES z błędem (lista w wynikach)."), "qsar", "qsar"),
    Topic("omics", "🔬 Proteomika, metabolomika, lipidomika", (
        "<b>Co wysłać</b>\n"
        "Tabelę z programu, który przetworzył surowe dane (DIA-NN, MS-DIAL, MZmine, Profinder), i najlepiej "
        "<b>arkusz próbek</b>: która kolumna to próbka, QC, blank, jaka grupa, kolejność iniekcji. Kilka plików "
        "spakuj w zip. Podpis: <code>omics</code>. Bez arkusza bot zgadnie typy z nazw (QC, Blank) i to "
        "napisze.\n\n"
        "<b>Ustawienia</b>\n"
        "<code>group=group</code> – kolumna z grupą; <code>ref=CONTROL</code> – grupa odniesienia.\n"
        "<code>pair=Donor_ID</code> – ta sama osoba w obu grupach (np. M1 i M12): testy dla par.\n"
        "<code>blank_ratio=10</code> – cecha musi być 10× wyższa niż w blanku (kryterium z zajęć z lipidomiki; "
        "domyślnie 3).\n"
        "<code>qc_cv=0.3</code> – odrzuć cechy z RSD w QC powyżej 30%.\n"
        "<code>covariates=age_years,sex</code> – uwzględnij w modelu liniowym.\n\n"
        "<b>Co robi po kolei</b>\n"
        "Audyt → złe iniekcje (wzorce wewnętrzne) → filtr blanka i braków → uzupełnienie braków → normalizacja → "
        "korekta dryfu i partii (po QC) → RSD w QC → PCA → test na każdą cechę z poprawką FDR → PLS-DA z VIP → "
        "adnotacja. W <code>wyniki/kroki.txt</code> jest, ile cech zostało po każdym kroku i dlaczego.\n\n"
        "<b>Jak czytać wynik</b>\n"
        "• <i>PCA</i>: próbki QC mają leżeć w ciasnej kupce. Rozrzucone QC = problem z pomiarem, nie z "
        "biologią.\n"
        "• <i>Mediana CV w QC</i> przed i po korekcie: po ma być niższa (dobrze poniżej 20–30%).\n"
        "• <i>Volcano</i>: w górze i na bokach są cechy istotne (FDR) i o dużej zmianie.\n"
        "• <i>FDR, nie p</i>: przy tysiącach cech samo p &lt; 0,05 daje setki fałszywych trafień.\n"
        "• <i>PLS-DA</i>: patrz na dokładność z walidacji krzyżowej i test permutacyjny, nie na ładny wykres.\n\n"
        "<b>Surowe pliki</b> (.wiff, .raw, .mzML) przetwórz najpierw u siebie w DIA-NN / MS-DIAL / MZmine; "
        "bot bierze tabele. Duże tabele: <code>/url LINK omics</code>."), "omics", "omics"),
    Topic("dock", "💊 Dokowanie", (
        "<b>Co wysłać</b>\n"
        "Ligandy jako <code>.smi</code>, <code>.csv</code> (kolumny smiles, name) albo <code>.sdf</code>, z "
        "podpisem <code>dock receptor=1STP</code> (kod PDB), albo zip: receptor .pdb + ligandy.\n\n"
        "<b>Gdzie dokować (pudełko)</b>\n"
        "<code>ligand_ref=BTN</code> – wokół ligandu, który już siedzi w strukturze (najpewniejsze).\n"
        "<code>residues=A:45,A:88</code> – wokół wskazanych reszt.\n"
        "<code>center=10,5,-3</code> – ręcznie [Å].\n"
        "Bez niczego: jeśli w strukturze jest jeden ligand, bot weźmie jego miejsce.\n\n"
        "<b>Inne ustawienia</b>\n"
        "<code>exhaustiveness=8</code> – dokładność szukania; 16–32 dla ważnych wyników, czas rośnie liniowo.\n"
        "<code>ph=7.4</code> – protonacja białka i ligandów.\n\n"
        "<b>Jak czytać wynik</b>\n"
        "• <i>Redocking</i>: bot najpierw dokuje ligand z kryształu. RMSD &lt; 2 Å = protokół odtwarza znaną pozę "
        "i reszcie wyników można ufać bardziej. Powyżej – zmień pudełko albo dokładność.\n"
        "• <i>Powinowactwo [kcal/mol]</i>: bardziej ujemne = lepiej, ale to <b>ranking, nie pomiar</b>. Różnice "
        "poniżej ~1 kcal/mol nic nie znaczą.\n"
        "• <i>LE</i> (efektywność liganda): powinowactwo na ciężki atom – nie faworyzuje dużych cząsteczek.\n"
        "• <i>kontakty.csv</i>: z którymi resztami ligand się styka i gdzie mogą być wiązania wodorowe.\n\n"
        "<b>Ile to trwa</b>\n"
        "Rząd minuty na ligand; setki ligandów to godziny. Po restarcie laptopa gotowe ligandy są pomijane.\n\n"
        "<b>Ograniczenia</b>\n"
        "Białko jest sztywne, wody i jony metali są usuwane, tautomery nie są wyliczane."), "dock", "dock"),
    Topic("charmm", "📦 Paczki CHARMM-GUI i NAMD", (
        "<b>CHARMM-GUI</b>\n"
        "Na charmm-gui.org przygotuj układ (Solution Builder), zaznacz silnik (GROMACS / OpenMM / NAMD) i "
        "pobierz <code>charmm-gui.tgz</code>. Wyślij ten plik botowi – rozpozna paczkę sam.\n\n"
        "<code>engine=gromacs</code> – który folder paczki uruchomić (auto: GROMACS, potem OpenMM, potem NAMD).\n"
        "<code>ns=5</code> – długość produkcji, <code>T=310</code> – temperatura; 0 = jak w paczce.\n\n"
        "<b>Do sprawozdania</b>\n"
        "W <code>wyniki/protokol_charmm.csv</code> i w podsumowaniu jest tabela: pole siłowe, woda, jony, "
        "termostat, barostat, więzy, kroki – gotowe do porównania z ręcznym przygotowaniem z "
        "<code>/guide md</code>.\n\n"
        "<b>NAMD</b>\n"
        "Wymaga rejestracji, więc nie instaluje się sam. Pobierz z ks.uiuc.edu wersję "
        "<i>Linux-x86_64-multicore</i> i wyślij plik <code>NAMD_…tar.gz</code> botowi (jeśli ma ponad 20 MB: "
        "<code>/url LINK</code>). Od tej chwili <code>namd3</code> jest dostępny w każdym zadaniu.\n\n"
        "<b>Na co uważać</b>\n"
        "Starsze paczki zakładają kartę graficzną – bot przestawia je na procesor, ale OpenMM i NAMD są tu "
        "wyraźnie wolniejsze niż GROMACS. Produkcja z paczki bywa długa: ustaw <code>ns</code> świadomie."),
          "charmm"),
    Topic("skrypty", "🛠 Własne skrypty i notebooki", (
        "<b>Kiedy</b>\n"
        "Gdy gotowy przepis nie pasuje: własny protokół GROMACS, skrypt Pythona, notebook z zajęć.\n\n"
        "<b>Co wysłać</b>\n"
        "• <b>zip z <code>run.sh</code></b> – folder z plikami i skryptem; wszystko, co skrypt zapisze, wraca.\n"
        "• <b>.py</b> – uruchomi się w środowisku z GROMACS, OpenMM, RDKit, Vina, MOPAC, scikit-learn, pandas.\n"
        "• <b>.ipynb</b> – wykona się od góry do dołu, wraca notebook z wynikami (kod z Colaba z GPU nie "
        "zadziała).\n"
        "• <b>.tpr</b> – od razu mdrun.\n\n"
        "<b>GROMACS, który przeżyje restart</b>\n"
        "W <code>run.sh</code> zamiast grompp + mdrun pisz:\n"
        "<pre>md_step nvt -f nvt.mdp -c em.gro -r em.gro -p topol.top\n"
        "md_step npt -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top\n"
        "md_step md  -f md.mdp  -c npt.gro -t npt.cpt -p topol.top</pre>\n"
        "Skończony etap jest pomijany, przerwany wznawia się z checkpointu.\n\n"
        "<b>Zasady</b>\n"
        "Skrypt może zostać uruchomiony drugi raz (po restarcie laptopa), więc niech nie kasuje wyników. "
        "Liczba wątków jest w <code>$COMPUTE_THREADS</code>. Plik wejściowy ponad 20 MB: "
        "<code>/url LINK skrypt</code> albo <code>curl</code> w skrypcie.\n\n"
        "<b>Czego laptop nie zrobi</b>\n"
        "Brak karty graficznej: AlphaFold, ESMFold i trenowanie dużych sieci zostają na Colabie.")),
    Topic("wyniki", "📥 Wyniki i duże pliki", (
        "<b>Co przychodzi po zakończeniu</b>\n"
        "1. Podsumowanie (<code>summary.txt</code>) – najważniejsze liczby słowami.\n"
        "2. Do czterech najważniejszych wykresów.\n"
        "3. <code>wyniki_NNNN.zip</code> – tabele CSV, wszystkie wykresy, log.\n\n"
        "<b>Gdy plik jest za duży</b> (Telegram: 50 MB na plik)\n"
        "• Do 50 MB, ale poza zipem: <code>/get 3 gromacs/md.xtc</code>.\n"
        "• Większe, z włączonym panelem: bot daje przycisk <b>⬇️</b> z linkiem ważnym 24 h (urządzenie z "
        "Tailscale albo po <code>/funnel on</code>); przerwane pobieranie da się wznowić.\n"
        "• Z SSH: bot podaje gotowe polecenie <code>scp</code> dla pliku i całego katalogu.\n"
        "• Bez panelu i SSH: <code>/get 3 plik czesci</code> – kawałki po 45 MB (do 1 GB) i polecenie do "
        "sklejenia.\n\n"
        "<b>Trajektorie MD</b>\n"
        "Pełna trajektoria bywa duża, dlatego w <code>wyniki/</code> jest też lekka wersja samego białka "
        "(<code>bialko.xtc</code> + <code>bialko_em.pdb</code>) do obejrzenia w VMD/PyMOL/ChimeraX.\n\n"
        "<b>Wrócić do starego zadania</b>\n"
        "<code>/jobs</code> → <code>/job 3</code> → 📦 Wyniki. Pliki leżą na laptopie, dopóki ich nie usuniesz.\n\n"
        "<b>Miejsce na dysku</b>\n"
        "<code>/system</code> pokazuje wolne miejsce. Poniżej 2 GB nowe zadania czekają, a bot o tym pisze. "
        "<code>/usun 3</code> usuwa jedno zadanie z wynikami, <code>/usun stare</code> – skończone ponad 7 dni "
        "temu (zawsze z potwierdzeniem). Zanim usuniesz, pobierz to, co potrzebne.")),
    Topic("zdalnie", "🖥 Panel, Tailscale i SSH", (
        "<b>Panel</b>\n"
        "Przycisk „Panel” obok pola wiadomości (albo <code>/panel</code>): kolejka na żywo z godziną końca, "
        "temperatura i dysk, wykresy i logi, formularze wszystkich ustawień, ten przewodnik.\n"
        "Działa, gdy telefon ma włączony Tailscale. Bez niego: <code>/funnel on</code> udostępnia panel "
        "publicznie (wejdzie tylko Twój Telegram), <code>/funnel off</code> – z powrotem prywatnie.\n\n"
        "<b>Terminal</b> (z Maca: <code>ssh vostro</code>)\n"
        "<pre>compute jobs                 # kolejka\n"
        "compute log 3 -f             # log na żywo\n"
        "compute submit folder/       # folder z run.sh\n"
        "compute submit dane.xlsx qsar max_desc=3\n"
        "compute md 1E0L T=310 ns=2\n"
        "cd \"$(compute where 3)\"      # katalog wyników\n"
        "compute-env                  # gmx, python, vina w powłoce</pre>\n"
        "To ta sama kolejka co w bocie: bot napisze o starcie i końcu także takich zadań.\n\n"
        "<b>VS Code i Jupyter</b>\n"
        "VS Code: Remote-SSH → <code>vostro</code>, interpreter "
        "<code>/var/lib/compute-bot/env/bin/python</code>.\n"
        "Jupyter: <code>ssh vostro-jupyter</code>, potem <code>compute-env; jupyter lab --no-browser</code> i "
        "link z tokenem otwierasz na Macu.\n\n"
        "<b>Uwaga na rdzenie</b>\n"
        "Ręczne liczenie w terminalu zabiera te same 2 rdzenie co kolejka: <code>compute pause</code> przed, "
        "<code>compute resume</code> po. Długie rzeczy w terminalu uruchamiaj w <code>tmux</code> albo wyślij "
        "do kolejki.")),
    Topic("problemy", "🧯 Gdy coś nie działa", (
        "<b>❌ Zadanie skończyło się błędem</b>\n"
        "Bot pokazuje koniec logu; ostatnia linia <code>ERROR:</code> mówi po polsku, co poprawić (np. brak "
        "kolumny, zły kod PDB). Popraw i <code>/retry 3</code> albo <code>/retry 3 klucz=wartość</code>. Pełny "
        "log: <code>/log 3</code> i <code>log.txt</code> w zipie.\n\n"
        "<b>🌡️ „Za gorąco, wstrzymuję”</b>\n"
        "Normalne zabezpieczenie: powyżej 92 °C zadanie czeka, aż procesor ostygnie do 75 °C, i rusza samo. "
        "Jeśli dzieje się ciągle – laptop stoi na czymś miękkim albo ma zakurzony wentylator.\n\n"
        "<b>🔁 „Wznowione po przerwie”</b>\n"
        "Laptop się zrestartował albo zaktualizował. Dynamika traci najwyżej kilka minut. Po trzech takich "
        "przerwach zadanie kończy się błędem, żeby nie kręcić się w kółko.\n\n"
        "<b>⚠️ „Runner nie odpowiada”</b>\n"
        "Usługa licząca stoi albo laptop śpi. Przez SSH: "
        "<code>sudo systemctl restart compute-runner</code>. Gdy nie ma SSH – trzeba poprosić gospodarzy o "
        "restart laptopa; zadania wznowią się same.\n\n"
        "<b>💾 „Za mało miejsca”</b>\n"
        "<code>/usun stare</code> albo <code>/usun 3</code> – najpierw pobierz, co potrzebne "
        "(<code>/guide wyniki</code>).\n\n"
        "<b>⏳ Nic się nie dzieje</b>\n"
        "<code>/system</code>: czy runner działa i czy nie ma pauzy (<code>/wznow</code>). <code>/jobs</code>: "
        "może liczy się wcześniejsze zadanie – <code>/top 5</code> przesuwa #5 na początek.\n\n"
        "<b>Bot milczy</b>\n"
        "Telegram bywa niedostępny z laptopa przez chwilę; obliczenia idą dalej, a wiadomości dojdą później.\n\n"
        "<b>Nowa wersja</b>\n"
        "Bot sam napisze, gdy wyjdzie; <code>/update</code> instaluje, <code>/rollback</code> cofa. Jeśli po "
        "aktualizacji poprosi o <code>/env</code> – to doinstalowanie pakietów jako zwykłe zadanie.")),
)
BY_ID = {topic.id: topic for topic in TOPICS}

TAG = re.compile(r"</?(?:b|i|code|pre)>")


def strip_tags(text):
    return TAG.sub("", text)


def commands_in(text):
    """Bot commands the text tells the owner to type."""
    return sorted(set(re.findall(r"<code>/([a-z]+)(?![a-z/])", text)))  # not paths like /var/lib


def params_in(topic):
    """key= names shown as recipe parameters in a topic about one recipe."""
    if topic.recipe is None:
        return []
    keys = set()
    for code in re.findall(r"<code>([^<]*)</code>", topic.text):
        keys.update(re.findall(r"(?:^|\s)([A-Za-z_]+)=", code))
    return sorted(keys)


def markdown():
    """The same guide as a Markdown file."""
    out = ["# Przewodnik", "",
           "Ten sam tekst jest w bocie (`/guide`) i w panelu. Plik generuje `python3 -m compute_bot.guide`.", ""]
    for topic in TOPICS:
        text = re.sub(r"<pre>(.*?)</pre>", lambda m: "\n```\n" + m[1] + "\n```\n", topic.text, flags=re.S)
        text = re.sub(r"<b>(.*?)</b>", r"**\1**", text)
        text = re.sub(r"<i>(.*?)</i>", r"*\1*", text)
        text = re.sub(r"<code>(.*?)</code>", r"`\1`", text)
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        lines = [line + ("  " if line and not line.startswith("```") else "") for line in text.split("\n")]
        out += [f"## {topic.title}", "", *lines, ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.write(markdown())
