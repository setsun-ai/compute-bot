# Proteomika / metabolomika – przepis `omics`

Bierze macierz intensywności (białka albo cechy LC-MS × próbki) i robi to, co na zajęciach robi się w MS-DIAL /
DIA-NN → MetaboAnalyst: audyt danych, odrzucenie złych iniekcji, odsiew tła z blanków i cech z brakami, imputację,
log2, normalizację, korektę dryfu po QC (QC-LOESS) i efektu partii, ocenę jakości cech (CV w QC, D-ratio), PCA
przed/po korekcie, test na każdą cechę z FDR (Benjamini–Hochberg), volcano, heatmapę, PCA i PLS-DA z VIP, walidacją
krzyżową i testem permutacyjnym, a dla białek adnotację z UniProt i wzbogacenie GO/KEGG/Reactome w g:Profiler.

## Co wysłać

Jeden plik albo zip z kilkoma plikami. Przepis sam rozpoznaje, co jest czym – po zawartości, nie po nazwie:

- **Macierz intensywności** (jedna na warstwę), jedno z:
  - DIA-NN `report.pg_matrix.tsv` (Protein.Group, Protein.Ids, Protein.Names, Genes, …, kolumny = pliki raw) albo
    `report.gg_matrix.tsv` (Genes, N.Sequences, …). Ścieżki typu `D:\SCIEX OS Data\…\P1_3.wiff` skracam do `P1_3`.
  - DIA-NN raport długi `report.tsv` (Run, Protein.Group, PG.MaxLFQ, Q.Value) – biorę PG.MaxLFQ, wiersze z
    Q.Value i PG.Q.Value ≤ 0.01. (`report.parquet` wymaga pakietu pyarrow – jeśli go nie ma, wyślij `report.tsv`.)
  - MS-DIAL eksport alignment (Height/Area/Normalized) z wierszami nad tabelą: Class, File type, Injection order,
    Batch ID – z nich biorę grupy, typy próbek, kolejność i partie. Kolumny średnich/odchyleń klas pomijam.
    Zero w MS-DIAL = niewykryte (brak).
  - Agilent Profinder / MassHunter MFE (csv albo xlsx, celowana i niecelowana): kolumny podsumowania
    (`Name`, `Formula`, `Mass (avg)`, `RT (avg)`, `Found`, …), potem bloki na próbkę `[Mass] <próbka>`, `[RT] …`,
    `[Area] …` / `[Height] …` / `[Vol] …`, `[Score] …`. Macierz = blok intensywności (Area, potem Height, potem
    Vol); masa i RT zostają jako opis cech. Pusty drugi wiersz nagłówka jest pomijany. Bez `Name` cecha ma
    identyfikator `masa@RT`. Profinder podaje masę neutralną, więc wykresy mają oś „masa”, nie m/z.
  - MZmine 3, eksport listy cech do CSV: stary układ (`row ID`, `row m/z`, `row retention time`,
    `<plik>.mzML Peak area` / `Peak height`) i nowy (`id`, `mz`, `rt`, `datafile:<plik>:area`). Biorę area
    (albo height), 0 = niewykryte; nazwa z `row identity (main ID)` / `compound_name`, jeśli jest.
  - Dowolna tabela: pierwsza kolumna = ID cechy, dalej kolumny próbek (np. `proteomics_intensities.csv`, braki jako
    `NA` albo puste). Albo odwrotnie: próbki w wierszach z kolumną `Label`/`Group` (format MetaboAnalyst).
- **Arkusz próbek** (csv/xlsx, opcjonalnie): jedna kolumna musi zawierać nazwy kolumn macierzy (albo nazwy plików,
  np. `A_1.raw`). Rozpoznaję kolumny: grupa (`group=` albo group/class/condition/…), typ próbki
  (sample_type/type: study/sample, QC/pooled_qc, blank/process_blank/solvent_blank, technical_replicate,
  conditioning_qc), kolejność (run_order/injection order), partia (batch), osoba (subject_id). Druga tabela z osobami
  (np. `subjects.csv` z grupą i kowariatami) dołącza się sama po wspólnej kolumnie (subject_id).
- **Skoroszyt metadanych z kilkoma arkuszami** (np. README, Donor_metadata, Timepoint_metadata,
  Sample_metadata, Sequence): czytam każdy arkusz, wiersz tytułu nad nagłówkiem jest pomijany. Arkusz próbek to ten,
  którego kolumna pasuje do nazw kolumn macierzy (`AB 1_2B 101` pasuje do `AB_1_2B_101.mzML`). Pozostałe arkusze
  dołączam po wspólnych nazwach kolumn: `Donor_ID` (cechy dawczyni), `Donor_ID + Timepoint` (cechy zmienne w czasie),
  `Sample_ID`/`File_name` (Sequence z `Injection_order`). Gdy próbka nie ma wiersza w arkuszu (albo pasuje tylko po
  numerze na końcu nazwy), grupę, osobę i partię odczytuję z nazwy pliku regułą nauczoną na opisanych próbkach –
  podsumowanie mówi, dla ilu plików, a rozbieżności arkusz ↔ nazwa trafiają do uwag.
- **Wzorce wewnętrzne w tej samej tabeli**: cechy z nazwą znakowaną izotopowo (`15:0-18:1(d7) PC`,
  `18:1 SM (d18:1/18:1)-d9`, `SM d9`, `Ceramide-d7`), `ISTD`/`IS` w nazwie albo deuter we wzorze (`C41 H73 D7 N O8 P`)
  są wyjmowane z macierzy przed filtrami (chronione, jak na ćw. 6) i służą do kontroli iniekcji, `normalize=is` i
  wykresu dryfu. `d18:1` w nazwie sfingolipidu to baza sfingoidowa, nie deuter.
- **Opis cech** (opcjonalnie): tabela, której kolumna pasuje do ID cech (uniprot_accession, gene_symbol,
  protein_name / metabolite_name, mz, rt_min, adduct, analytical_cluster_id…).
- **Wzorce wewnętrzne** (opcjonalnie): mała tabela z tymi samymi kolumnami próbek co macierz
  (`*_internal_standards.csv`) – do wykrywania złych iniekcji i do `normalize=is`.

Układ z zajęć (STUDENCI: `{proteomics,metabolomics}_intensities/_features/_injections/_internal_standards.csv` +
`subjects.csv`) działa wprost – obie warstwy liczą się w jednym zadaniu, wyniki w `wyniki/proteomics/` i
`wyniki/metabolomics/`.

**Bez arkusza próbek** typy i grupy zgaduję z nazw kolumn: `QC…`/`pool…` = QC, `Blank…` = blank, reszta = próbki;
grupa = nazwa bez końcowego numeru (`P1_3` → `P1`, `CTRL3` → `CTRL`). Kolejność = kolejność kolumn – wtedy korekty
dryfu nie robię (kolejność kolumn w DIA-NN to nie kolejność pomiaru). Podsumowanie mówi, co zostało zgadnięte;
arkusz próbek to nadpisuje.

Przykładowe podpisy pliku:

```
omics
omics ref=CONTROL_RISK covariates=age_years,sex,bmi_kg_m2
omics layer=metabolomics normalize=pqn impute=knn
omics group=condition ref=ctrl test=mwu fdr=0.1
omics organism=mmusculus annotate=nie
omics group=Timepoint pair=Donor_ID blank_ratio=10
omics group=Timepoint pair=Donor_ID test=wilcoxon lipids=tak normalize=is
```

Ostatnie dwa to układ z zajęć z lipidomiki (mleko M1 vs M12): ta sama dawczyni w obu punktach czasowych, replikaty
1A/1B i 2A/2B, QC i blank w kilku dniach. Kryterium blanka z ćw. 6 (sygnał w blanku ≥ 10 % sygnału w próbkach →
usuń) to `blank_ratio=10` (domyślnie jest łagodniejsze 3).

## Parametry

| parametr | domyślnie | znaczenie |
|---|---|---|
| `layer` | auto | proteomics / metabolomics – tylko ta warstwa (auto = wszystkie znalezione) |
| `group` | group | kolumna z grupą w arkuszu próbek |
| `ref` | – | grupa odniesienia; puste = nazwa w stylu control/ctrl/kontrola, inaczej pierwsza alfabetycznie |
| `covariates` | – | kowariaty, np. `age_years,sex` (tekstowe → zmienne 0/1); z nimi `test=auto` = model liniowy |
| `na_max` | 0.5 | cecha zostaje, jeśli przynajmniej w jednej grupie braków jest ≤ tyle |
| `blank_ratio` | 3 | mediana w próbkach musi być ≥ tyle × mediana w blankach (process blanki, gdy są; 0 = wyłącz) |
| `impute` | halfmin | halfmin (½ minimum cechy), min, knn (5 najbliższych próbek, na log2), none |
| `normalize` | median | median, pqn (iloraz do mediany QC), sum, is (wzorce wewnętrzne), none |
| `drift` | auto | QC-LOESS w każdej partii; auto = tylko gdy w partii ≥ 5 QC i znana kolejność; loess = od 4 QC |
| `batch` | auto | mediana QC w partii → mediana QC ogółem (gdy w partii < 2 QC: mediana próbek); auto = gdy > 1 partia |
| `qc_cv` | 0.3 | odrzuć cechę z CV w QC (po korekcie) powyżej |
| `test` | auto | auto: z `pair` → t dla par, 2 grupy → Welch, > 2 → ANOVA, z kowariatami → lm; także ttest, mwu, kruskal, `paired_t`, `wilcoxon` (rangowany test znaków Wilcoxona dla par) |
| `pair` | – | kolumna z osobą/parą (np. `Donor_ID`): ta sama wartość w obu grupach → test dla par; pary bez drugiej próbki odpadają (podsumowanie podaje ile i które) |
| `replicates` | mean | `mean`: iniekcje tej samej osoby w tej samej grupie (albo z oznaczeniem replikatu w nazwie: 1A/1B, `_rep2`) uśredniam na skali log2 przed testem – n = próbki biologiczne; `keep`: każda iniekcja osobno, z ostrzeżeniem o pseudoreplikacji (przy testach dla par zawsze uśredniam) |
| `lipids` | auto | analiza klas lipidów z nazw; auto = gdy ≥ 30 % nazwanych cech to lipidy; `tak` / `nie` |
| `fdr` | 0.05 | próg q (Benjamini–Hochberg) |
| `fc` | 1 | próg \|log2 FC\| na volcano i w liczbie „dużych” zmian |
| `annotate` | tak | UniProt dla istotnych białek (do 200) |
| `enrich` | tak | g:Profiler dla istotnych genów względem zmierzonego tła |
| `organism` | hsapiens | organizm g:Profiler (mmusculus, rnorvegicus, scerevisiae, …); też filtr UniProt |

## Kolejność kroków

1. **Audyt**: duplikaty kolumn/ID, zgodność ID między tabelami, próbki bez metadanych, braki wg typu próbki, braki
   wg intensywności (czy MNAR czy losowe), kolejność i liczba QC w partiach, rozkład grup w partiach.
2. **Złe iniekcje**: odporny z-score (mediana, MAD) sygnału wzorców wewnętrznych, sumy intensywności i odsetka
   braków; |z| > 3 → odrzucona. Odrzucona próbka z replikatem technicznym → w statystyce zastępuje ją replikat.
3. **Filtry cech**: blanki (`blank_ratio`), braki w grupach próbek (`na_max`), obecność w QC (cecha wykryta w
   ≥ 1 − `na_max` próbek QC; przy < 3 QC pominięte). Gdy wszystkie replikaty jednej próbki biologicznej odstają tak
   samo (braki, suma sygnału), to cecha próbki, a nie błąd iniekcji – zostają (chyba że odstają też wzorce IS).
4. **Imputacja** (`impute`) na skali liniowej, potem **log2**. Pozycje uzupełnione są zapamiętane: CV, D-ratio i
   dopasowanie dryfu liczą się tylko z wartości zmierzonych.
5. **Normalizacja** → **dryf** (lokalna regresja liniowa z wagami tricube, span 0.75, po kolejności iniekcji w
   partii, dopasowana do QC; poza zakresem QC – stała) → **partie**. Potem CV w QC i D-ratio
   (MAD QC / MAD próbek); cecha z CV > `qc_cv` albo D-ratio > 0.5 odpada. Uwaga: CV QC po korekcie liczonej na tych
   samych QC zawsze spada – uczciwiej ocenia ją powtarzalność replikatów technicznych (podaję, gdy są).
6. **Statystyka** na próbkach biologicznych (replikaty uśrednione, `replicates`), dla par na różnicach w parze: test, log2 FC vs `ref`, q (BH). Gdy cechy mają `analytical_cluster_id`
   (addukty/izotopy jednego związku), testuję jednego reprezentanta na klaster (najwyższa mediana). PCA
   (autoskalowanie), PLS-DA (autoskalowanie w każdej fałdzie, liczba składowych wg Q², dokładność w CV, do 200
   permutacji), VIP.
7. **Adnotacja**: UniProt (nazwa, funkcja, GO, lokalizacja), g:Profiler (wszystkie / w górę / w dół). Bez internetu
   krok jest pomijany z uwagą w podsumowaniu; odpowiedzi są w pamięci podręcznej `wyniki/cache/`. Dla metabolitów:
   nazwy i wzory z opisu cech.

8. **Wykresy LC-MS** (metabolomika): RT vs m/z (kolor = klasa lipidu albo istotność), wykres Kendricka (baza CH₂:
   KMD vs nominalna masa Kendricka – szeregi homologiczne leżą na poziomej linii), diagram Van Krevelena (O/C vs H/C
   ze wzorów, gdy są), Venn cech obecnych w grupach (2–3 grupy, wykryte w ≥ 50 % iniekcji grupy), dryf wzorców IS
   w kolejności iniekcji z RSD.
9. **Lipidy** (`lipids`): z nazwy klasa, suma C:DB, łańcuchy, baza d/t, eter O-/P-, lyso (`PC 34:1`,
   `PC(16:0/18:1)`, `15:0-18:1 PC`, `SMd34:1`, `Cer d18:1/24:0`, `Lyso-PC18:1`, `PEO-38:5`, `TG48:4 ISO1`). Sumy klas
   i udziały w sygnale lipidów w grupach z tym samym testem co cechy, profil nasycenia (SFA = 0, MUFA = 1,
   PUFA ≥ 2 wiązania podwójne w całej cząsteczce), rozkład długości łańcuchów i wiązań podwójnych istotnych lipidów,
   wzbogacenie klas wśród istotnych (test dokładny Fishera, jednostronny, tło = testowane lipidy, BH) – offline.
   Udziały to części sumy sygnału, nie stężenia (klasy jonizują różnie).

## Co wraca

- `summary.txt`: rozmiar danych, co zgadnięto, odrzucone iniekcje, ile cech odpadło na którym filtrze, jakość (CV w
  QC przed/po, D-ratio, replikaty), liczba istotnych przy FDR, najsilniejsze cechy z nazwami, PLS-DA, wzbogacone
  terminy, uwagi.
- Zdjęcia: PCA przed/po korekcie (partie, kolejność, QC), volcano, heatmapa top 50, dryf + rozkład CV w QC.
- `wyniki/` (dla dwóch warstw w podkatalogach):
  - `kroki.txt` – każdy krok: ile cech × próbek zostało i dlaczego;
  - `statystyka.csv` – na cechę: nazwa, średnie log2 w grupach, log2FC, p, q (> 2 grupy: p/q dla każdej grupy vs ref);
  - `macierz_po_obrobce.csv` – log2 po filtrach, normalizacji i korektach (cechy × iniekcje);
  - `qc_cechy.csv` – stosunek do blanków, braki w grupach, CV QC przed/po, D-ratio, decyzja; `qc_iniekcje.csv` –
    typ, grupa, partia, kolejność, z-score, czy odrzucona, czynnik normalizacji, czy w statystyce;
  - `plsda_vip.csv`, `pca_wyniki.csv`, `adnotacja_uniprot.csv`, `wzbogacenie.csv`, `metabolity_istotne.csv`;
  - LC-MS: `kmd_van_krevelen.csv` (m/z, RT, masa Kendricka, KMD, H/C, O/C), `venn.csv`, `wzorce_is.csv` (RSD w QC i
    próbkach, trend z kolejnością), wykresy `rt_mz.png`, `kmd.png`, `van_krevelen.png`, `venn.png`, `is_dryf.png`;
  - lipidy: `lipidy_adnotacja.csv`, `lipidy_klasy.csv` (udziały w grupach, log2FC i q sumy klasy i jej udziału),
    `lipidy_nasycenie.csv`, `lipidy_wzbogacenie_klas.csv`, `lipidy_istotne_rozklad.csv`, wykresy
    `lipidy_klasy.png`, `lipidy_nasycenie.png`, `lipidy_istotne_rozklad.png`. W trybie lipidowym zdjęcia to PCA
    przed/po, volcano, skład klas i RT vs masa;
  - wykresy: `qc_iniekcje.png`, `qc_dryf.png`, `pca_przed_po.png`, `volcano.png`, `heatmapa.png`, `pca_grupy.png`,
    `plsda.png`, `boxploty.png`.

## Na co uważać

- Mediana/PQN zakładają, że większość cech się nie zmienia między grupami.
- Gdy audyt mówi, że braki są losowe (MCAR), `impute=halfmin` zaniża wartości i rozdmuchuje wariancję – użyj `knn`.
- Korekta partii medianą próbek (gdy brak QC) przy grupach nierówno rozłożonych w partiach usuwa część biologii.
- 3 replikaty techniczne jednej próbki to nie 3 osoby: przy nazwach `P1_1…P1_3` grupa = próbka, a test porównuje
  próbki, nie warunki – do porównania warunków wyślij arkusz z kolumną grupy.
- Replikaty ekstrakcji (A/B) i biologiczne (1/2) tej samej dawczyni w tym samym punkcie czasowym to jedna jednostka
  w teście (`replicates=mean`): n = liczba dawczyń, nie plików. Wilcoxon dla par przy 5 parach nie da p < 0,0625 –
  nic nie przejdzie FDR; przy tak małym n użyj `paired_t`.
- Gdy dużo lipidów jednej klasy zmienia się w jedną stronę, mediana przesuwa resztę – rozważ `normalize=is`.
- Wzorce IS w tabeli niecelowanej (bez nazw) nie są rozpoznawane – szukanie po m/z trzeba zrobić samemu.
- Czas na laptopie: ok. 20–30 s na 8 500 białek × 168 iniekcji.
