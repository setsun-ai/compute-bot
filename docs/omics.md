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
  - Dowolna tabela: pierwsza kolumna = ID cechy, dalej kolumny próbek (np. `proteomics_intensities.csv`, braki jako
    `NA` albo puste). Albo odwrotnie: próbki w wierszach z kolumną `Label`/`Group` (format MetaboAnalyst).
- **Arkusz próbek** (csv/xlsx, opcjonalnie): jedna kolumna musi zawierać nazwy kolumn macierzy (albo nazwy plików,
  np. `A_1.raw`). Rozpoznaję kolumny: grupa (`group=` albo group/class/condition/…), typ próbki
  (sample_type/type: study/sample, QC/pooled_qc, blank/process_blank/solvent_blank, technical_replicate,
  conditioning_qc), kolejność (run_order/injection order), partia (batch), osoba (subject_id). Druga tabela z osobami
  (np. `subjects.csv` z grupą i kowariatami) dołącza się sama po wspólnej kolumnie (subject_id).
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
```

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
| `test` | auto | auto: 2 grupy → Welch, > 2 → ANOVA, z kowariatami → lm; także ttest, mwu, kruskal |
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
3. **Filtry cech**: blanki (`blank_ratio`), braki (`na_max`).
4. **Imputacja** (`impute`) na skali liniowej, potem **log2**. Pozycje uzupełnione są zapamiętane: CV, D-ratio i
   dopasowanie dryfu liczą się tylko z wartości zmierzonych.
5. **Normalizacja** → **dryf** (lokalna regresja liniowa z wagami tricube, span 0.75, po kolejności iniekcji w
   partii, dopasowana do QC; poza zakresem QC – stała) → **partie**. Potem CV w QC i D-ratio
   (MAD QC / MAD próbek); cecha z CV > `qc_cv` albo D-ratio > 0.5 odpada. Uwaga: CV QC po korekcie liczonej na tych
   samych QC zawsze spada – uczciwiej ocenia ją powtarzalność replikatów technicznych (podaję, gdy są).
6. **Statystyka** na jednej iniekcji na osobę: test, log2 FC vs `ref`, q (BH). Gdy cechy mają `analytical_cluster_id`
   (addukty/izotopy jednego związku), testuję jednego reprezentanta na klaster (najwyższa mediana). PCA
   (autoskalowanie), PLS-DA (autoskalowanie w każdej fałdzie, liczba składowych wg Q², dokładność w CV, do 200
   permutacji), VIP.
7. **Adnotacja**: UniProt (nazwa, funkcja, GO, lokalizacja), g:Profiler (wszystkie / w górę / w dół). Bez internetu
   krok jest pomijany z uwagą w podsumowaniu; odpowiedzi są w pamięci podręcznej `wyniki/cache/`. Dla metabolitów:
   nazwy i wzory z opisu cech.

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
  - wykresy: `qc_iniekcje.png`, `qc_dryf.png`, `pca_przed_po.png`, `volcano.png`, `heatmapa.png`, `pca_grupy.png`,
    `plsda.png`, `boxploty.png`.

## Na co uważać

- Mediana/PQN zakładają, że większość cech się nie zmienia między grupami.
- Gdy audyt mówi, że braki są losowe (MCAR), `impute=halfmin` zaniża wartości i rozdmuchuje wariancję – użyj `knn`.
- Korekta partii medianą próbek (gdy brak QC) przy grupach nierówno rozłożonych w partiach usuwa część biologii.
- 3 replikaty techniczne jednej próbki to nie 3 osoby: przy nazwach `P1_1…P1_3` grupa = próbka, a test porównuje
  próbki, nie warunki – do porównania warunków wyślij arkusz z kolumną grupy.
- Czas na laptopie: ok. 20–30 s na 8 500 białek × 168 iniekcji.
