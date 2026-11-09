# QSAR / QSPR (`qsar`)

Model zależności struktura–właściwość z tabeli: deskryptory → odsiew → podział → wybór zmiennych → modele →
walidacja wg OECD → domena stosowalności → prognozy dla związków bez odpowiedzi.

## Co wysłać

Jeden plik: `.xlsx` (`.xls` zapisz jako `.xlsx`), `.csv`, `.tsv`, `.txt` albo `.smi`.

- **Wiersz = związek.** Kolumny: nazwa, odpowiedź (y) i deskryptory liczbowe albo SMILES (albo jedno i drugie).
- **Dwa wiersze nagłówka** jak w arkuszu z zajęć (w 1. wierszu „Organic compounds”, „EXP LogKda”, „Zbiór”, w 2.
  nazwy deskryptorów `log D`, `Mw'`, `εα`…) są scalane automatycznie. Twarde spacje w nazwach, komórki `#VALUE!`,
  `#DIV/0!`, `NA`, `-` są traktowane jak puste. CSV może mieć `;` i przecinki dziesiętne.
- **Odpowiedź** jest zgadywana po nazwie (logK…, activity, aktywność, pIC50, target, odpowiedź, klasa, y); gdy nie
  wiadomo, przepis prosi o `target=`.
- **Związki do prognozy** = wiersze z pustą odpowiedzią (np. naftalen, nitrobenzen dopisane na końcu ze SMILES albo
  z deskryptorami). Dostają prognozę z każdego modelu i informację, czy są w domenie.
- **Odpowiedź tekstowa** z 2–10 wartościami (np. aktywny/nieaktywny) albo liczbowa 0/1 = **klasyfikacja**.
- `.smi`: `SMILES nazwa [odpowiedź]` w linii, opcjonalny nagłówek.

Przykładowy podpis pliku:

```
qsar descriptors=given,mopac max_desc=3 split=sorted every=3 scramble=200
```

## Parametry

| parametr | domyślnie | znaczenie |
|---|---|---|
| `target`, `smiles`, `name` | zgadnij | nazwy kolumn (wielkość liter i spacje bez znaczenia) |
| `split` | `sorted` | `sorted` – sortowanie po y i co `every`-ty związek do testu, najmniejszy i największy zawsze w uczącym; `ks` – Kennard–Stone na deskryptorach; `random` – losowy (w klasyfikacji warstwowy); `column` – z kolumny |
| `split_col` | `Zbiór`, `set`… | kolumna z podziałem: T/V, train/test, U/W, uczący/walidacyjny/testowy, cal/ext |
| `every` | 3 | przy `sorted` |
| `test` | 0.25 | udział testu przy `ks`/`random` |
| `descriptors` | `given` | `given` (kolumny liczbowe z pliku), `rdkit` (~210 deskryptorów 2D), `mordred` (~1600 2D), `mopac` (PM6), `morgan` (bity Morgana r=2, 1024 – raczej do klasyfikacji). Można łączyć: `given,mopac` |
| `max_desc` | 4 | najwięcej deskryptorów w modelu (i tak najwyżej n_uczący/5 – Topliss) |
| `select` | `auto` | `exhaustive` (wszystkie podzbiory), `forward`, `ga` (algorytm genetyczny), `enet` (elastic net), `all` (bez wyboru); `auto` = pełne przeszukanie, jeśli zajmie < 5 min, inaczej GA |
| `models` | wszystkie | `mlr,pls,ridge,lasso,enet,knn,dt,rf,svr,xgb`; w klasyfikacji odpowiednio regresja logistyczna (L2/L1/enet), PLS-DA, kNN, drzewo, las, SVM, XGBoost |
| `corr_max` | 0.95 | z pary deskryptorów skorelowanych mocniej zostaje ten lepiej skorelowany z y |
| `scramble` | 100 | liczba permutacji Y-scrambling |
| `seed` | 42 | ziarno (te same dane + ziarno = ten sam wynik) |

## Co się dzieje

1. **Deskryptory.** MOPAC: RDKit ETKDG → MMFF94 → `PM6 PRECISE` (HOMO, LUMO, przerwa, twardość, elektroujemność,
   elektrofilowość, potencjał jonizacji, moment dipolowy, ciepło tworzenia, energia całkowita, pole i objętość
   powierzchni COSMO, skrajne ładunki). Wyniki Mordred i MOPAC są w `cache/` – po restarcie laptopa gotowe
   cząsteczki są pomijane. Cząsteczka, której nie da się policzyć, jest wymieniona w podsumowaniu i nie wchodzi do
   modelu.
2. **Braki.** Deskryptor z jakimkolwiek brakiem wśród związków z odpowiedzią odpada (podsumowanie mówi, które
   związki to spowodowały – możesz je usunąć z pliku, jeśli te deskryptory są ważniejsze).
3. **Podział**, potem wszystko liczone tylko na zbiorze uczącym: odsiew stałych (> 90% tych samych wartości) i
   skorelowanych, autoskalowanie (średnia i SD zbioru uczącego).
4. **Wybór zmiennych** kryterium Q²LOO regresji MLR (klasyfikacja: zrównoważona dokładność w CV), kara za
   VIF > 10. Liczba deskryptorów: najmniejsza, której Q²LOO jest najwyżej 0,02 poniżej najlepszej.
   Zbiór testowy nie bierze udziału ani w wyborze, ani w strojeniu – jest prawdziwie zewnętrzny.
5. **Modele** – hiperparametry strojone walidacją krzyżową na zbiorze uczącym; Q² z LOO (powyżej 60 związków
   uczących: 10-krotna CV) przy ustalonych hiperparametrach.
6. **Walidacja (OECD 4):** R², R²adj, RMSE_C, MAE, Q²LOO, RMSE_CV; na teście Q²F1, Q²F2, Q²F3, CCC, R²ext (r²
   obserwowane–przewidziane), RMSE_P, MAE_P, rm² (średnie i Δ), kryteria Golbraikha–Tropshy;
   Y-scrambling: R²scr i Q²scr (max, średnia), cR²p. Dla MLR dodatkowo scrambling **z ponownym wyborem
   deskryptorów** (forward na każdej permutacji) – pokazuje, ile Q² daje sam przypadek przy wybieraniu k z wielu
   deskryptorów.
7. **Najlepszy model:** wśród modeli bez przeuczenia (R² − Q²LOO ≤ 0,3) najniższy błąd CV; każdy model w granicach
   jednego błędu standardowego od niego jest uznany za równie dobry i wygrywa najprostszy (MLR → PLS → ridge →
   lasso → enet → kNN → drzewo → SVR → RF → XGB). Nie R² (rośnie z samym dopasowaniem) i nie test (zostałby
   użyty do wyboru i przestałby być zewnętrzny). Podsumowanie mówi, który model wygrałby po samym Q²F3.
8. **Domena (OECD 3):** dźwignia h z h* = 3(p'+1)/n i reszty standaryzowane ±3 (wykres Williamsa) dla modeli
   liniowych; dla nieliniowych średnia odległość euklidesowa do 5 najbliższych związków uczących (próg: 95. centyl
   tej odległości w zbiorze uczącym).

## Co wraca

- `summary.txt` – liczby i wnioski (wysyłane jako wiadomość), do 4 wykresów jako zdjęcia.
- `wyniki/metrics.csv` – wiersz na model, wszystkie statystyki i hiperparametry, kolumna `best`.
- `wyniki/predictions.csv` – każdy związek × każdy model: zbiór (uczący/testowy/prognoza/wykluczony), y
  obserwowane, przewidziane, LOO, reszta, reszta standaryzowana, h, h*, odległość kNN, `in_ad`.
- `wyniki/selected.txt` – najlepszy podzbiór dla każdego k, wybrane deskryptory, równanie MLR ze współczynnikami
  (SE, t, p, β standaryzowane, VIF), F, s.
- `wyniki/mlr_coefficients.csv`, `wyniki/descriptors.csv` (wszystkie policzone deskryptory), `wyniki/scrambling.csv`.
- Wykresy: `obs_vs_pred_<model>.png`, `williams.png`, `scrambling.png`, `corr_heatmap.png`, `pca.png`
  (klasyfikacja: `confusion.png`, `roc.png`).

## Ograniczenia

- Nazwa związku nie wystarczy do policzenia deskryptorów – potrzebny SMILES.
- MOPAC liczy jedną konformację z RDKit (bez przeszukiwania konformacyjnego), w fazie gazowej.
- Pełne przeszukanie podzbiorów jest wykładnicze: przy setkach deskryptorów i `max_desc` ≥ 3 przepis przejdzie
  na algorytm genetyczny (zapisze szacowany czas).
- Przy kilkudziesięciu związkach różnice Q² między modelami rzędu 0,02–0,05 mieszczą się w szumie; dlatego reguła
  jednego błędu standardowego i preferencja prostszych modeli.
