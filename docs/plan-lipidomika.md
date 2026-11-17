# Plan: tryb lipidowy w `omics` (na później)

Źródło: zajęcia „Lipidomika i glikomika” (sem. 1) – projekt mleka kobiecego M1 vs M12, ćwiczenia 3–7 i X.
Statystyka i QC z tych zajęć już są w `omics`; brakuje rzeczy niżej. Kolejność = kolejność robienia.

| # | Co | Po co (z zajęć) | Zakres |
|---|---|---|---|
| 1 | Czytanie tabel Agilent Profinder / MFE (`[Mass]`, `[Height]`/`[Area]` na próbkę; tabela celowana z `Name`, `Formula`, dwa wiersze nagłówka) i eksportu MZmine 3 (`row m/z`, `row retention time`, `*.mzML Peak area`) | ćw. 4, 7: tak wyglądają wasze dane | ~1 dzień pracy: parsery + testy na syntetycznych tabelach w tym kształcie |
| 2 | Projekt sparowany: kolumna dawczyni/pary, test Wilcoxona dla par i t dla par (+FDR); uśrednianie replikatów technicznych (1A/1B) przed testem | ćw. 4: pseudoreplikacja; M1 i M12 tej samej dawczyni | ~0,5 dnia |
| 3 | Nazwy lipidów → klasa, C:DB, łańcuchy (`PC 34:1`, `15:0-18:1 PC`, `TG 52:2`, `SM d18:1/18:1`); sumy i udziały per klasa, profil nasycenia, wzbogacenie klas (Fisher, offline) | ćw. 5, X: interpretacja biologiczna | ~1 dzień, parser nazw z testami na przykładach z LIPID MAPS |
| 4 | Wykresy z ćw. X: RT vs m/z, KMD (CH₂), Van Krevelen (ze wzorów), Venn grup, dryf wzorców IS w kolejności | ćw. 6, X | ~0,5 dnia |
| 5 | Surowe .mzML: przepis `mzmine` (MZmine 3 w trybie wsadowym z plikiem batch z GUI, pliki przez `/url`) | ćw. 3, 7: dekonwolucja trwa – praca dla laptopa | osobno, ~1–2 dni; wymaga MZmine z rejestracją (jak NAMD), sprawdzić licencję i pamięć (8 GB) |

Razem 1–4: około 3–4 dni pracy z testami. 5 dopiero, gdy 1–4 działają na prawdziwych danych.

Jak sprawdzić: na kopii `nontargeted table 5 000 threshold.csv` i `targeted data.xlsx` z zajęć (lokalnie, nigdy w repo)
wynik ma się zgadzać z tym, co wyszło w MetaboAnalyst/Orange na zajęciach (liczba cech po filtrach, kierunek zmian M1→M12).
