# Kontrakt przepisu (recipe)

Jak runner uruchamia przepis i czego od niego oczekuje. Jedno miejsce prawdy dla wszystkich przepisów.

## Uruchomienie
- `python -m compute_bot.recipes.<name>` pythonem ze środowiska conda (`deploy/environment.yml`), katalog roboczy =
  katalog `work/` zadania, `PYTHONPATH` = katalog aplikacji, `PATH` zawiera `compute_bot/bin` (np. `md_step`).
- `params.json` w katalogu roboczym: parametry już sprawdzone i z domyślnymi wartościami, typy jak w
  `compute_bot/recipes/__init__.py` (`RECIPES[name].params`). Przepis nie waliduje ich drugi raz typowo, ale sprawdza
  sens danych wejściowych (np. czy kolumna istnieje) i kończy się czytelnym błędem.
- Pliki od właściciela leżą w katalogu roboczym: jeden plik albo rozpakowany zip (bez wspólnego folderu nadrzędnego).
- `COMPUTE_THREADS` – ile wątków wolno użyć (laptop: 2 rdzenie / 4 wątki, bez GPU, kilka GB RAM). Ustaw
  `OMP_NUM_THREADS` itp. według tego. Proces już ma niski priorytet (nice 19).
- Internet jest dostępny (pobieranie z RCSB/UniProt/g:Profiler), ale brak sieci nie może wywrócić przepisu, jeśli
  krok jest opcjonalny: zapisz ostrzeżenie w podsumowaniu.

## Wyjście
- Wyniki w `wyniki/` w katalogu roboczym (tabele CSV, wykresy PNG ~150 dpi, struktury).
- `summary.txt` – krótko, po polsku, zwykły tekst (bez HTML), do ~2500 znaków: najważniejsze liczby i wnioski.
  Bot wyśle go jako wiadomość po zakończeniu.
- `highlights.txt` – opcjonalnie: do 4 ścieżek PNG (po jednej w linii, względem katalogu roboczego), które bot wyśle
  jako zdjęcia.
- Postęp: linie `PROGRESS: <tekst>` na stdout (z `flush=True`), np. `PROGRESS: NPT 40%` albo `PROGRESS: ligand 12/300`.
  Bot pokazuje ostatnią.
- Kod wyjścia 0 = sukces. Błąd dla właściciela: ostatnia linia na stderr `ERROR: <po polsku, co poprawić>`, kod != 0.
  Nie łap wyjątków tylko po to, żeby je ukryć.

## Wznawianie
- Laptop może się zrestartować w trakcie: runner uruchomi przepis jeszcze raz w tym samym katalogu. Długie kroki muszą
  wtedy iść dalej, a nie od zera (GROMACS przez `md_step`, skrining: pomiń ligandy z gotowym wynikiem). Dwa
  uruchomienia pod rząd dają ten sam stan końcowy.

## Kod i testy
- Moduł `compute_bot/recipes/<name>.py`, ewentualnie pakiet `compute_bot/recipes/<name>/`. Logika obliczeń w czystych
  funkcjach (dane wchodzą, wynik wychodzi), wejście/wyjście plikowe tylko w `main()`.
- Testy w `tests/recipes/test_<name>.py` (pytest), wołają przepis tak jak runner (subprocess w katalogu tmp z
  `params.json`) i sprawdzają to, co zobaczy właściciel (summary, pliki, liczby). Do tego testy czystych funkcji na
  danych, gdzie znany jest wynik. Testy zależne od pakietów: `pytest.importorskip`. Wolne (sieć, dłuższe obliczenia):
  `@pytest.mark.slow`.
- Repozytorium jest publiczne: żadnych danych z zajęć, danych osobowych ani tokenów w repo. Dane testowe generuj
  syntetycznie w teście (albo małe publiczne struktury pobierane w testach `slow`).
- Styl: ruff (`ruff.toml`), linia do 120 znaków, komentarze po angielsku, komunikaty dla użytkownika po polsku.
- Dokumentacja dla użytkownika: `docs/<name>.md` po polsku – co wysłać, przykładowy podpis pliku, co wraca.
