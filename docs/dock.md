# Dokowanie (AutoDock Vina) – przepis `dock`

Dokuje ligandy do białka: przygotowuje receptor i ligandy, wyznacza pudełko, dokuje po jednym ligandzie (skrining
setek związków wznawia się po restarcie laptopa) i zwraca ranking, pozy, kontakty i wykresy. Jeśli w strukturze jest
ligand krystaliczny, najpierw go **redokuje** jako kontrolę i podaje RMSD najlepszej pozy względem kryształu – to
standardowe sprawdzenie, czy protokół w ogóle odtwarza znane wiązanie (< 2 Å = dobrze).

## Co wysłać

- **Receptor**: plik `.pdb` / `.cif` (kompleks z ligandem albo samo białko) **albo** parametr `receptor=KOD` (pobranie
  z RCSB). Gdy plików struktury jest kilka, nazwij właściwy np. `receptor.pdb`.
- **Ligandy** (dowolnie wiele plików, wszystko może być w jednym zipie z receptorem):
  - `.csv` / `.tsv` – kolumna `smiles` (albo podobna: `canonical_smiles`… – rozpoznaję też po zawartości) i
    opcjonalnie nazwa: `name`, `id`, `nazwa`, `compound_id`, `chembl_id`…; separator `,` `;` lub tabulator;
  - `.smi` – `SMILES nazwa` w linii, `#` = komentarz;
  - `.sdf` / `.mol` – nazwa z tytułu rekordu albo pola `name`/`id`; 3D z pliku nie jest używane (stereochemia tak);
  - `.mol2`.
- Bez pliku ligandów, a z ligandem krystalicznym – sam redocking (sprawdzenie protokołu).

Przykładowe podpisy pliku:

```
dock ligand_ref=STI exhaustiveness=16
dock receptor=1STP ligand_ref=BTN
dock residues=A:189,A:195,A:216 padding=8
dock center=12.5,3.1,-8.0 size=24,24,24 scoring=vinardo
```

## Parametry

| parametr | domyślnie | znaczenie |
|---|---|---|
| `receptor` | – | kod PDB, gdy nie wysyłasz struktury |
| `ligand_ref` | – | ligand krystaliczny (nazwa HETATM, np. `STI`, albo z łańcuchem `B:STI`): pudełko + redocking |
| `residues` | – | reszty kieszeni `A:45,A:88` (bez łańcucha `45`, jeśli jednoznaczne) |
| `center`, `size` | –, 22,22,22 | pudełko wprost [Å]; `size` działa tylko razem z `center` |
| `padding` | 6 | margines wokół ligandu/reszt z każdej strony [Å] (pudełko min. 10 Å) |
| `exhaustiveness` | 8 | dokładność przeszukiwania Vina (skrining: 8; ważne wyniki: 16–32) |
| `poses` | 9 | liczba zapisanych póz na ligand |
| `ph` | 7.4 | pH protonacji receptora (pdbfixer) i ligandów (Open Babel) |
| `keep_waters` | nie | zostaw wody krystaliczne w receptorze |
| `scoring` | vina | `vina` albo `vinardo` |
| `seed` | 42 | ziarno (te same dane + ziarno = te same wyniki) |

Kolejność wyboru pudełka: `center` → `ligand_ref` → `residues` → jedyny „lekopodobny” ligand w pliku (≥ 6 atomów
ciężkich, nie bufor/glikol/siarczan/cukier) – wtedy podsumowanie mówi, który wzięto. Gdy nie da się zgadnąć
(brak ligandu albo kilka różnych), przepis kończy się błędem z listą ligandów do wyboru.

## Co się dzieje

1. Receptor: pierwszy model, pierwsze altloci, tylko łańcuchy białkowe; wody (chyba że `keep_waters`) i wszystkie
   HETATM usunięte (ligand krystaliczny jest zapamiętany na pudełko i redocking); pdbfixer uzupełnia brakujące atomy
   (bez dobudowywania pętli) i wodory przy `ph`; PDBQT pisze Meeko (Open Babel tylko awaryjnie). Reszty, których
   Meeko nie przyjął, są usuwane i wypisane w podsumowaniu (z ostrzeżeniem, gdy leżą przy pudełku).
2. Ligandy: RDKit, największy fragment (bez soli), zobojętnienie, protonacja przy `ph` (reguły Open Babel), 3D
   ETKDG + MMFF94, PDBQT z Meeko. Błąd jednego ligandu trafia do `nieudane.csv`, reszta liczy się dalej.
3. Dokowanie: Vina (API Pythona), mapy liczone raz, wątki = `COMPUTE_THREADS`. Po każdym ligandzie zapis wyniku –
   po restarcie gotowe ligandy są pomijane (zmiana SMILES lub ustawień = liczy od nowa). Ligand, przy którym proces
   padł dwa razy, jest oznaczany jako nieudany. Postęp: `ligand 12/300 (ETA …)`.

## Co wraca (`wyniki/`)

- `ranking.csv` – nazwa, SMILES, najlepsza energia [kcal/mol], efektywność ligandu LE = −ΔG/atomy ciężkie, MW,
  logP, HBD/HBA, wiązania rotujące, naruszenia reguły Lipińskiego, SMILES po protonacji, ścieżki do póz;
  `ref_XXX` = redokowany ligand krystaliczny (kontrola).
- `pozy/<nazwa>.sdf` i `.pdbqt` – wszystkie pozy (energia w polach SDF), `ligandy_pdbqt/` – przygotowane ligandy.
- `kompleksy/NN_<nazwa>.pdb` – receptor + najlepsza poza dla czołówki i kontroli (otwórz w PyMOL).
- `kontakty.csv` – reszty ≤ 4 Å od ligandu i kandydaci na mostki wodorowe (N/O–N/O ≤ 3.5 Å, bez kątów).
- `receptor.pdb` (z wodorami), `receptor.pdbqt`, `box.txt` (format konfiguracji Vina), `box.pdb` (narożniki pudełka
  z krawędziami – `load box.pdb` w PyMOL i `show sticks`), `ligand_krystaliczny_XXX.sdf`.
- `histogram_energii.png`, `energia_vs_masa.png`; `summary.txt` – pudełko, RMSD redockingu, liczby, top 10,
  kontakty najlepszego, ostrzeżenia.

## Na co uważać

- Energia z Vina/Vinardo to przybliżenie (błąd rzędu 2 kcal/mol), nie zmierzone powinowactwo: dobra do odsiewu
  słabych związków, słaba do szeregowania podobnych. Duże cząsteczki dostają z natury lepsze wyniki – patrz LE.
- Receptor jest sztywny, bez wód i jonów (jon metalu w kieszeni jest usuwany – podsumowanie ostrzega; Vina go nie
  modeluje dobrze). Białka z metalem w miejscu aktywnym dokuj ostrożnie.
- Protonacja ligandów to reguły Open Babel, nie przewidywanie pKa; tautomery nie są wyliczane.
- Nieokreślone centra stereogeniczne w SMILES – RDKit wybiera jedno; podaj stereochemię, jeśli ją znasz.
- Laptop: ok. 2–60 s na ligand przy exhaustiveness=8 (zależnie od wielkości i giętkości ligandu i pudełka).
