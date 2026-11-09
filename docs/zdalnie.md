# Praca zdalna: Telegram albo SSH przez Tailscale

Obie drogi prowadzą do tej samej kolejki: zadanie wysłane z terminala widać w `/jobs`, a bot pisze o jego starcie,
postępie i końcu tak samo jak o wysłanym z telefonu.

## Telegram – na co dzień

- Plik albo `/md 1E0L` otwiera **kartę parametrów**: widzisz wartości, zmieniasz je przyciskami (np. `T: 300` →
  300 / 310 / 350 / 400 / 500 / 600) albo piszesz `T=450 ns=2`, potem **▶️ Start**. Zła wartość jest od razu
  odrzucona, więc 10-godzinne zadanie nie padnie na literówce.
- Po fakcie: `/retry N T=500` (to samo z inną wartością), `/dolicz N 5` (skończone MD o 5 ns dłużej, bez kopiowania
  plików), `/top N`, `/pauza`, `/usun stare`.
- Opcje GROMACS spoza karty: `mdp=nstxout-compressed=500;tcoupl=nose-hoover` (produkcja) albo własny `run.sh`.

## SSH przez Tailscale – gdy chcesz pracować jak przy laptopie

Jednorazowo na laptopie: `sudo bash ~/Documents/boty/compute-bot/deploy/remote.sh` (SSH, tmux, Tailscale; logujesz się
linkiem). Na Macu zainstaluj Tailscale tym samym kontem. Bez przekierowywania portów u gospodarzy.

```bash
ssh UZYTKOWNIK@NAZWA-LAPTOPA        # nazwę wypisze remote.sh
compute jobs                         # kolejka; compute log 3 -f – log na żywo
compute submit ~/symulacje/1E0L      # folder z run.sh; compute submit dane.xlsx qsar max_desc=3
compute md 1E0L T=600 ns=0.5 engine=openmm
compute-env                          # gmx, python, vina, obabel, namd3 w Twojej powłoce
cd "$(compute where 3)"              # katalog wyników zadania 3
```

- **VS Code**: rozszerzenie Remote-SSH → host z Tailscale → interpreter `/var/lib/compute-bot/env/bin/python`;
  notebooki liczą się na laptopie.
- **JupyterLab**: `ssh -L 8888:localhost:8888 UZYTKOWNIK@NAZWA` → `compute-env; jupyter lab --no-browser` → link
  z tokenem otwierasz na Macu.
- Ręczne obliczenia zabierają te same 2 rdzenie co kolejka: na ten czas `compute pause`, potem `compute resume`.
- Długie rzeczy w terminalu uruchamiaj w `tmux`, żeby przeżyły zerwane połączenie (albo po prostu wyślij je do kolejki).

## Silniki MD

| Silnik | Skąd | Jak użyć |
|---|---|---|
| GROMACS | środowisko conda | przepis `md` (domyślnie), `.tpr`, `run.sh` z `md_step` |
| OpenMM | środowisko conda | `md … engine=openmm`, `.py` |
| NAMD | po rejestracji na ks.uiuc.edu (licencja) | wyślij botowi `NAMD_…_Linux-x86_64-multicore.tar.gz` albo `sudo bash deploy/install-namd.sh PLIK`; potem `namd3` w `run.sh` i `charmm engine=namd` |
| CHARMM-GUI | paczka `charmm-gui.tgz` | wyślij botowi: przepis `charmm` uruchomi folder gromacs/, openmm/ albo namd/ |
