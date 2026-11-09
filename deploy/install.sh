#!/bin/bash
# Installs or updates compute-bot on Debian/Ubuntu (run as root from the cloned repo). Safe to run again:
# jobs, settings and the conda environment stay; the environment only gets what's new.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPT=/opt/compute-bot
HOME_DIR=/var/lib/compute-bot
ENV_PREFIX=$HOME_DIR/env
BOT_ENV=/etc/compute-bot.env
RUNNER_ENV=/etc/compute-runner.env
MM=$OPT/bin/micromamba

[ "$(id -u)" = 0 ] || { echo "Uruchom jako root (sudo)."; exit 1; }
[ "$(uname -m)" = x86_64 ] || { echo "Ten instalator jest dla x86_64."; exit 1; }
command -v systemctl >/dev/null || { echo "Potrzebny systemd."; exit 1; }

echo "Pakiety systemowe..."
apt-get install -y -qq python3 curl ca-certificates bzip2 >/dev/null
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Potrzebny Python ≥ 3.10."; exit 1; }

id compute >/dev/null 2>&1 || useradd --system --home-dir "$HOME_DIR" --shell /usr/sbin/nologin compute
# setgid: everything inside stays in group compute, so the owner's login (added below) can read results and
# submit from a terminal (compute submit ...), while the services keep their own user.
install -d -o compute -g compute -m 2770 "$HOME_DIR"
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
  usermod -aG compute "$SUDO_USER"
fi

free_gb=$(df -Pk "$HOME_DIR" | awk 'NR==2 {print int($4 / 1048576)}')
if [ ! -d "$ENV_PREFIX" ] && [ "$free_gb" -lt 6 ]; then
  echo "Na środowisko (~3 GB) i wyniki potrzeba co najmniej 6 GB wolnego, jest $free_gb GB."
  exit 1
fi

echo "Kod do $OPT/app..."
install -d "$OPT/bin"
rm -rf "$OPT/app.new"
mkdir -p "$OPT/app.new"
(cd "$HERE" && tar --exclude=.git --exclude=tests --exclude=__pycache__ -cf - .) | tar -C "$OPT/app.new" -xf -
if ! (cd "$OPT/app.new" && PYTHONPATH="$OPT/app.new" python3 -m compute_bot --selftest); then
  rm -rf "$OPT/app.new"
  echo "Selftest nie przeszedł, nic nie zmieniam."
  exit 1
fi
rm -rf "$OPT/app"
mv "$OPT/app.new" "$OPT/app"
chmod -R a+rX "$OPT"
# This install is what should run now, even if the bot downloaded something newer before.
rm -rf "$HOME_DIR/update"

if [ ! -x "$MM" ]; then
  echo "micromamba (menedżer pakietów conda)..."
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$OPT" bin/micromamba
fi

echo "Środowisko obliczeniowe (GROMACS, RDKit, Vina, scikit-learn...). Pierwszy raz trwa 10–30 min."
export MAMBA_ROOT_PREFIX=$HOME_DIR/mamba
# micromamba writes the pip part next to the yml, so it must sit where the compute user can write.
install -o compute -g compute -m 600 "$OPT/app/deploy/environment.yml" "$HOME_DIR/environment.yml"
if [ -d "$ENV_PREFIX/conda-meta" ]; then
  sudo -u compute env MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MM" env update -y -q -p "$ENV_PREFIX" \
    -f "$HOME_DIR/environment.yml"
else
  sudo -u compute env MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MM" create -y -q -p "$ENV_PREFIX" \
    -f "$HOME_DIR/environment.yml"
fi
sudo -u compute "$ENV_PREFIX/bin/python" -c "import rdkit, sklearn, MDAnalysis, vina, mordred, openmm" \
  || { echo "Środowisko nie działa, zobacz komunikaty wyżej."; exit 1; }
sudo -u compute "$ENV_PREFIX/bin/gmx" --version >/dev/null || { echo "GROMACS się nie uruchamia."; exit 1; }

if [ ! -f "$BOT_ENV" ]; then
  echo
  echo "Potrzebny NOWY bot od @BotFather (nie token AMP ani weight-bota: dwa programy na jednym tokenie się kłócą)."
  read -rsp "Token bota (nie będzie widoczny): " token
  echo
  read -rp "Napisz teraz cokolwiek do nowego bota na Telegramie i naciśnij Enter... " _
  owner=$(cd "$OPT/app" && TG_TOKEN="$token" python3 -m compute_bot.bot --whoami)
  case "$owner" in
    ''|*[!0-9]*) echo "Nie widzę Twojej wiadomości albo token jest zły: $owner"; exit 1 ;;
  esac
  ( umask 077; printf 'TG_TOKEN=%s\nTG_OWNER_ID=%s\n' "$token" "$owner" > "$BOT_ENV" )
  echo "Bot będzie odpowiadał tylko Tobie (ID $owner)."
fi
if [ ! -f "$RUNNER_ENV" ]; then
  printf 'COMPUTE_THREADS=%s\n' "$(nproc)" > "$RUNNER_ENV"
fi

install -m 755 "$OPT/app/deploy/compute" /usr/local/bin/compute
install -m 644 "$OPT/app/deploy/profile.sh" /etc/profile.d/compute-bot.sh
install -m 644 "$OPT/app/deploy/compute-bot.service" "$OPT/app/deploy/compute-runner.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q compute-bot compute-runner
systemctl restart compute-bot compute-runner
sleep 3
if systemctl is-active --quiet compute-bot && systemctl is-active --quiet compute-runner; then
  echo "Gotowe. Wersja $(cd "$OPT/app" && python3 -m compute_bot --version), wątków: $(cut -d= -f2 "$RUNNER_ENV")."
  echo "Wyślij botowi /help. Liczbę wątków zmienisz w $RUNNER_ENV (potem: systemctl restart compute-runner)."
  echo "W terminalu: compute jobs, compute submit FOLDER, compute-env (środowisko). Po pierwszej instalacji"
  echo "zaloguj się ponownie, żeby grupa compute zadziałała. Dostęp zdalny: sudo bash deploy/remote.sh"
else
  journalctl -u compute-bot -u compute-runner -n 20 --no-pager || true
  exit 1
fi
