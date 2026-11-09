#!/bin/bash
# Installs NAMD for compute-bot jobs from the tarball you downloaded after registering at
# https://www.ks.uiuc.edu/Research/namd/ (Linux-x86_64-multicore). Same as sending the file to the bot.
#   sudo bash install-namd.sh ~/Pobrane/NAMD_3.0.1_Linux-x86_64-multicore.tar.gz
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Uruchom jako root (sudo)."; exit 1; }
file=$(realpath "${1:?podaj plik NAMD_..._Linux-x86_64-multicore.tar.gz}")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$file" "$tmp/"
chown -R compute:compute "$tmp"
cd "$tmp"
sudo -u compute env COMPUTE_HOME=/var/lib/compute-bot PYTHONPATH=/opt/compute-bot/app \
  python3 -m compute_bot.tools namd "$(basename "$file")"
cat summary.txt
