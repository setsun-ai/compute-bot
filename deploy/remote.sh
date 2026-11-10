#!/bin/bash
# Remote work on the laptop without opening ports: Tailscale (private network) + SSH, then VS Code Remote-SSH,
# JupyterLab through an SSH tunnel and the `compute` command for the same queue as the bot. Run as root.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Uruchom jako root (sudo)."; exit 1; }

apt-get install -y -qq openssh-server tmux >/dev/null
systemctl enable -q --now ssh
if ! command -v tailscale >/dev/null; then
  echo "Instaluję Tailscale (oficjalny skrypt tailscale.com)..."
  curl -fsSL https://tailscale.com/install.sh | sh
fi
echo
echo "Logowanie do Tailscale: otwórz link poniżej (na telefonie albo Macu) i zaloguj się swoim kontem."
tailscale up --ssh
# The panel: private HTTPS on the tailnet name; the bot may switch it to public (Funnel) as user compute.
tailscale set --operator=compute
tailscale serve --bg --https=443 http://127.0.0.1:8787
name=$(tailscale status --self --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' || hostname)
user=${SUDO_USER:-$USER}
if grep -q '^WEB_URL=' /etc/compute-bot.env; then
  sed -i "s|^WEB_URL=.*|WEB_URL=https://$name/|" /etc/compute-bot.env
else
  echo "WEB_URL=https://$name/" >> /etc/compute-bot.env
fi
grep -q '^SSH_TARGET=' /etc/compute-bot.env && sed -i "s|^SSH_TARGET=.*|SSH_TARGET=$user@$name|" /etc/compute-bot.env \
  || echo "SSH_TARGET=$user@$name" >> /etc/compute-bot.env
systemctl restart compute-bot compute-web
cat <<INFO

Gotowe. Panel: przycisk „Panel” w czacie z botem (na telefonie włącz Tailscale; bez niego: /funnel on).
Z Maca (z zainstalowanym Tailscale, tym samym kontem):
  ssh $user@$name                 # terminal; długie rzeczy w tmux
  compute jobs                    # ta sama kolejka co w bocie; compute submit FOLDER
  compute-env                     # gmx, python, vina... w swojej powłoce
  VS Code: Remote-SSH → $user@$name, interpreter /var/lib/compute-bot/env/bin/python
  Jupyter:  ssh -L 8888:localhost:8888 $user@$name  →  compute-env; jupyter lab --no-browser
Ręczne obliczenia konkurują z kolejką o 2 rdzenie: na ten czas compute pause, potem compute resume.
INFO
