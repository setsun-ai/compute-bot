"""
Who can reach the panel: only devices in the owner's Tailscale network (tailscale serve, the default) or
anyone on the internet (tailscale funnel), always behind the Mini App signature check in web.py.
The service user may drive tailscale because deploy/remote.sh made it the operator.
"""
from __future__ import annotations

import subprocess


class ExposureError(Exception):
    """tailscale refused or is missing; the text goes to the owner."""


def shell(args):
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise ExposureError("nie ma tailscale: sudo bash deploy/remote.sh") from None
    except subprocess.TimeoutExpired:
        raise ExposureError("tailscale nie odpowiada") from None
    if done.returncode != 0:
        raise ExposureError((done.stderr or done.stdout).strip()[:300] or f"kod {done.returncode}")
    return done.stdout


class Exposure:
    def __init__(self, port, run=shell):
        self.target, self.run = f"http://127.0.0.1:{port}", run

    def status(self):
        return "Funnel on" in self.run(["tailscale", "funnel", "status"])

    def set_public(self, on):
        if on:
            self.run(["tailscale", "funnel", "--bg", "--https=443", self.target])
        else:
            self.run(["tailscale", "funnel", "--https=443", "off"])
            self.run(["tailscale", "serve", "--bg", "--https=443", self.target])
