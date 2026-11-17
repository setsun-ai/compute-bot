"""
Download links for result files too big for Telegram. The link is the permission: job, path and expiry signed
with a key derived from the bot token, so the web service can check it without any stored state.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

DAY = 86400


def _key(secret):
    return hmac.new(b"compute-bot download", secret.encode(), hashlib.sha256).digest()


def _sign(secret, body):
    return base64.urlsafe_b64encode(hmac.new(_key(secret), body.encode(), hashlib.sha256).digest()[:18]).decode()


def make(secret, job, rel, expires):
    body = base64.urlsafe_b64encode(json.dumps([job, rel, int(expires)]).encode()).decode().rstrip("=")
    return f"{body}.{_sign(secret, body)}"


def parse(secret, token, now):
    """(job, path) when the link is ours and still valid, else None."""
    try:
        body, signature = token.split(".")
        if not hmac.compare_digest(signature, _sign(secret, body)):
            return None
        job, rel, expires = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        return (int(job), str(rel)) if now <= expires else None
    except (ValueError, TypeError):
        return None
