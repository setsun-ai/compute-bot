"""The Telegram Bot API over urllib: enough for messages, buttons, photos and files."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class TelegramError(Exception):
    """Carries Telegram's own words or the error's type, never the URL: it contains the token."""


class Telegram:
    def __init__(self, token, base="https://api.telegram.org"):
        self.token, self.base = token, base

    def __call__(self, method, fields=None, files=None, timeout=60):
        url = f"{self.base}/bot{self.token}/{method}"
        fields = {k: str(v) for k, v in (fields or {}).items()}
        if files:
            boundary = "compute-" + os.urandom(8).hex()
            parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                     for k, v in fields.items()]
            for field, (filename, data) in files.items():
                quoted = urllib.parse.quote(filename)
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                             f"filename=\"{quoted}\"; filename*=UTF-8''{quoted}\r\n"
                             "Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
            body, kind = b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"
        else:
            body, kind = urllib.parse.urlencode(fields).encode(), "application/x-www-form-urlencoded"
        request = urllib.request.Request(url, data=body, headers={"Content-Type": kind})
        try:
            with urllib.request.urlopen(request, timeout=timeout + 10) as response:
                return json.loads(response.read()).get("result")
        except urllib.error.HTTPError as e:
            try:
                reason = json.loads(e.read()).get("description")
            except ValueError:
                reason = None
            raise TelegramError(f"{method}: {reason or e.code}") from None
        except (OSError, ValueError) as e:
            raise TelegramError(f"{method}: {type(e).__name__}") from None

    def download(self, file_id):
        path = self("getFile", {"file_id": file_id})["file_path"]
        try:
            with urllib.request.urlopen(f"{self.base}/file/bot{self.token}/{path}", timeout=120) as response:
                return response.read()
        except OSError as e:
            raise TelegramError(f"download: {type(e).__name__}") from None
