"""
The Mini App backend: a small HTTP server on 127.0.0.1 that `tailscale serve` (private) or `tailscale funnel`
(public, switched from the bot) puts behind HTTPS. Standard library only.

  python3 -m compute_bot.web     (TG_TOKEN, TG_OWNER_ID, COMPUTE_HOME, WEB_PORT=8787)

The page itself (web/index.html) carries no data. Every /api call must carry Telegram's signed initData in the
X-Init-Data header: valid HMAC for this bot's token, the owner's user id, signed less than a day ago. Anything else
is 401, and an address with too many failures gets 429 for a while. The API does only what the bot does, through
the same functions (actions.py); it never runs commands.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import VERSION, actions, guide, jobs, links
from .recipes import RECIPES, ParamError, quick_values

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
MAX_AGE = 86400  # seconds an initData signature stays good (a Mini App session can stay open long)
MAX_FAILURES, BLOCK_SECONDS = 10, 600
MAX_FILE = 50 * 1024 ** 2


def check_init_data(init_data, token, owner, now, max_age=MAX_AGE):
    """Telegram's Mini App signature: HMAC-SHA256 keyed by HMAC("WebAppData", token) over the sorted fields."""
    try:
        fields = dict(urllib.parse.parse_qsl(init_data, strict_parsing=True))
        given = fields.pop("hash")
        check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
        secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
        expected = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, given):
            return False
        if now - int(fields["auth_date"]) > max_age:
            return False
        return json.loads(fields["user"])["id"] == owner
    except (ValueError, KeyError, TypeError):
        return False


def job_json(store, spec):
    status = store.status(spec.id)
    return {"id": spec.id, "name": spec.name, "kind": spec.kind, "entry": spec.entry, "params": spec.params,
            "created": spec.created, **status._asdict()}


def recipes_json():
    return {name: {"title": r.title, "help": r.help, "inputs": list(r.inputs),
                   "params": {p: {"kind": v.kind, "default": v.default, "help": v.help, "choices": list(v.choices),
                                  "low": v.low, "high": v.high, "quick": quick_values(name, p)}
                              for p, v in r.params.items()}}
            for name, r in RECIPES.items() if name != "env"}


class App:
    """Requests → JSON answers; the HTTP layer below only parses and writes."""

    def __init__(self, store, token, owner, system_info, clock=time.time):
        self.store, self.token, self.owner = store, token, owner
        self.send_document = self.telegram_document
        self.system_info, self.clock = system_info, clock
        self.failures, self.lock = {}, threading.Lock()

    def telegram_document(self, name, data, caption):
        from .telegram import Telegram
        Telegram(self.token)("sendDocument", {"chat_id": self.owner, "caption": caption},
                             files={"document": (name, data)}, timeout=300)

    # --- access ------------------------------------------------------------------------------------

    def blocked(self, address):
        with self.lock:
            count, since = self.failures.get(address, (0, 0.0))
            if count >= MAX_FAILURES and self.clock() - since < BLOCK_SECONDS:
                return True
            if count >= MAX_FAILURES:
                self.failures.pop(address, None)
            return False

    def failed(self, address):
        with self.lock:
            count, since = self.failures.get(address, (0, self.clock()))
            self.failures[address] = (count + 1, since if count else self.clock())

    def allowed(self, init_data):
        return check_init_data(init_data or "", self.token, self.owner, self.clock())

    # --- reading ---------------------------------------------------------------------------------------

    def state(self):
        beat = self.store.runner_state()
        alive = bool(beat) and self.clock() - beat.get("time", 0) < 180
        statuses = self.store.statuses()
        return {"version": VERSION, "now": self.clock(), "hold": self.store.hold, "system": self.system_info(),
                "max_attempts": jobs.MAX_ATTEMPTS, "queue": jobs.queue_order(statuses, self.store.priorities()),
                "runner": {"alive": alive, **{k: beat.get(k) for k in ("time", "job", "blocked", "free", "threads")}},
                "jobs": [job_json(self.store, spec) for spec in reversed(self.store.specs()[-50:])]}

    def job(self, job):
        spec = self.store.spec(job)
        work = self.store.work(job)
        summary = work / "summary.txt"
        highlights = work / "highlights.txt"
        return {**job_json(self.store, spec),
                "summary": summary.read_text(errors="replace")[:20000] if summary.exists() else "",
                "highlights": highlights.read_text().split()[:8] if highlights.exists() else [],
                "files": [{"path": str(p.relative_to(work)), "size": size} for p, size in self.store.outputs(job)]}

    def drafts(self):
        try:
            state = json.loads((self.store.root / "bot.json").read_text())
        except (OSError, ValueError):
            return {}
        return {token: d for token, d in state.get("drafts", {}).items()
                if d.get("recipe") and not (self.store.root / "staging" / token / actions.CLOSED).exists()}

    # --- routing --------------------------------------------------------------------------------------

    def get(self, path, query):
        parts = path.strip("/").split("/")[1:]  # after "api"
        if parts == ["state"]:
            return self.state()
        if parts == ["recipes"]:
            return recipes_json()
        if parts == ["guide"]:
            return [topic._asdict() for topic in guide.TOPICS]
        if parts == ["drafts"]:
            return self.drafts()
        job = self.job_id(parts)
        if len(parts) == 2:
            return self.job(job)
        if parts[2:] == ["log"]:
            return {"text": self.store.log_tail(job, int(query.get("bytes", 20000)))}
        if parts[2:] == ["file"]:
            found = self.store.result_file(job, query.get("path", ""))
            if found is None or found.stat().st_size > MAX_FILE:
                raise LookupError
            return found
        raise LookupError

    def post(self, path, body):
        parts = path.strip("/").split("/")[1:]
        if parts == ["hold"]:
            self.store.set_hold(bool(body.get("on")))
            return {"hold": self.store.hold}
        if parts == ["cleanup"]:
            old = actions.old_finished(self.store, float(body.get("days", 7)), self.clock())
            for job in old:
                self.store.remove(job)
            return {"removed": old}
        if len(parts) == 3 and parts[0] == "drafts" and parts[2] == "discard":
            if parts[1] not in self.drafts():
                raise FileExistsError
            actions.close_staged(self.store.root / "staging" / parts[1])
            return {"ok": True}
        if len(parts) == 3 and parts[0] == "drafts":
            return self.start_draft(parts[1], body)
        job = self.job_id(parts)
        action = parts[2:] and parts[2]
        if action == "remove":
            self.store.remove(job)
            return {"ok": True}
        if action == "cancel":
            self.store.request_cancel(job)
            return {"ok": True}
        if action == "top":
            actions.top(self.store, job)
            return {"ok": True}
        if action == "retry":
            return {"id": actions.retry(self.store, job, body.get("params") or None).id}
        if action == "send":
            found = self.store.result_file(job, str(body.get("path", "")))
            if found is None or found.stat().st_size > MAX_FILE:
                raise LookupError
            self.send_document(found.name, found.read_bytes(), f"#{job}")
            return {"ok": True}
        if action == "extend":
            return {"id": actions.extend(self.store, job, body.get("ns")).id}
        raise LookupError

    def start_draft(self, token, body):
        draft = self.drafts().get(token)
        if not token.isdigit() or draft is None:
            raise FileExistsError
        typed = {**draft.get("params", {}), **{k: str(v) for k, v in (body.get("params") or {}).items()}}
        spec = actions.start_staged(self.store, self.store.root / "staging" / token, draft["recipe"],
                                    draft["name"], typed, draft.get("fetch") or [])
        return {"id": spec.id}

    def job_id(self, parts):
        if len(parts) >= 2 and parts[0] == "jobs" and parts[1].isdigit() and int(parts[1]) in self.store.ids():
            return int(parts[1])
        raise LookupError


TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".pdf": "application/pdf",
         ".csv": "text/csv; charset=utf-8", ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8"}


def make_server(app, host, port):
    class Handler(BaseHTTPRequestHandler):
        server_version = "compute-bot"

        def log_message(self, fmt, *args):  # no request lines with paths in the journal
            pass

        def address(self):
            # Behind tailscale serve/funnel every request comes from 127.0.0.1; the forwarded header names the client.
            # The proxy appends the real address last; anything before it is whatever the client wrote.
            return self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[-1].strip()

        def reply(self, code, payload=None, kind="application/json"):
            if isinstance(payload, Path):
                data, kind = payload.read_bytes(), TYPES.get(payload.suffix.lower(), "application/octet-stream")
            elif isinstance(payload, bytes):
                data = payload
            else:
                data = json.dumps(payload if payload is not None else {}, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def handle_api(self, method):
            address = self.address()
            if app.blocked(address):
                return self.reply(HTTPStatus.TOO_MANY_REQUESTS)
            if not app.allowed(self.headers.get("X-Init-Data")):
                app.failed(address)
                return self.reply(HTTPStatus.UNAUTHORIZED)
            url = urllib.parse.urlparse(self.path)
            try:
                if method == "GET":
                    query = dict(urllib.parse.parse_qsl(url.query))
                    return self.reply(HTTPStatus.OK, app.get(url.path, query))
                length = min(int(self.headers.get("Content-Length") or 0), 64 * 1024)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
                return self.reply(HTTPStatus.OK, app.post(url.path, body))
            except LookupError:
                return self.reply(HTTPStatus.NOT_FOUND)
            except FileExistsError:
                return self.reply(HTTPStatus.CONFLICT, {"error": "Ta karta jest już uruchomiona albo nieaktualna."})
            except ParamError as e:
                return self.reply(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            except ValueError:
                return self.reply(HTTPStatus.BAD_REQUEST, {"error": "zła treść zapytania"})

        def download(self):
            """A signed link from the bot (links.py): the file as an attachment, resumable with Range."""
            address = self.address()
            if app.blocked(address):
                return self.reply(HTTPStatus.TOO_MANY_REQUESTS)
            found = links.parse(app.token, self.path[len("/dl/"):].split("?")[0], app.clock())
            path = app.store.result_file(*found) if found and found[0] in app.store.ids() else None
            if path is None:
                app.failed(address)
                return self.reply(HTTPStatus.NOT_FOUND)
            size = path.stat().st_size
            start = 0
            match = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range") or "")
            if match and int(match[1]) < size:
                start = int(match[1])
            self.send_response(HTTPStatus.PARTIAL_CONTENT if start else HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size - start))
            self.send_header("Accept-Ranges", "bytes")
            if start:
                self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
            name = urllib.parse.quote(path.name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{name}; filename=\"{path.name}\"")
            self.end_headers()
            with open(path, "rb") as source:
                source.seek(start)
                while chunk := source.read(1024 * 1024):
                    try:
                        self.wfile.write(chunk)
                    except OSError:
                        return  # the browser went away; it can resume with Range

        def do_GET(self):
            if self.path.startswith("/dl/"):
                return self.download()
            if self.path.startswith("/api/"):
                return self.handle_api("GET")
            if urllib.parse.urlparse(self.path).path in ("/", "/index.html"):
                return self.reply(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            return self.reply(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path.startswith("/api/"):
                return self.handle_api("POST")
            return self.reply(HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), Handler)


def system_json(home):
    """Numbers for the panel's gauges (the chat gets the same as text from bot.system_info)."""
    from .runner import read_cpu_temp
    info = {"temp": read_cpu_temp(), "cpus": os.cpu_count()}
    try:
        info["load"] = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
        mem = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        info["mem_total"], info["mem_free"] = (int(mem[k].split()[0]) * 1024 for k in ("MemTotal", "MemAvailable"))
    except (OSError, KeyError, ValueError):
        pass
    import shutil
    disk = shutil.disk_usage(home)
    info["disk_total"], info["disk_free"] = disk.total, disk.free
    return info


def main():
    token, owner = os.environ.get("TG_TOKEN", ""), os.environ.get("TG_OWNER_ID", "")
    if not token or not owner.isdigit():
        print("Brak TG_TOKEN / TG_OWNER_ID", file=sys.stderr)
        return 2
    home = Path(os.environ.get("COMPUTE_HOME", "."))
    port = int(os.environ.get("WEB_PORT", "8787"))
    app = App(jobs.Store(home), token, int(owner), lambda: system_json(home))
    print(f"compute-web {VERSION} on 127.0.0.1:{port}", flush=True)
    make_server(app, "127.0.0.1", port).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
