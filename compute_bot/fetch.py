"""
The first step of a job whose input is a link: download into work/, unpack a zip, then become the real command.

  python -m compute_bot.fetch -- CMD...     (cwd = the job's work/, the links are in ../spec.json)

Runs inside the job, so it is logged, cancellable and reported like the job itself. A finished download is kept
(.part files resume with Range), and .fetch_done makes a restarted job skip the whole step: an unpacked file the
job has changed since is never overwritten. Standard library only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .jobs import ParamError, zip_layout

DONE = Path(".fetch_done")
CHUNK = 1024 * 1024


def direct_url(url):
    """Share links → links that give the file itself."""
    parsed = urllib.parse.urlparse(url)
    if "drive.google.com" in parsed.netloc:
        match = re.search(r"/d/([\w-]+)", parsed.path) or re.search(r"[?&]id=([\w-]+)", url)
        if match:
            return f"https://drive.usercontent.google.com/download?id={match[1]}&export=download&confirm=t"
    if "dropbox.com" in parsed.netloc:
        query = urllib.parse.parse_qs(parsed.query)
        query["dl"] = ["1"]
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))
    return url


def filename_from(headers, fallback):
    disposition = headers.get("Content-Disposition") or ""
    match = re.search(r"filename\*=UTF-8''([^;]+)", disposition) or re.search(r'filename="?([^";]+)"?', disposition)
    name = Path(urllib.parse.unquote(match[1])).name if match else ""
    return name or fallback


def download(url, name, say=print):
    """Downloads into name (or the server's own name when we only had a placeholder); returns the final name."""
    part = Path(name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(direct_url(url), headers={"User-Agent": "compute-bot"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:
            have = 0  # the server ignored Range: from the start
        total = response.headers.get("Content-Length")
        total = int(total) + have if total and total.isdigit() else None
        final = filename_from(response.headers, name) if name.startswith("pobrany") else name
        shown, done = -10, have
        with open(part, "ab" if have else "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                percent = int(done * 100 / total) if total else None
                if percent is not None and percent >= shown + 10:
                    shown = percent
                    say(f"PROGRESS: pobieranie {final} {percent}% ({done // CHUNK} MB)")
    os.replace(part, final)
    return final


def unpack_zip(name):
    with zipfile.ZipFile(name) as archive:
        for member, target in zip_layout(archive).items():
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(path, "wb") as dst:
                while chunk := src.read(CHUNK):
                    dst.write(chunk)
    Path(name).unlink()


def fetch_all(items, say=print):
    for item in items:
        name = item["name"]
        if Path(name).exists():
            continue
        final = download(item["url"], name, say)
        if final.lower().endswith(".zip") and zipfile.is_zipfile(final):
            say(f"PROGRESS: rozpakowuję {final}")
            unpack_zip(final)
    DONE.write_text("\n".join(item["url"] for item in items))


def main(argv):
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    if not DONE.exists():
        spec = json.loads(Path("../spec.json").read_text())
        try:
            fetch_all(spec.get("fetch") or [], lambda line: print(line, flush=True))
        except (urllib.error.URLError, OSError, ParamError, zipfile.BadZipFile) as e:
            print(f"ERROR: nie udało się pobrać pliku: {e}", file=sys.stderr)
            return 1
        if spec["kind"] == "script" and not Path(spec["entry"]).exists():
            print(f"ERROR: w pobranym pliku nie ma {spec['entry']}", file=sys.stderr)
            return 1
    if not command:
        return 0
    sys.stdout.flush()
    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
