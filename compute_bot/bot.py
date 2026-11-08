"""
Telegram side of the compute queue: files and commands in, job news and results out. Standard library only.

  python3 -m compute_bot.bot           run (TG_TOKEN, TG_OWNER_ID, COMPUTE_HOME)
  python3 -m compute_bot.bot --whoami  Telegram ID of whoever last wrote to the bot (the installer uses it)

Answers only its owner in a private chat. The bot creates jobs and asks to cancel them; the runner runs them.
The bot alone writes bot.json (what it already told the owner, the report interval, files waiting for a choice).
"""
from __future__ import annotations

import html
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

from . import VERSION, jobs
from .recipes import RECIPES, ParamError
from .telegram import Telegram, TelegramError

MAX_DOWNLOAD = 20 * 1024 ** 2  # Telegram's limit for files a bot can download
MAX_UPLOAD = 49 * 1024 ** 2  # and for files it can send
ICONS = {"queued": "⏳", "running": "⚙️", "paused": "🌡️", "done": "✅", "failed": "❌", "cancelled": "⏹️"}
HELP = (
    "🧪 <b>Kolejka obliczeń</b> na laptopie. Wyślij plik, a policzę i odeślę wyniki.\n\n"
    "📦 <b>zip z run.sh</b> – dowolny skrypt (GROMACS, Python, R…); w skrypcie <code>md_step nazwa opcje-grompp</code> "
    "wznawia mdrun po restarcie\n"
    "📄 <b>.tpr</b> – od razu mdrun · <b>.ipynb</b> – wykonany notebook wraca · <b>.py</b> – skrypt\n"
    "🧬 <b>.pdb</b> albo <code>/md 1E0L T=600 ns=0.5</code> – cała dynamika z analizą\n"
    "📊 <b>xlsx/csv</b> – zapytam: QSAR czy omika (albo podpisz plik: <code>qsar target=LogKd</code>)\n"
    "💊 <code>/dock 1STP</code> + zip z ligandami – dokowanie\n\n"
    "Parametry w podpisie pliku: <code>klucz=wartość</code>. Lista: /help md · /help qsar · /help omics · "
    "/help dock\n\n"
    "/jobs – kolejka · /job N – szczegóły · /log N · /get N [plik] · /cancel N · /retry N\n"
    "/co 3 – raport postępu co 3 h (0 = tylko start i koniec) · /system – temperatura, RAM, dysk\n"
    "/env – doinstaluj pakiety · /update · /rollback"
)


def esc(text):
    return html.escape(str(text), quote=False)


def keyboard(rows):
    return json.dumps({"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows if row]})


def duration(seconds):
    seconds = int(max(0, seconds or 0))
    hours, rest = divmod(seconds, 3600)
    if hours >= 24:
        return f"{hours // 24} d {hours % 24} h"
    return f"{hours} h {rest // 60} min" if hours else f"{rest // 60} min {rest % 60} s"


def size_text(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def system_info(home="."):
    """Load, temperature, memory, disk: what decides whether the laptop can take another job."""
    from .runner import read_cpu_temp
    lines = []
    temp = read_cpu_temp()
    lines.append(f"🌡️ CPU {temp:.0f} °C" if temp is not None else "🌡️ brak czujnika")
    try:
        load = open("/proc/loadavg").read().split()[:3]
        lines.append(f"📈 obciążenie {' / '.join(load)} (rdzeni: {os.cpu_count()})")
    except OSError:
        pass
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        total, free = (int(info[k].split()[0]) * 1024 for k in ("MemTotal", "MemAvailable"))
        lines.append(f"🧠 RAM wolne {size_text(free)} z {size_text(total)}")
    except (OSError, KeyError, ValueError):
        pass
    disk = shutil.disk_usage(home)
    lines.append(f"💾 dysk wolne {size_text(disk.free)} z {size_text(disk.total)}")
    try:
        up = float(open("/proc/uptime").read().split()[0])
        lines.append(f"⏱️ działa od {duration(up)}")
    except OSError:
        pass
    return "\n".join(lines)


class Bot:
    def __init__(self, store, api, owner, state_path, updater=None, clock=time.time, system_info=system_info):
        self.store, self.api, self.owner = store, api, owner
        self.state_path, self.updater, self.clock = Path(state_path), updater, clock
        self.system_info = system_info
        self.zip_budget = 45 * 1024 ** 2
        self.restart = False
        fresh = not self.state_path.exists()
        self.state = {"announced": {}, "reported": {}, "report_hours": 3.0, "pending": {}}
        if not fresh:
            self.state.update(json.loads(self.state_path.read_text()))
        else:  # first start (or lost state): don't repeat news about old jobs
            for job, status in store.statuses().items():
                if status.state in jobs.FINAL:
                    self.state["announced"][str(job)] = f"{status.state}:{status.attempt}"
            self.save()

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.write_json(self.state_path, self.state)

    # --- talking -----------------------------------------------------------------------------------

    def say(self, text, markup=None):
        fields = {"chat_id": self.owner, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": True}
        if markup:
            fields["reply_markup"] = markup
        return self.api("sendMessage", fields)

    def send_file(self, name, data, caption=""):
        fields = {"chat_id": self.owner, "caption": caption[:1000], "parse_mode": "HTML"}
        return self.api("sendDocument", fields, files={"document": (name, data)}, timeout=300)

    # --- what comes in --------------------------------------------------------------------------------

    def handle_update(self, update):
        if "callback_query" in update:
            query = update["callback_query"]
            if query.get("from", {}).get("id") != self.owner:
                return
            self.api("answerCallbackQuery", {"callback_query_id": query["id"]})
            return self.handle_tap(query["data"], query.get("message", {}).get("message_id"))
        msg = update.get("message")
        if not msg or msg.get("chat", {}).get("id") != self.owner or msg.get("from", {}).get("id") != self.owner:
            return
        try:
            if "document" in msg:
                return self.handle_document(msg["document"], msg.get("caption"))
            text = (msg.get("text") or "").strip()
            if text.startswith("/"):
                return self.handle_command(text)
            self.say("Wyślij plik albo komendę. /help pokaże, co umiem.")
        except ParamError as e:
            self.say(f"❌ {esc(e)}")

    def handle_document(self, document, caption):
        name = jobs.safe_name(document.get("file_name") or "plik")
        if (document.get("file_size") or 0) > MAX_DOWNLOAD:
            return self.say(f"❌ {esc(name)} ma {size_text(document['file_size'])}, a bot może pobrać najwyżej "
                            "20 MB. Spakuj mniej albo dodaj do zipa run.sh, który pobierze duże pliki (wget/curl).")
        word, params = jobs.parse_caption(caption)
        try:
            data = self.api.download(document["file_id"])
        except TelegramError as e:
            return self.say(f"❌ Nie mogę pobrać pliku: {esc(e)}")
        files = jobs.unpack(name, data)
        found = jobs.detect(name, word, files)
        if found is None:
            return self.ask_what(name, caption, files)
        kind, entry = found
        self.queue(kind, entry, name, params, files)

    def ask_what(self, name, caption, files):
        token = f"{int(self.clock() * 1000) % 10 ** 9}"
        folder = self.store.root / "staging" / token
        for rel, data in files.items():
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_bytes(data)
        self.state["pending"][token] = {"name": name, "caption": caption or ""}
        self.save()
        suffix = PurePosixPath(name).suffix.lower()
        options = [r for r in RECIPES.values() if suffix in r.inputs] or [r for r in RECIPES.values() if r.inputs]
        rows = [[(r.title, f"f:{token}:{r.name}")] for r in options] + [[("✖️ Nic", f"f:{token}:-")]]
        self.say(f"Co zrobić z <b>{esc(name)}</b>?", keyboard(rows))

    def queue(self, kind, entry, name, params, files):
        if kind == "recipe":
            recipe = RECIPES[entry]
            params = recipe.validate(params)
            if entry == "md" and not files and not params["pdb"]:
                raise ParamError("Podaj kod PDB (/md 1E0L) albo wyślij plik .pdb.")
            if entry == "dock" and not files and not params["receptor"]:
                raise ParamError("Dokowanie potrzebuje ligandów: wyślij zip z receptorem i ligandami albo plik "
                                 ".smi/.sdf z podpisem dock receptor=XXXX.")
        elif params:
            raise ParamError("Parametry klucz=wartość są tylko dla przepisów (md, qsar, omics, dock); "
                             "skrypt ustawia wszystko sam.")
        spec = self.store.create(kind, entry, name, params, files)
        ahead = sum(1 for job, s in self.store.statuses().items() if job != spec.id and s.state not in jobs.FINAL)
        where = "zaraz startuje" if not ahead else f"przed nim: {ahead}"
        what = RECIPES[entry].title if kind == "recipe" else {"script": "skrypt run.sh", "tpr": "mdrun",
                                                              "notebook": "notebook", "python": "skrypt Python"}[kind]
        self.say(f"📥 <b>#{spec.id}</b> {esc(name)} – {esc(what)} w kolejce ({where}).")
        return spec

    def handle_tap(self, data, message_id):
        parts = data.split(":")
        if parts[0] == "f" and len(parts) == 3:
            token, choice = parts[1], parts[2]
            pending = self.state["pending"].pop(token, None)
            self.save()
            folder = self.store.root / "staging" / token
            if pending is None or not folder.exists():
                return self.say("Ten plik już nie czeka. Wyślij go jeszcze raz.")
            files = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
            shutil.rmtree(folder, ignore_errors=True)
            if choice == "-":
                return self.say("Dobrze, nic nie robię.")
            try:
                _, params = jobs.parse_caption(pending["caption"])
                self.queue("recipe", choice, pending["name"], params, files)
            except ParamError as e:
                self.say(f"❌ {esc(e)}")
        elif parts[0] == "j" and len(parts) == 3 and parts[1].isdigit():
            self.handle_command(f"/{parts[2]} {parts[1]}")

    def handle_command(self, text):
        word, _, rest = text.partition(" ")
        word = word[1:].split("@")[0].lower()
        rest = rest.strip()
        number = int(rest.split()[0]) if rest.split() and rest.split()[0].isdigit() else None
        if word in ("start", "help", "menu", "pomoc"):
            if rest in RECIPES:
                return self.say(RECIPES[rest].usage())
            return self.say(HELP)
        if word in RECIPES:
            recipe = RECIPES[word]
            if recipe.needs_file and word != "md":
                return self.say(f"{esc(recipe.title)}: wyślij plik z podpisem <code>{word} …</code>.\n\n"
                                + recipe.usage())
            _, params = jobs.parse_caption(text)
            return self.queue("recipe", word, word if word == "env" else f"{word} {rest}".strip(), params, {})
        if word in ("jobs", "kolejka"):
            return self.list_jobs()
        if word == "system":
            return self.say(self.system_info() + "\n" + self.queue_line())
        if word == "co":
            try:
                hours = float(rest.replace(",", "."))
            except ValueError:
                return self.say(f"Raport co {self.state['report_hours']:g} h. Zmiana: /co 2 (0 = wyłącz).")
            self.state["report_hours"] = max(0.0, hours)
            self.save()
            return self.say("Raporty postępu wyłączone." if not hours else f"Raport postępu co {hours:g} h.")
        if word == "update":
            return self.update_command()
        if word == "rollback":
            return self.rollback_command()
        if number is None or number not in self.store.ids():
            return self.say("Podaj numer zadania, np. /job 3. Lista: /jobs")
        if word == "job":
            return self.job_view(number)
        if word == "log":
            return self.say(f"📜 <b>#{number}</b>\n<pre>{esc(self.store.log_tail(number, 3500))}</pre>")
        if word == "get":
            path = rest.split(maxsplit=1)[1] if len(rest.split(maxsplit=1)) > 1 else None
            return self.get_command(number, path)
        if word == "cancel":
            if self.store.status(number).state in jobs.FINAL:
                return self.say(f"#{number} już się skończyło.")
            self.store.request_cancel(number)
            return self.say(f"⏹️ Zatrzymuję #{number}…")
        if word == "retry":
            spec = self.store.spec(number)
            files = self.store.inputs(number)
            again = self.store.create(spec.kind, spec.entry, spec.name, spec.params, files)
            return self.say(f"🔁 #{number} jeszcze raz jako <b>#{again.id}</b>.")
        self.say("Nie znam tej komendy. /help")

    # --- views --------------------------------------------------------------------------------------

    def queue_line(self):
        states = [s.state for s in self.store.statuses().values()]
        return (f"⚙️ liczy: {states.count('running') + states.count('paused')} · "
                f"⏳ czeka: {states.count('queued')} · wersja {VERSION}")

    def list_jobs(self):
        lines = []
        for job in self.store.ids()[-12:]:
            spec, status = self.store.spec(job), self.store.status(job)
            extra = f" – {esc(status.progress)}" if status.state in ("running", "paused") and status.progress else ""
            lines.append(f"{ICONS[status.state]} <b>#{job}</b> {esc(spec.name)}{extra}")
        self.say("\n".join(lines) or "Kolejka jest pusta.")

    def job_view(self, job):
        spec, status = self.store.spec(job), self.store.status(job)
        lines = [f"{ICONS[status.state]} <b>#{job}</b> {esc(spec.name)} – {status.state}"]
        if status.started:
            end = status.finished or self.clock()
            lines.append(f"czas: {duration(end - status.started)}, próba {status.attempt}")
        if status.progress:
            lines.append(f"postęp: {esc(status.progress)}")
        if spec.params:
            shown = " ".join(f"{k}={v}" for k, v in spec.params.items() if v not in (None, "", []))
            lines.append(f"<code>{esc(shown[:800])}</code>")
        buttons = [("📜 Log", f"j:{job}:log")]
        buttons.append(("📦 Wyniki", f"j:{job}:get") if status.state in jobs.FINAL else ("⏹️ Stop", f"j:{job}:cancel"))
        if status.state in ("failed", "cancelled"):
            buttons.append(("🔁 Ponów", f"j:{job}:retry"))
        self.say("\n".join(lines), keyboard([buttons]))

    def get_command(self, job, rel):
        if rel is None:
            return self.send_results(job)
        path = self.store.result_file(job, rel)
        if path is None:
            return self.say(f"Nie ma pliku {esc(rel)} w #{job}.")
        if path.stat().st_size > MAX_UPLOAD:
            return self.say(f"{esc(rel)} ma {size_text(path.stat().st_size)}; Telegram przyjmie najwyżej 50 MB.")
        self.send_file(path.name, path.read_bytes(), f"#{job}")

    # --- news about jobs ------------------------------------------------------------------------------

    def tick(self):
        now = self.clock()
        changed = False
        for job, status in self.store.statuses().items():
            key = f"{status.state}:{status.attempt}"
            before = self.state["announced"].get(str(job))
            if key != before:
                self.state["announced"][str(job)] = key
                changed = True
                self.announce(job, status, before)
                self.state["reported"][str(job)] = now
            elif status.state == "running" and self.state["report_hours"] > 0:
                last = self.state["reported"].get(str(job), now)
                if now - last >= self.state["report_hours"] * 3600:
                    self.state["reported"][str(job)] = now
                    changed = True
                    elapsed = duration(now - (status.started or now))
                    self.say(f"⚙️ <b>#{job}</b> liczy od {elapsed}" +
                             (f": {esc(status.progress)}" if status.progress else ""))
        if changed:
            self.save()

    def announce(self, job, status, before):
        spec = self.store.spec(job)
        title = f"<b>#{job}</b> {esc(spec.name)}"
        old_state, _, old_attempt = (before or "").partition(":")
        if status.state == "running":
            if old_state == "paused" and str(status.attempt) == old_attempt:
                self.say(f"🌡️➡️⚙️ {title}: procesor ostygł, liczę dalej.")
            elif status.attempt > 1:
                self.say(f"🔁 {title} wznowione po przerwie (próba {status.attempt}).")
            else:
                self.say(f"▶️ {title} – start.")
        elif status.state == "paused":
            self.say(f"🌡️ {title}: za gorąco, wstrzymuję do ostygnięcia.")
        elif status.state == "done":
            self.send_results(job, header=f"✅ {title} gotowe w {duration(status.finished - status.started)}.")
        elif status.state == "failed":
            tail = self.store.log_tail(job, 1500)
            reason = f"\n{esc(status.progress)}" if status.progress else ""
            self.say(f"❌ {title} – błąd (kod {status.exit_code}).{reason}\n<pre>{esc(tail)}</pre>",
                     keyboard([[("🔁 Ponów", f"j:{job}:retry"), ("📦 Pliki", f"j:{job}:get")]]))
        elif status.state == "cancelled":
            self.say(f"⏹️ {title} zatrzymane.")

    def send_results(self, job, header=None):
        work = self.store.work(job)
        summary = work / "summary.txt"
        if summary.exists():
            body = summary.read_text(errors="replace")[:3200]
        else:
            body = self.store.log_tail(job, 1200)
        self.say((header + "\n\n" if header else "") + f"<pre>{esc(body)}</pre>")
        highlights = work / "highlights.txt"
        if highlights.exists():
            for rel in highlights.read_text().split()[:4]:
                path = self.store.result_file(job, rel)
                if path and path.stat().st_size < 10 * 1024 ** 2:
                    self.api("sendPhoto", {"chat_id": self.owner, "caption": rel},
                             files={"photo": (path.name, path.read_bytes())})
        archive, left_out = self.results_zip(job)
        self.send_file(f"wyniki_{job:04d}.zip", archive, f"#{job}")
        if left_out:
            shown = "\n".join(f"<code>/get {job} {esc(rel)}</code> ({size_text(size)})" for rel, size in left_out[:10])
            more = f"\n… i {len(left_out) - 10} innych" if len(left_out) > 10 else ""
            self.say(f"📎 Za duże do zipa, pobierz osobno (do 50 MB):\n{shown}{more}")

    def results_zip(self, job):
        """Smallest files first until the zip would get too big for Telegram; the log always goes in."""
        work = self.store.work(job)
        out = io.BytesIO()
        left_out = []
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("log.txt", self.store.log_tail(job, 2 * 1024 ** 2))
            for path, size in self.store.outputs(job):
                rel = str(path.relative_to(work))
                if out.tell() + size > self.zip_budget:
                    left_out.append((rel, size))
                    continue
                archive.write(path, rel)
        return out.getvalue(), left_out

    # --- updating itself --------------------------------------------------------------------------------

    def update_command(self):
        if self.updater is None:
            return self.say("Aktualizacje są wyłączone w tej instalacji.")
        from .updater import UpdateError, version_tuple
        try:
            tag = self.updater.latest()
            if version_tuple(tag[1:]) <= version_tuple(VERSION):
                return self.say(f"Masz najnowszą wersję ({VERSION}).")
            env_changed = self.updater.install(tag)
        except UpdateError as e:
            return self.say(f"❌ Aktualizacja nie wyszła: {esc(e)}")
        note = "\nNowe pakiety w środowisku: wyślij /env (to zadanie w kolejce)." if env_changed else ""
        self.say(f"⬆️ Zainstalowane {tag}. Bot startuje od nowa; runner przełączy się, gdy skończy bieżące "
                 f"zadanie.{note}")
        self.restart = True

    def rollback_command(self):
        if self.updater is None:
            return self.say("Aktualizacje są wyłączone w tej instalacji.")
        from .updater import UpdateError
        try:
            self.updater.rollback()
        except UpdateError as e:
            return self.say(f"❌ {esc(e)}")
        self.say("↩️ Wracam do poprzedniej wersji.")
        self.restart = True


def log(message):
    print(message, flush=True)


def whoami(api):
    try:
        updates = api("getUpdates", {"timeout": 0}) or []
    except TelegramError as e:
        print(e)
        return
    people = [u.get("message", {}).get("from", {}).get("id") for u in updates]
    people = [p for p in people if p]
    print(people[-1] if people else "Napisz coś do bota i uruchom to jeszcze raz.")


def run(api, bot):
    log(f"compute-bot {VERSION} started")
    try:
        api("setMyCommands", {"commands": json.dumps([
            {"command": "jobs", "description": "Kolejka zadań"},
            {"command": "system", "description": "Temperatura, RAM, dysk"},
            {"command": "help", "description": "Co umiem"}])})
    except TelegramError as e:
        log(f"start: {e}")
    offset, failures = None, 0
    while True:
        try:
            bot.tick()
            fields = {"timeout": 20, "allowed_updates": json.dumps(["message", "callback_query"])}
            if offset:
                fields["offset"] = offset
            updates = api("getUpdates", fields) or []
            failures = 0
        except (TelegramError, OSError) as e:
            failures += 1
            log(f"poll failed: {e}")
            time.sleep(min(60, 5 * failures))
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                bot.handle_update(update)
            except Exception as e:
                log(f"update failed: {type(e).__name__}: {e}")
                try:
                    bot.say(f"❌ Coś nie wyszło ({esc(type(e).__name__)}: {esc(e)}).")
                except Exception:
                    pass
        if bot.restart:
            try:
                api("getUpdates", {"offset": offset, "timeout": 0})
            except TelegramError as e:
                log(f"confirm: {e}")
            sys.exit(0)  # Restart=always starts the new version


def main(argv):
    token, owner = os.environ.get("TG_TOKEN", ""), os.environ.get("TG_OWNER_ID", "")
    if not token:
        print("Brak TG_TOKEN", file=sys.stderr)
        return 2
    api = Telegram(token)
    if "--whoami" in argv:
        whoami(api)
        return 0
    if not owner.isdigit():
        print("Brak TG_OWNER_ID (sprawdzisz go przez --whoami)", file=sys.stderr)
        return 2
    from .updater import Updater
    home = Path(os.environ.get("COMPUTE_HOME", "."))
    store = jobs.Store(home)
    bot = Bot(store, api, int(owner), home / "bot.json", updater=Updater(home / "update"),
              system_info=lambda: system_info(home))
    try:
        run(api, bot)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
