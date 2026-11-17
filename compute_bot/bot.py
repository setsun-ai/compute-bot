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
import shlex
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

from . import VERSION, actions, drafts, examples, guide, jobs, links
from .drafts import keyboard
from .recipes import RECIPES, ParamError
from .telegram import Telegram, TelegramError

MAX_DOWNLOAD = 20 * 1024 ** 2  # Telegram's limit for files a bot can download
MAX_UPLOAD = 49 * 1024 ** 2  # and for files it can send
PART = 45 * 1024 ** 2  # a big file sent in pieces: size of one piece
MAX_PARTS_TOTAL = 1024 ** 3  # beyond this, pieces through Telegram stop making sense
ICONS = {"queued": "⏳", "running": "⚙️", "paused": "🌡️", "done": "✅", "failed": "❌", "cancelled": "⏹️"}
HELP = (
    "🧪 <b>Kolejka obliczeń</b> na laptopie. Wyślij plik, a policzę i odeślę wyniki.\n\n"
    "📦 <b>zip z run.sh</b> – dowolny skrypt; w nim <code>md_step nazwa opcje-grompp</code> wznawia mdrun "
    "po restarcie\n"
    "📄 <b>.tpr</b> – od razu mdrun · <b>.ipynb</b> – wraca wykonany · <b>.py</b> – skrypt\n"
    "🧬 <b>.pdb</b> albo <code>/md 1E0L T=600 ns=0.5</code> – cała dynamika z analizą\n"
    "📊 <b>xlsx/csv</b> – zapytam: QSAR czy omika (albo podpis <code>qsar target=LogKd</code>)\n"
    "💊 zip z receptorem i ligandami + podpis <code>dock</code> – dokowanie\n"
    "🔗 <code>/url LINK omics …</code> – plik większy niż 20 MB (Drive, Dropbox, link)\n\n"
    "Parametry: /help md · /help qsar · /help omics · /help dock\n\n"
    "/guide – przewodnik krok po kroku z przykładami do wypróbowania\n"
    "/jobs – kolejka · /job N · /log N · /get N [plik] · /cancel N\n"
    "/retry N [klucz=wartość] – jeszcze raz, też ze zmianą · /top N – następne w kolejce\n"
    "/usun N · /usun stare – zwolnij dysk · /pauza · /wznow – wstrzymaj wszystko\n"
    "/co 3 – raport postępu co 3 h · /cisza 23-7 – nocą bez dźwięku · /system\n"
    "/panel – kolejka, wykresy i parametry w aplikacji · /funnel on|off – panel bez Tailscale\n"
    "/env – doinstaluj pakiety · /update · /rollback"
)


def esc(text):
    return html.escape(str(text), quote=False)


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
    def __init__(self, store, api, owner, state_path, updater=None, clock=time.time, system_info=system_info,
                 web_url=None, exposure=None, link_secret="", ssh_target=None):
        self.web_url, self.exposure = web_url, exposure
        self.link_secret, self.ssh_target = link_secret, ssh_target  # download links; user@host for scp
        self.store, self.api, self.owner = store, api, owner
        self.state_path, self.updater, self.clock = Path(state_path), updater, clock
        self.system_info = system_info
        self.zip_budget = 45 * 1024 ** 2
        self.restart = False
        fresh = not self.state_path.exists()
        self.local_hour = lambda: time.localtime(self.clock()).tm_hour
        self.state = {"announced": {}, "reported": {}, "report_hours": 3.0, "drafts": {}, "open_draft": None,
                      "quiet": [23, 7], "runner_alert": ""}
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

    def quiet_now(self):
        quiet = self.state["quiet"]
        if not quiet:
            return False
        start, end, hour = quiet[0], quiet[1], self.local_hour()
        return start <= hour < end if start < end else hour >= start or hour < end

    def fields(self, **extra):
        fields = {"chat_id": self.owner, "parse_mode": "HTML", **extra}
        if self.quiet_now():
            fields["disable_notification"] = True
        return fields

    def say(self, text, markup=None):
        fields = self.fields(text=text[:4000], disable_web_page_preview=True)
        if markup:
            fields["reply_markup"] = markup
        return self.api("sendMessage", fields)

    def send_file(self, name, data, caption=""):
        fields = self.fields(caption=caption[:1000])
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
            if "=" in text:
                return self.edit_open_draft(text)
            self.say("Wyślij plik albo komendę. /help pokaże, co umiem.")
        except ParamError as e:
            self.say(f"❌ {esc(e)}")

    def handle_document(self, document, caption):
        name = jobs.safe_name(document.get("file_name") or "plik")
        if (document.get("file_size") or 0) > MAX_DOWNLOAD:
            return self.say(f"❌ {esc(name)} ma {size_text(document['file_size'])}, a bot może pobrać najwyżej "
                            "20 MB. Wyślij link: <code>/url LINK …</code> (Drive, Dropbox), plik pobierze się "
                            "w zadaniu.")
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
        if kind == "recipe":
            return self.open_draft(entry, name, params, files)
        self.queue(kind, entry, name, params, files)

    # --- files waiting for a decision (staging/<token>/, drafts in bot.json) ---------------------------

    def stage(self, files):
        token = f"{int(self.clock() * 1000) % 10 ** 9}"
        while token in self.state["drafts"]:
            token = str(int(token) + 1)
        folder = self.store.root / "staging" / token
        folder.mkdir(parents=True, exist_ok=True)
        for rel, data in files.items():
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_bytes(data)
        return token

    def staged(self, token):
        folder = self.store.root / "staging" / token
        return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def drop(self, token):
        self.state["drafts"].pop(token, None)
        if self.state.get("open_draft") == token:
            self.state["open_draft"] = None
        self.save()
        shutil.rmtree(self.store.root / "staging" / token, ignore_errors=True)

    def ask_what(self, name, caption, files):
        token = self.stage(files)
        self.state["drafts"][token] = {"recipe": None, "name": name, "caption": caption or ""}
        self.save()
        suffix = PurePosixPath(name).suffix.lower()
        options = [r for r in RECIPES.values() if suffix in r.inputs] or [r for r in RECIPES.values() if r.inputs]
        rows = [[(r.title, f"f:{token}:{r.name}")] for r in options] + [[("✖️ Nic", f"f:{token}:-")]]
        self.say(f"Co zrobić z <b>{esc(name)}</b>?", keyboard(rows))

    def open_draft(self, recipe, name, typed, files=None, fetch=(), token=None):
        """A recipe job waits on a card until Start; env needs no choices and goes straight in."""
        if recipe == "env":
            return self.queue("recipe", "env", name, typed, {})
        draft = drafts.new(recipe, name, typed, fetch)
        token = token or self.stage(files or {})
        self.state["drafts"][token] = draft
        self.state["open_draft"] = token
        self.save()
        self.say(*drafts.card(token, draft, self.web_url))

    def show(self, message_id, text, markup):
        try:
            self.api("editMessageText", {"chat_id": self.owner, "message_id": message_id, "text": text,
                                         "parse_mode": "HTML", "reply_markup": markup})
        except TelegramError:  # too old to edit, or unchanged: a new message does the job
            self.say(text, markup)

    def draft_tap(self, action, token, rest, message_id):
        draft = self.state["drafts"].get(token)
        if draft is None or draft.get("recipe") is None:
            return self.say("Ta karta już nie jest aktualna. Wyślij plik jeszcze raz.")
        if action == "c":
            self.drop(token)
            return self.show(message_id, "✖️ Anulowane.", keyboard([]))
        if action == "s":
            spec = self.queue("recipe", draft["recipe"], draft["name"], draft["params"], self.staged(token),
                              draft["fetch"])
            if spec is not None:
                self.drop(token)
            return
        if action == "p":
            return self.show(message_id, *drafts.choices(token, draft, rest[0]))
        if action == "v":
            draft = drafts.choose(draft, rest[0], int(rest[1]))
            self.state["drafts"][token] = draft
            self.state["open_draft"] = token
            self.save()
        self.show(message_id, *drafts.card(token, draft, self.web_url))

    def edit_open_draft(self, text):
        token = self.state.get("open_draft")
        draft = self.state["drafts"].get(token) if token else None
        if not draft or draft.get("recipe") is None:
            return self.say("Nie ma otwartej karty do zmiany. Wyślij plik albo komendę (/help).")
        draft = drafts.edit_text(draft, text)
        self.state["drafts"][token] = draft
        self.save()
        self.say(*drafts.card(token, draft, self.web_url))

    def queue(self, kind, entry, name, params, files, fetch=()):
        if kind == "recipe":
            recipe = RECIPES[entry]
            params = recipe.validate(params)
            if entry == "md" and not files and not fetch and not params["pdb"]:
                raise ParamError("Podaj kod PDB (/md 1E0L) albo wyślij plik .pdb.")
            if entry == "dock" and not files and not fetch and not params["receptor"]:
                raise ParamError("Dokowanie potrzebuje ligandów: wyślij zip z receptorem i ligandami albo plik "
                                 ".smi/.sdf z podpisem dock receptor=XXXX.")
        elif params:
            raise ParamError("Parametry klucz=wartość są tylko dla przepisów (md, qsar, omics, dock); "
                             "skrypt ustawia wszystko sam.")
        spec = self.store.create(kind, entry, name, params, files, fetch)
        ahead = sum(1 for job, s in self.store.statuses().items() if job != spec.id and s.state not in jobs.FINAL)
        where = "zaraz startuje" if not ahead else f"przed nim: {ahead}"
        what = RECIPES[entry].title if kind == "recipe" else {
            "script": "skrypt run.sh", "tpr": "mdrun", "notebook": "notebook", "python": "skrypt Python",
            "tool": "instalacja NAMD"}[kind]
        self.say(f"📥 <b>#{spec.id}</b> {esc(name)} – {esc(what)} w kolejce ({where}).")
        return spec

    def handle_tap(self, data, message_id):
        action, _, rest = data.partition(":")
        parts = rest.split(":")
        try:
            if action == "f" and len(parts) == 2:
                return self.chosen(parts[0], parts[1])
            if action in ("p", "v", "s", "c", "b"):
                return self.draft_tap(action, parts[0], parts[1:], message_id)
            if action == "j" and len(parts) == 2 and parts[0].isdigit():
                return self.handle_command(f"/{parts[1]} {parts[0]}")
            if action == "d" and len(parts) == 1:
                return self.remove_confirmed(parts[0])
            if action == "g":
                return self.show(message_id, *self.guide_view(rest))
            if action == "t":
                return self.try_example(rest)
            if action == "h":
                return self.help_command(rest)
            if action == "u" and rest == "install":
                return self.update_command()
        except ParamError as e:
            self.say(f"❌ {esc(e)}")

    def chosen(self, token, choice):
        """The owner said what an ambiguous file is for: a card for that recipe, or nothing."""
        pending = self.state["drafts"].get(token)
        if pending is None or pending.get("recipe") is not None:
            return self.say("Ten plik już nie czeka. Wyślij go jeszcze raz.")
        if choice == "-":
            self.drop(token)
            return self.say("Dobrze, nic nie robię.")
        _, typed = jobs.parse_caption(pending["caption"])
        self.open_draft(choice, pending["name"], typed, token=token)

    def handle_command(self, text):
        word, _, rest = text.partition(" ")
        word, rest = word[1:].split("@")[0].lower(), rest.strip()
        if word in RECIPES:
            return self.recipe_command(word, text, rest)
        general = {"start": self.start_command, "help": self.help_command, "menu": self.help_command,
                   "pomoc": self.help_command, "guide": self.guide_command, "przewodnik": self.guide_command,
                   "jobs": self.list_jobs, "kolejka": self.list_jobs,
                   "system": self.system_command, "co": self.report_command, "cisza": self.quiet_command,
                   "pauza": self.hold_command, "wznow": self.release_command, "url": self.url_command,
                   "dolicz": self.extend_command, "panel": self.panel_command, "funnel": self.funnel_command,
                   "update": self.update_command, "rollback": self.rollback_command}
        if word in general:
            return general[word](rest)
        if word == "usun" and rest == "stare":
            return self.remove_old_command()
        numbered = {"job": self.job_view, "log": self.log_command, "get": self.get_command,
                    "skrypt": self.script_command,
                    "cancel": self.cancel_command, "retry": self.retry_command, "top": self.top_command,
                    "usun": self.remove_command}
        if word not in numbered:
            return self.say("Nie znam tej komendy. /help")
        first, _, more = rest.partition(" ")
        if not first.isdigit() or int(first) not in self.store.ids():
            return self.say(f"Podaj numer zadania, np. /{word} 3. Lista: /jobs")
        numbered[word](int(first), more.strip())

    def help_command(self, rest):
        self.say(RECIPES[rest].usage() if rest in RECIPES else HELP)

    def start_command(self, _rest=""):
        rows = [[("📖 Przewodnik", "g:"), ("▶️ Wypróbuj: krótka dynamika", "t:md")]]
        if self.web_url:
            rows.append([("🧾 Panel", self.web_url)])
        self.say("🧪 <b>Kolejka obliczeń</b> na laptopie.\nWyślij plik albo komendę, ustaw kartę, naciśnij Start – "
                 "wyniki wrócą tutaj. Zacznij od przewodnika albo od gotowego przykładu. Ściąga komend: /help",
                 keyboard(rows))

    def guide_view(self, topic_id=""):
        """(text, markup) of one topic, or of the list of topics."""
        topic = guide.BY_ID.get(topic_id)
        if topic is None:
            buttons = [(t.title, f"g:{t.id}") for t in guide.TOPICS]
            rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
            return "📖 <b>Przewodnik</b> – wybierz temat:", keyboard(rows)
        rows = []
        if topic.example:
            rows.append([("▶️ Wypróbuj na przykładzie", f"t:{topic.example}")])
        if topic.recipe:
            rows.append([("⚙️ Wszystkie ustawienia", f"h:{topic.recipe}")])
        rows.append([("↩️ Tematy", "g:")])
        return f"<b>{topic.title}</b>\n\n{topic.text}", keyboard(rows)

    def guide_command(self, rest=""):
        self.say(*self.guide_view(rest.strip().lower()))

    def try_example(self, name):
        example = examples.EXAMPLES.get(name)
        if example is None:
            return self.say("Nie ma takiego przykładu.")
        self.say(f"🧪 {esc(example.note)}")
        self.open_draft(example.recipe, example.name, dict(example.typed), example.files())

    def recipe_command(self, word, text, rest):
        recipe = RECIPES[word]
        if recipe.needs_file and word != "md":
            return self.say(f"{esc(recipe.title)}: wyślij plik z podpisem <code>{word} …</code> albo link: "
                            f"<code>/url LINK {word} …</code>.\n\n" + recipe.usage())
        _, params = jobs.parse_caption(text)
        self.open_draft(word, word if word == "env" else f"{word} {rest}".strip(), params)

    def system_command(self, _rest=""):
        beat = self.store.runner_state()
        runner = (f"runner: {'działa' if self.runner_alive(beat) else 'NIE ODPOWIADA'}"
                  + (" · ⏸️ pauza" if self.store.hold else ""))
        self.say(self.system_info() + "\n" + runner + "\n" + self.queue_line())

    def report_command(self, rest):
        try:
            hours = float(rest.replace(",", "."))
        except ValueError:
            return self.say(f"Raport co {self.state['report_hours']:g} h. Zmiana: /co 2 (0 = wyłącz).")
        self.state["report_hours"] = max(0.0, hours)
        self.save()
        self.say("Raporty postępu wyłączone." if not hours else f"Raport postępu co {hours:g} h.")

    def quiet_command(self, rest):
        if rest == "off":
            self.state["quiet"] = None
        else:
            try:
                start, end = (int(x) % 24 for x in rest.split("-"))
            except ValueError:
                quiet = self.state["quiet"]
                now = f"{quiet[0]}–{quiet[1]}" if quiet else "wyłączona"
                return self.say(f"Cisza nocna (wiadomości bez dźwięku): {now}. Zmiana: /cisza 23-7, /cisza off.")
            self.state["quiet"] = [start, end]
        self.save()
        quiet = self.state["quiet"]
        self.say(f"🌙 Cisza nocna {quiet[0]}:00–{quiet[1]}:00." if quiet else "🔔 Cisza nocna wyłączona.")

    def hold_command(self, _rest=""):
        self.store.set_hold(True)
        self.say("⏸️ Pauza: bieżące zadanie zatrzymane, nowe nie startują. /wznow puszcza dalej.")

    def release_command(self, _rest=""):
        self.store.set_hold(False)
        self.say("▶️ Wznawiam kolejkę.")

    def url_command(self, rest):
        link, _, caption = rest.partition(" ")
        if not link:
            return self.say("Użycie: <code>/url LINK [qsar|omics|dock|md|skrypt] [klucz=wartość…]</code>\n"
                            "Google Drive (udostępnione: każdy z linkiem), Dropbox albo zwykły link; plik "
                            "pobierze się w zadaniu, więc może być duży.")
        word, params = jobs.parse_caption(caption)
        kind, entry, name = jobs.detect_url(link, word)
        fetch = [{"url": link, "name": name}]
        if kind == "recipe":
            return self.open_draft(entry, name, params, fetch=fetch)
        self.queue(kind, entry, name, params, {}, fetch=fetch)

    def panel_command(self, _rest=""):
        if not self.web_url:
            return self.say("Panel działa przez Tailscale: na laptopie <code>sudo bash deploy/remote.sh</code>, "
                            "potem tu wróć. Do tego czasu wszystko jest dostępne w czacie.")
        self.say("🧾 Panel: kolejka na żywo, wykresy, logi, parametry. Na telefonie musi działać Tailscale "
                 "(albo /funnel on).", keyboard([[("🧾 Otwórz panel", self.web_url)]]))

    def funnel_command(self, rest):
        if not self.web_url or self.exposure is None:
            return self.panel_command()
        from .exposure import ExposureError
        try:
            if rest in ("on", "off"):
                self.exposure.set_public(rest == "on")
            public = self.exposure.status()
        except ExposureError as e:
            return self.say(f"❌ {esc(e)}")
        if public:
            self.say("🌍 Panel publiczny (Funnel włączony): otworzysz go bez Tailscale. Wchodzi tylko podpisany "
                     "Telegram z Twoim kontem; obcy dostaje „brak dostępu”. /funnel off – z powrotem prywatnie.")
        else:
            self.say("🔒 Panel prywatny: tylko urządzenia w Twoim Tailscale. /funnel on – publicznie.")

    def extend_command(self, rest):
        """/dolicz N NS: a finished md run goes on for NS more nanoseconds, sharing its files (hard links)."""
        words = rest.replace(",", ".").split()
        if len(words) != 2 or not words[0].isdigit() or int(words[0]) not in self.store.ids():
            return self.say("Użycie: /dolicz N ILE_NS, np. /dolicz 4 5 – zadanie #4 o 5 ns dłużej.")
        child = actions.extend(self.store, int(words[0]), words[1])
        self.say(f"➕ <b>#{child.id}</b>: #{words[0]} o {child.params['extend_ns']:g} ns dłużej, w kolejce.")

    def script_command(self, job, _rest=""):
        """A GROMACS md job as a zip to edit: inputs, .mdp files and run.sh with the same md_step stages."""
        name, data = actions.script_zip(self.store, job)
        self.send_file(name, data,
                       f"📝 #{job} jako skrypt: zmień pliki .mdp (albo run.sh) i odeślij ten zip – policzy się "
                       "od minimalizacji z Twoimi ustawieniami. /guide skrypty")

    def log_command(self, job, _rest=""):
        self.say(f"📜 <b>#{job}</b>\n<pre>{esc(self.store.log_tail(job, 3500))}</pre>")

    def cancel_command(self, job, _rest=""):
        if self.store.status(job).state in jobs.FINAL:
            return self.say(f"#{job} już się skończyło.")
        self.store.request_cancel(job)
        self.say(f"⏹️ Zatrzymuję #{job}…")

    def retry_command(self, job, rest=""):
        """The same files again; key=value after the number changes those parameters."""
        _, changes = jobs.parse_caption(rest)
        again = actions.retry(self.store, job, changes)
        self.say(f"🔁 #{job} jeszcze raz jako <b>#{again.id}</b>.")

    def top_command(self, job, _rest=""):
        actions.top(self.store, job)
        self.say(f"⏫ #{job} będzie następne.")

    def remove_command(self, job, _rest=""):
        if self.store.status(job).state not in jobs.FINAL:
            return self.say(f"#{job} jeszcze się nie skończyło: najpierw /cancel {job}.")
        size = sum(s for _, s in self.store.outputs(job))
        self.say(f"🗑️ Usunąć #{job} ({esc(self.store.spec(job).name)}, {size_text(size)}) razem z wynikami? "
                 "Tego nie da się cofnąć.", keyboard([[("🗑️ Tak, usuń", f"d:{job}"), ("Zostaw", "x")]]))

    def remove_old_command(self, days=7):
        now = self.clock()
        old = actions.old_finished(self.store, days, now)
        if not old:
            return self.say(f"Nie ma zadań skończonych ponad {days} dni temu.")
        size = sum(s for job in old for _, s in self.store.outputs(job))
        listed = ", ".join(f"#{job}" for job in old)
        self.say(f"🗑️ Usunąć {len(old)} zadań starszych niż {days} dni ({listed}; {size_text(size)})? "
                 "Tego nie da się cofnąć.",
                 keyboard([[("🗑️ Tak, usuń", "d:" + ",".join(map(str, old))), ("Zostaw", "x")]]))

    def remove_confirmed(self, listed):
        removed = []
        for part in listed.split(","):
            if part.isdigit() and int(part) in self.store.ids():
                try:
                    self.store.remove(int(part))
                    removed.append(f"#{part}")
                except ParamError as e:
                    self.say(f"❌ {esc(e)}")
        self.say("🗑️ Usunięte: " + ", ".join(removed) if removed else "Nic nie usunąłem.")

    # --- views --------------------------------------------------------------------------------------

    def queue_line(self):
        states = [s.state for s in self.store.statuses().values()]
        return (f"⚙️ liczy: {states.count('running') + states.count('paused')} · "
                f"⏳ czeka: {states.count('queued')} · wersja {VERSION}")

    def list_jobs(self, _rest=""):
        lines = []
        for job in self.store.ids()[-12:]:
            spec, status = self.store.spec(job), self.store.status(job)
            extra = f" – {esc(status.progress)}" if status.state in ("running", "paused") and status.progress else ""
            lines.append(f"{ICONS[status.state]} <b>#{job}</b> {esc(spec.name)}{extra}")
        self.say("\n".join(lines) or "Kolejka jest pusta.")

    def job_view(self, job, _rest=""):
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

    def get_command(self, job, rel=""):
        if not rel:
            return self.send_results(job)
        if rel.endswith(" czesci"):
            return self.send_parts(job, rel[:-len(" czesci")].strip())
        path = self.store.result_file(job, rel)
        if path is None:
            return self.say(f"Nie ma pliku {esc(rel)} w #{job}.")
        if path.stat().st_size > MAX_UPLOAD:
            return self.say(*self.big_files(job, [(rel, path.stat().st_size)]))
        self.send_file(path.name, path.read_bytes(), f"#{job}")

    def big_files(self, job, files):
        """How to get files Telegram won't carry: a link through the panel's address, scp, or pieces."""
        lines = ["📦 Za duże na Telegram (limit 50 MB):"]
        buttons = []
        for rel, size in files[:8]:
            lines.append(f"• <b>{esc(rel)}</b> ({size_text(size)})")
            if self.ssh_target:
                where = self.store.work(job) / rel
                lines.append(f"  <code>scp {esc(self.ssh_target)}:{esc(shlex.quote(str(where)))} .</code>")
            if self.web_url:
                token = links.make(self.link_secret, job, rel, self.clock() + links.DAY)
                buttons.append([(f"⬇️ {PurePosixPath(rel).name} ({size_text(size)})",
                                 f"url:{self.web_url.rstrip('/')}/dl/{token}")])
            elif not self.ssh_target and size <= MAX_PARTS_TOTAL:
                lines.append(f"  <code>/get {job} {esc(rel)} czesci</code> – w kawałkach po 45 MB")
        if len(files) > 8:
            lines.append(f"… i {len(files) - 8} innych: /job {job}")
        if self.web_url:
            lines.append("Linki działają 24 h, na urządzeniu z Tailscale (albo po /funnel on); przerwane pobieranie "
                         "można wznowić.")
        if self.ssh_target:
            lines.append(f"Cały katalog: <code>scp -r {esc(self.ssh_target)}:{esc(str(self.store.work(job)))} .</code>")
        if not self.web_url and not self.ssh_target:
            lines.append("Wygodniej: <code>sudo bash deploy/remote.sh</code> na laptopie – wtedy dostajesz link do "
                         "pobrania i scp.")
        return "\n".join(lines), keyboard(buttons) if buttons else None

    def send_parts(self, job, rel):
        path = self.store.result_file(job, rel)
        if path is None:
            return self.say(f"Nie ma pliku {esc(rel)} w #{job}.")
        size = path.stat().st_size
        if size > MAX_PARTS_TOTAL:
            return self.say(f"{esc(rel)} ma {size_text(size)}: to za dużo na kawałki przez Telegram. Na laptopie "
                            "<code>sudo bash deploy/remote.sh</code> da link do pobrania i scp.")
        count = -(-size // PART)
        with open(path, "rb") as source:
            for number in range(1, count + 1):
                self.send_file(f"{path.name}.part{number:02d}", source.read(PART), f"#{job} {number}/{count}")
        self.say(f"Sklej w folderze z kawałkami:\n<code>cat {esc(path.name)}.part* &gt; {esc(path.name)}</code>\n"
                 f"(Windows: <code>copy /b {esc(path.name)}.part* {esc(path.name)}</code>)")

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
        changed |= self.check_runner()
        changed |= self.check_for_update(now)
        for closed in (self.store.root / "staging").glob(f"*/{actions.CLOSED}"):
            self.drop(closed.parent.name)  # started or discarded in the Mini App (its own cards too)
        if changed:
            self.save()

    def check_for_update(self, now):
        """Once a day: tell about a new release once, with a button that installs it."""
        if self.updater is None or now - self.state.get("update_checked", 0) < 86400:
            return False
        self.state["update_checked"] = now
        from .updater import UpdateError, version_tuple
        try:
            tag = self.updater.latest()
        except UpdateError:
            return True
        if version_tuple(tag[1:]) > version_tuple(VERSION) and self.state.get("update_told") != tag:
            self.state["update_told"] = tag
            self.say(f"⬆️ Jest nowa wersja {esc(tag)} (masz {VERSION}). Trwające zadanie nie ucierpi: runner "
                     "przełączy się po jego końcu.", keyboard([[("⬆️ Zainstaluj", "u:install")]]))
        return True

    def runner_alive(self, beat):
        return bool(beat) and self.clock() - beat.get("time", 0) < 180

    def check_runner(self):
        """Say once when the runner is silent with work waiting, or can't start for lack of disk, and when it's over."""
        beat = self.store.runner_state()
        waiting = any(s.state == "queued" for s in self.store.statuses().values())
        if waiting and not self.runner_alive(beat):
            alert = "dead"
        elif beat.get("blocked") == "disk":
            alert = "disk"
        else:
            alert = ""
        before = self.state.get("runner_alert", "")
        if alert == before:
            return False
        self.state["runner_alert"] = alert
        if alert == "dead":
            self.say("⚠️ Runner nie odpowiada od kilku minut, a zadania czekają. Laptop śpi albo usługa padła: "
                     "<code>systemctl status compute-runner</code>.")
        elif alert == "disk":
            self.say(f"💾 Za mało miejsca na dysku ({size_text(beat.get('free', 0))} wolne), nowe zadania czekają. "
                     "Zwolnij miejsce: /usun stare albo /usun N.")
        elif before == "dead":
            self.say("✅ Runner znowu działa.")
        elif before == "disk":
            self.say("✅ Miejsce na dysku jest, kolejka rusza.")
        return True

    def announce(self, job, status, before):
        spec = self.store.spec(job)
        title = f"<b>#{job}</b> {esc(spec.name)}"
        old_state, _, old_attempt = (before or "").partition(":")
        if status.state == "running":
            if old_state == "paused" and str(status.attempt) == old_attempt:
                self.say(f"▶️ {title}: liczę dalej.")
            elif status.attempt > 1:
                self.say(f"🔁 {title} wznowione po przerwie (próba {status.attempt}).")
            else:
                self.say(f"▶️ {title} – start.")
        elif status.state == "paused" and status.pause == "hold":
            self.say(f"⏸️ {title} wstrzymane (/wznow).")
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
        rows = [[("📜 Log", f"j:{job}:log"), ("🔁 Ponów", f"j:{job}:retry"), ("🗑️ Usuń", f"j:{job}:usun")]]
        if (work / "gromacs" / "run.sh").exists():
            rows.append([("📝 Zrób z tego skrypt", f"j:{job}:skrypt")])
        self.say((header + "\n\n" if header else "") + f"<pre>{esc(body)}</pre>", keyboard(rows))
        highlights = work / "highlights.txt"
        if highlights.exists():
            for rel in highlights.read_text().split()[:4]:
                path = self.store.result_file(job, rel)
                if path and path.stat().st_size < 10 * 1024 ** 2:
                    self.api("sendPhoto", self.fields(caption=esc(rel)),
                             files={"photo": (path.name, path.read_bytes())})
        archive, left_out = self.results_zip(job)
        self.send_file(f"wyniki_{job:04d}.zip", archive, f"#{job}")
        small = [(rel, size) for rel, size in left_out if size <= MAX_UPLOAD]
        big = [(rel, size) for rel, size in left_out if size > MAX_UPLOAD]
        if small:
            shown = "\n".join(f"<code>/get {job} {esc(rel)}</code> ({size_text(size)})" for rel, size in small[:10])
            more = f"\n… i {len(small) - 10} innych" if len(small) > 10 else ""
            self.say(f"📎 Nie zmieściły się w zipie, pobierz osobno:\n{shown}{more}")
        if big:
            self.say(*self.big_files(job, big))

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

    def update_command(self, _rest=""):
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

    def rollback_command(self, _rest=""):
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
            {"command": "panel", "description": "Panel (Mini App)"},
            {"command": "guide", "description": "Przewodnik i przykłady"},
            {"command": "help", "description": "Ściąga komend"}])})
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
    from .exposure import Exposure
    web_url = os.environ.get("WEB_URL") or None  # written by deploy/remote.sh once Tailscale serves the panel
    bot = Bot(store, api, int(owner), home / "bot.json", updater=Updater(home / "update"),
              system_info=lambda: system_info(home), web_url=web_url, link_secret=token,
              ssh_target=os.environ.get("SSH_TARGET") or None,
              exposure=Exposure(int(os.environ.get("WEB_PORT", "8787"))))
    if web_url:
        try:  # the "Panel" button next to the message field, like BotFather's Open
            api("setChatMenuButton", {"chat_id": owner, "menu_button": json.dumps(
                {"type": "web_app", "text": "Panel", "web_app": {"url": web_url}})})
        except TelegramError as e:
            log(f"menu button: {e}")
    try:
        run(api, bot)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
