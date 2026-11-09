"""
The parameter card: a recipe job before it starts. The owner sees the values, changes them with buttons or by
typing key=value, and presses Start. Pure functions; the bot keeps drafts in bot.json and the files in staging/.

A draft is a dict: {"recipe", "name", "params" (as typed, strings), "fetch"}. Its parameters are always valid:
an edit that wouldn't validate is refused, so Start never fails on a parameter.
"""
from __future__ import annotations

import html
import json

from . import jobs
from .recipes import QUICK, RECIPES, quick_values

PER_ROW = 3


def button(text, data):
    """A tap goes back to the bot; an https address opens the Mini App instead."""
    if data.startswith("https://"):
        return {"text": text, "web_app": {"url": data}}
    return {"text": text, "callback_data": data}


def keyboard(rows):
    return json.dumps({"inline_keyboard": [[button(t, d) for t, d in row] for row in rows if row]})


def new(recipe, name, typed, fetch=()):
    """A draft from a caption's key=value; ParamError when they don't fit the recipe."""
    RECIPES[recipe].validate(typed)
    return {"recipe": recipe, "name": name, "params": dict(typed), "fetch": list(fetch)}


def values(draft):
    return RECIPES[draft["recipe"]].validate(draft["params"])


def shown(value):
    if isinstance(value, bool):
        return "tak" if value else "nie"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ",".join(map(str, value))
    return str(value) if value not in (None, "") else "—"


def card(token, draft, web_url=None):
    recipe = RECIPES[draft["recipe"]]
    current = values(draft)
    quick = QUICK.get(draft["recipe"], {})
    keys = list(quick) + [k for k in draft["params"] if k not in quick]
    lines = [f"🧾 <b>{html.escape(recipe.title)}</b> – {html.escape(draft['name'])}", ""]
    for key in keys:
        mark = "✏️ " if key in draft["params"] else ""
        lines.append(f"{mark}<code>{key}</code> = <b>{html.escape(shown(current[key]))}</b> "
                     f"<i>{html.escape(recipe.params[key].help)}</i>")
    lines += ["", f"Zmień przyciskiem albo napisz np. <code>{_example(draft['recipe'])}</code>. "
                  f"Wszystkie parametry: /help {draft['recipe']}"]
    buttons = [(f"{key}: {shown(current[key])}", f"p:{token}:{key}") for key in QUICK.get(draft["recipe"], {})]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([("▶️ Start", f"s:{token}"), ("✖️ Anuluj", f"c:{token}")])
    if web_url:
        rows.append([("🧾 Wszystkie parametry w panelu", web_url)])
    return "\n".join(lines), keyboard(rows)


def _example(recipe):
    return {"md": "T=450 ns=2", "qsar": "target=LogKd max_desc=3", "omics": "group=group ref=CONTROL",
            "dock": "ligand_ref=BTN exhaustiveness=16", "charmm": "ns=5"}.get(recipe, "klucz=wartość")


def choices(token, draft, key):
    options = quick_values(draft["recipe"], key)
    current = shown(values(draft)[key])
    buttons = [(("• " if v == current else "") + v, f"v:{token}:{key}:{i}") for i, v in enumerate(options)]
    rows = [buttons[i:i + PER_ROW] for i in range(0, len(buttons), PER_ROW)] + [[("↩️ Wróć", f"b:{token}")]]
    help_text = RECIPES[draft["recipe"]].params[key].help
    return f"<code>{key}</code>: {html.escape(help_text)}\nInna wartość: napisz <code>{key}=…</code>", keyboard(rows)


def choose(draft, key, index):
    return edit(draft, {key: quick_values(draft["recipe"], key)[index]})


def edit(draft, typed):
    """Changed parameters; ParamError (and the draft unchanged) when the new values don't validate."""
    params = {**draft["params"], **typed}
    RECIPES[draft["recipe"]].validate(params)
    return {**draft, "params": params}


def edit_text(draft, text):
    word, typed = jobs.parse_caption(text)
    if word is not None:
        raise jobs.ParamError(f"Na karcie zmieniam tylko parametry klucz=wartość, a nie „{word}”.")
    return edit(draft, typed)
