"""The Mini App page (web/index.html): one self-contained file that carries no data and only calls routes web.py has."""
import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from compute_bot import jobs, web
from compute_bot.recipes import ParamError

PAGE = web.PAGE.read_text(encoding="utf-8")
TELEGRAM_JS = "https://telegram.org/js/telegram-web-app.js"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.inline, self.tags, self.links = [], [], [], []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if tag == "script":
            self.scripts.append(attrs)
            self._in_script = "src" not in attrs
        if tag in ("link", "img", "iframe", "object", "embed", "source") and (attrs.get("href") or attrs.get("src")):
            self.links.append(attrs.get("href") or attrs.get("src"))

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.inline.append(data)


@pytest.fixture(scope="module")
def page():
    parsed = Page()
    parsed.feed(PAGE)
    parsed.close()
    return parsed


def test_page_parses_as_one_html_document(page):
    assert PAGE.lstrip().startswith("<!DOCTYPE html>")
    for tag in ("html", "head", "body", "main", "style", "script"):
        assert tag in page.tags
    assert page.inline and "'use strict'" in page.inline[0]


def test_only_external_script_is_telegrams_and_nothing_else_is_loaded_from_outside(page):
    external = [s["src"] for s in page.scripts if "src" in s]
    assert external == [TELEGRAM_JS]
    assert page.links == []  # no fonts, stylesheets, images or frames from anywhere
    others = set(re.findall(r"https?://[^\s'\"`)<>]+", PAGE)) - {TELEGRAM_JS}
    assert not others, others
    assert "@import" not in PAGE and "url(" not in PAGE


def test_page_carries_no_data(page):
    assert not any(s.get("type") in ("application/json", "application/ld+json") for s in page.scripts)
    assert not re.search(r"\bdata:[a-z]+/", PAGE)  # no embedded data: URIs
    for forbidden in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
        assert forbidden not in PAGE  # nothing kept on the phone between openings
    # the signature goes in the header, never in the URL
    assert "'X-Init-Data': initData" in PAGE
    assert not re.search(r"[?&](initData|tgWebAppData|hash)=", PAGE)


API_CALL = re.compile(r"api\('(GET|POST)',\s*(['`])(/api/.*?)\2")


def page_calls():
    calls = [(m[1], m[3]) for m in API_CALL.finditer(PAGE)]
    # every /api string in the page goes through api() (the one place that adds the header)
    assert len(calls) == PAGE.count("'/api/") + PAGE.count("`/api/"), "an /api path outside api()"
    assert "fetch(" in PAGE and PAGE.count("fetch(") == 1
    return calls


def web_route_words():
    """String constants App.get / App.post compare path parts with: the routing vocabulary of web.py."""
    tree = ast.parse(Path(web.__file__).read_text())
    app = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "App")
    words = set()
    for fn in app.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("get", "post", "job_id"):
            for node in ast.walk(fn):
                if isinstance(node, ast.Compare):
                    for c in ast.walk(node):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            words.add(c.value)
    return words


def test_every_path_word_the_page_uses_is_in_web_py_routing():
    words = web_route_words()
    assert {"state", "recipes", "drafts", "jobs", "log", "file", "hold",
            "cancel", "top", "retry", "send", "extend"} <= words
    for _, path in page_calls():
        static = [p for p in path.split("?")[0].strip("/").split("/")[1:] if "${" not in p]
        missing = [p for p in static if p not in words and p != "start"]  # drafts/TOKEN/<anything> starts the card
        assert not missing, (path, missing)


@pytest.fixture
def app(tmp_path):
    store = jobs.Store(tmp_path)
    store.create("recipe", "md", "1E0L", jobs.RECIPES["md"].validate({"pdb": "1E0L"}), {})
    store.set_status(1, jobs.Status("done", attempt=1))
    (store.work(1) / "a.txt").write_text("x")
    (tmp_path / "staging" / "777").mkdir(parents=True)
    (tmp_path / "staging" / "777" / "x.pdb").write_text("ATOM")
    (tmp_path / "bot.json").write_text(json.dumps({"drafts": {"777": {"recipe": "md", "name": "x.pdb", "params": {}}}}))
    a = web.App(store, "1:a", 1, system_info=lambda: {})
    a.send_document = lambda *args: None
    return a


def concrete(path):
    """A page path with its ${…} parts filled: job 1, card 777, file a.txt, log 100 bytes."""
    path = re.sub(r"/jobs/\$\{[^}]+\}", "/jobs/1", path)
    path = re.sub(r"/drafts/\$\{[^}]+\}", "/drafts/777", path)
    path = re.sub(r"path=\$\{[^}]+\}", "path=a.txt", path)
    path = re.sub(r"bytes=\$\{[^}]+\}", "bytes=100", path)
    assert "${" not in path, path
    url, _, query = path.partition("?")
    return url, dict(p.split("=", 1) for p in query.split("&") if p)


def test_every_api_call_of_the_page_is_routed_by_the_app(app):
    calls = page_calls()
    assert {(m, concrete(p)[0]) for m, p in calls} >= {
        ("GET", "/api/state"), ("GET", "/api/drafts"), ("GET", "/api/recipes"), ("GET", "/api/jobs/1"),
        ("GET", "/api/jobs/1/log"), ("GET", "/api/jobs/1/file"), ("POST", "/api/hold"), ("POST", "/api/jobs/1/cancel"),
        ("POST", "/api/jobs/1/top"), ("POST", "/api/jobs/1/retry"), ("POST", "/api/jobs/1/send"),
        ("POST", "/api/jobs/1/extend"), ("POST", "/api/drafts/777/start")}
    for method, path in calls:
        url, query = concrete(path)
        try:
            if method == "GET":
                app.get(url, query)
            else:
                app.post(url, {"path": "a.txt", "ns": 1, "on": False, "params": {}})
        except LookupError:
            pytest.fail(f"{method} {url} is not routed by web.App")
        except (ParamError, FileExistsError):
            pass  # routed; refused for this job's state, which is the page's business to show
