"""The Mini App server as Telegram's in-app browser calls it: only the owner's signed initData gets anything."""
import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from compute_bot import actions, jobs, web

TOKEN, OWNER = "123:abc", 100


def signed(user_id=OWNER, auth_date=None, token=TOKEN):
    fields = {"auth_date": str(int(auth_date or time.time())), "query_id": "q1",
              "user": json.dumps({"id": user_id, "first_name": "N"})}
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urllib.parse.urlencode(fields)


# --- the signature ---------------------------------------------------------------------------------------

def test_valid_owner_signature_passes():
    assert web.check_init_data(signed(), TOKEN, OWNER, time.time()) is True


@pytest.mark.parametrize("init_data", [
    signed(user_id=666),                         # someone else's Telegram
    signed(token="999:other"),                   # signed for another bot
    signed(auth_date=time.time() - 3 * 86400),   # too old
    signed().replace("query_id=q1", "query_id=q2"),  # changed after signing
    "", "hash=abc", "garbage%%%",
])
def test_anything_else_fails(init_data):
    assert web.check_init_data(init_data, TOKEN, OWNER, time.time()) is False


# --- the server ----------------------------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    store = jobs.Store(tmp_path)
    app = web.App(store, TOKEN, OWNER, system_info=lambda: {"temp": 55.0})
    httpd = web.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield store, f"http://127.0.0.1:{httpd.server_port}", app
    httpd.shutdown()


def call(base, path, body=None, init=None, raw=False):
    headers = {"X-Init-Data": signed() if init is None else init}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read()
            return response.status, payload if raw else json.loads(payload)
    except urllib.error.HTTPError as e:
        return e.code, None


def test_page_is_served_without_data_and_api_needs_the_signature(server):
    store, base, _ = server
    store.create("script", "run.sh", "secret-name.zip", {}, {"run.sh": b"true"})
    with urllib.request.urlopen(base + "/", timeout=5) as response:
        page = response.read().decode()
    assert "<html" in page.lower() and "secret-name" not in page
    assert call(base, "/api/state", init="")[0] == 401
    assert call(base, "/api/state", init=signed(user_id=666))[0] == 401


def test_state_shows_jobs_runner_and_system(server):
    store, base, _ = server
    store.create("recipe", "md", "1E0L", {"pdb": "1E0L"}, {})
    store.set_status(1, jobs.Status("running", attempt=1, started=1.0, progress="NPT 40%"))
    store.set_runner_state({"time": time.time(), "job": 1, "blocked": "", "free": 10 ** 10})
    code, state = call(base, "/api/state")
    assert code == 200
    (job,) = state["jobs"]
    assert (job["id"], job["state"], job["progress"], job["params"]["pdb"]) == (1, "running", "NPT 40%", "1E0L")
    assert state["runner"]["alive"] is True and state["system"]["temp"] == 55.0 and state["hold"] is False


def test_job_detail_log_and_files(server):
    store, base, _ = server
    store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    (store.work(1) / "wyniki").mkdir()
    (store.work(1) / "wyniki/rmsd.png").write_bytes(b"\x89PNG")
    (store.work(1) / "summary.txt").write_text("RMSD 0.2 nm")
    (store.work(1) / "highlights.txt").write_text("wyniki/rmsd.png\n")
    store.log_path(1).write_text("hello log")
    code, detail = call(base, "/api/jobs/1")
    assert detail["summary"] == "RMSD 0.2 nm" and detail["highlights"] == ["wyniki/rmsd.png"]
    assert {"path": "wyniki/rmsd.png", "size": 4} in detail["files"]
    assert call(base, "/api/jobs/1/log")[1]["text"] == "hello log"
    assert call(base, "/api/jobs/1/file?path=wyniki/rmsd.png", raw=True)[1] == b"\x89PNG"
    assert call(base, "/api/jobs/1/file?path=../spec.json", raw=True)[0] == 404
    assert call(base, "/api/jobs/9")[0] == 404


def test_recipes_describe_every_parameter(server):
    _, base, _ = server
    code, recipes = call(base, "/api/recipes")
    md = recipes["md"]["params"]
    assert md["T"]["kind"] == "float" and md["T"]["default"] == 300.0 and "300" in md["T"]["quick"]
    assert md["ff"]["choices"][0] == "amber99sb-ildn"


def test_actions_cancel_hold_top_retry_extend(server):
    store, base, _ = server
    store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    store.create("recipe", "md", "1E0L", RECIPES_MD := jobs.RECIPES["md"].validate({"pdb": "1E0L"}), {})
    assert call(base, "/api/jobs/1/cancel", {})[0] == 200 and store.cancel_requested(1)
    call(base, "/api/hold", {"on": True})
    assert store.hold
    assert call(base, "/api/jobs/2/top", {})[0] == 200 and store.priorities() == {2: 1}
    store.set_status(2, jobs.Status("done", attempt=1))
    code, made = call(base, "/api/jobs/2/retry", {"params": {"T": "500"}})
    assert code == 200 and store.spec(made["id"]).params["T"] == 500.0 and RECIPES_MD["T"] == 300.0
    code, made = call(base, "/api/jobs/2/extend", {"ns": 2})
    assert code == 200 and store.spec(made["id"]).params["extend_ns"] == 2.0
    code, _ = call(base, "/api/jobs/1/retry", {"params": {"T": "500"}})
    assert code == 400  # scripts have no parameters


def test_bad_parameters_come_back_as_a_message(server):
    store, base, _ = server
    store.create("recipe", "md", "1E0L", jobs.RECIPES["md"].validate({"pdb": "1E0L"}), {})
    store.set_status(1, jobs.Status("done", attempt=1))
    request = urllib.request.Request(base + "/api/jobs/1/retry", data=json.dumps({"params": {"T": "hot"}}).encode(),
                                     headers={"X-Init-Data": signed(), "Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 400 and "T=hot" in json.loads(error.value.read())["error"]


def test_start_a_draft_from_the_panel_and_the_bot_cleans_up(server, tmp_path):
    store, base, _ = server
    folder = tmp_path / "staging" / "777"
    folder.mkdir(parents=True)
    (folder / "1e0l.pdb").write_bytes(b"ATOM")
    (tmp_path / "bot.json").write_text(json.dumps({"drafts": {"777": {
        "recipe": "md", "name": "1e0l.pdb", "params": {"T": "600"}, "fetch": []}}}))
    code, drafts = call(base, "/api/drafts")
    assert drafts["777"]["params"]["T"] == "600"
    code, made = call(base, "/api/drafts/777/start", {"params": {"T": "450", "ns": "2"}})
    assert code == 200
    spec = store.spec(made["id"])
    assert (spec.params["T"], spec.params["ns"]) == (450.0, 2.0)
    assert (store.work(made["id"]) / "1e0l.pdb").read_bytes() == b"ATOM"
    assert (folder / actions.CLOSED).exists()
    assert call(base, "/api/drafts/777/start", {"params": {}})[0] == 409  # never twice


def test_repeated_bad_signatures_get_rate_limited(server):
    _, base, app = server
    codes = [call(base, "/api/state", init="hash=bad")[0] for _ in range(web.MAX_FAILURES + 3)]
    assert codes[0] == 401 and codes[-1] == 429
    assert call(base, "/api/state")[0] == 429  # even the right one waits out the block from that address


def test_a_file_can_be_sent_to_the_chat(server):
    store, base, app = server
    sent = []
    app.send_document = lambda name, data, caption: sent.append((name, data, caption))
    store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    (store.work(1) / "md.xtc").write_bytes(b"traj")
    assert call(base, "/api/jobs/1/send", {"path": "md.xtc"})[0] == 200
    assert sent == [("md.xtc", b"traj", "#1")]
    assert call(base, "/api/jobs/1/send", {"path": "../spec.json"})[0] == 404


def test_rate_limit_follows_the_address_the_proxy_added_not_the_one_the_client_wrote(server):
    _, base, _ = server
    for i in range(web.MAX_FAILURES + 2):
        request = urllib.request.Request(base + "/api/state", headers={
            "X-Init-Data": "hash=bad", "X-Forwarded-For": f"10.0.0.{i}, 100.64.0.9"})
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as e:
            last = e.code
    assert last == 429


def test_state_has_server_time_heartbeat_attempts_and_queue_order(server):
    store, base, _ = server
    for name in ("a.zip", "b.zip"):
        store.create("script", "run.sh", name, {}, {"run.sh": b"true"})
    store.set_priority(2, 1)
    store.set_runner_state({"time": 123.0, "job": None, "blocked": ""})
    _, state = call(base, "/api/state")
    assert state["now"] > 0 and state["runner"]["time"] == 123.0 and state["max_attempts"] == jobs.MAX_ATTEMPTS
    assert state["queue"] == [2, 1]


def test_remove_a_finished_job_and_old_ones(server):
    store, base, app = server
    for _ in range(3):
        store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    store.set_status(1, jobs.Status("done", attempt=1, finished=app.clock() - 10 * 86400))
    store.set_status(2, jobs.Status("done", attempt=1, finished=app.clock() - 3600))
    assert call(base, "/api/jobs/3/remove", {})[0] == 400  # still queued
    code, removed = call(base, "/api/cleanup", {"days": 7})
    assert code == 200 and removed["removed"] == [1] and store.ids() == [2, 3]
    assert call(base, "/api/jobs/2/remove", {})[0] == 200 and store.ids() == [3]


def test_discard_a_card(server, tmp_path):
    store, base, _ = server
    (tmp_path / "staging" / "55").mkdir(parents=True)
    (tmp_path / "bot.json").write_text(json.dumps({"drafts": {"55": {"recipe": "md", "name": "x", "params": {},
                                                                     "fetch": []}}}))
    assert call(base, "/api/drafts/55/discard", {})[0] == 200
    assert call(base, "/api/drafts")[1] == {}
    assert call(base, "/api/drafts/55/start", {"params": {}})[0] == 409


def test_panel_can_ask_for_a_download_link(server):
    store, base, app = server
    store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    (store.work(1) / "big.xtc").write_bytes(b"traj")
    code, made = call(base, "/api/jobs/1/link", {"path": "big.xtc"})
    assert code == 200 and made["url"].startswith("/dl/")
    with urllib.request.urlopen(base + made["url"], timeout=5) as response:  # no Telegram header needed
        assert response.read() == b"traj"
    assert call(base, "/api/jobs/1/link", {"path": "../spec.json"})[0] == 404


def test_panel_can_open_an_example_as_a_card_the_bot_then_shows(server, tmp_path):
    store, base, _ = server
    code, made = call(base, "/api/examples/qsar/draft", {})
    assert code == 200
    token = made["token"]
    assert (tmp_path / "staging" / token / "przyklad_qsar.csv").exists()
    _, drafts = call(base, "/api/drafts")
    assert drafts[token]["recipe"] == "qsar" and drafts[token]["params"] == {"max_desc": "2"}
    code, started = call(base, f"/api/drafts/{token}/start", {"params": {}})
    assert code == 200 and store.spec(started["id"]).entry == "qsar"
    assert call(base, "/api/examples/nope/draft", {})[0] == 404


def test_guide_is_served(server):
    _, base, _ = server
    code, topics = call(base, "/api/guide")
    assert code == 200 and {"start", "md", "wyniki"} <= {t["id"] for t in topics}
