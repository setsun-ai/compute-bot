"""Download links for results too big for Telegram: a signed, expiring capability, served with resume."""
import threading
import urllib.error
import urllib.request

import pytest

from compute_bot import jobs, links, web

TOKEN, OWNER = "123:abc", 100


def test_a_link_opens_its_file_until_it_expires_and_cannot_be_altered():
    token = links.make(TOKEN, 7, "gromacs/md.xtc", expires=1000.0)
    assert links.parse(TOKEN, token, now=999.0) == (7, "gromacs/md.xtc")
    assert links.parse(TOKEN, token, now=1001.0) is None
    assert links.parse("999:other", token, now=999.0) is None
    assert links.parse(TOKEN, token[:-2] + "xx", now=999.0) is None
    assert links.parse(TOKEN, "garbage", now=0.0) is None
    other = links.make(TOKEN, 8, "gromacs/md.xtc", expires=1000.0)
    body, sig = token.split(".")
    assert links.parse(TOKEN, other.split(".")[0] + "." + sig, now=999.0) is None


@pytest.fixture
def server(tmp_path):
    store = jobs.Store(tmp_path)
    store.create("script", "run.sh", "a.zip", {}, {"run.sh": b"true"})
    (store.work(1) / "big.xtc").write_bytes(bytes(range(256)) * 4000)  # ~1 MB
    app = web.App(store, TOKEN, OWNER, system_info=dict)
    httpd = web.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield store, f"http://127.0.0.1:{httpd.server_port}", app
    httpd.shutdown()


def test_download_without_telegram_whole_and_resumed(server):
    store, base, app = server
    url = base + "/dl/" + links.make(TOKEN, 1, "big.xtc", expires=app.clock() + 60)
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read()
        assert response.headers["Content-Disposition"].endswith('filename="big.xtc"')
        assert response.headers["Accept-Ranges"] == "bytes"
    assert data == (store.work(1) / "big.xtc").read_bytes()
    request = urllib.request.Request(url, headers={"Range": "bytes=1000000-"})
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 206 and response.read() == data[1000000:]


def test_expired_forged_or_escaping_links_get_nothing(server):
    store, base, app = server
    for token in (links.make(TOKEN, 1, "big.xtc", expires=app.clock() - 1),
                  links.make("9:x", 1, "big.xtc", expires=app.clock() + 60),
                  links.make(TOKEN, 1, "../spec.json", expires=app.clock() + 60),
                  links.make(TOKEN, 9, "big.xtc", expires=app.clock() + 60)):
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/dl/" + token, timeout=5)
        assert error.value.code == 404
