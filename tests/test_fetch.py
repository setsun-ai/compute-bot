"""Links people paste: share pages must become the file itself."""
from compute_bot.fetch import direct_url, filename_from


def test_google_drive_share_links_become_download_links():
    for link in ("https://drive.google.com/file/d/1AbC-x_9/view?usp=sharing",
                 "https://drive.google.com/open?id=1AbC-x_9"):
        assert direct_url(link) == "https://drive.usercontent.google.com/download?id=1AbC-x_9&export=download&confirm=t"


def test_dropbox_links_download_instead_of_preview():
    assert "dl=1" in direct_url("https://www.dropbox.com/s/abc/report.tsv?dl=0")


def test_other_links_stay():
    assert direct_url("https://x.org/a.tsv") == "https://x.org/a.tsv"


def test_server_name_replaces_a_placeholder():
    assert filename_from({"Content-Disposition": 'attachment; filename="report.pg_matrix.tsv"'}, "pobrany") == \
        "report.pg_matrix.tsv"
    encoded = {"Content-Disposition": "attachment; filename*=UTF-8''dane%20%C5%BC.xlsx"}
    assert filename_from(encoded, "x") == "dane ż.xlsx"
    assert filename_from({}, "pobrany") == "pobrany"
    assert filename_from({"Content-Disposition": 'attachment; filename="../../etc/passwd"'}, "x") == "passwd"
