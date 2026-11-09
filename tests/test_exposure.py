"""Switching the panel between private (tailscale serve) and public (tailscale funnel)."""
import pytest

from compute_bot.exposure import Exposure, ExposureError


class Shell:
    def __init__(self, outputs=None, fail=None):
        self.calls, self.outputs, self.fail = [], outputs or {}, fail

    def __call__(self, args):
        self.calls.append(args)
        if self.fail and self.fail in args:
            raise ExposureError("tailscale: brak uprawnień")
        return self.outputs.get(args[1], "")


def test_public_on_uses_funnel_and_off_goes_back_to_private_serve():
    shell = Shell()
    exposure = Exposure(8787, run=shell)
    exposure.set_public(True)
    assert shell.calls[-1] == ["tailscale", "funnel", "--bg", "--https=443", "http://127.0.0.1:8787"]
    exposure.set_public(False)
    assert shell.calls[-2:] == [["tailscale", "funnel", "--https=443", "off"],
                                ["tailscale", "serve", "--bg", "--https=443", "http://127.0.0.1:8787"]]


def test_status_reads_funnel_state():
    assert Exposure(8787, run=Shell({"funnel": "# Funnel on:\n#     - https://vostro.ts.net"})).status() is True
    assert Exposure(8787, run=Shell({"funnel": "No serve config"})).status() is False


def test_errors_reach_the_owner():
    with pytest.raises(ExposureError):
        Exposure(8787, run=Shell(fail="funnel")).set_public(True)
