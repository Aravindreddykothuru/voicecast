"""`tts setup` must run the connectivity monitor, not just construct it.

What went wrong. Runner.start() starts the monitor thread; cmd_setup built a
ConnectivityMonitor and never started it. Nothing then polled during a setup,
so connectivity was only re-evaluated from one place -- the retry path in
netqueue, after a download fails. Once every task was PAUSED (offline) there
was nothing left to fail, so nothing asked again and the queue waited forever
for news that never came.

Seen twice on the 3.7 GB gated download: the link dropped, came back within
seconds, and setup sat paused for over an hour with its own probe returning
True whenever it was called by hand. A runtime that needs a restart to notice
the network returned is not self-healing, which is the whole claim.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def spy_monitor(monkeypatch):
    """Replace ConnectivityMonitor with one that records start()/stop()."""
    import app.tts_runtime.netmon as netmon

    calls = {"started": 0, "stopped": 0, "instances": []}

    class Spy:
        def __init__(self, mode="auto", *a, **kw):
            self.mode = mode
            calls["instances"].append(self)

        def start(self):
            calls["started"] += 1

        def stop(self):
            calls["stopped"] += 1

        def is_up(self):
            return True

        def check_now(self):
            return True

    monkeypatch.setattr(netmon, "ConnectivityMonitor", Spy)
    return calls


def _run_setup(monkeypatch, tmp_path, setup_impl):
    """Drive cmd_setup with provision.setup replaced, so no network is used."""
    import app.tts_runtime.provision as provision
    from app.tts_runtime import cli

    monkeypatch.setattr(provision, "setup", setup_impl)

    class Args:
        config = None
        home = tmp_path
        timeout = 5

    return cli.cmd_setup(Args())


def test_setup_starts_the_connectivity_monitor(spy_monitor, monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_setup(cfg, home, store, netmon, timeout_s=None):
        seen["netmon"] = netmon
        return {"token": True, "models": {}, "tasks": {}}

    rc = _run_setup(monkeypatch, tmp_path, fake_setup)
    capsys.readouterr()

    assert rc == 0
    assert spy_monitor["started"] == 1, (
        "cmd_setup constructed a ConnectivityMonitor but never start()ed it, so nothing "
        "polls and a paused download never resumes"
    )
    assert seen["netmon"] is spy_monitor["instances"][0], "a different monitor was handed to setup()"


def test_the_monitor_is_stopped_even_when_setup_raises(spy_monitor, monkeypatch, tmp_path):
    def boom(cfg, home, store, netmon, timeout_s=None):
        raise RuntimeError("provisioning blew up")

    with pytest.raises(RuntimeError, match="provisioning blew up"):
        _run_setup(monkeypatch, tmp_path, boom)

    assert spy_monitor["started"] == 1
    assert spy_monitor["stopped"] == 1, "the polling thread must not outlive a failed setup"
