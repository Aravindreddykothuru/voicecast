"""Every SYSPIN voice loads at a pinned revision, and every file it uses is
checked against a pinned sha256 before use.

A release that changes upstream, a truncated download or a swapped file used
to load silently and produce different speech. BengaliFemale was published
without extra.py and crashed mid-job on Bengali dubs (issue #6). These tests
pin down that nothing loads unpinned, nothing loads tampered, and the one
known-incomplete release is reported rather than hidden.
"""
from __future__ import annotations

import re

import pytest

from app.capabilities import SUPPORTED_LANGUAGES
from app.providers.tts.syspin_manifest import (
    EXTRA_PY_DONORS,
    EXTRA_PY_SHA256,
    SYSPIN_MANIFEST,
    pin_for,
    supply_chain_warnings,
)

HEX40, HEX64 = re.compile(r"^[0-9a-f]{40}$"), re.compile(r"^[0-9a-f]{64}$")


def _advertised_syspin_voices():
    return {v.model for lang in SUPPORTED_LANGUAGES for v in lang.voices if v.engine == "syspin"}


def test_every_advertised_syspin_voice_is_pinned():
    """A voice the product can select but the manifest does not pin would
    refuse to load mid-job. Adding a voice means pinning it."""
    unpinned = _advertised_syspin_voices() - set(SYSPIN_MANIFEST)
    assert not unpinned, f"advertised but not pinned: {sorted(unpinned)}"


@pytest.mark.parametrize("repo", sorted(SYSPIN_MANIFEST))
def test_each_pin_names_an_exact_commit_and_hashes_every_file(repo):
    pin = SYSPIN_MANIFEST[repo]
    assert HEX40.match(pin.revision), f"{repo}: revision must be a full commit sha, got {pin.revision!r}"
    assert "chars.txt" in pin.files, f"{repo}: chars.txt is not pinned"
    weights = [f for f in pin.files if f.endswith(".pt")]
    assert len(weights) == 1, f"{repo}: expected exactly one pinned .pt, got {weights}"
    for name, digest in pin.files.items():
        assert HEX64.match(digest), f"{repo}/{name}: not a sha256: {digest!r}"


def test_extra_py_donors_are_complete_releases():
    """A borrowed extra.py must come from a release that actually ships it."""
    assert HEX64.match(EXTRA_PY_SHA256)
    for donor in EXTRA_PY_DONORS:
        assert pin_for(donor).ships_extra_py, f"{donor} is listed as a donor but ships no extra.py"


def test_the_incomplete_bengali_release_is_reported_not_hidden():
    incomplete = sorted(r for r, p in SYSPIN_MANIFEST.items() if not p.ships_extra_py)
    assert incomplete == ["SYSPIN/tts_vits_coquiai_BengaliFemale"], (
        f"the set of releases missing extra.py changed: {incomplete} -- update issue #6 and the docs")
    warnings = supply_chain_warnings(SYSPIN_MANIFEST)
    assert len(warnings) == 1 and "BengaliFemale" in warnings[0] and "borrowed" in warnings[0]


def test_an_unpinned_voice_refuses_with_a_precise_message():
    """Not a missing-dependency error: it names the manifest and what to add."""
    pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    from app.providers.loading import ModelLoadError
    from app.providers.tts.syspin_provider import SyspinVoice

    with pytest.raises(ModelLoadError, match=r"not pinned in app/providers/tts/syspin_manifest\.py"):
        SyspinVoice("SYSPIN/tts_vits_coquiai_GujaratiFemale")


def test_a_tampered_file_is_refused_naming_the_file_and_both_hashes(tmp_path):
    from app.providers.loading import ModelLoadError
    from app.providers.tts.syspin_provider import _sha256, _verified

    f = tmp_path / "chars.txt"
    f.write_text("abc", encoding="utf-8")
    good = _sha256(str(f))
    assert _verified(str(f), good, "voice/chars.txt") == str(f)
    f.write_text("abc ", encoding="utf-8")          # one byte changed
    with pytest.raises(ModelLoadError) as exc:
        _verified(str(f), good, "voice@rev/chars.txt")
    msg = str(exc.value)
    assert "voice@rev/chars.txt" in msg and good[:16] in msg and "does not match the pinned" in msg


def test_a_voice_whose_download_differs_from_its_pin_refuses_to_load(tmp_path, monkeypatch):
    """The voice itself, not just the helper: it downloads at the pinned
    commit and checks every pinned file before anything is executed."""
    pytest.importorskip("torch")
    hub = pytest.importorskip("huggingface_hub")
    from app.providers.loading import ModelLoadError
    from app.providers.tts.syspin_provider import SyspinVoice

    repo = "SYSPIN/tts_vits_coquiai_TeluguFemale"
    pin = pin_for(repo)
    for name in pin.files:
        (tmp_path / name).write_bytes(b"not the release")
    seen = {}

    def fake_snapshot(repo_id, revision=None, **kwargs):
        seen.update(repo_id=repo_id, revision=revision)
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    with pytest.raises(ModelLoadError, match=r"TeluguFemale@[0-9a-f]{12}/\S+: sha256 [0-9a-f]+ does not match the pinned"):
        SyspinVoice(repo)
    assert seen == {"repo_id": repo, "revision": pin.revision}


BENGALI_FEMALE = "SYSPIN/tts_vits_coquiai_BengaliFemale"


def _serve_donor_extra_py(tmp_path, monkeypatch, content: bytes):
    """A sibling release's extra.py, served by a stubbed hf_hub_download."""
    import huggingface_hub

    donor_file = tmp_path / "donor_extra.py"
    donor_file.write_bytes(content)
    calls = []

    def fake_download(repo_id, filename, revision=None, **kwargs):
        calls.append((repo_id, filename, revision))
        return str(donor_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return donor_file, calls


def test_a_missing_extra_py_is_borrowed_at_a_pinned_revision_and_logged_loudly(tmp_path, monkeypatch, caplog):
    """The workaround for issue #6 is allowed only if it is pinned, verified
    and loud: the donor file comes from the donor's pinned commit, not its
    moving head, and every load says so at WARNING."""
    pytest.importorskip("huggingface_hub")
    import logging

    from app.providers.tts import syspin_manifest
    from app.providers.tts.syspin_provider import _pinned_extra_py, _sha256

    donor_file, calls = _serve_donor_extra_py(tmp_path, monkeypatch, b"# the shared Coqui tokenizer\n")
    monkeypatch.setattr(syspin_manifest, "EXTRA_PY_SHA256", _sha256(str(donor_file)))
    with caplog.at_level(logging.WARNING, logger="app.providers.tts.syspin_provider"):
        path = _pinned_extra_py(str(tmp_path / "voice"), BENGALI_FEMALE, pin_for(BENGALI_FEMALE))

    donor = EXTRA_PY_DONORS[0]
    assert path == str(donor_file)
    assert calls[0] == (donor, "extra.py", pin_for(donor).revision), calls
    loud = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "SUPPLY CHAIN" in r.getMessage()]
    assert loud and BENGALI_FEMALE in loud[0] and donor in loud[0], caplog.text


def test_a_borrowed_extra_py_that_differs_from_the_pin_is_refused(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    from app.providers.loading import ModelLoadError
    from app.providers.tts.syspin_provider import _pinned_extra_py

    _serve_donor_extra_py(tmp_path, monkeypatch, b"# not the pinned file\n")
    with pytest.raises(ModelLoadError,
                       match=r"borrowed for SYSPIN/tts_vits_coquiai_BengaliFemale\).*does not match the pinned"):
        _pinned_extra_py(str(tmp_path), BENGALI_FEMALE, pin_for(BENGALI_FEMALE))


@pytest.fixture
def real_syspin(monkeypatch):
    from app.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "tts_provider", "real")
    monkeypatch.setattr(s, "tts_engine", "syspin")
    monkeypatch.setattr(s, "tts_require_commercial_license", True)
    return s


def test_capabilities_surfaces_the_borrowed_file(client, real_syspin):
    caps = client.get("/api/capabilities").json()
    warnings = caps["tts_voice_warnings"]
    assert len(warnings) == 1 and "BengaliFemale" in warnings[0], warnings


def test_capabilities_reports_nothing_for_the_mock_engine(client):
    assert client.get("/api/capabilities").json()["tts_voice_warnings"] == []
