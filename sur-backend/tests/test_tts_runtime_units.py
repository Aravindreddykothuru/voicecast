"""Unit tests for the TTS runtime's pure parts: config, licence allowlist,
store state machine, pin verification, triggers, circuit breaker, known-bad
learning, report. No model, no worker process."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from app.tts_runtime import config, licenses
from app.tts_runtime.breaker import CircuitBreaker
from app.tts_runtime.checks import Calibration, cer, reader_checks, signal_checks
from app.tts_runtime.config import BreakerConfig, ConfigError
from app.tts_runtime.knownbad import KnownBad, words
from app.tts_runtime.manifest import (
    FilePin,
    HashMismatch,
    git_blob_sha1_of,
    lfs_pointer,
    verify_file,
)
from app.tts_runtime.store import Store

BASE = {"chains": {"hi": ["syspin", "indic_parler", "indicf5"], "default": ["indic_parler", "indicf5"]}}


# --- licence allowlist ----------------------------------------------------------
def test_the_shipped_config_passes_the_allowlist():
    cfg = config.load()
    assert cfg.chain_for("hi") == ("syspin", "indic_parler", "indicf5")
    assert cfg.chain_for("ta") == ("indic_parler", "indicf5")      # no SYSPIN voice -> default chain


@pytest.mark.parametrize("repo,why", [
    ("facebook/mms-tts-hin", "CC-BY-NC"),
    ("SWivid/F5-TTS", "non-commercial"),
    ("coqui/XTTS-v2", "Coqui Public Model License"),
])
def test_forbidden_models_are_refused_with_the_reason(repo, why):
    with pytest.raises(licenses.LicenseError, match=why):
        licenses.check("indicf5", repo, "MIT")


def test_a_model_not_on_the_allowlist_is_refused():
    with pytest.raises(licenses.LicenseError, match="not on the licence allowlist"):
        licenses.check("some_new_tts", "someone/model", "Apache-2.0")


def test_a_licence_that_differs_from_the_record_is_refused():
    with pytest.raises(licenses.LicenseError, match="declares licence 'MIT'"):
        licenses.check("indic_parler", "ai4bharat/indic-parler-tts", "MIT")


def test_test_only_models_cannot_join_a_chain():
    data = {"chains": {"hi": ["fake:a"], "default": ["fake:a"]}}
    with pytest.raises(ConfigError, match="tests only"):
        config.parse(data, allow_test_models=False)
    assert config.parse(data, allow_test_models=True).chain_for("hi") == ("fake:a",)


def test_readers_must_be_allowlisted_readers():
    with pytest.raises(ConfigError, match="not an allowlisted reader"):
        config.parse({**BASE, "readers": {"hi": ["syspin"]}})


# --- config strictness ------------------------------------------------------------
@pytest.mark.parametrize("patch,match", [
    ({"chains": {"hi": ["syspin"]}}, "default"),
    ({"timeoutz": {}}, "unknown key"),
    ({"timeouts": {"synth": 5}}, "unknown key"),
    ({"known_bad": {"auto_add_after": "three of five"}}, "3 of last 5"),
    ({"known_bad": {"auto_remove": True}}, "must be false"),
    ({"offline_mode": "sometimes"}, "auto | force | off"),
    ({"circuit_breaker": {"fail_ratio": 1.5}}, "between 0 and 1"),
    ({"retries": {"network_backoff": [5, 2]}}, "non-decreasing"),
    ({"chains": {"hi": ["syspin", "syspin"], "default": ["syspin"]}}, "twice"),
    ({"chains": {"hi": ["nosuchmodel"], "default": ["syspin"]}}, "unknown model"),
])
def test_config_errors_are_precise(patch, match):
    with pytest.raises(ConfigError, match=match):
        config.parse({**BASE, **patch})


def test_yaml_off_is_read_as_offline_mode_off():
    """YAML 1.1 turns a bare `off` into False; the documented value must still work."""
    import yaml

    assert config.parse({**BASE, **yaml.safe_load("offline_mode: off")}).offline_mode == "off"


def test_known_bad_rule_is_parsed():
    cfg = config.parse({**BASE, "known_bad": {"auto_add_after": "2 of last 4", "auto_remove": False}})
    assert (cfg.known_bad.hits, cfg.known_bad.of_last) == (2, 4)


# --- store --------------------------------------------------------------------------
def test_store_runs_in_wal_mode_with_full_sync(tmp_path):
    s = Store(tmp_path / "s.sqlite3")
    assert s.q("PRAGMA journal_mode")[0][0] == "wal"
    assert s.q("PRAGMA synchronous")[0][0] == 2          # FULL


def test_job_creation_is_idempotent_and_lines_start_pending(tmp_path):
    s = Store(tmp_path / "s.sqlite3")
    lines = [{"text": "a", "lang": "hi"}, {"text": "b", "lang": "hi"}]
    assert s.create_job("j1", {}, str(tmp_path), lines) is True
    assert s.create_job("j1", {}, str(tmp_path), lines) is False
    assert s.counts("j1")["PENDING"] == 2


def test_unknown_line_states_are_rejected(tmp_path):
    s = Store(tmp_path / "s.sqlite3")
    s.create_job("j1", {}, str(tmp_path), [{"text": "a", "lang": "hi"}])
    with pytest.raises(ValueError):
        s.set_line("j1", 0, "HALF_DONE")


# --- pin verification ----------------------------------------------------------------
def test_small_file_verifies_by_git_blob_id(tmp_path):
    data = b'{"a": 1}\n'
    f = tmp_path / "config.json"
    f.write_bytes(data)
    pin = FilePin("config.json", len(data), git_blob_sha1_of(data), lfs=False)
    assert verify_file(f, pin) == hashlib.sha256(data).hexdigest()
    f.write_bytes(b'{"a": 2}\n')
    with pytest.raises(HashMismatch, match="does not match upstream"):
        verify_file(f, pin)


def test_lfs_file_verifies_through_its_pointer(tmp_path):
    data = bytes(range(256)) * 100
    f = tmp_path / "model.safetensors"
    f.write_bytes(data)
    oid = git_blob_sha1_of(lfs_pointer(hashlib.sha256(data).hexdigest(), len(data)))
    assert verify_file(f, FilePin("model.safetensors", len(data), oid, lfs=True))
    f.write_bytes(data[:-1] + b"\x00")
    with pytest.raises(HashMismatch):
        verify_file(f, FilePin("model.safetensors", len(data), oid, lfs=True))


# Published by Hugging Face for the ungated parler-tts-tiny-v1@fe1bd939 (tree
# API): (path, LFS sha256, size, git id of the pointer). The pointer rebuilt
# from the sha256 must give the published git id -- the fact that lets a
# gated repo's files be verified without seeing their sha256.
UPSTREAM_LFS = [
    ("model.safetensors", "2e549192e0ef60cc2627cbd9d2c54ef0985ded3576d8ae7bf8744267cd6d2427", 1266301840,
     "0a35330900fb5d373c509b66d4cad07c16d3b4a8"),
    ("spiece.model", "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86", 791656,
     "317a5ccbde45300f5d1d970d4d449af2108b147e"),
]


@pytest.mark.parametrize("path,sha256,size,git_oid", UPSTREAM_LFS)
def test_pointer_method_reproduces_published_upstream_ids(path, sha256, size, git_oid):
    assert git_blob_sha1_of(lfs_pointer(sha256, size)) == git_oid
    pin = next(f for f in json.loads((config.BACKEND_DIR / "app/tts_runtime/pins/parler_tiny.json")
                                     .read_text())["files"] if f["path"] == path)
    assert (pin["size"], pin["git_oid"], pin["lfs"]) == (size, git_oid, True)


def test_size_mismatch_is_reported_before_hashing(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    with pytest.raises(HashMismatch, match="size 3 != pinned 4"):
        verify_file(f, FilePin("x", 4, "0" * 40, False))


# --- triggers --------------------------------------------------------------------------
CAL = Calibration({"hi": {"short": {"cps_lo": 4, "cps_hi": 20}, "long": {"cps_lo": 8, "cps_hi": 20}}},
                  {"short": {"cps_lo": 3, "cps_hi": 30}, "long": {"cps_lo": 3, "cps_hi": 30}}, 0.5, 0.5)


def _speech(seconds, sr=16000, lead=0.05, trail=0.05, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    x = amp * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return np.concatenate([np.zeros(int(lead * sr)), x, np.zeros(int(trail * sr))]).astype(np.float32), sr


def test_a_good_clip_passes_every_trigger():
    x, sr = _speech(0.5)                 # "नमस्ते" is 6 chars -> 12 chars/s
    r = signal_checks(x, sr, "नमस्ते", "hi", CAL)
    assert r.passed, r.failures


@pytest.mark.parametrize("make,trigger", [
    (lambda: (np.zeros(0, np.float32), 16000), "empty"),
    (lambda: (np.full(1600, np.nan, np.float32), 16000), "nan"),
    (lambda: (np.zeros(16000, np.float32), 16000), "empty"),
    (lambda: _speech(3.0), "duration"),                      # 6 chars in 3 s = 2 chars/s
    (lambda: _speech(0.5, lead=1.0), "lead_silence"),
    (lambda: _speech(0.5, trail=1.0), "trail_silence"),
    (lambda: (np.sign(_speech(0.5)[0]) * 1.0, 16000), "clipping"),
])
def test_each_trigger_fires_with_a_precise_detail(make, trigger):
    x, sr = make()
    r = signal_checks(x, sr, "नमस्ते", "hi", CAL)
    assert not r.passed and trigger in r.triggers, r.failures
    assert all(f["detail"] for f in r.failures)


def test_uncalibrated_language_uses_default_range_and_says_so_when_none():
    x, sr = _speech(0.5)
    assert signal_checks(x, sr, "வணக்கம்", "ta", CAL).passed
    r = signal_checks(x, sr, "नमस्ते", "hi", None)
    assert r.passed and r.scores["uncalibrated"] is True


def test_cer_is_identical_to_the_gate():
    import importlib.util

    spec = importlib.util.spec_from_file_location("e2e", config.BACKEND_DIR / "scripts" / "e2e_dub.py")
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    for ref, hyp in [("సరే.", "సరి"), ("ठीक है।", "ठीक है"), ("क्यों?", "किम"), ("abc", "")]:
        assert cer(ref, hyp) == e2e._cer(ref, hyp)


def test_reader_trigger_and_disagreement():
    base = signal_checks(*_speech(0.5), "नमस्ते", "hi", CAL)
    assert reader_checks(base, "नमस्ते", {"r1": "नमस्ते"}).passed
    bad = signal_checks(*_speech(0.5), "नमस्ते", "hi", CAL)
    assert reader_checks(bad, "नमस्ते", {"r1": "किम"}).triggers == ["reader"]
    mixed = signal_checks(*_speech(0.5), "नमस्ते", "hi", CAL)
    assert reader_checks(mixed, "नमस्ते", {"r1": "नमस्ते", "r2": "किम"}).triggers == ["readers_disagree"]


# --- circuit breaker ----------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_breaker_opens_above_the_ratio_and_not_before_min_samples(tmp_path):
    s, clk, alerts = Store(tmp_path / "s.sqlite3"), Clock(), []
    b = CircuitBreaker(s, BreakerConfig(window=20, fail_ratio=0.30, cooldown_s=600, min_samples=10),
                       lambda k, m: alerts.append(k), clk)
    for _ in range(3):
        b.record("m", "hi", False)            # 3 of 3 failed, but below min_samples
    assert b.allows("m", "hi")
    for _ in range(7):
        b.record("m", "hi", True)             # 3 of 10 = 30%: not MORE than 30%
    assert b.allows("m", "hi")
    b.record("m", "hi", False)                # 4 of 11 = 36%
    assert not b.allows("m", "hi") and alerts == ["breaker_open"]


def test_breaker_half_opens_after_cooldown_and_closes_on_a_passing_trial(tmp_path):
    s, clk, alerts = Store(tmp_path / "s.sqlite3"), Clock(), []
    b = CircuitBreaker(s, BreakerConfig(20, 0.3, 600, 1), lambda k, m: alerts.append(k), clk)
    b.record("m", "hi", False)
    assert b.state("m", "hi") == "open"
    clk.t += 599
    assert b.state("m", "hi") == "open"
    clk.t += 2
    assert b.state("m", "hi") == "half_open"
    b.record("m", "hi", True)
    assert b.state("m", "hi") == "closed"
    assert alerts == ["breaker_open", "breaker_half_open", "breaker_close"]
    assert s.recent_outcomes("m", "hi", 20) == []       # window cleared on close


def test_failed_trial_reopens_and_restarts_the_cooldown(tmp_path):
    s, clk = Store(tmp_path / "s.sqlite3"), Clock()
    b = CircuitBreaker(s, BreakerConfig(20, 0.3, 600, 1), clock=clk)
    b.record("m", "hi", False)
    clk.t += 601
    assert b.state("m", "hi") == "half_open"
    b.record("m", "hi", False)
    assert b.state("m", "hi") == "open"
    clk.t += 300
    assert b.state("m", "hi") == "open"


def test_health_check_failure_opens_immediately(tmp_path):
    s = Store(tmp_path / "s.sqlite3")
    b = CircuitBreaker(s, BreakerConfig())
    b.health_failed("m", "te", "no audio")
    assert not b.allows("m", "te") and b.allows("m", "hi")
    assert s.events(kind="breaker_open")[0]["detail"] == "health check failed: no audio"


# --- known-bad ---------------------------------------------------------------------------
def test_word_splitting_keeps_indic_vowel_signs_and_drops_dandas():
    assert words("क्यों?") == ["क्यों"]
    assert words("रुकिए।") == ["रुकिए"]
    assert words("ঠিক আছে।") == ["ঠিক", "আছে"]
    assert words("क्या हाल है॥") == ["क्या", "हाल", "है"]


def test_word_added_after_3_of_last_5_failures_and_never_removed(tmp_path):
    s = Store(tmp_path / "s.sqlite3")
    kb = KnownBad(s, 3, 5, tmp_path / "kb")
    for ok in (False, True, False):
        assert kb.record("syspin", "hi", "क्यों?", ok) is None
    assert kb.record("syspin", "hi", "क्यों?", False) == "क्यों"
    assert "क्यों" in (tmp_path / "kb" / "hi.txt").read_text(encoding="utf-8")
    for _ in range(10):
        kb.record("syspin", "hi", "क्यों?", True)
    assert KnownBad(s, 3, 5, tmp_path / "kb").hits_in("hi", "तुम क्यों आए?") == ["क्यों"]


def test_only_syspin_single_word_lines_teach_known_bad(tmp_path):
    s = Store(tmp_path / "s.sqlite3")
    kb = KnownBad(s, 1, 1, tmp_path / "kb")
    assert kb.record("syspin", "hi", "तुम क्यों आए?", False) is None      # sentence: which word?
    assert kb.record("indic_parler", "hi", "क्यों?", False) is None      # not SYSPIN
    assert kb.words_for("hi") == set()


def test_the_seeded_lists_carry_evidence():
    from app.tts_runtime.knownbad import DEFAULT_DIR

    for lang, expected in (("hi", {"क्यों", "रुकिए"}), ("mr", {"आत्ता", "खरच"})):
        text = (DEFAULT_DIR / f"{lang}.txt").read_text(encoding="utf-8")
        assert KnownBad(None, directory=DEFAULT_DIR).words_for(lang) == expected
        for w in expected:
            line = next(ln for ln in text.splitlines() if ln.startswith(w))
            assert "#" in line and "N=" in line or "of 40" in line or "CER" in line, line
