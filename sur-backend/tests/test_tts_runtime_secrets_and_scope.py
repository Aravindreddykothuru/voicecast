"""The Hugging Face token stays in the parent process, and a language the
product does not support cannot enter the queue.

The token is a long-lived credential: it must not reach a model worker (which
never authenticates), nor any file the runtime writes -- store, heartbeat,
worker log or run report. Urdu was dropped from scope on 2026-09-23, so it
must be refused at every door rather than queued and flagged forever.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from app.tts_runtime import config
from app.tts_runtime.checks import Calibration
from app.tts_runtime.pool import WorkerHandle
from app.tts_runtime.runner import Runner

SENTINEL = "hf_" + "S3nt1nelTokenValueNeverToBeLogged1234"
CAL = Calibration({}, {"short": {"cps_lo": 4, "cps_hi": 30}, "long": {"cps_lo": 4, "cps_hi": 30}}, 1.0, 1.0)


def _runner(tmp_path, chains=None):
    cfg = config.parse({"chains": chains or {"hi": ["fake:a"], "default": ["fake:a"]}, "offline_mode": "off",
                        "retries": {"per_model": 0}, "scene_consistency": False},
                       allow_test_models=True, source="test")
    return Runner(dataclasses.replace(cfg, synth_timeout_s=20), tmp_path / "home", calibration=CAL,
                  known_bad_dir=tmp_path / "kb", heartbeat_s=0.5)


@pytest.mark.parametrize("var", WorkerHandle.SECRET_ENV)
def test_a_worker_never_receives_the_token(tmp_path, monkeypatch, var):
    monkeypatch.setenv(var, SENTINEL)
    h = WorkerHandle("fake:a", {}, None, tmp_path)
    env = h.worker_env()
    assert var not in env
    assert not any(SENTINEL in str(v) for v in env.values()), "token leaked through another variable"


def test_the_token_never_reaches_the_store_logs_or_report(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    r = _runner(tmp_path)
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(3)]})
    r.run(jid)
    written = [p for p in (tmp_path / "home").rglob("*") if p.is_file()]
    assert written, "the run wrote nothing; the test would pass vacuously"
    for p in written:
        assert SENTINEL.encode() not in p.read_bytes(), f"token found in {p}"


def test_urdu_cannot_enter_the_queue(tmp_path):
    """Not "flagged forever" -- refused at the door, because no model in any
    chain speaks it. The real chains are used here, not the fake ones."""
    cfg = config.load()
    r = Runner(cfg, tmp_path / "home", calibration=CAL, known_bad_dir=tmp_path / "kb", heartbeat_s=0.5)
    with pytest.raises(ValueError, match=r"no model in the 'ur' chain speaks 'ur'"):
        r.submit({"lines": [{"text": "\u0627\u0631\u062f\u0648", "lang": "ur"}]})
    assert not r.store.q("SELECT 1 FROM lines"), "nothing was queued"
    # The product's own door is shut too.
    from app.capabilities import require_language

    with pytest.raises(ValueError, match="Unsupported target language 'ur'"):
        require_language("ur")


def test_a_language_whose_models_are_merely_unprovisioned_is_still_queueable(tmp_path):
    """The distinction that matters: Tamil has no provisioned model on this
    box (both gated), but Indic Parler and IndicF5 do speak it -- so a Tamil
    line queues and is skipped-with-a-reason at render time, recoverable the
    moment the weights arrive. Urdu is refused outright."""
    cfg = config.load()
    r = Runner(cfg, tmp_path / "home", calibration=CAL, known_bad_dir=tmp_path / "kb", heartbeat_s=0.5)
    jid = r.submit({"lines": [{"text": "\u0b95\u0bbe\u0bb2\u0bc8", "lang": "ta"}]})
    assert r.store.counts(jid)["PENDING"] == 1


def test_urdu_is_gone_from_the_product():
    from app.capabilities import LANGUAGES_BY_CODE, SUPPORTED_LANGUAGES

    assert "ur" not in LANGUAGES_BY_CODE
    assert "Urdu" not in {lang.name for lang in SUPPORTED_LANGUAGES}
    cfg = config.load()
    assert "ur" not in cfg.chains
    for f in ("benchmark_sentences.json", "warmup_sentences.json"):
        data = json.loads((config.BACKEND_DIR / "app/tts_runtime/data" / f).read_text(encoding="utf-8"))
        assert "ur" not in data.get("langs", data)


def test_each_model_gets_its_own_measured_timeout():
    cfg = config.load()
    assert cfg.synth_timeout_for("syspin") == 60
    assert cfg.synth_timeout_for("indic_parler") == cfg.synth_timeout_for("indicf5") == 600
    assert cfg.synth_timeout_for("fake:a") == cfg.synth_timeout_s, "unlisted models keep the default"


def test_a_per_model_timeout_must_be_a_positive_number():
    base = {"chains": {"hi": ["syspin"], "default": ["syspin"]}}
    with pytest.raises(config.ConfigError, match="positive number of seconds"):
        config.parse({**base, "timeouts": {"per_model": {"syspin": 0}}})
    with pytest.raises(config.ConfigError, match="unknown key"):
        config.parse({**base, "timeouts": {"per_model": {"nosuchmodel": 10}}})


def test_the_pool_uses_the_per_model_timeout(tmp_path, monkeypatch):
    """A model slower than the default bound must not be killed at the default."""
    from app.tts_runtime.pool import ModelPool

    cfg = config.parse({"chains": {"hi": ["fake:a"], "default": ["fake:a"]},
                        "timeouts": {"synth_s": 1, "per_model": {"fake": 30}}, "offline_mode": "off"},
                       allow_test_models=True, source="test")
    pool = ModelPool(cfg, tmp_path, lambda m: {"behavior": {"sleep_s": 3.0}})
    try:
        r = pool.synth("fake:a", "नमस्ते", "hi", None, None, 0, str(tmp_path / "o.wav"))
        assert r["ok"], "killed at the 1 s default instead of the model's 30 s bound"
    finally:
        pool.close_all()


def test_setup_exits_non_zero_and_names_a_model_it_could_not_provision(capsys, monkeypatch, tmp_path):
    """A model named in a chain but left unprovisioned is a half-set-up
    deployment. `tts setup` must say which repo and why, and fail -- never
    exit 0 having quietly provisioned only the model that happened to work."""
    from app.tts_runtime import cli

    report = {
        "token": False, "warnings": [],
        "models": {"syspin": {"status": "cached"},
                   "indic_parler": {"queued": 0, "verified": 5,
                                    "skipped_gated": ["ai4bharat/indic-parler-tts"]}},
        "tasks": {},
    }
    monkeypatch.setattr("app.tts_runtime.provision.setup", lambda *a, **k: report)
    args = type("A", (), {"config": None, "home": tmp_path, "timeout": None})()
    assert cli.cmd_setup(args) == 1
    err = capsys.readouterr().err
    assert "indic_parler NOT provisioned" in err and "ai4bharat/indic-parler-tts" in err
    assert "HF_TOKEN is not set" in err


def test_setup_exits_zero_when_every_model_is_provisioned(monkeypatch, tmp_path):
    from app.tts_runtime import cli

    report = {"token": True, "warnings": [],
              "models": {"syspin": {"status": "cached"},
                         "indic_parler": {"queued": 0, "verified": 8, "skipped_gated": []}},
              "tasks": {"dl:x": {"state": "DONE", "attempts": 1, "error": None}}}
    monkeypatch.setattr("app.tts_runtime.provision.setup", lambda *a, **k: report)
    args = type("A", (), {"config": None, "home": tmp_path, "timeout": None})()
    assert cli.cmd_setup(args) == 0
