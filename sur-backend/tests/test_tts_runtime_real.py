"""The runtime with real models.

  * Regression: SYSPIN through the runtime renders byte-for-byte what the
    pipeline renders today, so the per-language pass table cannot move.
  * Offline: with every worker's network blocked at the socket level, SYSPIN
    renders and the te reader reads -- and the block itself is proven on.
  * parler_tiny: the Parler adapter end to end in its own interpreter
    (skipped unless `.venv-parler` exists and the model is provisioned).

Skipped, not failed, where the weights are not on this machine.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from app.tts_runtime import config
from app.tts_runtime.adapters.syspin import SyspinAdapter
from app.tts_runtime.checks import Calibration
from app.tts_runtime.netmon import ConnectivityMonitor
from app.tts_runtime.runner import Runner

pytest.importorskip("torch")
SYSPIN_OK, SYSPIN_WHY = SyspinAdapter.availability({})
needs_syspin = pytest.mark.skipif(not SYSPIN_OK, reason=f"SYSPIN not cached: {SYSPIN_WHY}")
WIDE = Calibration({}, {"short": {"cps_lo": 1, "cps_hi": 60}, "long": {"cps_lo": 1, "cps_hi": 60}}, 3.0, 3.0)


@needs_syspin
@pytest.mark.parametrize("lang,text", [("te", "సరే."), ("hi", "रुकिए।"), ("bn", "সত্যিই?"),
                                       ("hi", "मैं कल सुबह तुमसे स्टेशन पर मिलूँगा।")])
def test_syspin_through_the_runtime_is_byte_identical_to_production(lang, text):
    from app.capabilities import require_language
    from app.providers.tts.common import render_line
    from app.providers.tts.syspin_provider import SAMPLE_RATE, SyspinVoice
    from app.tts_runtime.adapters.syspin import _repo_for

    a = SyspinAdapter(None, {"carrier": True})
    a.load("cpu")
    got = a.synth(text, lang, None, None, draw=0)
    lg = require_language(lang)
    voice = SyspinVoice(_repo_for(lang, None))
    want = render_line(voice.render, text, SAMPLE_RATE, carrier=lg.tts_carrier,
                       merge_closures=lg.tts_carrier_merge_closures)
    assert got.sr == SAMPLE_RATE and np.array_equal(got.samples, np.asarray(want, dtype=np.float32))


def _offline_runner(tmp_path, chains, readers=None, options=None):
    cfg = config.parse({"chains": {**chains, "default": chains[next(iter(chains))]}, "offline_mode": "force",
                        "retries": {"per_model": 0}, "scene_consistency": False, "readers": readers or {}},
                       allow_test_models=True, source="test")
    cfg = dataclasses.replace(cfg, synth_timeout_s=300, check_timeout_s=300)
    return Runner(cfg, tmp_path / "home", calibration=WIDE, options=options or {},
                  env_extra={"TTS_RUNTIME_DENY_NETWORK": "1"}, netmon=ConnectivityMonitor("force"),
                  known_bad_dir=tmp_path / "kb")


def test_the_offline_guard_really_blocks_the_network(tmp_path):
    r = _offline_runner(tmp_path, {"hi": ["fake:a"]}, options={"fake:a": {"behavior": {"default": "network"}}})
    jid = r.submit({"lines": [{"text": "नमस्ते", "lang": "hi"}]})
    r.run(jid)
    detail = r.store.attempts(jid, 0)[0]["detail"]
    assert "network access denied" in detail, detail


@needs_syspin
def test_rendering_and_reading_need_no_network(tmp_path):
    r = _offline_runner(tmp_path, {"hi": ["syspin"], "te": ["syspin"]}, readers={"te": ["vakyansh"]})
    lines = [{"text": "సరే.", "lang": "te"}, {"text": "అది భయంకరమైన వార్త.", "lang": "te"},
             {"text": "ठीक है।", "lang": "hi"}, {"text": "मैं कल सुबह तुमसे स्टेशन पर मिलूँगा।", "lang": "hi"}]
    jid = r.submit({"lines": lines})
    res = r.run(jid)
    assert res["status"] == "DONE"
    attempts = r.store.attempts(jid)
    assert not any("network access denied" in (a["detail"] or "") for a in attempts)
    assert all(ln["output_path"] for ln in r.store.lines(jid)), "every line has audio"
    import json

    te = [json.loads(a["scores"]) for a in attempts if a["idx"] in (0, 1) and a["scores"]]
    assert all("cer_vakyansh" in s for s in te), "the te reader ran offline"
    assert not r.store.events(jid, "reader_error")


PARLER_PY = config.BACKEND_DIR / ".venv-parler" / "Scripts" / "python.exe"
if not PARLER_PY.exists():
    PARLER_PY = config.BACKEND_DIR / ".venv-parler" / "bin" / "python"


def _parler_ready() -> tuple[bool, str]:
    from app.tts_runtime.adapters.indic_parler import ParlerTinyAdapter

    if not Path(PARLER_PY).exists():
        return False, "no .venv-parler"
    return ParlerTinyAdapter.availability({})


PARLER_OK, PARLER_WHY = _parler_ready()


@pytest.mark.skipif(not PARLER_OK, reason=f"parler_tiny not ready: {PARLER_WHY}")
def test_the_parler_adapter_renders_in_its_own_interpreter_offline(tmp_path):
    from app.tts_runtime.adapters.base import runtime_home

    cfg_patch = {"interpreters": {"parler_tiny": str(PARLER_PY)}}
    cfg = config.parse({"chains": {"en": ["parler_tiny"], "default": ["parler_tiny"]}, "offline_mode": "force",
                        "retries": {"per_model": 0}, "scene_consistency": False, **cfg_patch},
                       allow_test_models=True, source="test")
    cfg = dataclasses.replace(cfg, synth_timeout_s=900)
    r = Runner(cfg, tmp_path / "home", calibration=WIDE, options={"parler_tiny": {"home": str(runtime_home())}},
               env_extra={"TTS_RUNTIME_DENY_NETWORK": "1"}, netmon=ConnectivityMonitor("force"),
               known_bad_dir=tmp_path / "kb")
    jid = r.submit({"lines": [{"text": "Hello, this is a test.", "lang": "en"}]})
    res = r.run(jid)
    assert res["status"] == "DONE", r.store.attempts(jid)
    from app.tts_runtime.audio import UNIFIED_SR, read_wav

    out = read_wav(r.store.line(jid, 0)["output_path"])
    assert out.sr == UNIFIED_SR and 0.5 < out.seconds < 10
