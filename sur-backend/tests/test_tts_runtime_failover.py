"""Failover with real worker processes and fake models that fail on command.

Model 1 crashes / returns garbage / times out -> the line comes from model 2
and the batch completes; every model fails -> the line is flagged, the batch
completes and the report lists it. Plus: unavailable models, known-bad skip
and last resort, circuit breaker, health checks, the OOM ladder, scene
consistency, the allowlist enforced inside the worker, idempotent reruns.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from app.tts_runtime import config, knownbad
from app.tts_runtime.checks import Calibration
from app.tts_runtime.runner import Runner

CAL = Calibration({}, {"short": {"cps_lo": 4, "cps_hi": 30}, "long": {"cps_lo": 4, "cps_hi": 30}}, 1.0, 1.0)


def make_runner(tmp_path, chains, options=None, **overrides):
    cfg = config.parse({"chains": {**chains, "default": chains[next(iter(chains))]}, "offline_mode": "off",
                        "retries": {"per_model": overrides.pop("retries", 1)},
                        "circuit_breaker": overrides.pop("breaker", {"window": 20, "fail_ratio": 0.3,
                                                                     "cooldown_s": 600, "min_samples": 10}),
                        "scene_consistency": overrides.pop("scenes", False)},
                       allow_test_models=True, source="test")
    cfg = dataclasses.replace(cfg, synth_timeout_s=overrides.pop("timeout", 20), **overrides)
    return Runner(cfg, tmp_path / "home", calibration=CAL, options=options or {}, known_bad_dir=tmp_path / "kb",
                  heartbeat_s=0.5)


def run(r, lines):
    jid = r.submit({"lines": [ln if isinstance(ln, dict) else {"text": ln, "lang": "hi"} for ln in lines]})
    return jid, r.run(jid)


def states(r, jid):
    return [(ln["state"], ln["model"]) for ln in r.store.lines(jid)]


@pytest.mark.parametrize("mode", ["crash", "garbage", "exception", "nan", "empty", "clip", "lead_silence"])
def test_model_1_failing_hands_the_line_to_model_2(tmp_path, mode):
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:a": {"behavior": {"texts": {"खराब": mode}}}})
    jid, res = run(r, ["नमस्ते", "खराब", "ठीक है"])
    assert res["status"] == "DONE"
    assert states(r, jid) == [("DONE", "fake:a"), ("DONE", "fake:b"), ("DONE", "fake:a")]
    switches = r.store.events(jid, "switch")
    assert len(switches) == 1 and json.loads(switches[0]["detail"])["to"] == "fake:b"
    outcomes = [a["outcome"] for a in r.store.attempts(jid, 1) if a["model"] == "fake:a"]
    assert len(outcomes) == 2, "one retry per model"


def test_a_hung_model_is_killed_at_the_timeout_and_the_next_model_speaks(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:a": {"behavior": {"texts": {"रुको": "hang"}}}},
                    timeout=4, retries=0)
    jid, res = run(r, ["रुको", "नमस्ते"])
    assert res["status"] == "DONE"
    assert states(r, jid) == [("DONE", "fake:b"), ("DONE", "fake:a")]
    assert [a["outcome"] for a in r.store.attempts(jid, 0)][0] == "timeout"


def test_every_model_failing_flags_the_line_keeps_the_best_audio_and_finishes_the_batch(tmp_path):
    opts = {"fake:a": {"behavior": {"texts": {"बुरा": "garbage"}}},
            "fake:b": {"behavior": {"texts": {"बुरा": "lead_silence"}}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts)
    jid, res = run(r, ["नमस्ते", "बुरा", "ठीक है"])
    assert res["status"] == "DONE" and res["counts"]["FLAGGED"] == 1 and res["counts"]["DONE"] == 2
    flagged = r.store.line(jid, 1)
    assert flagged["state"] == "FLAGGED" and flagged["output_path"]            # best attempt kept
    assert flagged["model"] == "fake:b", "lead silence (severity 3) beats garbage duration (20)"
    report = json.loads((tmp_path / "home" / "jobs" / jid / "report.json").read_text(encoding="utf-8"))
    assert [f["idx"] for f in report["flagged"]] == [1] and "every model failed" in report["flagged"][0]["reason"]
    assert "line 1" in (tmp_path / "home" / "jobs" / jid / "report.md").read_text(encoding="utf-8")


def test_an_unavailable_model_is_skipped_and_reported_not_fatal(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:a": {"behavior": {"available": False}}})
    jid, res = run(r, ["नमस्ते"])
    assert states(r, jid) == [("DONE", "fake:b")]
    report = json.loads((tmp_path / "home" / "jobs" / jid / "report.json").read_text(encoding="utf-8"))
    assert report["unavailable_models"][0]["model"] == "fake:a"


def test_a_chain_with_no_usable_model_flags_every_line_without_audio(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a"]}, {"fake:a": {"behavior": {"available": False}}})
    jid, res = run(r, ["नमस्ते", "ठीक है"])
    assert res["status"] == "DONE" and res["counts"]["FLAGGED"] == 2
    assert all(ln["output_path"] is None for ln in r.store.lines(jid))


def test_known_bad_word_skips_the_model_then_uses_it_as_flagged_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr(knownbad, "applies_to", lambda m: m == "fake:a")   # like SYSPIN: one voice
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "hi.txt").write_text("क्यों  # test evidence\n", encoding="utf-8")
    # model 2 is unavailable: nothing else can say the word
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:b": {"behavior": {"available": False}}})
    jid, res = run(r, ["तुम क्यों आए?", "नमस्ते"])
    first = r.store.line(jid, 0)
    assert first["state"] == "FLAGGED" and first["model"] == "fake:a" and "known-bad" in first["flag_reason"]
    assert r.store.line(jid, 1)["state"] == "DONE"
    kinds = [a["outcome"] for a in r.store.attempts(jid, 0)]
    assert kinds[0] == "known_bad" and "ok" in kinds
    assert r.store.events(jid, "known_bad_last_resort")


def test_known_bad_word_goes_to_the_next_model_when_one_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(knownbad, "applies_to", lambda m: m == "fake:a")   # like SYSPIN: one voice
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "hi.txt").write_text("क्यों\n", encoding="utf-8")
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]})
    jid, _ = run(r, ["क्यों?"])
    assert states(r, jid) == [("DONE", "fake:b")]


def test_known_bad_is_learned_after_3_of_5_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(knownbad, "applies_to", lambda m: m == "fake:a")   # like SYSPIN: one voice
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:a": {"behavior": {"texts": {"खरच": "garbage"}}}},
                    retries=0)
    jid, _ = run(r, ["खरच", "खरच", "खरच", "खरच"])
    added = r.store.events(None, "known_bad_added")
    assert len(added) == 1 and json.loads(added[0]["detail"])["word"] == "खरच"
    last = [a["outcome"] for a in r.store.attempts(jid, 3)]
    assert last[0] == "known_bad", "the fourth line skips fake:a straight away"


def test_circuit_breaker_opens_skips_the_model_and_alerts(tmp_path):
    opts = {"fake:a": {"behavior": {"default": "garbage"}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts, retries=0,
                    breaker={"window": 4, "fail_ratio": 0.3, "cooldown_s": 600, "min_samples": 2})
    jid, res = run(r, [f"पंक्ति {i}" for i in range(5)])
    assert res["status"] == "DONE" and all(m == "fake:b" for _s, m in states(r, jid))
    assert [k for k, _m in r.alerts if k.startswith("breaker")] == ["breaker_open"]
    skipped = [a["outcome"] for a in r.store.attempts(jid) if a["model"] == "fake:a"]
    assert skipped.count("breaker_open") == 3, skipped


def test_a_failed_health_check_opens_the_breaker_before_any_line(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:a": {"behavior": {"health": "fail"}}})
    jid, _ = run(r, ["नमस्ते", "ठीक है"])
    assert states(r, jid) == [("DONE", "fake:b"), ("DONE", "fake:b")]
    assert "health check failed" in r.store.events(None, "breaker_open")[0]["detail"]


def test_oom_unloads_the_least_recently_used_model_and_retries_once(tmp_path):
    # Line 1 loads both models (a fails, b speaks). Line 2: a fails again, then
    # b runs out of memory once -> a (least recently used) is unloaded, b retried.
    opts = {"fake:a": {"behavior": {"default": "garbage"}},
            "fake:b": {"behavior": {"texts": {"तीन": "oom_once"}, "state_dir": str(tmp_path / "st")}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts, max_models_loaded=2, retries=0,
                    breaker={"window": 20, "fail_ratio": 0.3, "cooldown_s": 600, "min_samples": 20})
    jid, res = run(r, ["दो", "तीन"])
    assert res["status"] == "DONE" and states(r, jid) == [("DONE", "fake:b"), ("DONE", "fake:b")]
    evs = [(e["kind"], e["model"]) for e in r.store.events(jid)]
    assert ("oom", "fake:b") in evs and ("unload", "fake:a") in evs
    assert evs.index(("oom", "fake:b")) < evs.index(("unload", "fake:a"))


def test_second_oom_falls_back_to_cpu(tmp_path):
    opts = {"fake:a": {"device": "cuda-sim", "behavior": {"default": "oom_on_gpu"}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts, retries=0)
    jid, res = run(r, ["नमस्ते"])
    assert states(r, jid) == [("DONE", "fake:a")]
    assert r.store.events(None, "oom_cpu_fallback") and "fake:a" in r.pool.cpu_forced


def test_oom_with_no_cpu_fallback_moves_to_the_next_model(tmp_path):
    opts = {"fake:a": {"behavior": {"default": "oom"}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts, retries=0)
    jid, res = run(r, ["नमस्ते"])
    # fake supports CPU fallback, but plain "oom" fails on CPU too
    assert states(r, jid) == [("DONE", "fake:b")]
    assert [a["outcome"] for a in r.store.attempts(jid, 0)][0] == "oom"


def test_a_mixed_scene_is_re_rendered_with_the_highest_ranked_model_that_passes_all(tmp_path):
    opts = {"fake:a": {"behavior": {"texts": {"खराब": "garbage"}}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts, scenes=True)
    jid, _ = run(r, [{"text": t, "lang": "hi", "scene": "s1"} for t in ("नमस्ते", "खराब", "ठीक है")]
                 + [{"text": "अलग", "lang": "hi", "scene": "s2"}])
    assert states(r, jid) == [("DONE", "fake:b")] * 3 + [("DONE", "fake:a")]
    assert r.store.events(jid, "scene_rerender")


def test_the_allowlist_is_enforced_inside_the_worker_too(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]})
    r.cfg = dataclasses.replace(r.cfg, allow_test_models=False)     # the worker is told: no test models
    r.pool.cfg = r.cfg
    jid, res = run(r, ["नमस्ते"])
    assert res["counts"]["FLAGGED"] == 1
    a = r.store.attempts(jid, 0)[0]
    assert a["outcome"] == "unavailable" and "allowlisted for tests only" in a["detail"]
    assert not r.store.events(None, "breaker_open"), "a licence refusal is not a flaky model"


def test_rerunning_the_same_lines_reuses_verified_renders(tmp_path):
    r = make_runner(tmp_path, {"hi": ["fake:a"]})
    jid, _ = run(r, ["नमस्ते", "ठीक है"])
    r2 = make_runner(tmp_path, {"hi": ["fake:a"]})
    jid2 = r2.submit({"job_id": "again", "lines": [{"text": "नमस्ते", "lang": "hi"}, {"text": "ठीक है", "lang": "hi"}]})
    r2.run(jid2)
    assert [a["outcome"] for a in r2.store.attempts(jid2)] == ["cache_hit", "cache_hit"]


def test_the_health_endpoint_and_heartbeat_report_the_running_job(tmp_path):
    import socket
    import threading
    import time
    import urllib.request

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = make_runner(tmp_path, {"hi": ["fake:a"]}, {"fake:a": {"behavior": {"sleep_s": 0.5}}})
    r.health_port = port
    jid = r.submit({"lines": [{"text": f"पंक्ति {i}", "lang": "hi"} for i in range(6)]})
    t = threading.Thread(target=r.run, args=(jid,))
    t.start()
    snap = None
    for _ in range(100):
        try:
            snap = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read())
            if snap.get("line") is not None:
                break
        except OSError:
            pass
        time.sleep(0.2)
    t.join(120)
    assert snap and snap["job"] == jid
    for key in ("queue", "network", "models", "last_error", "loaded"):
        assert key in snap, key
    assert snap["queue"]["pending_lines"] >= 0 and "fake:a" in snap["models"]
    hb = json.loads((tmp_path / "home" / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["job"] == jid and hb["state"] == "stopped"


def test_crashes_count_against_the_breaker_but_never_teach_known_bad(tmp_path, monkeypatch):
    monkeypatch.setattr(knownbad, "applies_to", lambda m: m == "fake:a")
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, {"fake:a": {"behavior": {"texts": {"खरच": "crash"}}}},
                    retries=0)
    run(r, ["खरच"] * 4)
    assert not r.store.events(None, "known_bad_added"), "a crash says nothing about pronunciation"


def test_out_of_memory_while_loading_follows_the_ladder_to_cpu(tmp_path):
    opts = {"fake:a": {"device": "cuda-sim", "behavior": {"load": "oom_on_gpu"}}}
    r = make_runner(tmp_path, {"hi": ["fake:a", "fake:b"]}, opts, retries=0)
    jid, res = run(r, ["नमस्ते"])
    assert states(r, jid) == [("DONE", "fake:a")]
    kinds = [e["kind"] for e in r.store.events()]
    assert "oom" in kinds and "oom_cpu_fallback" in kinds and "fake:a" in r.pool.cpu_forced


def test_the_pool_itself_recovers_from_an_out_of_memory_at_load(tmp_path):
    """The pool's own contract, with no health check in front of it: a model
    that cannot load on the GPU is loaded on the CPU instead."""
    from app.tts_runtime.pool import ModelPool

    cfg = config.parse({"chains": {"hi": ["fake:a"], "default": ["fake:a"]}, "offline_mode": "off"},
                       allow_test_models=True, source="test")
    opts = {"device": "cuda-sim", "behavior": {"load": "oom_on_gpu"}}
    pool = ModelPool(cfg, tmp_path, lambda m: opts)
    try:
        r = pool.synth("fake:a", "नमस्ते", "hi", None, None, 0, str(tmp_path / "out.wav"))
        assert r["ok"] and (tmp_path / "out.wav").exists()
        assert pool._workers["fake:a"].device == "cpu" and "fake:a" in pool.cpu_forced
    finally:
        pool.close_all()
