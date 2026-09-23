"""`tts benchmark`: the evidence behind chain order and trigger thresholds.

Measures exactly what the runtime runs (same pool, adapters, checks).

Test set per language: the pipeline's short lines (41 across te/hi/kn/mr/bn),
the known-bad words, and 20 sentences (data/benchmark_sentences.json).
Draw sets: A = draws 0,1,2 and B = draws 3,4,5 -- three seeds each, two
independent sets. Draw 0 is exactly what production renders.

Harness first: before any number is trusted, a known-perfect sample must
score perfectly (reader CER 0 on a clip measured to read back exactly),
CER must be 0 for every reference against itself, and silence/noise must
fail the signal checks. If any of that fails, the benchmark stops.

Calibration: the duration (chars per second of speech) and silence
thresholds are derived from set A with a margin fixed before looking
(25% on speech rate, 50% on silence), then evaluated on held-out set B
(false-positive rate) and on corrupted copies of set B (detection rate).

Reader CER only where a reader is reliable: full sentences in all five
SYSPIN languages; short lines only in te and kn (CONTRACTS.md #7).
Chain order: a model is moved ahead of another only if its sentence CER
95% CI lies entirely below the other's in BOTH sets and its sanity pass
rate is not significantly worse. Otherwise the order stays.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from app.tts_runtime.checks import (
    CALIBRATION_PATH,
    Calibration,
    cer,
    envelope,
    length_class,
    n_chars,
    signal_checks,
)

logger = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parent / "data"
SET_A, SET_B = (0, 1, 2), (3, 4, 5)
RATE_MARGIN, SILENCE_MARGIN = 1.25, 1.5          # fixed before looking at any data
SHORT_READER_LANGS = {"te", "kn"}
SENTENCE_READER_LANGS = {"te", "kn", "hi", "mr", "bn"}
KNOWN_PERFECT = ("te", "అవును.")                  # CONTRACTS.md #7 table: CER 0.000 on the production render


class HarnessError(RuntimeError):
    pass


# --- statistics ----------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_mean_ci(values, iters: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(iters, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- test set ----------------------------------------------------------------------------
def test_set(lang: str, known_bad_dir: Path) -> list[tuple[str, str]]:
    from app.tts_runtime.knownbad import KnownBad

    short = json.loads((DATA / "benchmark_short_words.json").read_text(encoding="utf-8"))["langs"].get(lang, [])
    sents = json.loads((DATA / "benchmark_sentences.json").read_text(encoding="utf-8"))["langs"].get(lang, [])
    kb = sorted(KnownBad(None, directory=known_bad_dir).words_for(lang))
    rows = [("short", t) for t in short] + [("known_bad", w) for w in kb if w not in {t.rstrip("?.।") for t in short}]
    return rows + [("sentence", t) for t in sents]


# --- collection ------------------------------------------------------------------------------
def collect(runner, models: list[str], langs: list[str], draws=SET_A + SET_B, log=print) -> list[dict]:
    """Render every (model, lang, text, draw); resumable (rows persist in bench/rows.jsonl)."""
    from app.tts_runtime.adapters import adapter_class
    from app.tts_runtime.pool import WorkerError

    bench = runner.home / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    rows_path = bench / "rows.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[(r["model"], r["lang"], r["text"], r["draw"])] = r
            except json.JSONDecodeError:
                pass
    out = open(rows_path, "a", encoding="utf-8")
    rows = list(done.values())
    for model in models:
        cls = adapter_class(model)
        for lang in langs:
            if not cls.supports(lang):
                continue
            ok, why = cls.availability(runner.options_for(model))
            if not ok:
                rows.append({"model": model, "lang": lang, "not_runnable": why})
                log(f"  {model}/{lang}: not runnable -- {why}")
                continue
            for kind, text in test_set(lang, runner.knownbad.dir):
                for d in draws:
                    if (model, lang, text, d) in done:
                        continue
                    tag = hashlib.sha256(f"{model}:{lang}:{text}".encode()).hexdigest()[:12]
                    wav = bench / "audio" / model.replace(":", "_") / lang / f"{tag}-d{d}.wav"
                    row = {"model": model, "lang": lang, "kind": kind, "text": text, "draw": d,
                           "set": "A" if d in SET_A else "B"}
                    try:
                        r = runner.pool.synth(model, text, lang, None, None, d, str(wav))
                        row.update(path=str(wav), secs=round(r.get("secs", 0), 3), rss_mb=r.get("rss_mb"),
                                   vram_mb=r.get("vram_mb"), awake_s=r.get("awake_s"),
                                   suspended_s=r.get("suspended_s"))
                    except WorkerError as e:
                        row.update(error=f"{e.kind}: {e.detail[:200]}")
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(row)
            log(f"  {model}/{lang}: rendered")
    out.close()
    return rows


def read_back(runner, rows: list[dict], log=print) -> None:
    """Reader CER where reliable; cached in the row files."""
    import soundfile as sf

    from app.tts_runtime.audio import as_mono_float, resample
    from app.tts_runtime.pool import WorkerError

    cache_path = runner.home / "bench" / "read.json"
    cached = {}
    if cache_path.exists():
        try:
            cached = {c["path"]: c for c in json.loads(cache_path.read_text(encoding="utf-8")) if c.get("path")}
        except (json.JSONDecodeError, KeyError):
            cached = {}
    for r in rows:
        if "path" in r and r["path"] in cached and "cer" in cached[r["path"]]:
            for k in ("audio_s", "speech_s", "lead_s", "trail_s", "cps", "clip_fraction", "cls", "cer", "hyp",
                      "reader_error"):
                if k in cached[r["path"]]:
                    r[k] = cached[r["path"]][k]
    for r in rows:
        if "path" not in r or "cer" in r:
            continue
        reliable = (r["kind"] == "sentence" and r["lang"] in SENTENCE_READER_LANGS) or \
                   (r["kind"] != "sentence" and r["lang"] in SHORT_READER_LANGS)
        x, sr = sf.read(r["path"], dtype="float32")
        sp, lead, trail = envelope(x, sr)
        r.update(audio_s=round(len(x) / sr, 3), speech_s=round(sp, 3), lead_s=round(lead, 3), trail_s=round(trail, 3),
                 cps=round(n_chars(r["text"]) / sp, 3) if sp > 0 else None,
                 clip_fraction=float(np.mean(np.abs(x) >= 0.999)), cls=length_class(r["text"]))
        if not reliable:
            r["cer"] = None
            continue
        w16 = Path(r["path"]).with_suffix(".16k.wav")
        if not w16.exists():
            sf.write(str(w16), resample(as_mono_float(x), sr, 16000), 16000, subtype="FLOAT")
        try:
            hyp = runner.pool.read("vakyansh", str(w16), r["lang"])
            r.update(hyp=hyp, cer=round(cer(r["text"], hyp), 4))
        except WorkerError as e:
            r.update(cer=None, reader_error=f"{e.kind}: {e.detail[:200]}")
    (runner.home / "bench" / "read.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


# --- harness verification ------------------------------------------------------------------
def verify_harness(runner, rows: list[dict]) -> dict:
    out = {}
    texts = {r["text"] for r in rows if "text" in r}
    bad = [t for t in texts if cer(t, t) != 0]
    if bad:
        raise HarnessError(f"CER of a reference against itself is not 0 for {bad[:3]}")
    out["self_cer_zero"] = len(texts)
    lang, text = KNOWN_PERFECT
    kp = [r for r in rows if r.get("model", "").startswith("syspin") and r.get("lang") == lang
          and r.get("text") == text and r.get("draw") == 0]
    if kp and kp[0].get("cer") is not None:
        if kp[0]["cer"] != 0:
            raise HarnessError(f"known-perfect sample {text!r} read back as {kp[0].get('hyp')!r} "
                               f"(CER {kp[0]['cer']}), expected 0 -- the reader or decode is broken")
        out["known_perfect"] = f"{text!r} -> {kp[0].get('hyp')!r}, CER 0"
    sr = 22050
    silence = np.zeros(sr, dtype=np.float32)
    noise = (0.3 * np.random.default_rng(0).standard_normal(10 * sr)).astype(np.float32)
    cal = Calibration({}, {"short": {"cps_lo": 3, "cps_hi": 30}, "long": {"cps_lo": 3, "cps_hi": 30}}, 1.0, 1.0)
    if signal_checks(silence, sr, "నమస్తే", "te", cal).passed:
        raise HarnessError("silence passed the signal checks")
    if signal_checks(noise, sr, "నమస్తే", "te", cal).passed:
        raise HarnessError("10 s of noise for a 6-character word passed the signal checks")
    out["null_samples_rejected"] = 2
    return out


# --- calibration ---------------------------------------------------------------------------
def derive_calibration(rows: list[dict], langs: list[str]) -> dict:
    A = [r for r in rows if r.get("set") == "A" and r.get("cps") and r.get("model", "").startswith("syspin")]
    ranges, allc = {}, {"short": [], "long": []}
    for lang in langs:
        for cls in ("short", "long"):
            v = [r["cps"] for r in A if r["lang"] == lang and r["cls"] == cls]
            if v:
                ranges.setdefault(lang, {})[cls] = {"cps_lo": round(min(v) / RATE_MARGIN, 3),
                                                    "cps_hi": round(max(v) * RATE_MARGIN, 3), "n": len(v)}
                allc[cls] += v
    default = {cls: {"cps_lo": round(min(v) / RATE_MARGIN, 3), "cps_hi": round(max(v) * RATE_MARGIN, 3), "n": len(v)}
               for cls, v in allc.items() if v}
    lead = max(r["lead_s"] for r in A) * SILENCE_MARGIN
    trail = max(r["trail_s"] for r in A) * SILENCE_MARGIN
    return {"method": f"set A (draws {SET_A}) SYSPIN renders, all of them; speech-rate range = observed min/max "
                      f"widened by {RATE_MARGIN}x, silence = observed max x {SILENCE_MARGIN}; margins fixed "
                      f"before looking. Evaluated on held-out set B.",
            "generated_by": "tts benchmark", "created": time.strftime("%Y-%m-%d"),
            "ranges": ranges, "default": default,
            "silence": {"max_lead_s": round(max(lead, 0.3), 3), "max_trail_s": round(max(trail, 0.3), 3)},
            "clip": {"level": 0.999, "max_fraction": 0.001}}


def _corruptions(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    import librosa

    return {"slowed_x2": librosa.effects.time_stretch(x, rate=0.5),
            "sped_x2": librosa.effects.time_stretch(x, rate=2.0),
            "truncated_40pct": x[: int(0.4 * len(x))],
            "noise_10s": (0.3 * np.random.default_rng(1).standard_normal(10 * sr)).astype(np.float32),
            "lead_silence_2_5s": np.concatenate([np.zeros(int(2.5 * sr), np.float32), x]),
            "clipped": np.clip(x * 20, -1, 1)}


def evaluate_calibration(rows: list[dict], cal: Calibration) -> dict:
    import soundfile as sf

    B = [r for r in rows if r.get("set") == "B" and r.get("path") and r.get("model", "").startswith("syspin")]
    fp = [r for r in B if not signal_checks(sf.read(r["path"], dtype="float32")[0], 22050, r["text"], r["lang"], cal).passed]
    out = {"set_B_renders": len(B), "false_positives": len(fp), "fpr_ci95": wilson(len(fp), len(B)),
           "false_positive_examples": [{"lang": r["lang"], "text": r["text"], "draw": r["draw"]} for r in fp[:10]]}
    det: dict[str, list[int]] = {}
    for r in B[::3]:                                   # every third render: enough for a CI, keeps it fast
        x, sr = sf.read(r["path"], dtype="float32")
        for name, y in _corruptions(x, sr).items():
            det.setdefault(name, []).append(int(not signal_checks(y.astype(np.float32), sr, r["text"], r["lang"], cal).passed))
    out["detection"] = {k: {"caught": sum(v), "n": len(v), "ci95": wilson(sum(v), len(v))} for k, v in det.items()}
    return out


# --- summary -----------------------------------------------------------------------------------
def latency_of(r: dict, timeout_s: float) -> float | None:
    """Seconds the model actually computed. Awake time when the worker reports
    it; otherwise wall time, unless it exceeds the synth timeout -- the parent
    would have killed any real render that long, so the machine was asleep."""
    if r.get("awake_s") is not None:
        return r["awake_s"]
    s = r.get("secs")
    return None if s is None or s > timeout_s else s


def summarize(rows: list[dict], cal: Calibration, timeout_s: float = 60.0) -> list[dict]:
    import soundfile as sf

    table = []
    keys = sorted({(r["model"], r["lang"]) for r in rows})
    for model, lang in keys:
        nr = [r for r in rows if r["model"] == model and r["lang"] == lang and "not_runnable" in r]
        if nr:
            table.append({"model": model, "lang": lang, "not_runnable": nr[0]["not_runnable"]})
            continue
        for kind in ("short", "known_bad", "sentence"):
            for s in ("A", "B"):
                R = [r for r in rows if r["model"] == model and r["lang"] == lang and r.get("kind") == kind
                     and r.get("set") == s]
                if not R:
                    continue
                ok = [r for r in R if r.get("path")]
                sane = sum(signal_checks(sf.read(r["path"], dtype="float32")[0], 22050 if model.startswith("syspin")
                                         else sf.info(r["path"]).samplerate, r["text"], lang, cal).passed for r in ok)
                by_line: dict[str, list[float]] = {}
                for r in ok:
                    if r.get("cer") is not None:
                        by_line.setdefault(r["text"], []).append(r["cer"])
                line_cer = [float(np.mean(v)) for v in by_line.values()]
                cer_m, cer_lo, cer_hi = boot_mean_ci(line_cer)
                passes = [int(c <= 0.35) for v in by_line.values() for c in v]
                timed = [r for r in ok if latency_of(r, timeout_s) is not None]
                lat = boot_mean_ci([latency_of(r, timeout_s) for r in timed])
                rtf = boot_mean_ci([latency_of(r, timeout_s) / r["audio_s"] for r in timed if r.get("audio_s")])
                table.append({
                    "model": model, "lang": lang, "kind": kind, "set": s, "renders": len(R),
                    "errors": len(R) - len(ok), "lines": len({r["text"] for r in R}),
                    "sanity_pass": sane, "sanity_n": len(ok), "sanity_ci95": wilson(sane, len(ok)),
                    "cer_mean": cer_m if line_cer else None, "cer_ci95": (cer_lo, cer_hi) if line_cer else None,
                    "reader_pass": sum(passes) if passes else None, "reader_n": len(passes) or None,
                    "reader_pass_ci95": wilson(sum(passes), len(passes)) if passes else None,
                    "latency_s": lat[0], "latency_ci95": (lat[1], lat[2]), "rtf": rtf[0],
                    "latency_excluded": len(ok) - len(timed),
                    "rss_mb": max((r.get("rss_mb") or 0) for r in ok) if ok else None,
                    "vram_mb": max((r.get("vram_mb") or 0) for r in ok) if ok else None})
    return table


def chain_order(table: list[dict], cfg) -> dict[str, dict]:
    """The pre-registered rule; returns lang -> {order, reason}."""
    out = {}
    for lang in sorted(cfg.chains):
        chain = list(cfg.chain_for(lang))
        runnable = [m for m in chain if any(t["model"] == m and t["lang"] == lang and "not_runnable" not in t
                                            for t in table)]
        if len(runnable) < 2:
            why = ("only runnable model: " + runnable[0]) if runnable else "no runnable model"
            out[lang] = {"order": chain, "changed": False,
                         "reason": f"{why}; the others are not runnable here (see table) -- order unchanged"}
            continue

        def sent(m, s, lang=lang):
            return next((t for t in table if t["model"] == m and t["lang"] == lang and t.get("kind") == "sentence"
                         and t["set"] == s), None)

        order = list(chain)
        changed = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                if a not in runnable or b not in runnable:
                    continue
                better = all(sent(b, s) and sent(a, s) and sent(b, s)["cer_ci95"] and sent(a, s)["cer_ci95"]
                             and sent(b, s)["cer_ci95"][1] < sent(a, s)["cer_ci95"][0]
                             and sent(b, s)["sanity_ci95"][1] >= sent(a, s)["sanity_ci95"][0] for s in ("A", "B"))
                if better:
                    order[i], order[j] = b, a
                    changed = True
        out[lang] = {"order": order, "changed": changed,
                     "reason": "reordered: sentence-CER CIs separate in both sets" if changed
                     else "no model's sentence CER is better with non-overlapping CIs in both sets -- order unchanged"}
    return out


def markdown(res: dict) -> str:
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))   # noqa: E731
    ci = lambda c: "-" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"                           # noqa: E731
    lines = ["# TTS benchmark", "", f"Run {res['created']} on {res['host']}.", "",
             "## Harness verification", ""] + [f"- {k}: {v}" for k, v in res["harness"].items()]
    ev = res["calibration_eval"]
    lines += ["", "## Trigger calibration (derived on set A, evaluated on held-out set B)", "",
              f"False positives on set B: {ev['false_positives']}/{ev['set_B_renders']} "
              f"(95% CI {ci(ev['fpr_ci95'])})", "", "| corruption | caught | 95% CI |", "|---|---|---|"]
    lines += [f"| {k} | {v['caught']}/{v['n']} | {ci(v['ci95'])} |" for k, v in ev["detection"].items()]
    excl = sum(t.get("latency_excluded", 0) for t in res["table"])
    lines += ["", "## Results", "",
              f"Latency is compute time: {excl} render(s) whose wall time spanned a machine sleep are excluded "
              "from latency (their audio and CER are kept).", "",
              "| model | lang | kind | set | lines x seeds | sanity pass | reader CER mean [95% CI] | "
              "reader pass | latency s | RTF | peak RSS MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in res["table"]:
        if "not_runnable" in t:
            lines.append(f"| {t['model']} | {t['lang']} | - | - | - | not runnable: {t['not_runnable'][:80]} | | | | | |")
            continue
        rp = "-" if t["reader_n"] is None else f"{t['reader_pass']}/{t['reader_n']}"
        lines.append(f"| {t['model']} | {t['lang']} | {t['kind']} | {t['set']} | {t['lines']}x{t['renders'] // max(t['lines'], 1)} "
                     f"| {t['sanity_pass']}/{t['sanity_n']} | {f(t['cer_mean'])} {ci(t['cer_ci95'])} | {rp} | "
                     f"{f(t['latency_s'])} | {f(t['rtf'])} | {f(t['rss_mb'])} |")
    lines += ["", "## Chain order", ""] + [f"- **{k}**: {' > '.join(v['order'])} -- {v['reason']}"
                                          for k, v in res["chain_order"].items()]
    return "\n".join(lines) + "\n"


def run(runner, models: list[str], langs: list[str], write_calibration: bool = True, log=print) -> dict:
    import platform

    t0 = time.time()
    rows = collect(runner, models, langs, log=log)
    log(f"collected {len(rows)} rows in {time.time() - t0:.0f}s; reading back")
    read_back(runner, rows, log=log)
    harness = verify_harness(runner, rows)
    log(f"harness verified: {harness}")
    cal_dict = derive_calibration(rows, langs)
    cal_path = runner.home / "bench" / "calibration.json"
    cal_path.write_text(json.dumps(cal_dict, indent=1), encoding="utf-8")
    cal = Calibration.load(cal_path)
    ev = evaluate_calibration(rows, cal)
    table = summarize(rows, cal, runner.cfg.synth_timeout_s)
    res = {"created": time.strftime("%Y-%m-%d %H:%M"), "host": f"{platform.system()} {platform.machine()}, CPU only",
           "harness": harness, "calibration": cal_dict, "calibration_eval": ev, "table": table,
           "chain_order": chain_order(table, runner.cfg), "seconds": round(time.time() - t0)}
    (runner.home / "bench" / "benchmark.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str),
                                                          encoding="utf-8")
    (runner.home / "bench" / "benchmark.md").write_text(markdown(res), encoding="utf-8")
    if write_calibration:
        CALIBRATION_PATH.write_text(json.dumps(cal_dict, indent=1), encoding="utf-8")
    return res
