"""Run one dub end to end through the public API, then check the output.

    python scripts/e2e_dub.py --video clip.mp4 --target te [--clone] [--asr-check]
    python scripts/e2e_dub.py --project <id> [--asr-check]     # verify an existing run

Drives the API exactly as the frontend does (create -> presigned upload ->
confirm -> process -> language gate -> poll -> download), so it needs the API
and both workers running (scripts/run-local.ps1). Exits non-zero if any check
fails; writes result.json and report.json to --out.

Checks: output duration equals the source; speech energy at every placed clip
and silence before the first line and between lines; no clip overruns or
overlapping speech (from the QA report); every spoken segment rendered.
--asr-check adds Whisper large-v3 (slow on CPU): the dubbed track is
auto-detected as the target language, and speech read back per line (with
context) scores corpus CER <= 0.25, every line of 3+ words <= 0.35, and every 1-2 word line <= 0.35 scored
inside the surrounding dub (short clips alone sit below Whisper's floor).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app.pipeline.ffmpeg_utils import tool_path  # noqa: E402


class Api:
    def __init__(self, base: str, email: str) -> None:
        self.base = base.rstrip("/")
        self.client = httpx.Client(base_url=self.base, headers={"X-User-Email": email}, timeout=300)

    def call(self, method: str, path: str, **kw):
        for attempt in range(4):
            try:
                resp = self.client.request(method, path, **kw)
                break
            except httpx.TransportError:
                # uvicorn drops idle keep-alive connections after 5s; only
                # idempotent requests are retried.
                if method != "GET" or attempt == 3:
                    raise
                time.sleep(1)
        if resp.status_code >= 400:
            sys.exit(f"HTTP {resp.status_code} {method} {path}: {resp.text}")
        return resp.json() if resp.content and "json" in resp.headers.get("content-type", "") else resp

    def url(self, maybe_relative: str) -> str:
        return maybe_relative if maybe_relative.startswith("http") else self.base + maybe_relative


def run_dub(api: Api, args) -> str:
    project = api.call("POST", "/api/projects", json={"title": f"e2e {os.path.basename(args.video)}",
                                                     "target_languages": [args.target]})
    pid = project["id"]
    print("project", pid, flush=True)
    up = api.call("POST", f"/api/projects/{pid}/upload",
                  json={"filename": os.path.basename(args.video), "content_type": "video/mp4"})
    with open(args.video, "rb") as f:
        httpx.put(api.url(up["upload_url"]), content=f.read(), headers={"Content-Type": "video/mp4"},
                  timeout=600).raise_for_status()
    api.call("POST", f"/api/projects/{pid}/upload/confirm", json={"source_video_id": up["source_video_id"]})
    api.call("POST", f"/api/projects/{pid}/process", json={
        "preserve_emotion": True, "clone_voice": args.clone, "lip_sync_aware": False,
        "review_language": args.source is None, "source_language": args.source,
    })

    started, last = time.time(), None
    while True:
        p = api.call("GET", f"/api/projects/{pid}")
        if (p["status"], p["current_stage"]) != last:
            last = (p["status"], p["current_stage"])
            print(f"[{time.time() - started:7.1f}s] {p['status']} {p['current_stage'] or ''}", flush=True)
        if p["status"] == "failed":
            sys.exit(f"pipeline failed: {p['error_message']} (permanent={p['error_is_permanent']})")
        if p["status"] == "awaiting_language_confirmation":
            p = api.call("POST", f"/api/projects/{pid}/confirm-language", json={})
            print(f"   accepted detected source language: {p['source_language']}", flush=True)
        if p["status"] == "ready":
            return pid
        time.sleep(3)


def download(api: Api, pid: str, out: str) -> dict:
    project = api.call("GET", f"/api/projects/{pid}")
    segments = api.call("GET", f"/api/projects/{pid}/segments")
    export = api.call("GET", f"/api/projects/{pid}/export")
    video = httpx.get(api.url(export["output_url"]), timeout=600)
    video.raise_for_status()
    with open(os.path.join(out, "dubbed.mp4"), "wb") as f:
        f.write(video.content)
    result = {"project": project, "segments": segments, "export": export}
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# --- checks ------------------------------------------------------------------
def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", s.lower())
                   if not unicodedata.category(c).startswith(("P", "Z", "S")))


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def _cer_spoken_numerals(ref: str, hyp: str) -> float:
    """CER that does not charge for a number written as a digit.

    The pipeline spells numerals out before translating ("at 9" -> "at nine"
    -> the Telugu word), because the TTS vocabularies have no digits. Whisper
    then writes the number it hears back as "9", and comparing that with the
    eight-character word the dub actually speaks cost ~8 edits on a
    36-character line: 0.17 CER of pure orthography, enough to fail a line
    whose speech was fine.

    So each digit-only token in the hypothesis is dropped along with at most
    one reference word -- whichever choice scores best. One word, not a free
    wildcard: a hypothesis of nothing but "9" has to stay a total miss rather
    than absorbing the whole line."""
    h_tokens = hyp.split()
    digits = [t for t in h_tokens if t.strip("().,!?;:-").isdigit()]
    if not digits:
        return _cer(ref, hyp)
    kept_hyp = " ".join(t for t in h_tokens if t not in digits)
    r_tokens = ref.split()
    best = _cer(ref, kept_hyp)
    for drop in range(len(r_tokens)):
        best = min(best, _cer(" ".join(r_tokens[:drop] + r_tokens[drop + 1:]), kept_hyp))
    return best


# --- how short is too short to ask a recogniser about ------------------------
# Measured 2026-09-20: fifteen correctly-rendered SYSPIN Telugu clips, each
# padded with 1.1s of silence exactly as the dub places a line, scored the way
# this gate scores a short line.
#
#    372ms 1.000 | 418ms 0.500 | 511ms 0.250 |  534ms 1.333 |  627ms 0.429
#    673ms 0.833 | 766ms 0.800 | 789ms 0.500 |  998ms 0.167 | 1022ms 0.400
#   1022ms 0.857 | 1091ms 0.750 || 1207ms 0.083 | 1231ms 0.250 | 1602ms 0.312
#
# Ten of the twelve clips under 1100ms failed the 0.35 bar on audio that is
# correct -- the same word reads back correctly inside a carrier sentence, and
# every one of these clips measures as ordinary speech (voiced fraction
# 0.44-0.88). Every clip from 1207ms up passed. The floor sits above every
# observed false failure and below every reliable read.
#
# Under it, a line is not scored by ASR at all: at that length the number
# measures the recogniser, not the dub. It gets a presence check instead --
# there is speech here, at level, where the timeline says it should be.
ASR_FLOOR_MS = 1200
# Same bar as speech_at_every_clip, so "present" means the same thing twice.
PRESENCE_MIN_RMS_DB = -45.0
# Real speech measured 0.44-0.88 across those fifteen clips and 0.76-0.92
# across the dub's own lines. Silence, white noise and a click all measure
# 0.00 -- pitch is what separates speech from something merely audible. (A
# pure tone would pass at 0.97, which is what the mock TTS renders; a mocked
# run fails the corpus CER long before it reaches here.)
PRESENCE_MIN_VOICED = 0.30


def scores_by_asr(fitted_ms: int) -> bool:
    """Whether this clip is long enough for its CER to mean anything."""
    return fitted_ms >= ASR_FLOOR_MS


def presence_verdict(rms_db: float, voiced_fraction: float) -> bool:
    """Speech is here: audible, and pitched rather than merely audible."""
    return rms_db > PRESENCE_MIN_RMS_DB and voiced_fraction >= PRESENCE_MIN_VOICED


def measure_presence(wav, sr: int) -> dict:
    import librosa
    import numpy as np

    rms_db = float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-12))
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=400, sr=sr, frame_length=1024)
    voiced_fraction = float(np.mean(voiced & np.isfinite(f0))) if len(f0) else 0.0
    return {"rms_db": round(rms_db, 1), "voiced_fraction": round(voiced_fraction, 2)}


def verify(result: dict, out: str, source_duration_s: float | None, asr_check: bool) -> bool:
    ffmpeg, ffprobe = tool_path("ffmpeg"), tool_path("ffprobe")
    dub = os.path.join(out, "dubbed.mp4")
    report: dict = {}

    def check(name, ok, detail):
        report[name] = {"ok": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + f"{name}: {detail}", flush=True)

    def rms_db(start_s: float, dur_s: float) -> float:
        err = subprocess.run([ffmpeg, "-hide_banner", "-ss", f"{start_s:.3f}", "-t", f"{max(dur_s, 0.05):.3f}",
                              "-i", dub, "-af", "astats=metadata=1", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        vals = re.findall(r"RMS level dB:\s*(-?[\d.]+|-inf)", err)
        return -120.0 if not vals or vals[0] == "-inf" else float(vals[0])

    qa = result["export"]["qa_report"]
    fits = {r["segment_id"]: r for r in qa["segments"]}
    spoken = [s for s in result["segments"] if (s["source_text"] or "").strip()]
    placed = sorted((s for s in result["segments"] if s["tts_audio_url"]), key=lambda s: s["start_ms"])

    probe = json.loads(subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", dub],
                                      capture_output=True, check=True).stdout)
    dub_s = float(probe["format"]["duration"])
    expected_s = source_duration_s or qa["overall"]["duration_ms"] / 1000
    check("duration_matches_source", abs(dub_s - expected_s) <= 0.1, f"expected {expected_s:.3f}s, got {dub_s:.3f}s")
    check("every_spoken_segment_rendered", all(s["tts_audio_url"] and s["translated_text"] for s in spoken),
          f"{len(spoken)} spoken of {len(result['segments'])}")
    levels = [(s["index"], round(rms_db(s["start_ms"] / 1000 + 0.05, fits[s["id"]]["fitted_ms"] / 1000 - 0.15), 1))
              for s in placed]
    check("speech_at_every_clip", all(db > -45 for _, db in levels), levels)
    if placed and placed[0]["start_ms"] > 200:
        lead = rms_db(0, placed[0]["start_ms"] / 1000 - 0.05)
        check("silence_before_first_line", lead < -60, f"{lead:.1f} dB")
    gaps = []
    for a, b in zip(placed, placed[1:]):
        end_a = a["start_ms"] + fits[a["id"]]["fitted_ms"]
        if b["start_ms"] - end_a > 400:
            gaps.append((a["index"], round(rms_db(end_a / 1000 + 0.12, (b["start_ms"] - end_a) / 1000 - 0.24), 1)))
    check("silence_between_lines", all(db < -60 for _, db in gaps), gaps)
    check("no_overruns", qa["overall"]["total_overrun_ms"] == 0, qa["overall"])
    overlaps = [(a["index"], b["index"]) for a, b in zip(placed, placed[1:])
                if a["start_ms"] + fits[a["id"]]["fitted_ms"] > b["start_ms"]]
    check("no_overlapping_speech", not overlaps, overlaps)

    if asr_check:
        import soundfile as sf
        from faster_whisper import WhisperModel

        target = result["project"]["target_languages"][0]
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "dub16k.wav")
            subprocess.run([ffmpeg, "-y", "-v", "error", "-i", dub, "-ac", "1", "-ar", "16000", wav], check=True)
            audio, sr = sf.read(wav, dtype="float32")
        _, det = model.transcribe(audio, beam_size=1)
        check("dub_language_detected", det.language == target and det.language_probability >= 0.8,
              f"{det.language} ({det.language_probability:.2f})")
        # Per line with context, aggregated -- a single long-form decode of the
        # whole track can skip lines outright.
        lines, dist, chars = [], 0.0, 0
        for i, s in enumerate(placed):
            ref = s["translated_text"]
            fitted_ms = fits[s["id"]]["fitted_ms"]
            start_ms, end_ms = s["start_ms"], s["start_ms"] + fitted_ms
            if not scores_by_asr(fitted_ms):
                # Too short to transcribe meaningfully -- check it is there.
                clip = audio[int(start_ms / 1000 * sr):min(int(end_ms / 1000 * sr), len(audio))]
                row = {"index": s["index"], "words": len(ref.split()), "scored": "presence",
                       "fitted_ms": fitted_ms, "ref": ref, **measure_presence(clip, sr)}
                row["ok"] = presence_verdict(row["rms_db"], row["voiced_fraction"])
                lines.append(row)
                print(f"   line {s['index']} {fitted_ms}ms < {ASR_FLOOR_MS}ms floor -- presence check "
                      f"{'ok' if row['ok'] else 'FAILED'}: {row['rms_db']} dBFS, "
                      f"voiced {row['voiced_fraction']:.2f}", flush=True)
                print(f"     ref: {ref}", flush=True)
                continue
            a = max(int((start_ms - 300) / 1000 * sr), 0)
            b = int((end_ms + 300) / 1000 * sr)
            hyp = " ".join(x.text.strip() for x in model.transcribe(audio[a:b], language=target, beam_size=5)[0])
            # The source had a numeral only if the pipeline spelled one out.
            score = _cer_spoken_numerals if any(ch.isdigit() for ch in (s["source_text"] or "")) else _cer
            cer, n = score(ref, hyp), len(_norm(ref))
            dist, chars = dist + cer * n, chars + n
            row = {"index": s["index"], "words": len(ref.split()), "scored": "asr",
                   "fitted_ms": fitted_ms, "cer": round(cer, 3), "ref": ref, "hyp": hyp}
            if row["words"] < 3:
                # A one- or two-word clip decoded on its own is below what
                # Whisper recognises reliably, which measures the recogniser,
                # not the voice. Short lines are scored instead inside the dub
                # around them (previous line start .. next line end), keeping
                # the words Whisper places within this line's own window --
                # and then held to the same bar as full sentences.
                ctx_a = placed[i - 1]["start_ms"] if i else max(start_ms - 3000, 0)
                ctx_b = (placed[i + 1]["start_ms"] + fits[placed[i + 1]["id"]]["fitted_ms"]
                         if i + 1 < len(placed) else end_ms + 3000)
                segs, _ = model.transcribe(audio[int(ctx_a / 1000 * sr):min(int(ctx_b / 1000 * sr), len(audio))],
                                           language=target, beam_size=5, word_timestamps=True)
                w0, w1 = (start_ms - ctx_a) / 1000 - 0.12, (end_ms - ctx_a) / 1000 + 0.12
                # Anchored on each word's END, not its midpoint. Whisper
                # stretches a word's start back over whatever silence precedes
                # it -- the dub's first word was timed from 0.00 though the
                # line starts at 0.908 -- which pushed the midpoint outside the
                # window and dropped a word that was read correctly (CER 0.29
                # standalone, 0.64 after the filter threw half of it away).
                # Lines are separated by silence here, so a word ending inside
                # this line's window belongs to this line.
                row["context_hyp"] = " ".join(w.word.strip() for seg in segs for w in (seg.words or [])
                                              if w0 <= w.end <= w1)
                row["context_cer"] = round(score(ref, row["context_hyp"]), 3)
            lines.append(row)
            print(f"   line {s['index']} CER {cer:.3f}" + (f" (in context {row['context_cer']:.3f})" if "context_cer" in row else "")
                  + f"\n     ref: {ref}\n     hyp: {hyp}", flush=True)
        scored = [r for r in lines if r["scored"] == "asr"]
        by_presence = [r for r in lines if r["scored"] == "presence"]
        check("dub_intelligible", dist / max(chars, 1) <= 0.25,
              f"corpus CER {dist / max(chars, 1):.3f} over {len(scored)} of {len(lines)} lines")
        check("full_sentences_intelligible", all(r["cer"] <= 0.35 for r in scored if r["words"] >= 3),
              [(r["index"], r["cer"]) for r in scored if r["words"] >= 3])
        check("short_lines_intelligible", all(r["context_cer"] <= 0.35 for r in scored if r["words"] < 3),
              [(r["index"], r["cer"], r["context_cer"]) for r in scored if r["words"] < 3])
        # Lines below the floor are held to presence, and counted, so that
        # "not scored" can never quietly become "not checked".
        check("short_clips_present", all(r["ok"] for r in by_presence),
              [(r["index"], r["fitted_ms"], r["rms_db"], r["voiced_fraction"]) for r in by_presence])
        report["lines"] = lines
        report["asr_floor_ms"] = ASR_FLOOR_MS

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return all(v["ok"] for k, v in report.items() if isinstance(v, dict) and "ok" in v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="source video to dub")
    src.add_argument("--project", help="verify an existing, finished project instead of running one")
    ap.add_argument("--target", default="te")
    ap.add_argument("--source", default=None, help="pin the source language (default: detect, then accept)")
    ap.add_argument("--clone", action="store_true", help="clone the original speakers' voices")
    ap.add_argument("--asr-check", action="store_true", help="also score intelligibility with Whisper large-v3")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--user-email", default="e2e@sur.local")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    api = Api(args.api, args.user_email)
    pid = args.project or run_dub(api, args)
    out = args.out or os.path.join("data", "e2e", pid)
    os.makedirs(out, exist_ok=True)
    result = download(api, pid, out)

    source_s = None
    if args.video:
        probe = subprocess.run([tool_path("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json",
                                args.video], capture_output=True, check=True).stdout
        source_s = float(json.loads(probe)["format"]["duration"])
    ok = verify(result, out, source_s, args.asr_check)
    print(("E2E PASSED" if ok else "E2E FAILED") + f" -> {out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
