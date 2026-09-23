"""Triggers: the evidence that a model got a line wrong.

Run on the raw model output, before any gain:
  empty / nan        nothing usable came back
  duration           speech rate (characters per second of speech, silence
                     trimmed) outside the range calibrated for the language
                     and line length on known-good renders (calibration.json)
  lead_silence / trail_silence   more silence than calibrated
  clipping           too many samples at full scale
  reader             a reader's CER against the text above the 0.35 bar
                     (only languages with a reliable reader: te, kn)
  readers_disagree   two readers reach different verdicts

Thresholds are never hand-picked here: they come from calibration.json,
written by `tts calibrate` from measured renders, with the method and the
sample sizes recorded in the file itself.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"
READER_MAX_CER = 0.35          # the same bar every other line is held to (e2e_dub.SHORT_LINE_MAX_CER)
SHORT_LINE_CHARS = 12          # common.SHORT_LINE_CHARS
SILENCE_TOP_DB = 40.0


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", s.lower())
                   if not unicodedata.category(c).startswith(("P", "Z", "S")))


def cer(ref: str, hyp: str) -> float:
    """Character error rate, identical to scripts/e2e_dub.py::_cer."""
    r, h = _norm(ref), _norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def n_chars(text: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFC", text) if unicodedata.category(c)[0] in "LMN")


def length_class(text: str) -> str:
    return "short" if len(text.strip()) <= SHORT_LINE_CHARS else "long"


@dataclass
class Calibration:
    ranges: dict            # lang -> {"short": {"cps_lo", "cps_hi"}, "long": {...}}
    default: dict           # same shape, used for languages without their own data
    max_lead_s: float
    max_trail_s: float
    clip_level: float = 0.999
    clip_max_fraction: float = 0.001
    source: str = ""

    def cps_range(self, lang: str, cls: str) -> tuple[float, float] | None:
        r = (self.ranges.get(lang) or {}).get(cls) or self.default.get(cls)
        return (float(r["cps_lo"]), float(r["cps_hi"])) if r else None

    @classmethod
    def load(cls, path: str | Path = CALIBRATION_PATH) -> Calibration:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["ranges"], d.get("default", {}), float(d["silence"]["max_lead_s"]),
                   float(d["silence"]["max_trail_s"]), float(d["clip"]["level"]),
                   float(d["clip"]["max_fraction"]), str(path))


@dataclass
class CheckResult:
    passed: bool
    failures: list[dict] = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    @property
    def triggers(self) -> list[str]:
        return [f["trigger"] for f in self.failures]

    def severity(self) -> float:
        """Lower is better: used to keep the best attempt when every model fails."""
        hard = {"empty": 100, "nan": 100, "exception": 100, "timeout": 100, "oom": 100, "crash": 100,
                "clipping": 30, "duration": 20, "reader": 10, "readers_disagree": 5,
                "lead_silence": 3, "trail_silence": 3}
        return sum(hard.get(t, 50) for t in self.triggers) + float(self.scores.get("cer") or 0)


def envelope(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    """(speech seconds, leading silence s, trailing silence s) at SILENCE_TOP_DB below peak."""
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    frame = max(1, int(0.010 * sr))
    n = len(x) // frame
    if n == 0:
        return 0.0, 0.0, 0.0
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-12
    db = 20 * np.log10(rms / rms.max())
    loud = np.flatnonzero(db > -SILENCE_TOP_DB)
    if len(loud) == 0:
        return 0.0, len(x) / sr, 0.0
    first, last = loud[0], loud[-1]
    return (last - first + 1) * frame / sr, first * frame / sr, (len(x) - (last + 1) * frame) / sr


def signal_checks(x: np.ndarray, sr: int, text: str, lang: str, calib: Calibration | None) -> CheckResult:
    res = CheckResult(True)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        res.failures.append({"trigger": "empty", "detail": "no samples"})
    elif not np.isfinite(x).all():
        res.failures.append({"trigger": "nan", "detail": f"{int((~np.isfinite(x)).sum())} non-finite samples"})
    if res.failures:
        res.passed = False
        return res
    peak = float(np.max(np.abs(x)))
    if peak < 1e-4:
        res.failures.append({"trigger": "empty", "detail": f"silent output (peak {peak:.1e})"})
        res.passed = False
        return res
    speech_s, lead_s, trail_s = envelope(x, sr)
    chars = n_chars(text)
    cps = chars / speech_s if speech_s > 0 else float("inf")
    cls = length_class(text)
    level = calib.clip_level if calib else 0.999
    clip_frac = float(np.mean(np.abs(x) >= level))
    res.scores.update(seconds=round(len(x) / sr, 3), speech_s=round(speech_s, 3), lead_s=round(lead_s, 3),
                      trail_s=round(trail_s, 3), chars=chars, cps=round(cps, 2), length_class=cls,
                      clip_fraction=round(clip_frac, 5), peak=round(peak, 4))
    if calib is None:
        res.scores["uncalibrated"] = True
    else:
        rng = calib.cps_range(lang, cls)
        if rng is None:
            res.scores["uncalibrated"] = True
        elif not rng[0] <= cps <= rng[1]:
            res.failures.append({"trigger": "duration", "detail": f"{cps:.1f} chars/s outside the calibrated "
                                 f"{rng[0]:.1f}-{rng[1]:.1f} for {lang} {cls} lines ({speech_s:.2f}s of speech, {chars} chars)"})
        if lead_s > calib.max_lead_s:
            res.failures.append({"trigger": "lead_silence", "detail": f"{lead_s:.2f}s > {calib.max_lead_s:.2f}s"})
        if trail_s > calib.max_trail_s:
            res.failures.append({"trigger": "trail_silence", "detail": f"{trail_s:.2f}s > {calib.max_trail_s:.2f}s"})
        if clip_frac > calib.clip_max_fraction:
            res.failures.append({"trigger": "clipping", "detail": f"{clip_frac:.2%} of samples at full scale"})
    res.passed = not res.failures
    return res


def reader_checks(res: CheckResult, text: str, hyps: dict[str, str]) -> CheckResult:
    """Fold reader verdicts into `res`. hyps: reader name -> transcript."""
    verdicts = {}
    for name, hyp in hyps.items():
        c = cer(text, hyp)
        verdicts[name] = c <= READER_MAX_CER
        res.scores[f"cer_{name}"] = round(c, 3)
        res.scores[f"hyp_{name}"] = hyp
    if hyps:
        res.scores["cer"] = min(res.scores[f"cer_{n}"] for n in hyps)
        if not any(verdicts.values()):
            worst = ", ".join(f"{n} read {res.scores[f'hyp_{n}']!r} (CER {res.scores[f'cer_{n}']:.2f})" for n in hyps)
            res.failures.append({"trigger": "reader", "detail": worst})
        elif len(set(verdicts.values())) > 1:
            res.failures.append({"trigger": "readers_disagree", "detail": json.dumps(verdicts)})
    res.passed = not res.failures
    return res
