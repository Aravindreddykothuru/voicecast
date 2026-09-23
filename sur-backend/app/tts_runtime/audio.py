"""The one audio contract every model's output is converted to: mono,
UNIFIED_SR, float32, loudness-normalised (RMS -20 dBFS with a 0.97 peak
limit -- the same level the pipeline already uses, common.TARGET_RMS_DBFS).

Checks run on the *raw* model output (clipping must be judged before any
gain is applied); only a line that passes is converted and written.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from app.tts_runtime.fsutil import atomic_write, sha256_file

UNIFIED_SR = 24000
TARGET_RMS_DBFS = -20.0
PEAK_LIMIT = 0.97


@dataclass
class Audio:
    samples: np.ndarray     # float32, mono
    sr: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / float(self.sr) if self.sr else 0.0


def as_mono_float(samples) -> np.ndarray:
    x = np.asarray(samples)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    x = x.astype(np.float32, copy=False)
    if x.ndim > 1:
        # (channels, n) or (n, channels): average the short axis.
        x = x.mean(axis=0 if x.shape[0] < x.shape[-1] else 1)
    return np.ascontiguousarray(x.reshape(-1))


def resample(x: np.ndarray, sr: int, target: int = UNIFIED_SR) -> np.ndarray:
    if sr == target or len(x) == 0:
        return x.astype(np.float32, copy=False)
    import soxr

    return soxr.resample(x, sr, target, quality="HQ").astype(np.float32)


def loudness_normalize(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    if rms < 1e-6:
        return x.astype(np.float32)
    out = x * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = float(np.max(np.abs(out)))
    if peak > PEAK_LIMIT:
        out = out * (PEAK_LIMIT / peak)
    return out.astype(np.float32)


def to_unified(samples, sr: int) -> Audio:
    return Audio(loudness_normalize(resample(as_mono_float(samples), sr)), UNIFIED_SR)


def write_wav_atomic(path, audio: Audio, subtype: str = "PCM_16") -> str:
    """Write `audio` atomically; returns the sha256 of the file written."""
    import soundfile as sf

    def _write(f):
        buf = io.BytesIO()
        sf.write(buf, audio.samples, audio.sr, subtype=subtype, format="WAV")
        f.write(buf.getvalue())

    atomic_write(path, _write)
    return sha256_file(path)


def read_wav(path) -> Audio:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return Audio(as_mono_float(x), int(sr))
