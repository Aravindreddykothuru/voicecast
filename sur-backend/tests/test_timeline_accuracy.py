"""Regression tests for CONTRACTS.md invariant #3 (timeline accuracy).

The bug these exist to prevent: mux_export concatenated the rendered clips
back-to-back, throwing away every segment's start_ms and every gap between
utterances, so the dub slid progressively out of sync with the picture. The
output was a valid video of the right-ish length, so nothing failed -- you
had to listen to it to notice.

These tests do what a reviewer's ear would: render, then measure where the
audio energy actually is. A concatenation regression fails the build.

They use real ffmpeg deliberately -- the bug lived in the ffmpeg invocation,
so mocking it would test nothing.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.pipeline import ffmpeg_utils

pytestmark = pytest.mark.timeline

_HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_IN_CI = os.environ.get("CI", "").lower() in ("1", "true", "yes")


def test_ffmpeg_is_available_in_ci():
    """These tests are the only thing standing between a concat regression and
    production, so CI must actually run them. Locally they may skip."""
    if _IN_CI:
        assert _HAVE_FFMPEG, (
            "ffmpeg/ffprobe missing in CI: the timeline-accuracy tests would skip, "
            "leaving the sync invariant unguarded. Install ffmpeg in the CI image."
        )


requires_ffmpeg = pytest.mark.skipif(
    not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed (guarded by test_ffmpeg_is_available_in_ci)"
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _silent_video(path: str, seconds: int) -> None:
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={seconds}",
          "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", path])


def _tone(path: str, seconds: float, freq: int = 440) -> None:
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
          "-ar", "24000", "-ac", "1", path])


def _rms_db(path: str, start_s: float, dur_s: float) -> float:
    """Mean RMS in dB over a window. -inf (silence) is normalised to -120."""
    out = subprocess.run(
        ["ffmpeg", "-ss", str(start_s), "-t", str(dur_s), "-i", path,
         "-af", "astats=metadata=1", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    for line in out.splitlines():
        if "RMS level dB" in line:
            val = line.split()[-1]
            return -120.0 if val in ("-inf", "") else float(val)
    return -120.0


def _duration_s(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


SPEECH_DB = -60.0   # anything above this is audible content
SILENCE_DB = -80.0  # anything below this is effectively silence


@requires_ffmpeg
def test_audio_energy_lands_at_each_segment_timestamp(tmp_path):
    """Segments at 2s and 6s must produce energy at 2s and 6s -- not 0s and 1s.

    This is the assertion the original code fails: concatenation puts clip A
    at 0.0 and clip B immediately after it.
    """
    video = str(tmp_path / "v.mp4")
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    out = str(tmp_path / "out.mp4")
    _silent_video(video, 10)
    _tone(a, 1.0, 440)
    _tone(b, 1.0, 880)

    ffmpeg_utils.mux_timeline(video, [(a, 2000), (b, 6000)], out)

    # Energy where the segments were placed...
    assert _rms_db(out, 2.05, 0.9) > SPEECH_DB, "no audio at segment 1's 2000ms timecode"
    assert _rms_db(out, 6.05, 0.9) > SPEECH_DB, "no audio at segment 2's 6000ms timecode"

    # ...and silence everywhere they weren't. The leading window is what
    # catches concatenation: it puts clip A at 0.0.
    assert _rms_db(out, 0.05, 1.5) < SILENCE_DB, "audio before the first segment (concatenated?)"
    assert _rms_db(out, 3.5, 2.0) < SILENCE_DB, "inter-utterance gap was not preserved"
    assert _rms_db(out, 7.5, 2.0) < SILENCE_DB, "audio after the last segment"


@requires_ffmpeg
def test_output_duration_matches_source_video(tmp_path):
    """The dub must not shorten or stretch the picture."""
    video = str(tmp_path / "v.mp4")
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    out = str(tmp_path / "out.mp4")
    _silent_video(video, 10)
    _tone(a, 1.0)
    _tone(b, 1.0)

    ffmpeg_utils.mux_timeline(video, [(a, 1000), (b, 5000)], out)

    src, dst = _duration_s(video), _duration_s(out)
    assert abs(dst - src) <= 0.1, f"output {dst:.3f}s vs source {src:.3f}s (>100ms drift)"


@requires_ffmpeg
def test_single_segment_keeps_its_offset(tmp_path):
    """The narrowest form of the bug: one segment that doesn't start at zero."""
    video = str(tmp_path / "v.mp4")
    a = str(tmp_path / "a.wav")
    out = str(tmp_path / "out.mp4")
    _silent_video(video, 6)
    _tone(a, 1.0)

    ffmpeg_utils.mux_timeline(video, [(a, 3000)], out)

    assert _rms_db(out, 0.05, 2.5) < SILENCE_DB, "clip was moved to 0s"
    assert _rms_db(out, 3.05, 0.9) > SPEECH_DB, "clip is not at its 3000ms timecode"


@requires_ffmpeg
def test_segment_order_does_not_change_placement(tmp_path):
    """Placement must key off start_ms, not list position."""
    video = str(tmp_path / "v.mp4")
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    out = str(tmp_path / "out.mp4")
    _silent_video(video, 10)
    _tone(a, 1.0, 440)
    _tone(b, 1.0, 880)

    # Deliberately out of chronological order.
    ffmpeg_utils.mux_timeline(video, [(b, 6000), (a, 2000)], out)

    assert _rms_db(out, 2.05, 0.9) > SPEECH_DB
    assert _rms_db(out, 6.05, 0.9) > SPEECH_DB
    assert _rms_db(out, 0.05, 1.5) < SILENCE_DB


@requires_ffmpeg
def test_many_segments_are_all_placed_when_mixed_in_batches(tmp_path):
    """One ffmpeg input per clip hit Windows' command-line limit at a few
    hundred segments; the mux now mixes in batches. No clip may be lost at a
    batch boundary."""
    video = str(tmp_path / "v.mp4")
    out = str(tmp_path / "out.mp4")
    tone = str(tmp_path / "t.wav")
    _silent_video(video, 42)
    _tone(tone, 0.4)
    starts = [i * 1000 for i in range(40)]  # > one batch of 32
    ffmpeg_utils.mux_timeline(video, [(tone, s) for s in starts], out)

    for s in (0, 31_000, 32_000, 39_000):
        assert _rms_db(out, s / 1000 + 0.05, 0.3) > SPEECH_DB, f"clip at {s}ms missing"
    assert _rms_db(out, 40.6, 1.0) < SILENCE_DB
    assert abs(_duration_s(out) - 42) <= 0.1


@requires_ffmpeg
def test_tempo_shortens_a_clip_without_moving_its_start(tmp_path):
    video = str(tmp_path / "v.mp4")
    a = str(tmp_path / "a.wav")
    out = str(tmp_path / "out.mp4")
    _silent_video(video, 6)
    _tone(a, 2.0)

    ffmpeg_utils.mux_timeline(video, [ffmpeg_utils.ClipPlacement(a, 1000, 2.0)], out)

    assert _rms_db(out, 0.05, 0.9) < SILENCE_DB
    assert _rms_db(out, 1.05, 0.9) > SPEECH_DB
    assert _rms_db(out, 2.2, 1.5) < SILENCE_DB, "2s clip at tempo 2.0 must end by ~2s"


@requires_ffmpeg
def test_non_mp4_video_codec_is_reencoded_not_rejected(tmp_path):
    video = str(tmp_path / "v.webm")
    a = str(tmp_path / "a.wav")
    out = str(tmp_path / "out.mp4")
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=3",
          "-an", "-c:v", "libvpx", "-b:v", "200k", video])
    _tone(a, 1.0)

    ffmpeg_utils.mux_timeline(video, [(a, 500)], out)
    assert _rms_db(out, 0.55, 0.8) > SPEECH_DB
    assert abs(_duration_s(out) - 3) <= 0.15


@requires_ffmpeg
def test_concat_audio_joins_clips_end_to_end(tmp_path):
    """Builds a speaker's voice-clone reference from several of their lines."""
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    out = str(tmp_path / "ref.wav")
    _tone(a, 0.5, 440)
    _tone(b, 0.7, 880)
    ffmpeg_utils.concat_audio([a, b], out)
    assert abs(_duration_s(out) - 1.2) <= 0.05


def test_mux_timeline_rejects_empty_placements():
    with pytest.raises(ValueError, match="no audio placements"):
        ffmpeg_utils.mux_timeline("v.mp4", [], "out.mp4")
