"""Thin wrappers around ffmpeg/ffprobe -- the only place in the codebase
that shells out to media tooling, so provider/task code stays testable
without ffmpeg installed (tests monkeypatch these)."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


class MediaToolMissingError(LookupError):
    """ffmpeg/ffprobe is not installed where settings say it is.

    A LookupError on purpose: pipeline tasks treat LookupError as permanent
    (app/pipeline/tasks.py _PERMANENT), and retrying a missing binary three
    times with backoff only delays the one message the operator needs.
    """


class MediaToolError(RuntimeError):
    """ffmpeg/ffprobe ran and exited non-zero. Carries the tail of stderr --
    subprocess.CalledProcessError on its own says only "exit status 1",
    which is how media failures used to reach the project's error_message."""


def _tool(name: str) -> str:
    from app.config import get_settings, resolve_backend_path

    settings = get_settings()
    configured = settings.ffmpeg_binary if name == "ffmpeg" else settings.ffprobe_binary
    env_var = f"{name.upper()}_BINARY"
    if any(sep in configured for sep in ("/", "\\")):
        path = resolve_backend_path(configured)
        if os.path.isfile(path):
            return path
        raise MediaToolMissingError(f"{env_var}={configured!r} does not exist (resolved to {path}).")
    found = shutil.which(configured)
    if found is None:
        raise MediaToolMissingError(
            f"{configured!r} is not on PATH. Install ffmpeg, or set {env_var} to the "
            f"binary's path (e.g. .tools/ffmpeg-9.0.1-essentials_build/bin/{name}.exe)."
        )
    return found


def tool_path(name: str) -> str:
    """Resolved path of "ffmpeg" or "ffprobe", honouring FFMPEG_BINARY /
    FFPROBE_BINARY. For scripts that need to run their own media commands."""
    return _tool(name)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-6:]
        raise MediaToolError(
            f"{os.path.basename(args[0])} exited {exc.returncode}: " + " | ".join(tail)
        ) from exc


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    return _run([_tool("ffmpeg"), "-hide_banner", "-nostdin", "-y", *args])


def _probe_json(path: str, *entries: str) -> dict:
    out = _run([_tool("ffprobe"), "-v", "error", *entries, "-of", "json", path])
    return json.loads(out.stdout or b"{}")


def probe_duration_ms(path: str) -> int:
    """Container duration in ms. Raises rather than returning 0.

    This used to swallow every exception and return 0 "for unit tests", which
    turned a missing ffprobe or an unreadable upload into a project whose
    video silently had no duration. Tests stub this function instead.
    """
    data = _probe_json(path, "-show_entries", "format=duration")
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        # ValueError: an input ffprobe can read but reports no duration for
        # will never grow one on retry.
        raise ValueError(f"ffprobe reported no duration for {path}") from exc
    return int(round(duration * 1000))


def probe_video_codec(path: str) -> str | None:
    data = _probe_json(path, "-select_streams", "v:0", "-show_entries", "stream=codec_name")
    streams = data.get("streams") or []
    return streams[0].get("codec_name") if streams else None


def extract_audio_wav(video_path: str, out_path: str, sample_rate: int = 16000) -> None:
    _ffmpeg("-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", out_path)


@contextlib.contextmanager
def extract_audio_slice(audio_path: str, start_ms: int, end_ms: int) -> Iterator[str]:
    """Cut [start_ms, end_ms) out of `audio_path` into a temp file, yielding
    its path; used by providers that only accept a single utterance at a
    time (faster-whisper, the emotion classifier)."""
    # NamedTemporaryFile(delete=True) keeps its handle open for the `with`
    # block's whole lifetime; on Windows (unlike POSIX) a second process can't
    # open that same path for writing while we still hold it, so ffmpeg's own
    # -y open would fail with a sharing violation. Create+close it up front
    # (delete=False) so ffmpeg can write to the bare path, then clean up
    # ourselves.
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _ffmpeg("-i", audio_path, "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}", path)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def concat_audio(paths: Sequence[str], out_path: str, sample_rate: int = 16000) -> None:
    """Join audio files end to end into one mono WAV."""
    if not paths:
        raise ValueError("concat_audio: no inputs")
    args: list[str] = []
    for p in paths:
        args += ["-i", p]
    joined = "".join(f"[{i}:a]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[out]"
    _ffmpeg(*args, "-filter_complex", joined, "-map", "[out]", "-ar", str(sample_rate), "-ac", "1", out_path)


@dataclass(frozen=True)
class ClipPlacement:
    """One rendered clip on the output timeline.

    `tempo` > 1 plays the clip faster (pitch-preserving, ffmpeg atempo) so it
    fits the time the picture gives it -- see app/pipeline/timeline.py, which
    decides it. Plain (path, start_ms) tuples are accepted as tempo 1.0.
    """

    path: str
    start_ms: int
    tempo: float = 1.0


# Inputs per ffmpeg invocation when mixing. One `-i` per clip on a single
# command line hits Windows' 32,767-character limit at a few hundred
# segments (a long video), and every input is an open decoder. Larger
# timelines are mixed in batches, then the batches are mixed.
_MIX_BATCH = 32
_MIX_RATE = 48000


def _mix_clips(clips: Sequence[ClipPlacement], out_wav: str) -> None:
    """Place each clip at its own start_ms (adelay) and sum them (amix).

    Concatenating the clips back-to-back instead (what the mux used to do)
    throws away every segment's timecode -- the leading offset and every gap
    between utterances -- so the dub slides progressively out of sync with
    the picture. CONTRACTS.md #3.
    """
    args: list[str] = []
    filters: list[str] = []
    labels = ""
    for i, clip in enumerate(clips):
        args += ["-i", clip.path]
        chain = [f"aresample={_MIX_RATE}", "aformat=sample_fmts=fltp:channel_layouts=mono"]
        if abs(clip.tempo - 1.0) > 1e-3:
            chain.append(f"atempo={clip.tempo:.4f}")
        # all=1 applies the delay to every channel.
        chain.append(f"adelay={max(clip.start_ms, 0)}:all=1")
        filters.append(f"[{i}:a]{','.join(chain)}[a{i}]")
        labels += f"[a{i}]"
    # normalize=0 keeps amix from attenuating each input by 1/N, which would
    # make a many-segment dub progressively quieter.
    filters.append(f"{labels}amix=inputs={len(clips)}:normalize=0:dropout_transition=0[out]")
    _ffmpeg(*args, "-filter_complex", ";".join(filters), "-map", "[out]",
            "-c:a", "pcm_s16le", "-ar", str(_MIX_RATE), "-ac", "1", out_wav)


# Codecs an .mp4 can carry as-is. Anything else (VP8 from a .webm, ProRes
# from a .mov, ...) is re-encoded rather than failing the whole export on a
# stream-copy the MP4 muxer refuses.
_MP4_COPYABLE_VIDEO = {"h264", "hevc", "av1", "vp9", "mpeg4"}


def mux_timeline(
    video_path: str,
    placements: Sequence[ClipPlacement | tuple[str, int] | tuple[str, int, float]],
    out_path: str,
) -> None:
    """Lay each rendered segment's audio at its own start offset on a silent
    bed, then mux the result onto the original video's picture track.

    Output duration equals the source video's: the dub track is padded with
    silence (apad) and the output ends with the picture (-shortest). Without
    apad, -shortest truncated the *video* to wherever the last utterance
    ended -- a 10s clip whose last line landed at 6s came out 6s long.
    """
    clips = [
        p if isinstance(p, ClipPlacement)
        else ClipPlacement(str(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 1.0)
        for p in placements
    ]
    if not clips:
        raise ValueError("mux_timeline: no audio placements given")

    with tempfile.TemporaryDirectory(prefix="sur-mux-") as tmp:
        layer, level = clips, 0
        while True:
            batches = [layer[i:i + _MIX_BATCH] for i in range(0, len(layer), _MIX_BATCH)]
            outputs = []
            for b, batch in enumerate(batches):
                wav = os.path.join(tmp, f"mix_{level}_{b}.wav")
                _mix_clips(batch, wav)
                outputs.append(ClipPlacement(wav, 0))
            if len(outputs) == 1:
                dub_wav = outputs[0].path
                break
            layer, level = outputs, level + 1

        codec = probe_video_codec(video_path)
        if codec is None:
            raise ValueError(f"{video_path} has no video stream to dub onto")
        video_codec = ["-c:v", "copy"] if codec in _MP4_COPYABLE_VIDEO else [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]

        _ffmpeg(
            "-i", video_path, "-i", dub_wav,
            "-filter_complex", "[1:a]apad[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            *video_codec,
            "-c:a", "aac", "-b:a", "192k", "-ar", str(_MIX_RATE),
            "-shortest", "-movflags", "+faststart",
            out_path,
        )
