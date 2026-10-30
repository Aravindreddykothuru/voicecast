"""Timeline planning: pure functions, no I/O, no ffmpeg.

Two decisions live here so they can be tested exhaustively without media:

1. `normalize_turns` -- turning raw diarization turns into segments that are
   safe to dub. Diarization output overlaps (two speakers at once), splits
   one sentence into several same-speaker fragments, and emits blips of a
   few hundred ms. Each overlap becomes overlapping dubbed speech; each
   fragment becomes a separate ASR call on half a word.

2. `plan_timeline` -- where each rendered clip goes and how fast it plays.
   Placing a clip at its segment's start_ms is necessary (CONTRACTS.md #3)
   but not sufficient: translated speech is routinely longer than the
   source (Telugu runs ~20-40% longer than English), so a clip laid at its
   start can run straight over the next line. The planner gives each clip
   the time up to the next voiced clip, speeds it up (pitch-preserving) only
   as much as needed and never beyond MAX_TEMPO, and reports whatever still
   doesn't fit instead of hiding it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from app.providers.base import SpeakerChunk

# Same-speaker fragments closer than this are one utterance.
MERGE_GAP_MS = 400
# Turns shorter than this are absorbed into the nearest same-speaker
# neighbour, or dropped if they are isolated -- faster-whisper on a 150ms
# slice returns hallucinated filler ("Thank you."), which would then be
# translated and spoken.
MIN_TURN_MS = 350

# Faster than this and dubbed speech stops sounding like speech.
MAX_TEMPO = 1.35
# Silence kept between one clip's end and the next clip's start.
GUARD_MS = 60


def normalize_turns(turns: Iterable[SpeakerChunk]) -> list[SpeakerChunk]:
    """Sorted, non-overlapping, merged speaker turns."""
    ordered = sorted((t for t in turns if t.end_ms > t.start_ms), key=lambda t: (t.start_ms, t.end_ms))

    # 1. Overlaps: the later turn keeps its start (a mouth starts moving
    #    there); the earlier turn is cut back to it. A same-speaker overlap
    #    is simply one longer turn.
    resolved: list[SpeakerChunk] = []
    for turn in ordered:
        if resolved and turn.start_ms < resolved[-1].end_ms:
            prev = resolved[-1]
            if prev.speaker_tag == turn.speaker_tag:
                resolved[-1] = replace(prev, end_ms=max(prev.end_ms, turn.end_ms))
                continue
            if turn.end_ms <= prev.end_ms:
                # Fully contained interjection inside another speaker's
                # turn: dubbing it would put two voices on top of each
                # other. The containing turn wins.
                continue
            resolved[-1] = replace(prev, end_ms=turn.start_ms)
        resolved.append(turn)

    # 2. Merge same-speaker fragments separated by a short pause.
    merged: list[SpeakerChunk] = []
    for turn in resolved:
        if (
            merged
            and merged[-1].speaker_tag == turn.speaker_tag
            and turn.start_ms - merged[-1].end_ms <= MERGE_GAP_MS
        ):
            merged[-1] = replace(merged[-1], end_ms=turn.end_ms)
        else:
            merged.append(turn)

    # 3. Blips: absorb into an adjacent same-speaker turn, else drop.
    out: list[SpeakerChunk] = []
    for i, turn in enumerate(merged):
        if turn.end_ms - turn.start_ms >= MIN_TURN_MS:
            out.append(turn)
            continue
        nxt = merged[i + 1] if i + 1 < len(merged) else None
        if out and out[-1].speaker_tag == turn.speaker_tag:
            out[-1] = replace(out[-1], end_ms=turn.end_ms)
        elif nxt is not None and nxt.speaker_tag == turn.speaker_tag:
            merged[i + 1] = replace(nxt, start_ms=turn.start_ms)
    return [t for t in out if t.end_ms - t.start_ms > 0]


@dataclass(frozen=True)
class TimelineSlot:
    segment_id: str
    start_ms: int
    end_ms: int
    # Duration of the rendered clip; None when nothing was rendered because
    # the segment contains no speech.
    clip_ms: int | None


@dataclass(frozen=True)
class PlannedClip:
    segment_id: str
    start_ms: int
    clip_ms: int
    # Time available before the next voiced clip (or the end of the video).
    window_ms: int
    tempo: float
    fitted_ms: int
    # > 0 when the clip is still longer than its window at MAX_TEMPO. Kept
    # visible in the QA report rather than silently trimming words.
    overrun_ms: int


def plan_timeline(
    slots: Sequence[TimelineSlot],
    video_duration_ms: int,
    *,
    max_tempo: float = MAX_TEMPO,
    guard_ms: int = GUARD_MS,
) -> list[PlannedClip]:
    voiced = sorted((s for s in slots if s.clip_ms), key=lambda s: (s.start_ms, s.end_ms))
    plan: list[PlannedClip] = []
    for i, slot in enumerate(voiced):
        clip_ms = int(slot.clip_ms or 0)
        next_start = voiced[i + 1].start_ms if i + 1 < len(voiced) else video_duration_ms
        window = max(next_start - slot.start_ms - guard_ms, 1)
        tempo = 1.0 if clip_ms <= window else min(clip_ms / window, max_tempo)
        fitted = math.ceil(clip_ms / tempo)
        plan.append(
            PlannedClip(
                segment_id=slot.segment_id,
                start_ms=slot.start_ms,
                clip_ms=clip_ms,
                window_ms=window,
                tempo=round(tempo, 4),
                fitted_ms=fitted,
                overrun_ms=max(0, fitted - window),
            )
        )
    return plan
