"""Pure timeline decisions (app/pipeline/timeline.py): no ffmpeg, no DB.

normalize_turns exists because raw pyannote output, measured on a real
two-speaker clip, splits one sentence into same-speaker fragments 270-355ms
apart and emits overlapping turns when voices meet -- each fragment an ASR
call on half a sentence, each overlap two dubbed voices at once.

plan_timeline exists because translated speech is routinely longer than the
source: laying a clip at its start_ms (CONTRACTS.md #3) is necessary but a
long clip then runs over the next line.
"""
from __future__ import annotations

import pytest

from app.pipeline.timeline import (
    GUARD_MS,
    MAX_TEMPO,
    TimelineSlot,
    normalize_turns,
    plan_timeline,
)
from app.providers.base import SpeakerChunk


def _c(start, end, spk="A"):
    return SpeakerChunk(start_ms=start, end_ms=end, speaker_tag=spk)


def _spans(chunks):
    return [(c.start_ms, c.end_ms, c.speaker_tag) for c in chunks]


def test_real_pyannote_fragments_merge_into_sentences():
    # Verbatim turns pyannote 3.1 produced for the two-speaker test clip.
    raw = [
        _c(908, 2140, "S0"), _c(2984, 4890, "S0"),
        _c(6730, 8873, "S1"),
        _c(10695, 12889, "S0"), _c(13159, 14745, "S0"),
        _c(16568, 17834, "S1"), _c(18627, 19572, "S1"),
        _c(21327, 22440, "S0"), _c(22795, 25056, "S0"),
        _c(26845, 27334, "S1"), _c(28110, 30287, "S1"),
    ]
    out = normalize_turns(raw)
    # Gaps of 270ms and 355ms are one utterance; ~800ms sentence pauses stay split.
    assert (10695, 14745, "S0") in _spans(out)
    assert (21327, 25056, "S0") in _spans(out)
    assert (908, 2140, "S0") in _spans(out) and (2984, 4890, "S0") in _spans(out)
    assert len(out) == 9


def test_input_order_does_not_matter():
    turns = [_c(5000, 6000, "B"), _c(0, 1000, "A"), _c(2000, 3000, "A")]
    assert _spans(normalize_turns(turns)) == _spans(normalize_turns(sorted(turns, key=lambda t: t.start_ms)))


def test_output_never_overlaps():
    turns = [_c(0, 3000, "A"), _c(2500, 5000, "B"), _c(4800, 7000, "A"), _c(6900, 9000, "B")]
    out = normalize_turns(turns)
    for a, b in zip(out, out[1:]):
        assert a.end_ms <= b.start_ms, f"{a} overlaps {b}"


def test_cross_speaker_overlap_keeps_the_later_speakers_start():
    out = normalize_turns([_c(0, 3000, "A"), _c(2500, 5000, "B")])
    assert _spans(out) == [(0, 2500, "A"), (2500, 5000, "B")]


def test_interjection_inside_another_turn_is_dropped_not_double_voiced():
    out = normalize_turns([_c(0, 6000, "A"), _c(2000, 2600, "B")])
    assert _spans(out) == [(0, 6000, "A")]


def test_same_speaker_overlap_is_one_turn():
    assert _spans(normalize_turns([_c(0, 3000), _c(2000, 4000)])) == [(0, 4000, "A")]


def test_isolated_blip_is_dropped_and_adjacent_blip_absorbed():
    out = normalize_turns([_c(0, 2000, "A"), _c(3000, 3150, "B"), _c(5000, 7000, "A"), _c(7600, 7800, "A")])
    assert _spans(out) == [(0, 2000, "A"), (5000, 7800, "A")]


def test_blip_before_its_own_speakers_turn_extends_that_turn():
    out = normalize_turns([_c(0, 2000, "A"), _c(2600, 2800, "B"), _c(3300, 6000, "B")])
    assert _spans(out) == [(0, 2000, "A"), (2600, 6000, "B")]


def test_empty_and_zero_length_turns():
    assert normalize_turns([]) == []
    assert normalize_turns([_c(1000, 1000)]) == []


# --------------------------------------------------------------------------
def _slot(seg_id, start, end, clip):
    return TimelineSlot(segment_id=seg_id, start_ms=start, end_ms=end, clip_ms=clip)


def test_clip_that_fits_plays_at_normal_speed_at_its_start():
    [p] = plan_timeline([_slot("a", 1000, 3000, 1800)], 10_000)
    assert (p.start_ms, p.tempo, p.overrun_ms) == (1000, 1.0, 0)


def test_long_clip_may_use_the_following_silence_before_speeding_up():
    # Segment is 2s but the next line starts 4s later: a 3.5s clip fits untouched.
    plan = plan_timeline([_slot("a", 0, 2000, 3500), _slot("b", 4000, 6000, 1000)], 10_000)
    assert plan[0].tempo == 1.0 and plan[0].overrun_ms == 0


def test_long_clip_is_sped_up_just_enough_to_clear_the_next_line():
    plan = plan_timeline([_slot("a", 0, 2000, 2400), _slot("b", 2000, 4000, 1000)], 10_000)
    window = 2000 - GUARD_MS
    assert plan[0].tempo == pytest.approx(2400 / window, abs=1e-3)
    assert plan[0].fitted_ms <= window + 1
    assert plan[0].overrun_ms == 0


def test_speedup_is_capped_and_the_remainder_is_reported_not_hidden():
    plan = plan_timeline([_slot("a", 0, 1000, 4000), _slot("b", 1000, 2000, 500)], 10_000)
    assert plan[0].tempo == MAX_TEMPO
    assert plan[0].overrun_ms > 0


def test_last_clip_is_bounded_by_the_video_end():
    [p] = plan_timeline([_slot("a", 8000, 9000, 2500)], 10_000)
    assert p.window_ms == 2000 - GUARD_MS
    assert p.tempo > 1.0


def test_silent_segments_get_no_clip_and_do_not_shrink_neighbours():
    plan = plan_timeline(
        [_slot("a", 0, 1000, 2500), _slot("silence", 1200, 1800, None), _slot("b", 3000, 4000, 900)],
        10_000,
    )
    assert [p.segment_id for p in plan] == ["a", "b"]
    assert plan[0].tempo == 1.0, "a silent segment must not count as the next line"


def test_placement_keys_off_start_not_list_position():
    plan = plan_timeline([_slot("b", 5000, 6000, 800), _slot("a", 1000, 2000, 800)], 10_000)
    assert [(p.segment_id, p.start_ms) for p in plan] == [("a", 1000), ("b", 5000)]
