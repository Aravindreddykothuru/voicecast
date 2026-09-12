"""Numbers are spelled out before translation.

Found in the end-to-end run: ASR transcribed "...tomorrow morning at nine" as
"at 9.", and the MMS voices' tokenizers drop every digit, so the number was
silently missing from the dub.
"""
from __future__ import annotations

import pytest

from app.pipeline.text_normalize import number_to_words, spell_out_numbers_en


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Let us meet again tomorrow morning at 9.", "Let us meet again tomorrow morning at nine."),
        ("not in 40 years", "not in forty years"),
        ("It costs 1,250 now.", "It costs one thousand two hundred fifty now."),
        ("about 3.5 hours", "about three point five hours"),
        ("up 25% this year", "up twenty-five percent this year"),
        ("only $12", "only twelve dollars"),
        ("₹500 per month", "five hundred rupees per month"),
        ("see you at 9:30", "see you at nine thirty"),
        ("at 10:05", "at ten oh five"),
        ("by 7:00", "by seven o'clock"),
        ("the 21st floor", "the twenty-first floor"),
        ("his 3rd try", "his third try"),
        ("the 12th day", "the twelfth day"),
        ("the 40th time", "the fortieth time"),
        ("back in 1998", "back in nineteen ninety-eight"),
        ("since 2005", "since two thousand five"),
        ("in 2026", "in twenty twenty-six"),
        ("room 101 and 7 people", "room one hundred one and seven people"),
    ],
)
def test_spells_out_numbers(text, expected):
    assert spell_out_numbers_en(text) == expected


def test_text_without_digits_is_untouched():
    s = "Good morning everyone. Thank you for coming."
    assert spell_out_numbers_en(s) is s


def test_large_numbers():
    assert number_to_words(0) == "zero"
    assert number_to_words(1_000_000) == "one million"
    assert number_to_words(2_345_678) == "two million three hundred forty-five thousand six hundred seventy-eight"


def test_words_with_digits_are_left_alone():
    assert spell_out_numbers_en("Use COVID19 and mp3 files") == "Use COVID19 and mp3 files"


def test_translate_stage_sends_spelled_out_numbers_for_english_sources():
    from types import SimpleNamespace
    from unittest.mock import patch

    from app.pipeline.tasks import translate_segment

    seen = []

    class Spy:
        def translate(self, text, target_lang, src_lang="en"):
            seen.append(text)
            return SimpleNamespace(text="x")

    seg = SimpleNamespace(id="s1", source_text="meet at 9", translated_text=None, status=None)
    with patch("app.pipeline.tasks.get_translation_provider", lambda: Spy()):
        translate_segment(seg, "te", "en")
        seg.source_text = "९ बजे"  # a non-English source is passed through untouched
        translate_segment(seg, "te", "hi")
    assert seen == ["meet at nine", "९ बजे"]
