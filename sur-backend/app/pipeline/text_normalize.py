"""Text normalization between ASR and translation.

Why numbers: Whisper writes numerals ("at 9", "40 years", "25%"). IndicTrans2
tends to copy digits through untouched, and the MMS voices that speak the
result have letters-only vocabularies -- the tokenizer drops every digit, so
"meet at 9" was dubbed as "meet at". Spelling numbers out in the *source*
language hands the translator words it knows how to translate.

Dependency-free on purpose: this runs in the main worker, which has no
inflect/num2words, and the rules needed for spoken English numbers are few.
"""
from __future__ import annotations

import re

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_IRREGULAR_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _below_thousand(n: int) -> str:
    words = []
    if n >= 100:
        words += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n or not words:
        words.append(_ONES[n])
    return " ".join(words)


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 1000:
        return _below_thousand(n)
    parts = []
    for value, name in _SCALES:
        if n >= value:
            parts.append(f"{number_to_words(n // value)} {name}")
            n %= value
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def _ordinal(n: int) -> str:
    words = number_to_words(n)
    head, sep, last = words.rpartition("-" if "-" in words.split(" ")[-1] else " ")
    if last in _IRREGULAR_ORDINALS:
        last = _IRREGULAR_ORDINALS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}"


def _year(n: int) -> str:
    if 2000 <= n <= 2009:
        return number_to_words(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{number_to_words(hi)} hundred"
    return f"{number_to_words(hi)} {'oh ' if lo < 10 else ''}{number_to_words(lo)}"


def _amount(text: str) -> str:
    whole, _, frac = text.replace(",", "").partition(".")
    words = number_to_words(int(whole))
    if frac:
        words += " point " + " ".join(_ONES[int(d)] for d in frac)
    return words


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_PATTERNS: list[tuple[re.Pattern, object]] = [
    # 9:30, 12:05 pm
    (re.compile(r"\b(\d{1,2}):(\d{2})\b"),
     lambda m: number_to_words(int(m[1])) + (
         " o'clock" if m[2] == "00" else f" {'oh ' if m[2][0] == '0' else ''}{number_to_words(int(m[2]))}")),
    (re.compile(rf"\$\s?({_NUM})"), lambda m: f"{_amount(m[1])} dollars"),
    (re.compile(rf"₹\s?({_NUM})"), lambda m: f"{_amount(m[1])} rupees"),
    (re.compile(rf"({_NUM})\s?%"), lambda m: f"{_amount(m[1])} percent"),
    (re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE), lambda m: _ordinal(int(m[1]))),
    # Four-digit years read as years ("nineteen ninety-eight"), not quantities.
    (re.compile(r"\b(1[1-9]\d{2}|20\d{2})\b(?![.,]\d)"), lambda m: _year(int(m[1]))),
    (re.compile(rf"(?<![\w.])({_NUM})(?![\w])"), lambda m: _amount(m[1])),
]


def spell_out_numbers_en(text: str) -> str:
    """English numerals -> English words; text without digits is returned as-is."""
    if not any(ch.isdigit() for ch in text):
        return text
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text
