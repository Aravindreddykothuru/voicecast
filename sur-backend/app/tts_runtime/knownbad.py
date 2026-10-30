"""Words SYSPIN mispronounces: skip it for lines containing them.

known_bad/<lang>.txt is the source of truth -- one word per line, `#`
comments carry the evidence. The runtime appends a word after it failed in
`hits` of its last `of_last` SYSPIN renders (default 3 of 5), and never
removes one: removal is a person's decision, made with new evidence.

Words are learned only from single-word lines. A failed sentence does not
say which of its words was wrong, and blaming all of them would teach the
runtime to skip SYSPIN for "है".
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path

from app.tts_runtime.config import BACKEND_DIR
from app.tts_runtime.fsutil import atomic_write_bytes

logger = logging.getLogger(__name__)
DEFAULT_DIR = BACKEND_DIR / "known_bad"
# Models the known-bad list applies to (skip + learn). SYSPIN only: the list
# describes one voice's mispronunciations, not the words themselves.
LEARNS_FROM = ("syspin",)


def applies_to(model: str) -> bool:
    return model.split(":", 1)[0] in LEARNS_FROM


# \w alone misses Indic vowel signs (category M), which would split "क्यों"
# apart; the Indic blocks are listed explicitly, minus the danda and double
# danda (U+0964/5), which are punctuation. ZWJ/ZWNJ occur inside words.
_WORD = re.compile(r"[\w\u0600-\u06FF\u0900-\u0963\u0966-\u0DFF\u200c\u200d]+")


def words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text))


class KnownBad:
    def __init__(self, store, hits: int = 3, of_last: int = 5, directory: str | Path = DEFAULT_DIR):
        self.store = store
        self.hits = hits
        self.of_last = of_last
        self.dir = Path(directory)
        self._cache: dict[str, set[str]] = {}

    def _file(self, lang: str) -> Path:
        return self.dir / f"{lang}.txt"

    def words_for(self, lang: str) -> set[str]:
        if lang not in self._cache:
            out: set[str] = set()
            p = self._file(lang)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    w = line.split("#", 1)[0].strip()
                    if w:
                        out.add(unicodedata.normalize("NFC", w))
            self._cache[lang] = out
        return self._cache[lang]

    def hits_in(self, lang: str, text: str) -> list[str]:
        bad = self.words_for(lang)
        return [w for w in words(text) if w in bad]

    def record(self, model: str, lang: str, text: str, ok: bool) -> str | None:
        """Log a render; returns the word if this render made it known-bad."""
        if not applies_to(model):
            return None
        ws = words(text)
        if len(ws) != 1:
            return None
        w = ws[0]
        self.store.record_word(lang, w, model, ok)
        if w in self.words_for(lang):
            return None
        recent = self.store.word_recent(lang, w, model, self.of_last)
        fails = recent.count(0)
        if fails >= self.hits:
            self._add(lang, w, f"auto: failed {fails} of its last {len(recent)} {model} renders")
            return w
        return None

    def _add(self, lang: str, word: str, evidence: str) -> None:
        p = self._file(lang)
        old = p.read_text(encoding="utf-8") if p.exists() else f"# SYSPIN known-bad words for {lang}\n"
        if not old.endswith("\n"):
            old += "\n"
        stamp = time.strftime("%Y-%m-%d")
        atomic_write_bytes(p, (old + f"{word}  # {evidence} ({stamp})\n").encode("utf-8"))
        self.words_for(lang).add(word)
        self.store.event("known_bad_added", lang=lang, detail={"word": word, "evidence": evidence})
        logger.warning("KNOWN-BAD: added %r for %s -- %s", word, lang, evidence)
