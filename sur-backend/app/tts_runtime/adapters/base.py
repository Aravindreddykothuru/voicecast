"""The contract every model implements.

Instances live inside a worker process (app/tts_runtime/worker.py), one model
per process, so a crash, hang or out-of-memory in one model cannot take the
batch down. The parent only ever calls the classmethods -- availability and
version -- which must stay cheap and import nothing heavy.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np


class ModelCorrupt(RuntimeError):
    """A model file failed its hash check at load. It is never loaded."""

    def __init__(self, path: str, detail: str):
        super().__init__(f"{path}: {detail}")
        self.path = path


class Unavailable(RuntimeError):
    """The model cannot run here (not provisioned, no interpreter, no reference)."""


@dataclass
class SynthOutput:
    samples: np.ndarray       # raw model output, float32 mono, native rate
    sr: int
    meta: dict = field(default_factory=dict)


def runtime_home() -> Path:
    from app.tts_runtime.config import BACKEND_DIR

    return Path(os.environ.get("TTS_RUNTIME_HOME", BACKEND_DIR / ".tts_runtime"))


class Adapter(ABC):
    name: ClassVar[str]
    repo: ClassVar[str]                 # primary weights repo, checked against the allowlist
    license: ClassVar[str]              # SPDX id; must equal the allowlist's record
    languages: ClassVar[frozenset[str]]
    kind: ClassVar[str] = "tts"         # tts | reader
    supports_cpu_fallback: ClassVar[bool] = False
    needs_token_for_setup: ClassVar[bool] = False

    def __init__(self, variant: str | None = None, options: dict | None = None):
        self.variant = variant
        self.options = options or {}
        self.device = "cpu"

    # --- parent side: cheap, no heavy imports -------------------------------
    @classmethod
    def supports(cls, lang: str) -> bool:
        return lang in cls.languages

    @classmethod
    def availability(cls, options: dict) -> tuple[bool, str]:
        """(can this model run here right now, why not)."""
        return True, ""

    @classmethod
    def version_for(cls, lang: str, speaker: str | None, options: dict) -> str:
        """Everything that changes the audio for a given text, as a string."""
        raise NotImplementedError

    @classmethod
    def sha256_manifest(cls, options: dict) -> dict[str, str]:
        return {}

    # --- worker side ------------------------------------------------------------
    @abstractmethod
    def load(self, device: str) -> dict:
        """Load weights onto `device` ("cpu", "cuda", ...). Returns info for the log."""

    def synth(self, text: str, lang: str, speaker: str | None, emotion: str | None,
              draw: int = 0) -> SynthOutput:
        raise NotImplementedError

    def read(self, samples: np.ndarray, sr: int, lang: str) -> str:
        raise NotImplementedError

    def health_check(self) -> dict:
        return {"ok": True}

    def unload(self) -> None:  # noqa: B027 -- optional hook; most adapters have nothing to free
        return None
