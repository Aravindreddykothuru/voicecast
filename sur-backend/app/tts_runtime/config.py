"""tts_chains.yaml, read strictly.

Every key the runtime understands is listed here; an unknown key is an error
(a typo must not silently fall back to a default -- CONTRACTS.md #5), and so
is a chain naming a model that is unknown or outside the licence allowlist.

A model id is an adapter name, optionally with a variant after a colon
("fake:b"); the allowlist is keyed by the adapter name.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.tts_runtime import licenses

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = BACKEND_DIR / "tts_chains.yaml"
ALLOW_TEST_MODELS_ENV = "TTS_RUNTIME_ALLOW_TEST_MODELS"


class ConfigError(ValueError):
    pass


def adapter_name(model_id: str) -> str:
    return model_id.split(":", 1)[0]


@dataclass(frozen=True)
class BreakerConfig:
    window: int = 20
    fail_ratio: float = 0.30
    cooldown_s: float = 600.0
    min_samples: int = 10


@dataclass(frozen=True)
class KnownBadConfig:
    hits: int = 3
    of_last: int = 5
    auto_remove: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    chains: dict[str, tuple[str, ...]]
    default_chain: tuple[str, ...]
    synth_timeout_s: float = 60.0
    check_timeout_s: float = 30.0
    retries_per_model: int = 1
    network_backoff_s: tuple[float, ...] = (2, 4, 8, 16, 32, 60, 120, 300)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    known_bad: KnownBadConfig = field(default_factory=KnownBadConfig)
    scene_consistency: bool = True
    max_models_loaded: int = 2
    offline_mode: str = "auto"
    carrier: dict[str, bool | None] = field(default_factory=dict)
    min_free_disk_mb: int = 2048
    readers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    interpreters: dict[str, str] = field(default_factory=dict)
    allow_test_models: bool = False
    source: str = ""

    def chain_for(self, lang: str) -> tuple[str, ...]:
        return self.chains.get(lang, self.default_chain)

    def models(self) -> list[str]:
        seen: list[str] = []
        for chain in (*self.chains.values(), self.default_chain):
            for m in chain:
                if m not in seen:
                    seen.append(m)
        return seen

    def uses_carrier(self, model_id: str) -> bool:
        return bool(self.carrier.get(adapter_name(model_id)))

    def interpreter_for(self, model_id: str) -> str | None:
        """The python that runs this model's worker, or None for the current one."""
        raw = self.interpreters.get(adapter_name(model_id))
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        if not p.exists():
            # A venv made on the other OS: Scripts/python.exe <-> bin/python.
            alt = (p.parent.parent / "bin" / "python") if p.parent.name == "Scripts" else \
                  (p.parent.parent / "Scripts" / "python.exe")
            if alt.exists():
                return str(alt)
        return str(p)


_TOP_KEYS = {"chains", "timeouts", "retries", "circuit_breaker", "known_bad", "scene_consistency",
             "max_models_on_gpu", "offline_mode", "carrier", "disk", "readers", "interpreters"}


def _only(d: dict, allowed: set[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return d


def parse(data: dict, *, source: str = "<dict>", allow_test_models: bool | None = None) -> RuntimeConfig:
    if allow_test_models is None:
        allow_test_models = os.environ.get(ALLOW_TEST_MODELS_ENV, "") == "1"
    if allow_test_models:
        logger.warning("TEST MODELS ENABLED (%s=1): test-only adapters may appear in chains. "
                       "Never set this in production.", ALLOW_TEST_MODELS_ENV)
    _only(data, _TOP_KEYS, source)
    raw_chains = data.get("chains")
    if not isinstance(raw_chains, dict) or "default" not in raw_chains:
        raise ConfigError(f"{source}: `chains` must be a mapping with a `default` entry")
    chains: dict[str, tuple[str, ...]] = {}
    for lang, chain in raw_chains.items():
        if not isinstance(chain, list) or not chain or not all(isinstance(m, str) for m in chain):
            raise ConfigError(f"{source}: chains.{lang} must be a non-empty list of model names")
        if len(set(chain)) != len(chain):
            raise ConfigError(f"{source}: chains.{lang} lists a model twice: {chain}")
        chains[str(lang)] = tuple(chain)
    default_chain = chains.pop("default")

    t = _only(data.get("timeouts", {}), {"synth_s", "check_s"}, f"{source}: timeouts")
    r = _only(data.get("retries", {}), {"per_model", "network_backoff"}, f"{source}: retries")
    b = _only(data.get("circuit_breaker", {}), {"window", "fail_ratio", "cooldown_s", "min_samples"},
              f"{source}: circuit_breaker")
    k = _only(data.get("known_bad", {}), {"auto_add_after", "auto_remove"}, f"{source}: known_bad")
    disk = _only(data.get("disk", {}), {"min_free_mb"}, f"{source}: disk")
    carrier_raw = _only(data.get("carrier", {}), set(licenses.ALLOWLIST), f"{source}: carrier")
    readers_raw = data.get("readers", {}) or {}
    interp = _only(data.get("interpreters", {}) or {}, set(licenses.ALLOWLIST), f"{source}: interpreters")

    m = re.fullmatch(r"\s*(\d+)\s+of\s+last\s+(\d+)\s*", str(k.get("auto_add_after", "3 of last 5")))
    if not m or int(m.group(1)) > int(m.group(2)) or int(m.group(1)) < 1:
        raise ConfigError(f"{source}: known_bad.auto_add_after must read like '3 of last 5', "
                          f"got {k.get('auto_add_after')!r}")
    if k.get("auto_remove", False):
        raise ConfigError(f"{source}: known_bad.auto_remove must be false -- a word is only ever "
                          "removed from known_bad/<lang>.txt by a person, with evidence.")
    backoff = tuple(float(x) for x in r.get("network_backoff", (2, 4, 8, 16, 32, 60, 120, 300)))
    if not backoff or any(x <= 0 for x in backoff) or list(backoff) != sorted(backoff):
        raise ConfigError(f"{source}: retries.network_backoff must be positive and non-decreasing")
    ratio = float(b.get("fail_ratio", 0.30))
    if not 0 < ratio < 1:
        raise ConfigError(f"{source}: circuit_breaker.fail_ratio must be between 0 and 1")
    mode = data.get("offline_mode", "auto")
    if mode is False:
        mode = "off"          # YAML 1.1 reads a bare `off` as the boolean false
    if mode not in ("auto", "force", "off"):
        raise ConfigError(f"{source}: offline_mode must be auto | force | off, got {mode!r}")
    carrier: dict[str, bool | None] = {}
    for name, v in carrier_raw.items():
        if v not in (True, False, "unmeasured"):
            raise ConfigError(f"{source}: carrier.{name} must be true, false or unmeasured")
        carrier[name] = None if v == "unmeasured" else v
    readers = {str(lang): tuple(v) for lang, v in readers_raw.items()}

    cfg = RuntimeConfig(
        chains=chains, default_chain=default_chain,
        synth_timeout_s=float(t.get("synth_s", 60)), check_timeout_s=float(t.get("check_s", 30)),
        retries_per_model=int(r.get("per_model", 1)), network_backoff_s=backoff,
        breaker=BreakerConfig(int(b.get("window", 20)), ratio, float(b.get("cooldown_s", 600)),
                              int(b.get("min_samples", 10))),
        known_bad=KnownBadConfig(int(m.group(1)), int(m.group(2)), False),
        scene_consistency=bool(data.get("scene_consistency", True)),
        max_models_loaded=int(data.get("max_models_on_gpu", 2)),
        offline_mode=mode, carrier=carrier, min_free_disk_mb=int(disk.get("min_free_mb", 2048)),
        readers=readers, interpreters={str(n): str(p) for n, p in interp.items()},
        allow_test_models=allow_test_models, source=source,
    )
    _check_models(cfg)
    return cfg


def _check_models(cfg: RuntimeConfig) -> None:
    from app.tts_runtime.adapters import ADAPTERS

    for model_id in cfg.models():
        name = adapter_name(model_id)
        if name not in ADAPTERS:
            raise ConfigError(f"{cfg.source}: unknown model {model_id!r}; known: {sorted(ADAPTERS)}")
        cls = ADAPTERS[name]
        try:
            licenses.check(name, cls.repo, cls.license, allow_test_only=cfg.allow_test_models)
        except licenses.LicenseError as e:
            raise ConfigError(f"{cfg.source}: {e}") from None
        if licenses.ALLOWLIST[name].kind != "tts":
            raise ConfigError(f"{cfg.source}: {model_id} is a {licenses.ALLOWLIST[name].kind}, not a TTS model")
    for lang, names in cfg.readers.items():
        for name in names:
            if name not in licenses.ALLOWLIST or licenses.ALLOWLIST[name].kind != "reader":
                raise ConfigError(f"{cfg.source}: readers.{lang}: {name!r} is not an allowlisted reader")


def load(path: str | os.PathLike | None = None, **kw) -> RuntimeConfig:
    import yaml

    p = Path(path) if path else DEFAULT_CONFIG
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"TTS runtime config not found: {p}") from None
    return parse(data, source=str(p), **kw)
