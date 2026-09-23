"""Adapter registry: model name -> class, resolved lazily so the parent never
imports a model library it does not need."""
from __future__ import annotations

import importlib


class _Lazy(dict):
    _PATHS = {
        "syspin": "app.tts_runtime.adapters.syspin:SyspinAdapter",
        "indic_parler": "app.tts_runtime.adapters.indic_parler:IndicParlerAdapter",
        "indicf5": "app.tts_runtime.adapters.indicf5:IndicF5Adapter",
        "vakyansh": "app.tts_runtime.adapters.vakyansh:VakyanshReader",
        "parler_tiny": "app.tts_runtime.adapters.indic_parler:ParlerTinyAdapter",
        "fake": "app.tts_runtime.adapters.fake:FakeAdapter",
    }

    def __contains__(self, name) -> bool:
        return name in self._PATHS

    def __iter__(self):
        return iter(self._PATHS)

    def __len__(self) -> int:
        return len(self._PATHS)

    def keys(self):
        return self._PATHS.keys()

    def __getitem__(self, name: str):
        if name not in self._PATHS:
            raise KeyError(name)
        if not dict.__contains__(self, name):
            mod, cls = self._PATHS[name].split(":")
            dict.__setitem__(self, name, getattr(importlib.import_module(mod), cls))
        return dict.__getitem__(self, name)


ADAPTERS = _Lazy()


def adapter_class(model_id: str):
    from app.tts_runtime.config import adapter_name

    return ADAPTERS[adapter_name(model_id)]


def variant_of(model_id: str) -> str | None:
    return model_id.split(":", 1)[1] if ":" in model_id else None
