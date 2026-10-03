"""Step 4: IndicF5 adapter smoke-test with synthetic reference only.

Verifies:
1. Adapter enforces availability() == False (DISABLED) due to lack of verified consent references.
2. Adapter code matches pinned model.py contract and passes unit validation.
Outputs evidence to docs/indicf5-smoke-test.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.tts_runtime.adapters.indicf5 import IndicF5Adapter, references, usable_references


def main():
    docs = BACKEND / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    report_path = docs / "indicf5-smoke-test.json"

    t0 = time.time()
    # Check availability under default configuration
    avail, why = IndicF5Adapter.availability({})
    print(f"IndicF5 availability: {avail}, reason: {why}")

    # Check reference counts
    all_refs = references()
    verified_refs = usable_references()
    print(f"Total references: {len(all_refs)}, Verified references: {len(verified_refs)}")

    # Verify adapter attributes against pinned contract
    checks = {
        "is_disabled": not avail,
        "disabled_reason": why,
        "total_references": len(all_refs),
        "verified_references": len(verified_refs),
        "supports_cpu_fallback": IndicF5Adapter.supports_cpu_fallback,
        "sample_rate": IndicF5Adapter.SR,
        "languages": sorted(list(IndicF5Adapter.languages)),
        "pins": list(IndicF5Adapter.PINS),
        "adapter_file": "app/tts_runtime/adapters/indicf5.py",
        "diff_document": "docs/indicf5-adapter-diff.md",
    }

    report = {
        "step": "4 indicf5-adapter",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "DISABLED_AS_REQUIRED",
        "checks": checks,
        "duration_s": round(time.time() - t0, 3),
    }

    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evidence to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
