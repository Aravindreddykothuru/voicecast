"""Write the API's OpenAPI schema into the frontend as the contract snapshot.

    python scripts/export_openapi.py

The frontend's contract test (sur-frontend/src/lib/api.contract.test.ts)
checks every request the UI builds and every response type it reads against
this file, and tests/test_openapi_contract_snapshot.py fails whenever the
backend's schema drifts from it. Change the API -> run this -> the frontend
tests show exactly which UI code no longer matches. That replaces the old
"keep types.ts in sync by hand", which is how the UI came to send fields the
backend ignored and read fields it never sent.
"""
from __future__ import annotations

import json
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.path.join(os.path.dirname(BACKEND), "sur-frontend", "src", "lib", "api-contract", "openapi.json")


def current_schema() -> str:
    sys.path.insert(0, BACKEND)
    os.environ.setdefault("TTS_PROVIDER", "mock")
    from app.main import app

    return json.dumps(app.openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
    with open(SNAPSHOT, "w", encoding="utf-8", newline="\n") as f:
        f.write(current_schema())
    print("wrote", SNAPSHOT)
