"""The frontend's copy of the API contract must match the API.

sur-frontend/src/lib/api-contract/openapi.json is what the frontend's
contract test validates the UI against. If this fails, the API changed:
run `python scripts/export_openapi.py`, then run the frontend tests to see
which UI code the change breaks.
"""
from __future__ import annotations

import importlib.util
import json
import os

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_frontend_contract_snapshot_matches_the_api():
    spec = importlib.util.spec_from_file_location("export_openapi", os.path.join(BACKEND, "scripts", "export_openapi.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    assert os.path.exists(module.SNAPSHOT), "no frontend contract snapshot; run python scripts/export_openapi.py"
    with open(module.SNAPSHOT, encoding="utf-8") as f:
        committed = json.load(f)
    current = json.loads(module.current_schema())
    assert committed == current, (
        "The API schema changed but the frontend contract snapshot did not. Run "
        "`python scripts/export_openapi.py` and then `npm test` in sur-frontend."
    )
