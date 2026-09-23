"""Generate a model pin from Hugging Face's public tree API.

    python scripts/pin_hf_repo.py <name> <repo> <revision> <license> [--include GLOB ...]

Writes app/tts_runtime/pins/<name>.json: the exact commit and, per file, its
size and git object id. Works for gated repositories without a token (the
tree is public). Nothing is downloaded; see app/tts_runtime/manifest.py for
how the ids verify a download exactly.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import urllib.request
from pathlib import Path

PINS = Path(__file__).resolve().parents[1] / "app" / "tts_runtime" / "pins"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("license")
    ap.add_argument("--include", action="append", default=[])
    a = ap.parse_args(argv)
    if len(a.revision) != 40:
        ap.error("revision must be a full 40-character commit sha")
    base = f"https://huggingface.co/api/models/{a.repo}"
    meta = json.load(urllib.request.urlopen(base, timeout=30))
    tree = json.load(urllib.request.urlopen(f"{base}/tree/{a.revision}?recursive=true", timeout=60))
    files = []
    for f in tree:
        if f.get("type") != "file":
            continue
        if a.include and not any(fnmatch.fnmatchcase(f["path"], g) for g in a.include):
            continue
        files.append({"path": f["path"], "size": int(f["size"]), "git_oid": f["oid"], "lfs": bool(f.get("lfs"))})
    if not files:
        print("no files matched", file=sys.stderr)
        return 1
    out = {"name": a.name, "repo": a.repo, "revision": a.revision, "license": a.license,
           "gated": bool(meta.get("gated")), "files": sorted(files, key=lambda x: x["path"])}
    PINS.mkdir(parents=True, exist_ok=True)
    (PINS / f"{a.name}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{a.name}: {a.repo}@{a.revision[:12]}, {len(files)} files, "
          f"{sum(f['size'] for f in files) / 1e6:.1f} MB, gated={out['gated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
