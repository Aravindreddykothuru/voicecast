"""Download a pinned repo and verify every file against its pin, then lock it.

The runtime's own `setup` only provisions models named in a chain plus the
configured readers. IndicConformer is neither yet -- it has no adapter and
no allowlist entry, which is exactly what steps 3 and 5 need it for -- so
this fetches it the same way `setup` would and leaves it in the same place,
verified to the same standard, without first pretending it is wired in.

Resumable: a file whose git object id already matches is skipped, so an
interrupted run continues instead of re-downloading 2.4 GB. Nothing is
trusted on size alone; `verify_file` recomputes the git object id (and for
LFS files the sha256 behind the pointer) exactly as the runtime does.

    python scripts/fetch_pinned_repo.py indic_conformer
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
os.environ.setdefault("STORAGE_BACKEND", "local")
# Downloading is the point of this script, so offline must not be inherited.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--home", default=str(BACKEND / ".tts_runtime"))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    from app.tts_runtime.manifest import HashMismatch, load_pin, model_dir, verify_file, write_lock

    pin = load_pin(args.name)
    home = pathlib.Path(args.home)
    folder = model_dir(home, pin)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{pin.name}: {pin.repo}@{pin.revision[:12]}, {len(pin.files)} files -> {folder}", flush=True)

    sha256s: dict[str, str] = {}
    done = skipped = 0
    t0 = time.time()
    for i, f in enumerate(pin.files, 1):
        target = folder / f.path
        if target.exists():
            try:
                sha256s[f.path] = verify_file(target, f)
                skipped += 1
                continue
            except HashMismatch:
                target.unlink()      # a bad copy is worse than no copy

        for attempt in (1, 2, 3):
            try:
                src = hf_hub_download(pin.repo, f.path, revision=pin.revision)
                target.parent.mkdir(parents=True, exist_ok=True)
                # copy rather than move: the hub cache keeps its own copy and
                # a symlink here would break the lock's meaning.
                target.write_bytes(pathlib.Path(src).read_bytes())
                sha256s[f.path] = verify_file(target, f)
                done += 1
                break
            except HashMismatch as e:
                print(f"  {f.path}: HASH MISMATCH ({e}); refetching", flush=True)
                target.unlink(missing_ok=True)
                if attempt == 3:
                    raise
            except Exception as e:                       # noqa: BLE001 -- network
                print(f"  {f.path}: {type(e).__name__} {str(e)[:120]}; retry {attempt}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)

        if i % 25 == 0 or i == len(pin.files):
            print(f"  {i}/{len(pin.files)} ({done} fetched, {skipped} already verified, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    write_lock(folder, pin, sha256s)
    print(f"locked {len(sha256s)} files in {folder}/VERIFIED.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
