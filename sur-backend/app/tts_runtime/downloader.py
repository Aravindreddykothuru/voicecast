"""Resumable, verified downloads.

Bytes land in <dest>.partial. An interrupted download resumes from the
partial file with an HTTP Range request -- it never starts over unless the
server ignores Range, which is logged. Only a file that verifies against its
pin is renamed into place (atomically, after fsync); one that fails is
deleted and never used.

Network failures raise NetworkDown with the partial kept; the task queue
(netqueue.py) pauses and retries on the backoff schedule. A 401/403 raises
AuthRequired: a gated repo without HF_TOKEN, which no retry will fix.
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from app.tts_runtime.fsutil import fsync_dir
from app.tts_runtime.manifest import FilePin, HashMismatch, verify_file

logger = logging.getLogger(__name__)

# Bytes received after the last complete chunk are lost when a connection
# drops mid-chunk; 64 KiB bounds that loss (1 MiB chunks threw away up to
# a megabyte per interruption -- found by the mid-download cut test).
CHUNK = 64 * 1024
# fsync the partial file this often, so a power cut can cost at most this
# much re-download (the final hash check catches any damage regardless).
FSYNC_EVERY = 32 * 1024 * 1024


class NetworkDown(ConnectionError):
    pass


class AuthRequired(PermissionError):
    pass


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "sur-tts-runtime/1"
    return s


def download(url: str, dest: str | Path, pin: FilePin, *, token: str | None = None,
             timeout: float = 30.0, session=None, on_progress=None) -> tuple[str, dict]:
    """Fetch `url` to `dest`, verified against `pin`. Returns (sha256, stats).

    stats: {"resumed_from": bytes already on disk, "fetched": bytes this call,
            "restarted": True if the server ignored Range}.
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stats = {"resumed_from": 0, "fetched": 0, "restarted": False, "already_present": False}
    if dest.exists():
        try:
            sha = verify_file(dest, pin)
            stats["already_present"] = True
            return sha, stats
        except HashMismatch as e:
            logger.warning("existing %s fails verification (%s); deleting and re-fetching", dest, e)
            dest.unlink()

    part = dest.with_name(dest.name + ".partial")
    pos = part.stat().st_size if part.exists() else 0
    if pos > pin.size:
        logger.warning("%s is larger than the pinned file; discarding it", part)
        part.unlink()
        pos = 0
    stats["resumed_from"] = pos

    if pos < pin.size:
        headers = {}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        if token:
            headers["Authorization"] = f"Bearer {token}"   # never logged
        s = session or _session()
        try:
            with s.get(url, headers=headers, stream=True, timeout=(10, timeout), allow_redirects=True) as r:
                if r.status_code in (401, 403):
                    raise AuthRequired(f"{url}: HTTP {r.status_code} -- a gated repository needs HF_TOKEN "
                                       "with its licence accepted on huggingface.co")
                if r.status_code == 416 and pos == pin.size:
                    pass
                elif pos and r.status_code == 200:
                    logger.warning("%s ignored the Range request; restarting %s from byte 0", url, dest.name)
                    stats["restarted"] = True
                    pos = 0
                elif pos and r.status_code == 206:
                    start = int((r.headers.get("Content-Range", "bytes 0-").split()[1]).split("-")[0])
                    if start != pos:
                        raise NetworkDown(f"{url}: server resumed at {start}, asked for {pos}")
                elif r.status_code not in (200, 206):
                    raise NetworkDown(f"{url}: HTTP {r.status_code}")
                mode = "ab" if pos else "wb"
                with open(part, mode) as f:
                    since_sync = 0
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        stats["fetched"] += len(chunk)
                        since_sync += len(chunk)
                        if since_sync >= FSYNC_EVERY:
                            f.flush()
                            os.fsync(f.fileno())
                            since_sync = 0
                        if on_progress:
                            on_progress(pos + stats["fetched"], pin.size)
                    f.flush()
                    os.fsync(f.fileno())
        except NetworkDown:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                ConnectionError, TimeoutError) as e:
            raise NetworkDown(f"{url}: {type(e).__name__}: {e} (kept {part.name}, "
                              f"{part.stat().st_size if part.exists() else 0} bytes)") from e

    got = part.stat().st_size if part.exists() else 0
    if got < pin.size:
        raise NetworkDown(f"{url}: connection ended at {got} of {pin.size} bytes (partial kept)")
    try:
        sha = verify_file(part, pin)
    except HashMismatch:
        with contextlib.suppress(OSError):
            part.unlink()
        raise
    os.replace(part, dest)
    fsync_dir(dest.parent)
    return sha, stats
