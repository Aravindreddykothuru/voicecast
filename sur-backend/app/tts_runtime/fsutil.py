"""Crash-safe file writes and disk checks.

A file the runtime produces is either absent or complete: written to a temp
name in the same directory, flushed, fsync'd, then atomically renamed over
the destination. A crash, kill -9 or power cut leaves at worst a stray temp
file (swept on the next run), never a half-written output under a real name.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import shutil
import uuid
from pathlib import Path

TMP_MARK = ".tmp-"


class DiskFull(OSError):
    """ENOSPC while writing; the temp file has been removed, nothing replaced."""


def fsync_dir(path: str | os.PathLike) -> None:
    # Makes the rename itself durable on POSIX. Windows cannot open a
    # directory for fsync; NTFS journals the rename metadata itself.
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: str | os.PathLike, write) -> None:
    """Call write(fileobj) on a temp file, then atomically move it to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{TMP_MARK}{uuid.uuid4().hex[:12]}-{path.name}"
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except OSError as e:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        if e.errno == errno.ENOSPC:
            raise DiskFull(errno.ENOSPC, f"disk full writing {path}; nothing was replaced") from e
        raise
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    atomic_write(path, lambda f: f.write(data))


def sweep_temp_files(directory: str | os.PathLike) -> int:
    """Remove temp files a crash left behind. Returns how many."""
    n = 0
    d = Path(directory)
    if not d.exists():
        return 0
    for p in d.rglob(f"{TMP_MARK}*"):
        with contextlib.suppress(OSError):
            p.unlink()
            n += 1
    return n


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def free_mb(path: str | os.PathLike) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / (1024 * 1024)
