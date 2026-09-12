from __future__ import annotations

import shutil
from pathlib import Path

from app.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Local filesystem storage backend for development without MinIO/S3.

    The root comes from settings.local_storage_root, anchored at the backend
    directory rather than the process's working directory: the API writes
    uploads here and separate worker processes read them, so a CWD-relative
    root silently split them into two different trees.
    """

    def __init__(self, root_dir: str | None = None) -> None:
        if root_dir is None:
            from app.config import get_settings, resolve_backend_path

            root_dir = resolve_backend_path(get_settings().local_storage_root)
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str, *, create_parent: bool = False) -> Path:
        path = (self.root / key.strip("/")).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"storage key escapes the storage root: {key!r}")
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def presigned_put_url(self, key: str, content_type: str, expires_in: int = 3600) -> str:
        # Served by app/api/routes_storage.py, standing in for a presigned URL.
        return f"/api/storage/upload/{key}"

    def presigned_get_url(self, key: str, expires_in: int = 3600) -> str:
        return f"/api/storage/files/{key}"

    def upload_file(self, key: str, local_path: str, content_type: str | None = None) -> None:
        shutil.copyfile(local_path, self._resolve_path(key, create_parent=True))

    def download_file(self, key: str, local_path: str) -> None:
        # Raising matters: this used to return silently when the object was
        # missing, leaving `local_path` absent or empty, so the failure
        # surfaced one step later as an unrelated ffmpeg "Invalid data found"
        # instead of naming the key that was never written. S3Storage raises
        # here too (botocore 404); both backends must fail the same way.
        target = self._resolve_path(key)
        if not target.is_file():
            raise FileNotFoundError(f"no object at key {key!r} (local storage root: {self.root})")
        shutil.copyfile(target, local_path)

    def exists(self, key: str) -> bool:
        return self._resolve_path(key).is_file()

    def delete(self, key: str) -> None:
        self._resolve_path(key).unlink(missing_ok=True)

    def get_size(self, key: str) -> int:
        target = self._resolve_path(key)
        if not target.is_file():
            raise FileNotFoundError(f"no object at key {key!r}")
        return target.stat().st_size
