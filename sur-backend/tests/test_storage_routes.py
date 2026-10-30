"""The local storage router stands in for a presigned S3/MinIO URL, so it has
to answer the same methods one does.

A real presigned GET URL answers HEAD with the object's Content-Length. The
local route registered GET only, so the export screen's size check returned
405 and logged a console error that could never happen against S3 -- a dev
stand-in that behaves differently from the thing it replaces hides bugs in
one direction and invents them in the other.
"""
from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings, resolve_backend_path
from app.main import app


@pytest.fixture
def local_file(monkeypatch, tmp_path):
    """A real file under the configured local storage root."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path))
    get_settings.cache_clear()

    key = "projects/p1/exports/e1/output.mp4"
    target = pathlib.Path(resolve_backend_path(str(tmp_path))) / key
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"not really an mp4, but a real byte count" * 7
    target.write_bytes(payload)
    yield key, len(payload)
    get_settings.cache_clear()


def test_head_returns_the_size_without_the_body(local_file):
    key, size = local_file
    with TestClient(app) as client:
        res = client.head(f"/api/storage/files/{key}")

    assert res.status_code == 200, f"HEAD must not 405: {res.status_code}"
    assert res.headers["content-length"] == str(size)
    assert res.content == b"", "HEAD must send headers only"


def test_get_still_returns_the_bytes(local_file):
    key, size = local_file
    with TestClient(app) as client:
        res = client.get(f"/api/storage/files/{key}")

    assert res.status_code == 200
    assert len(res.content) == size


def test_head_on_a_missing_file_is_404_not_405(local_file):
    with TestClient(app) as client:
        res = client.head("/api/storage/files/projects/p1/exports/e1/nope.mp4")

    assert res.status_code == 404


def test_segment_audio_fields_are_urls_not_storage_keys(monkeypatch, tmp_path):
    """`tts_audio_url` must be fetchable by a client, not a bare storage key.

    The column holds a key ("projects/<id>/segments/<id>/tts.wav"). GET
    /export already presigned its key before returning it; the segment routes
    returned the raw row, so the client resolved the key against the API base,
    got a 404, and the browser blocked it as ORB. No segment audio played at
    all under STORAGE_BACKEND=local -- and with S3 it worked by accident,
    because a presigned URL is absolute, which is why nobody saw it.
    """
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path))
    get_settings.cache_clear()

    from app.schemas.segment import segment_read

    class Row:
        id = "s1"
        project_id = "p1"
        speaker_id = None
        index = 0
        start_ms = 0
        end_ms = 1000
        source_text = "hi"
        detected_language = "en"
        detected_language_confidence = 0.9
        translated_text = "namaste"
        emotion_label = None
        emotion_score = None
        emotion_overridden = False
        source_audio_url = "projects/p1/segments/s1/source.wav"
        tts_audio_url = "projects/p1/segments/s1/tts.wav"
        tts_duration_ms = 900
        sync_offset_pct = 0.0
        status = "synthesized"
        error_message = None
        created_at = __import__("datetime").datetime(2026, 1, 1)
        updated_at = __import__("datetime").datetime(2026, 1, 1)

    read = segment_read(Row())

    for field in ("tts_audio_url", "source_audio_url"):
        value = getattr(read, field)
        assert value, f"{field} disappeared"
        assert value.startswith("/") or value.startswith("http"), (
            f"{field} is still a raw storage key ({value!r}); a client cannot fetch that"
        )
    get_settings.cache_clear()
