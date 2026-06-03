"""Opening a MemoryService: scope resolution and keyless degradation."""

from __future__ import annotations

from pathlib import Path

from mneme.service.factory import build_llm_client, open_service


def test_build_llm_client_none_without_key():
    assert build_llm_client({}) is None


def test_open_service_is_capture_only_without_key(tmp_path: Path):
    service = open_service(tmp_path, {})

    assert service.can_consolidate is False
    assert (tmp_path / ".mneme" / "memory.db").exists()


def test_open_service_honours_explicit_db(tmp_path: Path):
    db = tmp_path / "custom.db"

    service = open_service(tmp_path, {"MNEME_DB": str(db)})

    service.capture("user", "hello")
    assert db.exists()
