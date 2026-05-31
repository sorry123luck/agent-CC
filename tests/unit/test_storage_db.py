"""Tests for SQLite database engine configuration."""

from __future__ import annotations

from sqlalchemy.pool import QueuePool, StaticPool

from src.storage.db import Database


def test_file_sqlite_uses_non_static_pool_for_background_threads(tmp_path):
    db = Database(str(tmp_path / "openclaw.db"))
    db.init()

    assert isinstance(db.engine.pool, QueuePool)


def test_memory_sqlite_keeps_static_pool():
    db = Database(":memory:")
    db.init()

    assert isinstance(db.engine.pool, StaticPool)
