"""Tests for migration_evidence — 迁移脚本。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    return db


class TestMigrationEvidence:
    def test_empty_db_creates_tables(self, db):
        created = migrate(db)
        assert "visual_assets" in created
        assert "visual_observations" in created
        assert "candidate_evidence" in created
        assert "candidate_confidence_profiles" in created

    def test_existing_tables_skipped(self, db):
        migrate(db)
        created_again = migrate(db)
        assert created_again == []

    def test_stable_candidate_keys_has_new_columns(self, db):
        from sqlalchemy import inspect
        db.create_all()
        migrate(db)
        inspector = inspect(db.engine)
        columns = {col["name"] for col in inspector.get_columns("stable_candidate_keys")}
        assert "permanence_state" in columns
        assert "state_changed_at" in columns
        assert "fail_count" in columns
        assert "coordinate_drift" in columns
        assert "latest_profile_id" in columns

    def test_idempotent(self, db):
        migrate(db)
        migrate(db)
        migrate(db)
        # Should not raise
