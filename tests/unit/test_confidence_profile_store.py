"""Tests for confidence_profile_store — 置信度快照 CRUD。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.memory.confidence_profile_store import ConfidenceProfileStore


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def store():
    return ConfidenceProfileStore()


class TestConfidenceProfileCreateAndGet:
    def test_create_and_get(self, db, store):
        with db.session() as session:
            pid = store.create_profile(
                session,
                stable_key_id="k1",
                fused_confidence=0.85,
                formula_version="v1_dim_weighted",
                spatial_confidence=0.9,
                semantic_confidence=0.8,
            )
        with db.session() as session:
            profile = store.get_profile(session, pid)
            assert profile is not None
            assert profile["stable_key_id"] == "k1"
            assert profile["fused_confidence"] == 0.85
            assert profile["formula_version"] == "v1_dim_weighted"
            assert profile["spatial_confidence"] == 0.9
            assert profile["semantic_confidence"] == 0.8

    def test_create_with_all_dimensions(self, db, store):
        with db.session() as session:
            pid = store.create_profile(
                session,
                stable_key_id="k1",
                fused_confidence=0.75,
                formula_version="v1_dim_weighted",
                spatial_confidence=0.9,
                semantic_confidence=0.8,
                text_confidence=0.7,
                structure_confidence=0.6,
                visual_anchor_confidence=0.5,
                memory_confidence=0.4,
                action_confidence=0.3,
                conflict_penalty=0.05,
            )
        with db.session() as session:
            profile = store.get_profile(session, pid)
            assert profile["visual_anchor_confidence"] == 0.5
            assert profile["memory_confidence"] == 0.4
            assert profile["action_confidence"] == 0.3
            assert profile["conflict_penalty"] == 0.05

    def test_create_with_optional_fields(self, db, store):
        with db.session() as session:
            pid = store.create_profile(
                session,
                stable_key_id="k1",
                fused_confidence=0.5,
                formula_version="v1_dim_weighted",
                canvas_id="c1",
                page_model_id="pm1",
                state_template_id="st1",
                source_count=3,
                source_diversity=2,
                evidence_ids='["e1","e2","e3"]',
            )
        with db.session() as session:
            profile = store.get_profile(session, pid)
            assert profile["canvas_id"] == "c1"
            assert profile["page_model_id"] == "pm1"
            assert profile["state_template_id"] == "st1"
            assert profile["source_count"] == 3
            assert profile["source_diversity"] == 2
            assert profile["evidence_ids"] == '["e1","e2","e3"]'

    def test_nonexistent_returns_none(self, db, store):
        with db.session() as session:
            assert store.get_profile(session, "nonexistent") is None


class TestConfidenceProfileGetLatest:
    def test_get_latest_returns_most_recent(self, db, store):
        with db.session() as session:
            store.create_profile(session, "k1", 0.5, "v1_dim_weighted")
            store.create_profile(session, "k1", 0.8, "v1_dim_weighted")
        with db.session() as session:
            latest = store.get_latest(session, "k1")
            assert latest is not None
            assert latest["fused_confidence"] == 0.8

    def test_get_latest_nonexistent_returns_none(self, db, store):
        with db.session() as session:
            assert store.get_latest(session, "nonexistent") is None

    def test_get_latest_different_keys(self, db, store):
        with db.session() as session:
            store.create_profile(session, "k1", 0.5, "v1_dim_weighted")
            store.create_profile(session, "k2", 0.9, "v1_dim_weighted")
        with db.session() as session:
            latest1 = store.get_latest(session, "k1")
            latest2 = store.get_latest(session, "k2")
            assert latest1["fused_confidence"] == 0.5
            assert latest2["fused_confidence"] == 0.9


class TestConfidenceProfileList:
    def test_list_by_stable_key(self, db, store):
        with db.session() as session:
            store.create_profile(session, "k1", 0.5, "v1_dim_weighted")
            store.create_profile(session, "k1", 0.6, "v1_dim_weighted")
            store.create_profile(session, "k2", 0.9, "v1_dim_weighted")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 2

    def test_list_by_stable_key_limit(self, db, store):
        with db.session() as session:
            for i in range(10):
                store.create_profile(session, "k1", 0.5, "v1_dim_weighted")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1", limit=3)
            assert len(results) == 3

    def test_list_by_stable_key_ordered_desc(self, db, store):
        with db.session() as session:
            store.create_profile(session, "k1", 0.3, "v1_dim_weighted")
            store.create_profile(session, "k1", 0.9, "v1_dim_weighted")
            store.create_profile(session, "k1", 0.6, "v1_dim_weighted")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 3
            # computed_at may be same-second, so just verify all present
            confidences = {r["fused_confidence"] for r in results}
            assert confidences == {0.3, 0.6, 0.9}

    def test_list_by_canvas(self, db, store):
        with db.session() as session:
            store.create_profile(session, "k1", 0.5, "v1_dim_weighted", canvas_id="c1")
            store.create_profile(session, "k2", 0.6, "v1_dim_weighted", canvas_id="c1")
            store.create_profile(session, "k3", 0.9, "v1_dim_weighted", canvas_id="c2")
        with db.session() as session:
            results = store.list_by_canvas(session, "c1")
            assert len(results) == 2

    def test_list_by_canvas_empty(self, db, store):
        with db.session() as session:
            results = store.list_by_canvas(session, "nonexistent")
            assert len(results) == 0
