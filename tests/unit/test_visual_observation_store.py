"""Tests for visual_observation_store — 视觉观察记录 CRUD。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.memory.visual_observation_store import VisualObservationStore


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def store():
    return VisualObservationStore()


class TestVisualObservationCreateAndGet:
    def test_create_and_get(self, db, store):
        with db.session() as session:
            oid = store.create_observation(
                session,
                asset_id_a="a1",
                match_status="exact_match",
                dhash_distance=0,
                method="dhash_only",
            )
        with db.session() as session:
            obs = store.get_observation(session, oid)
            assert obs is not None
            assert obs["asset_id_a"] == "a1"
            assert obs["match_status"] == "exact_match"
            assert obs["dhash_distance"] == 0
            assert obs["method"] == "dhash_only"

    def test_create_with_asset_id_b(self, db, store):
        with db.session() as session:
            oid = store.create_observation(
                session,
                asset_id_a="a1",
                match_status="strong_match",
                asset_id_b="a2",
                dhash_distance=3,
                ssim_score=0.95,
                template_score=0.88,
                coordinate_drift=2.5,
                method="dhash+ssim+template",
            )
        with db.session() as session:
            obs = store.get_observation(session, oid)
            assert obs["asset_id_b"] == "a2"
            assert obs["ssim_score"] == 0.95
            assert obs["template_score"] == 0.88
            assert obs["coordinate_drift"] == 2.5

    def test_create_with_nullable_asset_id_b(self, db, store):
        """asset_id_b 可为 NULL（当前截图尚未持久化时）。"""
        with db.session() as session:
            oid = store.create_observation(
                session,
                asset_id_a="a1",
                match_status="no_match",
                asset_id_b=None,
                observed_canvas_id="c1",
                observed_bounds_json="[10,20,100,50]",
                method="dhash_only",
            )
        with db.session() as session:
            obs = store.get_observation(session, oid)
            assert obs["asset_id_b"] is None
            assert obs["observed_canvas_id"] == "c1"
            assert obs["observed_bounds_json"] == "[10,20,100,50]"

    def test_create_with_all_fields(self, db, store):
        with db.session() as session:
            oid = store.create_observation(
                session,
                asset_id_a="a1",
                match_status="drift_detected",
                asset_id_b="a2",
                stable_key_id="k1",
                canvas_id_a="c1",
                canvas_id_b="c2",
                observed_canvas_id="c2",
                observed_bounds_json="[10,20,100,50]",
                observed_crop_path="/tmp/crop.png",
                dhash_distance=8,
                ssim_score=0.72,
                template_score=0.65,
                coordinate_drift=12.0,
                roi_bounds_json="[5,15,110,60]",
                method="dhash+ssim+template",
                metadata='{"note":"test"}',
            )
        with db.session() as session:
            obs = store.get_observation(session, oid)
            assert obs["stable_key_id"] == "k1"
            assert obs["canvas_id_a"] == "c1"
            assert obs["canvas_id_b"] == "c2"
            assert obs["observed_canvas_id"] == "c2"
            assert obs["observed_crop_path"] == "/tmp/crop.png"
            assert obs["roi_bounds_json"] == "[5,15,110,60]"
            assert obs["metadata"] == '{"note":"test"}'

    def test_nonexistent_returns_none(self, db, store):
        with db.session() as session:
            assert store.get_observation(session, "nonexistent") is None


class TestVisualObservationQueries:
    def test_list_by_stable_key(self, db, store):
        with db.session() as session:
            store.create_observation(session, "a1", "exact_match", stable_key_id="k1", method="dhash_only")
            store.create_observation(session, "a1", "strong_match", stable_key_id="k1", method="dhash_only")
            store.create_observation(session, "a2", "no_match", stable_key_id="k2", method="dhash_only")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 2

    def test_list_by_stable_key_limit(self, db, store):
        with db.session() as session:
            for i in range(10):
                store.create_observation(session, "a1", "exact_match", stable_key_id="k1", method="dhash_only")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1", limit=3)
            assert len(results) == 3

    def test_list_by_stable_key_ordered_desc(self, db, store):
        """结果按 compared_at 降序排列。"""
        with db.session() as session:
            store.create_observation(session, "a1", "exact_match", stable_key_id="k1", method="dhash_only")
            store.create_observation(session, "a1", "no_match", stable_key_id="k1", method="dhash_only")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert results[0]["match_status"] == "no_match"
            assert results[1]["match_status"] == "exact_match"

    def test_list_by_asset_pair(self, db, store):
        with db.session() as session:
            store.create_observation(session, "a1", "exact_match", asset_id_b="a2", method="dhash_only")
            store.create_observation(session, "a1", "exact_match", asset_id_b="a3", method="dhash_only")
            store.create_observation(session, "a2", "no_match", asset_id_b="a3", method="dhash_only")
        with db.session() as session:
            results = store.list_by_asset_pair(session, "a1", "a2")
            assert len(results) == 1

    def test_list_by_asset_pair_empty(self, db, store):
        with db.session() as session:
            results = store.list_by_asset_pair(session, "nonexistent", "also_nonexistent")
            assert len(results) == 0

    def test_get_latest_match(self, db, store):
        with db.session() as session:
            store.create_observation(session, "a1", "exact_match", stable_key_id="k1", method="dhash_only")
            store.create_observation(session, "a1", "no_match", stable_key_id="k1", method="dhash_only")
        with db.session() as session:
            latest = store.get_latest_match(session, "k1")
            assert latest is not None
            assert latest["match_status"] == "no_match"

    def test_get_latest_match_nonexistent(self, db, store):
        with db.session() as session:
            assert store.get_latest_match(session, "nonexistent") is None


class TestVisualObservationDelete:
    def test_delete_old_observations(self, db, store):
        with db.session() as session:
            store.create_observation(session, "a1", "exact_match", stable_key_id="k1", method="dhash_only")
        with db.session() as session:
            # 删除未来日期之前的所有记录 → 应该删除刚创建的
            count = store.delete_old_observations(session, "2099-01-01T00:00:00")
            assert count == 1
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 0

    def test_delete_old_observations_keeps_new(self, db, store):
        with db.session() as session:
            store.create_observation(session, "a1", "exact_match", stable_key_id="k1", method="dhash_only")
        with db.session() as session:
            # 删除过去日期之前的所有记录 → 不应该删除刚创建的
            count = store.delete_old_observations(session, "2020-01-01T00:00:00")
            assert count == 0
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 1

    def test_delete_old_observations_empty(self, db, store):
        with db.session() as session:
            count = store.delete_old_observations(session, "2099-01-01T00:00:00")
            assert count == 0
