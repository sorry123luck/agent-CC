"""Tests for visual_asset_store — 视觉资产 CRUD。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.memory.visual_asset_store import VisualAssetStore


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def store():
    return VisualAssetStore()


class TestVisualAssetCRUD:
    def test_create_and_get(self, db, store):
        with db.session() as session:
            aid = store.create_asset(
                session,
                asset_type="warm_snapshot",
                path="warm/test.jpg",
                format="jpeg",
                canvas_id="c1",
            )
        with db.session() as session:
            asset = store.get_asset(session, aid)
            assert asset is not None
            assert asset["asset_type"] == "warm_snapshot"
            assert asset["path"] == "warm/test.jpg"

    def test_list_by_stable_key(self, db, store):
        with db.session() as session:
            store.create_asset(session, "control_crop", "c1.png", "png", stable_key_id="k1")
            store.create_asset(session, "control_crop", "c2.png", "png", stable_key_id="k1")
            store.create_asset(session, "control_crop", "c3.png", "png", stable_key_id="k2")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 2

    def test_list_by_stable_key_filter_type(self, db, store):
        with db.session() as session:
            store.create_asset(session, "control_crop", "c1.png", "png", stable_key_id="k1")
            store.create_asset(session, "warm_snapshot", "w1.jpg", "jpeg", stable_key_id="k1")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1", asset_type="control_crop")
            assert len(results) == 1

    def test_list_by_canvas(self, db, store):
        with db.session() as session:
            store.create_asset(session, "warm_snapshot", "w1.jpg", "jpeg", canvas_id="c1")
            store.create_asset(session, "warm_snapshot", "w2.jpg", "jpeg", canvas_id="c2")
        with db.session() as session:
            results = store.list_by_canvas(session, "c1")
            assert len(results) == 1

    def test_mark_stale(self, db, store):
        with db.session() as session:
            aid = store.create_asset(session, "warm_snapshot", "w1.jpg", "jpeg")
        with db.session() as session:
            assert store.mark_stale(session, aid)
            asset = store.get_asset(session, aid)
            assert asset["status"] == "stale"

    def test_archive(self, db, store):
        with db.session() as session:
            aid = store.create_asset(session, "warm_snapshot", "w1.jpg", "jpeg")
        with db.session() as session:
            assert store.archive(session, aid)
            asset = store.get_asset(session, aid)
            assert asset["status"] == "archived"

    def test_update_last_seen(self, db, store):
        with db.session() as session:
            aid = store.create_asset(session, "warm_snapshot", "w1.jpg", "jpeg")
        with db.session() as session:
            asset_before = store.get_asset(session, aid)
        with db.session() as session:
            assert store.update_last_seen(session, aid)
        with db.session() as session:
            asset_after = store.get_asset(session, aid)
            assert asset_after["last_seen_at"] >= asset_before["last_seen_at"]

    def test_increment_version(self, db, store):
        with db.session() as session:
            store.create_asset(session, "control_crop", "c1.png", "png", stable_key_id="k1")
        with db.session() as session:
            v = store.next_version(session, "k1", "control_crop")
            assert v == 2

    def test_nonexistent_returns_none(self, db, store):
        with db.session() as session:
            assert store.get_asset(session, "nonexistent") is None

    def test_mark_stale_nonexistent_returns_false(self, db, store):
        with db.session() as session:
            assert not store.mark_stale(session, "nonexistent")


class TestVisualAssetDHash:
    def test_get_by_dhash_with_filter(self, db, store):
        with db.session() as session:
            store.create_asset(session, "control_crop", "c1.png", "png",
                               stable_key_id="k1", dhash="abcd1234abcd5678")
            store.create_asset(session, "control_crop", "c2.png", "png",
                               stable_key_id="k1", dhash="abcd1234abcd5678")
            store.create_asset(session, "control_crop", "c3.png", "png",
                               stable_key_id="k2", dhash="abcd1234abcd5678")
        with db.session() as session:
            results = store.get_by_dhash(session, "abcd1234abcd5678", stable_key_id="k1")
            assert len(results) == 2

    def test_get_by_dhash_requires_filter(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="requires at least one filter"):
                store.get_by_dhash(session, "abcd1234abcd5678")


class TestVisualAssetDelete:
    def test_delete_nonexistent_returns_false(self, db, store):
        with db.session() as session:
            assert not store.delete_asset(session, "nonexistent")

    def test_delete_existing(self, db, store):
        with db.session() as session:
            aid = store.create_asset(session, "warm_snapshot", "w1.jpg", "jpeg")
        with db.session() as session:
            assert store.delete_asset(session, aid)
            assert store.get_asset(session, aid) is None

    def test_delete_orphaned_dry_run_forced(self, db, store):
        with db.session() as session:
            store.create_asset(session, "warm_snapshot", "/nonexistent/path.jpg", "jpeg")
        with db.session() as session:
            orphans = store.delete_orphaned_assets(session, dry_run=True)
            assert len(orphans) == 1

    def test_delete_orphaned_real_delete_raises(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="dry_run=True"):
                store.delete_orphaned_assets(session, dry_run=False)
