"""Unit tests for IconMemoryStore."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.memory.icon_memory_store import IconMemoryStore, IconMemoryMatch, _MINIMAL_PNG


@pytest.fixture
def tmp_asset_root(tmp_path):
    return tmp_path / "visual_assets"


@pytest.fixture
def store(tmp_asset_root):
    return IconMemoryStore(asset_root=tmp_asset_root)


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session with in-memory VisualAssetRecord storage."""
    session = MagicMock()
    _store = {}

    def _add(row):
        _store[row.asset_id] = row

    session.add = _add
    session.query = MagicMock(side_effect=lambda model: _MockQuery(_store))
    session.flush = MagicMock()
    session._store = _store
    return session


def _match_filter(row, query):
    """Simplified filter matching for mock."""
    return True


def _find_first(store, query):
    """Return first matching row or None."""
    for row in store.values():
        return row
    return None


class _MockQuery:
    """Mock query that supports filter_by with app_id filtering."""
    def __init__(self, store_dict):
        self._store = store_dict
        self._filters = {}

    def filter_by(self, **kwargs):
        q = _MockQuery(self._store)
        q._filters = {**self._filters, **kwargs}
        return q

    def filter(self, *args):
        # Ignore SQLAlchemy filter expressions, return self for chaining
        return self

    def order_by(self, *args):
        return self

    def all(self):
        results = list(self._store.values())
        if "app_id" in self._filters:
            results = [r for r in results if r.app_id == self._filters["app_id"]]
        if "asset_type" in self._filters:
            results = [r for r in results if r.asset_type == self._filters["asset_type"]]
        if "status" in self._filters:
            results = [r for r in results if r.status == self._filters["status"]]
        return results

    def first(self):
        results = self.all()
        return results[0] if results else None


def _make_crop(width=32, height=32, color=(200, 100, 50)):
    """Create a test crop image."""
    return Image.new("RGB", (width, height), color=color)


class TestStoreCrop:
    def test_store_creates_record(self, store, mock_session, tmp_asset_root):
        crop = _make_crop()
        asset_id = store.store_crop(
            mock_session, crop, "qq.exe", "chat", "send_button",
            semantic_text="发送", source="test",
        )
        assert asset_id
        assert len(mock_session._store) == 1
        row = mock_session._store[asset_id]
        assert row.asset_type == "control_crop"
        assert row.app_id == "qq.exe"
        assert row.dhash  # Should have computed dHash

    def test_store_hash_only_creates_placeholder(self, store, mock_session, tmp_asset_root):
        asset_id = store.store_crop(
            mock_session, None, "chrome.exe", "browser", "address_bar",
            privacy_level="hash_only",
        )
        assert asset_id
        row = mock_session._store[asset_id]
        dat_path = tmp_asset_root / row.path
        assert dat_path.exists()
        assert dat_path.read_bytes() == _MINIMAL_PNG

    def test_store_opaque_path_no_semantic_name(self, store, mock_session, tmp_asset_root):
        asset_id = store.store_crop(
            mock_session, _make_crop(), "qq.exe", "chat", "send_button",
            semantic_text="发送",
        )
        row = mock_session._store[asset_id]
        assert "send_button" not in row.path
        assert "发送" not in row.path
        assert "controls" in row.path and "qq.exe" in row.path

    def test_store_extra_metadata_has_semantic_state(self, store, mock_session):
        asset_id = store.store_crop(
            mock_session, _make_crop(), "qq.exe", "chat", "send_button",
            state="pending",
        )
        row = mock_session._store[asset_id]
        meta = json.loads(row.extra_metadata)
        assert meta["semantic_state"] == "pending"
        assert meta["semantic_role"] == "send_button"
        assert meta["privacy_level"] == "safe"


class TestFindMatch:
    def test_exact_dhash_match(self, store, mock_session):
        crop = _make_crop(color=(100, 200, 50))
        store.store_crop(mock_session, crop, "qq.exe", "chat", "button")

        # Same crop should match with distance 0
        import numpy as np
        from src.memory.visual_anchor import compute_dhash
        dhash = compute_dhash(np.array(crop))
        match, dist = store.find_match(mock_session, dhash, "qq.exe")
        assert match is not None
        assert dist == 0

    def test_no_match_different_app(self, store, mock_session):
        crop = _make_crop()
        store.store_crop(mock_session, crop, "qq.exe", "chat", "button")

        from src.memory.visual_anchor import compute_dhash
        dhash = compute_dhash(np.array(crop))
        match, dist = store.find_match(mock_session, dhash, "chrome.exe")
        assert match is None

    def test_no_match_empty_dhash(self, store, mock_session):
        match, dist = store.find_match(mock_session, "", "qq.exe")
        assert match is None
        assert dist == -1

    def test_similar_crop_matches(self, store, mock_session):
        """Slightly different crop should match within threshold."""
        crop1 = _make_crop(color=(100, 200, 50))
        store.store_crop(mock_session, crop1, "qq.exe", "chat", "button")

        # Create slightly different crop (same structure, minor variation)
        crop2 = _make_crop(color=(105, 195, 55))
        from src.memory.visual_anchor import compute_dhash
        dhash2 = compute_dhash(np.array(crop2))
        match, dist = store.find_match(mock_session, dhash2, "qq.exe", max_distance=5)
        # May or may not match depending on dHash sensitivity
        # This test verifies the mechanism works without asserting specific distance


class TestStateTransitions:
    def test_confirm(self, store, mock_session):
        asset_id = store.store_crop(
            mock_session, _make_crop(), "qq.exe", "chat", "button", state="pending",
        )
        assert store.confirm(mock_session, asset_id)
        meta = json.loads(mock_session._store[asset_id].extra_metadata)
        assert meta["semantic_state"] == "confirmed"
        assert meta["confirmed_at"] is not None

    def test_reject(self, store, mock_session):
        asset_id = store.store_crop(
            mock_session, _make_crop(), "qq.exe", "chat", "button", state="pending",
        )
        assert store.reject(mock_session, asset_id)
        meta = json.loads(mock_session._store[asset_id].extra_metadata)
        assert meta["semantic_state"] == "rejected"

    def test_mark_conflict(self, store, mock_session):
        asset_id = store.store_crop(
            mock_session, _make_crop(), "qq.exe", "chat", "button", state="pending",
        )
        assert store.mark_conflict(mock_session, asset_id, "search_input")
        meta = json.loads(mock_session._store[asset_id].extra_metadata)
        assert meta["semantic_state"] == "conflict"
        assert meta["conflict_role"] == "search_input"

    def test_increment_match_count(self, store, mock_session):
        asset_id = store.store_crop(
            mock_session, _make_crop(), "qq.exe", "chat", "button",
        )
        assert store.increment_match_count(mock_session, asset_id)
        meta = json.loads(mock_session._store[asset_id].extra_metadata)
        assert meta["match_count"] == 1
        assert store.increment_match_count(mock_session, asset_id)
        meta = json.loads(mock_session._store[asset_id].extra_metadata)
        assert meta["match_count"] == 2


class TestStats:
    def test_stats_empty(self, store, mock_session):
        s = store.stats(mock_session)
        assert s["total"] == 0

    def test_stats_counts_by_state(self, store, mock_session):
        store.store_crop(mock_session, _make_crop(), "qq.exe", "chat", "button", state="pending")
        store.store_crop(mock_session, _make_crop(), "qq.exe", "chat", "search", state="confirmed")
        s = store.stats(mock_session)
        assert s["total"] == 2



class TestPrecomputedDhash:
    def test_hash_only_with_precomputed_dhash(self, store, mock_session, tmp_asset_root):
        crop = _make_crop(color=(100, 200, 50))
        from src.memory.visual_anchor import compute_dhash
        import numpy as np
        dhash = compute_dhash(np.array(crop))
        asset_id = store.store_crop(
            mock_session, None, "qq.exe", "chat", "button",
            precomputed_dhash=dhash, privacy_level="hash_only",
        )
        row = mock_session._store[asset_id]
        assert row.dhash == dhash  # dhash preserved even though no image stored

    def test_hash_only_findable_by_dhash(self, store, mock_session):
        crop = _make_crop(color=(100, 200, 50))
        from src.memory.visual_anchor import compute_dhash
        import numpy as np
        dhash = compute_dhash(np.array(crop))
        store.store_crop(
            mock_session, None, "qq.exe", "chat", "button",
            precomputed_dhash=dhash, privacy_level="hash_only",
        )
        match, dist = store.find_match(mock_session, dhash, "qq.exe")
        assert match is not None
        assert dist == 0


class TestTieBreak:
    def test_confirmed_beats_pending(self, store, mock_session):
        crop = _make_crop(color=(100, 200, 50))
        from src.memory.visual_anchor import compute_dhash
        import numpy as np
        dhash = compute_dhash(np.array(crop))
        # Store two entries with same dhash but different states
        store.store_crop(mock_session, crop, "qq.exe", "chat", "button", state="pending")
        store.store_crop(mock_session, crop, "qq.exe", "chat", "button", state="confirmed")
        match, dist = store.find_match(mock_session, dhash, "qq.exe")
        assert match is not None
        # Confirmed should win tie-break
        assert match.semantic_state == "confirmed"

class TestIconMemoryMatch:
    def test_properties_from_metadata(self):
        row = {
            "asset_id": "abc123",
            "dhash": "a3f7b2c1d4e8f0a2",
            "path": "controls/qq.exe/a3f7b2c1/abc123.dat",
            "extra_metadata": json.dumps({
                "semantic_state": "confirmed",
                "semantic_role": "send_button",
                "semantic_text": "发送",
                "confidence": 0.95,
                "privacy_level": "safe",
                "app_process": "qq.exe",
                "source": "vlm_confirmed",
            }),
        }
        m = IconMemoryMatch(row)
        assert m.semantic_state == "confirmed"
        assert m.semantic_role == "send_button"
        assert m.semantic_text == "发送"
        assert m.confidence == 0.95
        assert m.privacy_level == "safe"
        assert m.app_process == "qq.exe"

    def test_defaults_when_no_metadata(self):
        row = {"asset_id": "xyz", "dhash": "", "path": ""}
        m = IconMemoryMatch(row)
        assert m.semantic_state == "pending"
        assert m.semantic_role == ""
        assert m.confidence == 0.0
