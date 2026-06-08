"""Unit tests for Icon Memory Management API (Phase 4C)."""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.integration.api_server import app
    return TestClient(app)


class TestListIconMemory:
    def test_list_empty(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.list_assets.return_value = []
                resp = client.get("/api/v1/icon-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["items"] == []

    def test_list_with_filters(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.list_assets.return_value = [
                    {"asset_id": "a1", "semantic_role": "button", "semantic_state": "confirmed"},
                ]
                resp = client.get("/api/v1/icon-memory?app_process=qq.exe&state=confirmed&role=button&limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        # Verify filters passed to store
        call_kwargs = si.list_assets.call_args[1]
        assert call_kwargs["app_process"] == "qq.exe"
        assert call_kwargs["state"] == "confirmed"
        assert call_kwargs["role"] == "button"
        assert call_kwargs["limit"] == 10


class TestGetIconMemory:
    def test_get_found(self, client):
        mock_match = MagicMock()
        mock_match.asset_id = "abc123"
        mock_match.meta = {"semantic_role": "button", "semantic_state": "confirmed"}
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.get_asset.return_value = mock_match
                resp = client.get("/api/v1/icon-memory/abc123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asset_id"] == "abc123"
        assert data["semantic_role"] == "button"

    def test_get_not_found(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.get_asset.return_value = None
                resp = client.get("/api/v1/icon-memory/nonexistent")
        assert resp.status_code == 404


class TestConfirmIconMemory:
    def test_confirm_success(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.confirm.return_value = True
                resp = client.post("/api/v1/icon-memory/abc123/confirm")
        assert resp.status_code == 200
        assert resp.json()["status"] == "confirmed"

    def test_confirm_rejected_fails(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.confirm.return_value = False
                resp = client.post("/api/v1/icon-memory/abc123/confirm")
        assert resp.status_code == 400


class TestRejectIconMemory:
    def test_reject_success(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.reject.return_value = True
                resp = client.post("/api/v1/icon-memory/abc123/reject")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_reject_not_found(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.reject.return_value = False
                resp = client.post("/api/v1/icon-memory/nonexistent/reject")
        assert resp.status_code == 404


class TestMarkLowQuality:
    def test_mark_success(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.mark_low_quality.return_value = True
                resp = client.post("/api/v1/icon-memory/abc123/mark-low-quality?reason=oversized")
        assert resp.status_code == 200
        assert resp.json()["quality_flag"] == "oversized"

    def test_mark_not_found(self, client):
        with patch("src.storage.db.Session") as ms:
            sess = MagicMock()
            ms.return_value.__enter__ = MagicMock(return_value=sess)
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                si = MagicMock(); mc.return_value = si
                si.mark_low_quality.return_value = False
                resp = client.post("/api/v1/icon-memory/nonexistent/mark-low-quality")
        assert resp.status_code == 404
