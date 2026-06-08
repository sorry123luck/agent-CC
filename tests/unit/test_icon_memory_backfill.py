"""Unit tests for _icon_memory_backfill (Phase 3 conservative backfill)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.memory.icon_memory_store import IconMemoryMatch


def _mk_elem(eid="e1", role="button", text="", bounds=(10, 10, 50, 40)):
    e = MagicMock()
    e.element_id = eid
    e.semantic_role = MagicMock()
    e.semantic_role.value = role
    e.text = text
    e.bounds = bounds
    e.attributes = {}
    return e


def _mk_canvas(elems=None):
    c = MagicMock()
    c.elements = elems or []
    c.artifacts = {}
    return c


def _mk_ss():
    return Image.new("RGB", (200, 100), color=(128, 128, 128))


def _mk_match(role="send_button", conf=0.9, state="confirmed", app_process="test_app"):
    return IconMemoryMatch({
        "asset_id": "abc", "dhash": "x", "path": "x",
        "extra_metadata": json.dumps({
            "semantic_state": state, "semantic_role": role,
            "semantic_text": role.replace("_", " "), "confidence": conf,
            "app_process": app_process,
        }),
    })


class TestDefaultDisabled:
    def test_no_artifact(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_ICON_MEMORY_BACKFILL", raising=False)
        from src.integration.api_server import _icon_memory_backfill
        c = _mk_canvas([_mk_elem()])
        _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert "icon_memory_backfills" not in c.artifacts

    def test_no_element_modification(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_ICON_MEMORY_BACKFILL", raising=False)
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button")
        c = _mk_canvas([e])
        _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert e.semantic_role.value == "button"


class TestEnabled:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_ICON_MEMORY_BACKFILL", "1")

    def test_exact_confirmed_backfills(self):
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (_mk_match("send_button", 0.9, "confirmed"), 0)
                _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 1
        assert c.artifacts["icon_memory_backfills"]["details"][0]["new_role"] == "send_button"

    def test_strong_confirmed_backfills(self):
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (_mk_match("search_input", 0.9, "confirmed"), 2)
                _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 1

    def test_weak_not_backfilled(self):
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (_mk_match("send_button", 0.9, "confirmed"), 4)
                _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_weak"] == 1

    def test_pending_not_backfilled(self):
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (_mk_match("send_button", 0.9, "pending"), 0)
                _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_pending"] == 1

    def test_protected_role_not_backfilled(self):
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="send_button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (_mk_match("search_input", 0.9, "confirmed"), 0)
                _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_protected"] == 1

    def test_no_screenshot_skips(self):
        from src.integration.api_server import _icon_memory_backfill
        c = _mk_canvas([_mk_elem()])
        _icon_memory_backfill(c, None, "qq.exe")
        assert "icon_memory_backfills" not in c.artifacts

    def test_artifact_name(self):
        from src.integration.api_server import _icon_memory_backfill
        c = _mk_canvas([])
        _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert "icon_memory_backfills" in c.artifacts

    def test_orphan_app_skipped(self):
        """Match with different app_process should not backfill."""
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        # Match with app_process="other_app" but function called with "test_app"
        m = _mk_match("send_button", 0.9, "confirmed", app_process="other_app")
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (m, 0)
                _icon_memory_backfill(c, _mk_ss(), "test_app.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_orphan_app"] == 1

    def test_empty_role_skipped(self):
        """Match with empty semantic_role should not backfill."""
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        m = IconMemoryMatch({
            "asset_id": "abc", "dhash": "x", "path": "x",
            "extra_metadata": json.dumps({
                "semantic_state": "confirmed", "semantic_role": "",
                "confidence": 0.9, "app_process": "chrome",
            }),
        })
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (m, 0)
                _icon_memory_backfill(c, _mk_ss(), "chrome.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_empty_role"] == 1

    def test_oversized_match_skipped(self):
        """Match with large relative_bounds should not backfill."""
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        m = IconMemoryMatch({
            "asset_id": "abc", "dhash": "x", "path": "x",
            "extra_metadata": json.dumps({
                "semantic_state": "confirmed", "semantic_role": "button",
                "confidence": 0.9, "app_process": "chrome",
                "relative_bounds": [0.0, 0.0, 0.5, 0.5],  # 25% area > 10%
            }),
        })
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (m, 0)
                _icon_memory_backfill(c, _mk_ss(), "chrome.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_oversized"] == 1

    def test_low_quality_state_skipped(self):
        """Match with quality_state=rejected should not backfill."""
        from src.integration.api_server import _icon_memory_backfill
        e = _mk_elem(role="button", bounds=(10, 10, 50, 40))
        c = _mk_canvas([e])
        m = IconMemoryMatch({
            "asset_id": "abc", "dhash": "x", "path": "x",
            "extra_metadata": json.dumps({
                "semantic_state": "confirmed", "semantic_role": "button",
                "confidence": 0.9, "app_process": "chrome",
                "quality_state": "rejected",
            }),
        })
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (m, 0)
                _icon_memory_backfill(c, _mk_ss(), "chrome.exe")
        assert c.artifacts["icon_memory_backfills"]["backfilled"] == 0
        assert c.artifacts["icon_memory_backfills"]["skipped_low_quality"] == 1
