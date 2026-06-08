"""Unit tests for _icon_memory_diagnostics (Phase 2 read-only pre-VLM diagnostics)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.memory.icon_memory_store import IconMemoryMatch


def _mk_elem(eid="e1", role="button", bounds=(10, 10, 50, 40)):
    e = MagicMock()
    e.element_id = eid
    e.semantic_role = MagicMock()
    e.semantic_role.value = role
    e.text = "OK"
    e.bounds = bounds
    return e


def _mk_canvas(elems=None):
    c = MagicMock()
    c.elements = elems or []
    c.artifacts = {}
    return c


def _mk_ss():
    return Image.new("RGB", (200, 100), color=(128, 128, 128))


def _mk_match(state="confirmed", role="button", conf=0.9, dist=0):
    m = IconMemoryMatch({
        "asset_id": "abc", "dhash": "x", "path": "x",
        "extra_metadata": json.dumps({"semantic_state": state, "semantic_role": role, "confidence": conf}),
    })
    return m, dist


class TestDefaultDisabled:
    def test_no_artifact(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_ICON_MEMORY_DIAGNOSTICS", raising=False)
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem()])
        _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert "icon_memory_pre_vlm_matches" not in c.artifacts

    def test_no_db_opened(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_ICON_MEMORY_DIAGNOSTICS", raising=False)
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem()])
        with patch("src.storage.db.Session") as ms:
            _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
            ms.assert_not_called()


class TestEnabled:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_ICON_MEMORY_DIAGNOSTICS", "1")

    def test_artifact_name(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([])
        _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert "icon_memory_pre_vlm_matches" in c.artifacts

    def test_no_screenshot(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem()])
        _icon_memory_diagnostics(c, None, "qq.exe")
        assert c.artifacts["icon_memory_pre_vlm_matches"]["total_checked"] == 0

    def test_no_elements(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([])
        _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert c.artifacts["icon_memory_pre_vlm_matches"]["total_checked"] == 0

    def test_non_fixed_skipped(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem(role="text")])
        _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert c.artifacts["icon_memory_pre_vlm_matches"]["total_checked"] == 0

    def test_match_level_exact(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem(role="button")])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = _mk_match(dist=0)
                _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert c.artifacts["icon_memory_pre_vlm_matches"]["details"][0]["match_level"] == "exact"

    def test_match_level_strong(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem(role="button")])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = _mk_match(dist=2)
                _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert c.artifacts["icon_memory_pre_vlm_matches"]["details"][0]["match_level"] == "strong"

    def test_match_level_weak(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([_mk_elem(role="button")])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = _mk_match(dist=4)
                _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert c.artifacts["icon_memory_pre_vlm_matches"]["details"][0]["match_level"] == "weak"

    def test_no_conflict_field(self):
        from src.integration.api_server import _icon_memory_diagnostics
        c = _mk_canvas([])
        _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert "conflict" not in c.artifacts["icon_memory_pre_vlm_matches"]

    def test_no_element_modification(self):
        from src.integration.api_server import _icon_memory_diagnostics
        e = _mk_elem(role="button")
        orig = e.semantic_role.value
        c = _mk_canvas([e])
        with patch("src.storage.db.Session") as ms:
            ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ms.return_value.__exit__ = MagicMock(return_value=False)
            with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                s = MagicMock(); mc.return_value = s
                s.find_match.return_value = (None, -1)
                _icon_memory_diagnostics(c, _mk_ss(), "qq.exe")
        assert e.semantic_role.value == orig
