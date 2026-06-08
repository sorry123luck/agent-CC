"""Unit tests for _icon_memory_vlm_confirm (Phase 4B)."""

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image


def _mk_canvas(eligible_candidates=None):
    c = MagicMock()
    c.elements = []
    c.artifacts = {
        "icon_memory_roi_candidates": {
            "icon_memory_candidates": eligible_candidates or [],
            "rejected_candidates": [],
        }
    }
    c.window_width = 1920
    c.window_height = 1080
    c.get_region = MagicMock(return_value=None)
    return c


def _mk_eligible(eid="e1", role="button", text="", bounds=(10, 10, 50, 40)):
    return {
        "element_id": eid, "role": role, "text": text, "bounds": list(bounds),
        "area": (bounds[2]-bounds[0]) * (bounds[3]-bounds[1]),
        "region_id": "", "vlm_eligible": True, "priority_score": 80.0,
        "memory_candidate_type": "icon_control",
    }


def _mk_ss():
    return Image.new("RGB", (200, 100), color=(128, 128, 128))


def _mock_provider():
    """Create a mock VLM provider."""
    provider = MagicMock()
    provider.name = "test_provider"
    provider.model_id = "test/model"
    provider.is_available.return_value = True
    return provider


def _run_vlm_confirm(c, ss, process="qq.exe", vlm_return=None):
    """Run _icon_memory_vlm_confirm with all necessary mocks."""
    with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
        vlm.return_value = vlm_return or {"is_ui_control": False, "reject_reason": "test"}
        with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
            cp.return_value = _mock_provider()
            with patch("src.storage.db.Session") as ms:
                sess = MagicMock()
                ms.return_value.__enter__ = MagicMock(return_value=sess)
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock()
                    mc.return_value = si
                    si.find_match.return_value = (None, -1)
                    si.store_crop.return_value = "asset123"
                    sess.query.return_value.filter_by.return_value.first.return_value = None
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, ss, process)


def _run_with_existing(ex, dist=0, vlm_return=None):
    """Run with an existing match in store."""
    c = _mk_canvas([_mk_eligible()])
    with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
        vlm.return_value = vlm_return or {"is_ui_control": False}
        with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
            cp.return_value = _mock_provider()
            with patch("src.storage.db.Session") as ms:
                ms.return_value.__enter__ = MagicMock(return_value=MagicMock())
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock()
                    mc.return_value = si
                    si.find_match.return_value = (ex, dist)
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
    return c


class TestDefaultDisabled:
    def test_no_artifact_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_ICON_MEMORY_CONFIRM", raising=False)
        from src.integration.api_server import _icon_memory_vlm_confirm
        c = _mk_canvas([_mk_eligible()])
        _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        assert "icon_memory_vlm_confirmations" not in c.artifacts

    def test_no_db_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_ICON_MEMORY_CONFIRM", raising=False)
        from src.integration.api_server import _icon_memory_vlm_confirm
        c = _mk_canvas([_mk_eligible()])
        with patch("src.storage.db.Session") as ms:
            _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
            ms.assert_not_called()


class TestEnabled:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_ICON_MEMORY_CONFIRM", "1")

    def test_no_screenshot_skips(self):
        from src.integration.api_server import _icon_memory_vlm_confirm
        c = _mk_canvas([_mk_eligible()])
        _icon_memory_vlm_confirm(c, None, "qq.exe")
        assert "icon_memory_vlm_confirmations" not in c.artifacts

    def test_no_eligible_candidates(self):
        c = _mk_canvas([])
        _run_vlm_confirm(c, _mk_ss())
        diag = c.artifacts.get("icon_memory_vlm_confirmations", {})
        assert diag.get("processed", 0) == 0

    def test_limit_applied(self):
        elems = [_mk_eligible(eid=f"e{i}") for i in range(10)]
        c = _mk_canvas(elems)
        _run_vlm_confirm(c, _mk_ss())
        diag = c.artifacts.get("icon_memory_vlm_confirmations", {})
        assert diag.get("processed") == 3

    def test_is_ui_control_false_rejects(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={"is_ui_control": False, "reject_reason": "avatar"})
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["rejected_by_vlm"] == 1
        assert diag["stored_pending"] == 0
        det = diag["details"][0]
        assert det["status"] == "rejected_by_vlm"
        assert det["reject_reason"] == "avatar"

    def test_valid_control_stored_pending(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={
            "is_ui_control": True, "semantic_role": "button",
            "semantic_text": "OK", "confidence": 0.9,
            "visual_state": "normal", "control_family": "action",
        })
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["stored_pending"] == 1
        assert diag["rejected_by_vlm"] == 0

    def test_low_confidence_rejected(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={
            "is_ui_control": True, "semantic_role": "button", "confidence": 0.3,
        })
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["rejected_low_confidence"] == 1
        det = diag["details"][0]
        assert det["status"] == "rejected_low_confidence"
        assert det["confidence"] == 0.3

    def test_existing_confirmed_skipped(self):
        ex = MagicMock()
        ex.semantic_state = "confirmed"
        ex.asset_id = "ex123"
        ex.semantic_role = "button"
        c = _run_with_existing(ex, dist=0)
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["skipped_existing_confirmed"] == 1
        assert diag["stored_pending"] == 0
        det = diag["details"][0]
        assert det["status"] == "skipped_existing_confirmed"
        assert det["asset_id"] == "ex123"
        assert det["dhash_distance"] == 0
        assert det["existing_role"] == "button"
        assert det["semantic_state"] == "confirmed"

    def test_existing_pending_skipped(self):
        import json as _json
        ex = MagicMock()
        ex.semantic_state = "pending"
        ex.asset_id = "p1"
        ex.semantic_role = "icon_button"
        ex.extra_metadata = _json.dumps({"semantic_state": "pending", "semantic_role": "icon_button", "match_count": 0})
        c = _run_with_existing(ex, dist=2)
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["skipped_existing_pending"] == 1
        assert diag["stored_pending"] == 0
        det = diag["details"][0]
        assert det["status"] == "skipped_existing_pending"
        assert det["asset_id"] == "p1"
        assert det["dhash_distance"] == 2
        assert det["existing_role"] == "icon_button"
        assert det["semantic_state"] == "pending"

    def test_invalid_role_rejected(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={
            "is_ui_control": True, "semantic_role": "invalid_xyz", "confidence": 0.9,
        })
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["rejected_by_vlm"] == 1
        det = diag["details"][0]
        assert det["status"] == "rejected_by_vlm"
        assert det["reject_reason"] == "invalid_role"

    def test_stored_pending_metadata(self):
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as v:
            v.return_value = {"is_ui_control": True, "semantic_role": "button",
                "semantic_text": "OK", "confidence": 0.9,
                "visual_state": "normal", "control_family": "action"}
            with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
                cp.return_value = _mock_provider()
                with patch("src.storage.db.Session") as ms:
                    s = MagicMock()
                    ms.return_value.__enter__ = MagicMock(return_value=s)
                    ms.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                        si = MagicMock(); mc.return_value = si
                        si.find_match.return_value = (None, -1)
                        si.store_crop.return_value = "a1"
                        row = MagicMock(); row.extra_metadata = "{}"
                        s.query.return_value.filter_by.return_value.first.return_value = row
                        from src.integration.api_server import _icon_memory_vlm_confirm
                        _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        d = c.artifacts["icon_memory_vlm_confirmations"]
        assert d["stored_pending"] == 1
        meta = json.loads(row.extra_metadata)
        assert meta["source_element_id"] == "e1"
        assert meta["vlm_prompt_version"] == "roi_v2"
        assert meta["visual_state"] == "normal"
        assert "roi_bounds" in meta and "dhash" in meta

    def test_http_error_counted(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={
            "is_ui_control": False, "reject_reason": "http_error", "error": "429 rate limited",
        })
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["http_error"] == 1
        assert diag["rejected_by_vlm"] == 0
        det = diag["details"][0]
        assert det["status"] == "http_error"
        assert "429" in det["error"]

    def test_invalid_json_counted(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={
            "is_ui_control": False, "reject_reason": "invalid_json", "error": "Expecting value",
        })
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["invalid_json"] == 1
        det = diag["details"][0]
        assert det["status"] == "invalid_json"
        assert "Expecting" in det["error"]

    def test_timeout_counted(self):
        c = _mk_canvas([_mk_eligible()])
        _run_vlm_confirm(c, _mk_ss(), vlm_return={
            "is_ui_control": False, "reject_reason": "timeout", "error": "timed out",
        })
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["timed_out"] == 1
        det = diag["details"][0]
        assert det["status"] == "timed_out"

    def test_global_error_on_exception(self):
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "button", "confidence": 0.9}
            with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
                cp.return_value = _mock_provider()
                with patch("src.storage.db.Session") as ms:
                    ms.side_effect = RuntimeError("db connection lost")
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag.get("global_error_count") == 1
        assert "db connection lost" in diag.get("global_error_message", "")
        assert diag["processed"] == 0

    def test_http_error_in_vlm_call(self):
        """_confirm_icon_with_vlm returns http_error for provider errors."""
        from src.integration.api_server import _confirm_icon_with_vlm
        crop = Image.new("RGB", (50, 50))
        with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
            provider = MagicMock()
            provider.name = "test"
            provider.model_id = "test/model"
            provider.is_available.return_value = True
            provider.analyze_page.side_effect = RuntimeError("429 Too Many Requests")
            cp.return_value = provider
            result = _confirm_icon_with_vlm(crop)
        assert result["reject_reason"] == "http_error"
        assert "429" in result["error"]

    def test_default_privacy_hash_only(self):
        """Default: privacy_level=hash_only, no raw crop saved."""
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as v:
            v.return_value = {"is_ui_control": True, "semantic_role": "button",
                "confidence": 0.9, "visual_state": "normal", "control_family": "action"}
            with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
                cp.return_value = _mock_provider()
                with patch("src.storage.db.Session") as ms:
                    sess = MagicMock()
                    ms.return_value.__enter__ = MagicMock(return_value=sess)
                    ms.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                        si = MagicMock(); mc.return_value = si
                        si.find_match.return_value = (None, -1)
                        si.store_crop.return_value = "a1"
                        sess.query.return_value.filter_by.return_value.first.return_value = None
                        from src.integration.api_server import _icon_memory_vlm_confirm
                        _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        # Verify store_crop called with hash_only
        call_kwargs = si.store_crop.call_args
        assert call_kwargs.kwargs.get("privacy_level") == "hash_only" or call_kwargs[1].get("privacy_level") == "hash_only"
        # Verify detail records privacy_level
        det = c.artifacts["icon_memory_vlm_confirmations"]["details"][0]
        assert det["privacy_level"] == "hash_only"

    def test_raw_crop_with_env_var(self, monkeypatch):
        """With OPENCLAW_ICON_MEMORY_STORE_RAW_CROP=1: privacy_level=safe."""
        monkeypatch.setenv("OPENCLAW_ICON_MEMORY_STORE_RAW_CROP", "1")
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as v:
            v.return_value = {"is_ui_control": True, "semantic_role": "button",
                "confidence": 0.9, "visual_state": "normal", "control_family": "action"}
            with patch("src.vlm.roi_provider_worker.create_configured_roi_provider") as cp:
                cp.return_value = _mock_provider()
                with patch("src.storage.db.Session") as ms:
                    sess = MagicMock()
                    ms.return_value.__enter__ = MagicMock(return_value=sess)
                    ms.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                        si = MagicMock(); mc.return_value = si
                        si.find_match.return_value = (None, -1)
                        si.store_crop.return_value = "a1"
                        sess.query.return_value.filter_by.return_value.first.return_value = None
                        from src.integration.api_server import _icon_memory_vlm_confirm
                        _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        call_kwargs = si.store_crop.call_args
        assert call_kwargs.kwargs.get("privacy_level") == "safe" or call_kwargs[1].get("privacy_level") == "safe"
        det = c.artifacts["icon_memory_vlm_confirmations"]["details"][0]
        assert det["privacy_level"] == "safe"

    def test_auto_promote_success(self):
        """Pending match with strong dHash + VLM confirm → promoted."""
        import json
        ex = MagicMock()
        ex.semantic_state = "pending"
        ex.asset_id = "p1"
        ex.semantic_role = "button"
        ex.extra_metadata = json.dumps({
            "semantic_state": "pending", "semantic_role": "button",
            "match_count": 2, "confidence": 0.9,
        })
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "button", "confidence": 0.9}
            with patch("src.storage.db.Session") as ms:
                sess = MagicMock()
                ms.return_value.__enter__ = MagicMock(return_value=sess)
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock(); mc.return_value = si
                    si.find_match.return_value = (ex, 0)  # exact match
                    sess.query.return_value.filter_by.return_value.first.return_value = ex
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["promoted_to_confirmed"] == 1
        assert diag["skipped_existing_pending"] == 0
        det = diag["details"][0]
        assert det["status"] == "promoted_to_confirmed"

    def test_auto_promote_blocked_high_risk(self):
        """High-risk role never auto-promotes."""
        import json
        ex = MagicMock()
        ex.semantic_state = "pending"
        ex.asset_id = "p1"
        ex.semantic_role = "send_button"
        ex.extra_metadata = json.dumps({
            "semantic_state": "pending", "semantic_role": "send_button",
            "match_count": 2, "confidence": 0.9,
        })
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "send_button", "confidence": 0.9}
            with patch("src.storage.db.Session") as ms:
                sess = MagicMock()
                ms.return_value.__enter__ = MagicMock(return_value=sess)
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock(); mc.return_value = si
                    si.find_match.return_value = (ex, 0)
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["promoted_to_confirmed"] == 0
        assert diag["skipped_existing_pending"] == 1

    def test_auto_promote_blocked_weak_match(self):
        """dHash distance > 2 → no promotion."""
        import json
        ex = MagicMock()
        ex.semantic_state = "pending"
        ex.asset_id = "p1"
        ex.semantic_role = "button"
        ex.extra_metadata = json.dumps({
            "semantic_state": "pending", "semantic_role": "button",
            "match_count": 2, "confidence": 0.9,
        })
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "button", "confidence": 0.9}
            with patch("src.storage.db.Session") as ms:
                sess = MagicMock()
                ms.return_value.__enter__ = MagicMock(return_value=sess)
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock(); mc.return_value = si
                    si.find_match.return_value = (ex, 5)  # weak match
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["promoted_to_confirmed"] == 0
        assert diag["skipped_existing_pending"] == 1

    def test_auto_promote_blocked_low_match_count(self):
        """match_count < 1 → no promotion (first time seeing)."""
        import json
        ex = MagicMock()
        ex.semantic_state = "pending"
        ex.asset_id = "p1"
        ex.semantic_role = "button"
        ex.extra_metadata = json.dumps({
            "semantic_state": "pending", "semantic_role": "button",
            "match_count": 0, "confidence": 0.9,
        })
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "button", "confidence": 0.9}
            with patch("src.storage.db.Session") as ms:
                sess = MagicMock()
                ms.return_value.__enter__ = MagicMock(return_value=sess)
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock(); mc.return_value = si
                    si.find_match.return_value = (ex, 0)
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["promoted_to_confirmed"] == 0
        assert diag["skipped_existing_pending"] == 1

    def test_auto_promote_blocked_quality_flag(self):
        """quality_flag set → no promotion."""
        import json
        ex = MagicMock()
        ex.semantic_state = "pending"
        ex.asset_id = "p1"
        ex.semantic_role = "button"
        ex.extra_metadata = json.dumps({
            "semantic_state": "pending", "semantic_role": "button",
            "match_count": 2, "confidence": 0.9, "quality_flag": "low_quality",
        })
        c = _mk_canvas([_mk_eligible()])
        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "button", "confidence": 0.9}
            with patch("src.storage.db.Session") as ms:
                sess = MagicMock()
                ms.return_value.__enter__ = MagicMock(return_value=sess)
                ms.return_value.__exit__ = MagicMock(return_value=False)
                with patch("src.memory.icon_memory_store.IconMemoryStore") as mc:
                    si = MagicMock(); mc.return_value = si
                    si.find_match.return_value = (ex, 0)
                    from src.integration.api_server import _icon_memory_vlm_confirm
                    _icon_memory_vlm_confirm(c, _mk_ss(), "qq.exe")
        diag = c.artifacts["icon_memory_vlm_confirmations"]
        assert diag["promoted_to_confirmed"] == 0
        assert diag["skipped_existing_pending"] == 1


class TestParseIconConfirmResponse:
    """Tests for _parse_icon_confirm_response parser."""

    def test_valid_json(self):
        from src.integration.api_server import _parse_icon_confirm_response
        result = _parse_icon_confirm_response('{"is_ui_control": true, "semantic_role": "button", "confidence": 0.9}')
        assert result["is_ui_control"] is True
        assert result["semantic_role"] == "button"
        assert result["confidence"] == 0.9

    def test_fenced_json(self):
        from src.integration.api_server import _parse_icon_confirm_response
        fenced = '```json\n{"is_ui_control": true, "semantic_role": "icon_button"}\n```'
        result = _parse_icon_confirm_response(fenced)
        assert result["is_ui_control"] is True
        assert result["semantic_role"] == "icon_button"

    def test_invalid_json_returns_invalid_json(self):
        from src.integration.api_server import _parse_icon_confirm_response
        result = _parse_icon_confirm_response("not json at all")
        assert result["reject_reason"] == "invalid_json"
        assert result["is_ui_control"] is False

    def test_non_dict_returns_invalid_json(self):
        from src.integration.api_server import _parse_icon_confirm_response
        result = _parse_icon_confirm_response('"just a string"')
        assert result["reject_reason"] == "invalid_json"
        assert "response_not_dict" in result.get("error", "")

    def test_defaults_applied(self):
        from src.integration.api_server import _parse_icon_confirm_response
        result = _parse_icon_confirm_response('{"is_ui_control": true}')
        assert result["semantic_role"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["visual_state"] == "unknown"
        assert result["control_family"] == "unknown"


class TestAutoPromoteWithRealMatch:
    """Tests for _try_auto_promote using real IconMemoryMatch."""

    def test_promote_with_real_match(self):
        from src.integration.api_server import _try_auto_promote
        from src.memory.icon_memory_store import IconMemoryMatch

        # Create real IconMemoryMatch
        match = IconMemoryMatch({
            "asset_id": "test123",
            "extra_metadata": json.dumps({
                "semantic_state": "pending",
                "semantic_role": "button",
                "match_count": 2,
                "confidence": 0.9,
            }),
        })

        store = MagicMock()
        session = MagicMock()
        provider = _mock_provider()

        with patch("src.integration.api_server._confirm_icon_with_vlm") as vlm:
            vlm.return_value = {"is_ui_control": True, "semantic_role": "button", "confidence": 0.9}
            result = _try_auto_promote(match, 0, Image.new("RGB", (50, 50)), "test", store, session, provider=provider)
        assert result is True

    def test_no_promote_with_real_match_high_risk(self):
        from src.integration.api_server import _try_auto_promote
        from src.memory.icon_memory_store import IconMemoryMatch

        match = IconMemoryMatch({
            "asset_id": "test123",
            "extra_metadata": json.dumps({
                "semantic_state": "pending",
                "semantic_role": "send_button",
                "match_count": 2,
                "confidence": 0.9,
            }),
        })

        store = MagicMock()
        session = MagicMock()
        provider = _mock_provider()

        result = _try_auto_promote(match, 0, Image.new("RGB", (50, 50)), "test", store, session, provider=provider)
        assert result is False

    def test_no_promote_with_real_match_low_count(self):
        from src.integration.api_server import _try_auto_promote
        from src.memory.icon_memory_store import IconMemoryMatch

        match = IconMemoryMatch({
            "asset_id": "test123",
            "extra_metadata": json.dumps({
                "semantic_state": "pending",
                "semantic_role": "button",
                "match_count": 0,
                "confidence": 0.9,
            }),
        })

        store = MagicMock()
        session = MagicMock()
        provider = _mock_provider()

        result = _try_auto_promote(match, 0, Image.new("RGB", (50, 50)), "test", store, session, provider=provider)
        assert result is False
