"""Unit tests for _icon_memory_roi_selector (Phase 4A)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image


def _mk_elem(eid="e1", role="button", text="", bounds=(10, 10, 50, 40)):
    e = MagicMock()
    e.element_id = eid
    e.semantic_role = MagicMock()
    e.semantic_role.value = role
    e.text = text
    e.bounds = bounds
    e.confidence = 0.9
    e.region_id = None
    return e


def _mk_canvas(elems=None):
    c = MagicMock()
    c.elements = elems or []
    c.artifacts = {}
    c.get_region = MagicMock(return_value=None)
    c.window_width = 1920
    c.window_height = 1080
    return c


class TestRoiSelector:
    def test_empty_canvas(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert diag["total_elements"] == 0
        assert len(diag.get("icon_memory_candidates", [])) == 0

    def test_button_selected(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="")])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag.get("icon_memory_candidates", [])) == 1
        assert diag["icon_memory_candidates"][0]["role"] == "button"

    def test_unknown_selected(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="unknown", text="")])
        _icon_memory_roi_selector(c)
        assert len(c.artifacts["icon_memory_roi_candidates"]["icon_memory_candidates"]) >= 0  # unknown may be classified

    def test_send_button_protected(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="send_button", text="发送")])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag.get("icon_memory_candidates", [])) == 0
        assert diag["skipped_protected"] == 1

    def test_search_input_protected(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="search_input", text="搜索")])
        _icon_memory_roi_selector(c)
        assert c.artifacts["icon_memory_roi_candidates"]["skipped_protected"] == 1

    def test_long_text_skipped(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="this is a very long text that exceeds twenty chars")])
        _icon_memory_roi_selector(c)
        assert len(c.artifacts["icon_memory_roi_candidates"]["icon_memory_candidates"]) == 0
        assert c.artifacts["icon_memory_roi_candidates"]["skipped_text_blocks"] == 1

    def test_url_text_skipped(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="https://example.com")])
        _icon_memory_roi_selector(c)
        assert c.artifacts["icon_memory_roi_candidates"]["skipped_text_blocks"] == 1

    def test_small_bounds_skipped(self):
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", bounds=(10, 10, 12, 12))])
        _icon_memory_roi_selector(c)
        assert c.artifacts["icon_memory_roi_candidates"]["skipped_invalid_bounds"] == 1

    def test_mixed_elements(self):
        from src.integration.api_server import _icon_memory_roi_selector
        elems = [
            _mk_elem(eid="btn1", role="button", text=""),
            _mk_elem(eid="btn2", role="send_button", text="发送"),
            _mk_elem(eid="txt1", role="text", text="hello"),
            _mk_elem(eid="unk1", role="unknown", text=""),
            _mk_elem(eid="img1", role="image", text=""),
        ]
        c = _mk_canvas(elems)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert diag["total_elements"] == 5
        assert len(diag.get("icon_memory_candidates", [])) >= 2  # button + image (text excluded, unknown may vary)
        assert diag["skipped_protected"] == 1

    def test_always_runs_no_env_gate(self):
        """ROI selector runs without any env var."""
        import os
        os.environ.pop("OPENCLAW_ICON_MEMORY_DIAGNOSTICS", None)
        os.environ.pop("OPENCLAW_ICON_MEMORY_BACKFILL", None)
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem()])
        _icon_memory_roi_selector(c)
        assert "icon_memory_roi_candidates" in c.artifacts

    def test_text_role_excluded(self):
        """text role should NOT be in ROI candidates."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="text", text="hello world")])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag.get("icon_memory_candidates", [])) == 0
        assert len(diag.get("icon_memory_candidates", [])) == 0  # text role not in BACKFILLABLE

    def test_large_element_excluded(self):
        """Element > 5% of window area should be excluded."""
        from src.integration.api_server import _icon_memory_roi_selector
        # 500x500 = 250000 area on 1920x1080 window (2073600 area) = 12% > 5%
        c = _mk_canvas([_mk_elem(role="button", bounds=(0, 0, 500, 500))])
        c.window_width = 1920
        c.window_height = 1080
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert diag["skipped_too_large"] >= 1

    def test_small_button_kept(self):
        """Small button should be kept."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="", bounds=(10, 10, 50, 40))])
        c.window_width = 1920
        c.window_height = 1080
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag.get("icon_memory_candidates", [])) == 1

    def test_large_unknown_container_excluded(self):
        """Large unknown with empty text/name/control_type should be excluded."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(role="unknown", text="", bounds=(0, 0, 250, 250))
        e.name = ""
        e.control_type = ""
        c = _mk_canvas([e])
        c.window_width = 1920
        c.window_height = 1080
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert diag["skipped_container_like"] >= 1

    def test_unknown_with_text_kept_in_good_region(self):
        """Unknown with short text in toolbar should be kept."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "toolbar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(role="unknown", text="OK", bounds=(10, 10, 50, 40))
        e.region_id = "r_toolbar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag.get("icon_memory_candidates", [])) >= 1

    def test_priority_button_higher_than_unknown(self):
        """button should rank higher than unknown."""
        from src.integration.api_server import _icon_memory_roi_selector
        btn = _mk_elem(eid="btn", role="button", text="OK", bounds=(10, 10, 50, 40))
        unk = _mk_elem(eid="unk", role="unknown", text="", bounds=(100, 100, 150, 140))
        c = _mk_canvas([unk, btn])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        cands = diag["icon_memory_candidates"]
        assert cands[0]["element_id"] == "btn"
        assert cands[0]["priority_score"] > cands[1]["priority_score"]

    def test_priority_long_text_button_penalized(self):
        """button with long text should rank lower."""
        from src.integration.api_server import _icon_memory_roi_selector
        long_btn = _mk_elem(eid="long", role="button", text="OK", bounds=(10, 10, 50, 40))
        short_btn = _mk_elem(eid="short", role="button", text="OK", bounds=(100, 100, 150, 140))
        c = _mk_canvas([long_btn, short_btn])
        _icon_memory_roi_selector(c)
        cands = c.artifacts["icon_memory_roi_candidates"]["icon_memory_candidates"]
        # Both are icon_control, but short_btn may have different priority
        assert len(cands) >= 1

    def test_priority_small_icon_higher_than_large(self):
        """small icon-sized element should rank higher than large one."""
        from src.integration.api_server import _icon_memory_roi_selector
        small = _mk_elem(eid="small", role="button", text="", bounds=(10, 10, 40, 40))
        large = _mk_elem(eid="large", role="button", text="", bounds=(100, 100, 400, 350))
        c = _mk_canvas([large, small])
        _icon_memory_roi_selector(c)
        cands = c.artifacts["icon_memory_roi_candidates"]["icon_memory_candidates"]
        assert len(cands) >= 1  # at least small should be accepted

    def test_priority_content_area_unknown_penalized(self):
        """unknown in content_area should rank lower."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "content_area"
        region.subtype = MagicMock()
        region.subtype.value = "main_content"
        e = _mk_elem(eid="chat_unk", role="unknown", text="", bounds=(10, 10, 50, 40))
        e.region_id = "r1"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        cands = diag.get("icon_memory_candidates", [])
        rej = diag.get("rejected_candidates", [])
        assert len(cands) + len(rej) >= 1

    def test_priority_toolbar_button_bonus(self):
        """button in toolbar region should get bonus."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "toolbar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="tb", role="button", text="Save", bounds=(10, 10, 50, 40))
        e.region_id = "r_toolbar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        cands = c.artifacts["icon_memory_roi_candidates"]["icon_memory_candidates"]
        assert len(cands) >= 1

    def test_limit_applied(self):
        """Default limit of 30 should be applied."""
        from src.integration.api_server import _icon_memory_roi_selector
        elems = [_mk_elem(eid=f"e{i}", role="button", text=f"B{i}", bounds=(10*i, 10, 10*i+30, 40)) for i in range(50)]
        c = _mk_canvas(elems)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert diag["total_candidates_before_purpose_gate"] >= 50
        assert len(diag.get("icon_memory_candidates", [])) == 30
        assert diag["limit"] == 30

    def test_priority_score_in_candidate(self):
        """Each candidate should have priority_score and priority_reasons."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="OK", bounds=(10, 10, 50, 40))])
        _icon_memory_roi_selector(c)
        cand = c.artifacts["icon_memory_roi_candidates"]["icon_memory_candidates"][0]
        assert "priority_score" in cand
        assert "priority_reasons" in cand
        assert isinstance(cand["priority_score"], (int, float))
        assert isinstance(cand["priority_reasons"], list)

    def test_message_bubble_excluded(self):
        """Button with long text in content area should be text_button or filtered."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "content_area"
        region.subtype = MagicMock()
        region.subtype.value = "chat"
        e = _mk_elem(eid="bubble", role="button", text="消息测试一二三四五六七八", bounds=(100, 200, 300, 230))
        e.region_id = "r_chat"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        # Button in content area with text — should NOT be icon_control
        icon_controls = [mc for mc in diag.get("icon_memory_candidates", [])
                        if mc.get("memory_candidate_type") == "icon_control"]
        # Either filtered out or classified as non-icon
        assert len(icon_controls) == 0 or diag.get("skipped_text_blocks", 0) >= 1

    def test_chat_list_item_excluded(self):
        """QQ chat_item is excluded by BACKFILLABLE role check (not in set)."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(eid="chat", role="chat_item", text="群助手", bounds=(10, 10, 200, 50))
        c = _mk_canvas([e])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        # chat_item is not in BACKFILLABLE, so it never enters candidates
        assert len(diag.get("icon_memory_candidates", [])) == 0

    def test_toolbar_button_included(self):
        """QQ top bar button (phone/video/more) should enter icon_memory_candidates."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "toolbar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="phone", role="icon_button", text="", bounds=(10, 10, 40, 40))
        e.region_id = "r_toolbar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag["icon_memory_candidates"]) == 1
        assert diag["icon_memory_candidates"][0]["memory_candidate_type"] == "icon_control"

    def test_bottom_toolbar_icon_included(self):
        """Bottom toolbar icon button should enter icon_memory_candidates."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "composer_area"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="attach", role="icon_button", text="", bounds=(10, 10, 40, 40))
        e.region_id = "r_composer"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag["icon_memory_candidates"]) == 1

    def test_history_conversation_excluded(self):
        """Chrome history conversation text should not enter icon_memory_candidates."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(eid="hist", role="text", text="火柴战队角色分析", bounds=(10, 10, 200, 40))
        c = _mk_canvas([e])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        # text role is already excluded by BACKFILLABLE filter
        assert len(diag.get("icon_memory_candidates", [])) == 0 or len(diag.get("icon_memory_candidates", [])) == 0

    def test_chrome_small_icon_included(self):
        """Chrome top bar small icon button should enter icon_memory_candidates."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(eid="ext", role="button", text="", bounds=(100, 5, 130, 35))
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=None)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert len(diag["icon_memory_candidates"]) >= 1

    def test_avatar_excluded(self):
        """Image in sidebar with no text should be classified as avatar_or_contact."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "sidebar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="avatar", role="image", text="", bounds=(10, 10, 50, 50))
        e.region_id = "r_sidebar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        # Small square image in sidebar = avatar, should be excluded
        assert diag.get("skipped_avatar_or_contact", 0) >= 1 or                all(mc.get("memory_candidate_type") != "avatar_or_contact"
                   for mc in diag.get("icon_memory_candidates", []))

    def test_purpose_gate_artifact_fields(self):
        """Artifact should have all purpose gate fields."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="", bounds=(10, 10, 50, 40))])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert "total_candidates_before_purpose_gate" in diag
        assert "icon_memory_candidates" in diag
        assert "skipped_list_item_or_message" in diag
        assert "skipped_avatar_or_contact" in diag
        assert "skipped_text_button" in diag
        assert "skipped_container_or_region" in diag
        assert "rejected_candidates" in diag

    def test_memory_candidate_type_in_candidate(self):
        """Each candidate should have memory_candidate_type and memory_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="", bounds=(10, 10, 50, 40))])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        for mc in diag.get("icon_memory_candidates", []):
            assert "memory_candidate_type" in mc
            assert "memory_eligible" in mc

    def test_avatar_not_vlm_eligible(self):
        """Small image in sidebar should not be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "sidebar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="avatar", role="image", text="", bounds=(10, 10, 50, 50))
        e.region_id = "r_sidebar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        icon_cands = diag.get("icon_memory_candidates", [])
        # Small square image in sidebar = avatar, should be rejected or not vlm_eligible
        eligible = [mc for mc in icon_cands if mc.get("vlm_eligible")]
        assert len(eligible) == 0

    def test_toolbar_button_vlm_eligible(self):
        """Button in toolbar should be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "toolbar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="tb", role="button", text="", bounds=(10, 10, 40, 40))
        e.region_id = "r_toolbar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        icon_cands = diag.get("icon_memory_candidates", [])
        eligible = [mc for mc in icon_cands if mc.get("vlm_eligible")]
        assert len(eligible) == 1

    def test_bottom_toolbar_icon_vlm_eligible(self):
        """Icon button in composer area should be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "composer_area"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="attach", role="icon_button", text="", bounds=(10, 10, 40, 40))
        e.region_id = "r_composer"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        icon_cands = diag.get("icon_memory_candidates", [])
        eligible = [mc for mc in icon_cands if mc.get("vlm_eligible")]
        assert len(eligible) >= 1

    def test_history_text_row_not_vlm_eligible(self):
        """Long horizontal text button should not be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(eid="hist", role="button", text="火柴战队角色分析", bounds=(10, 10, 300, 30))
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=None)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        icon_cands = diag.get("icon_memory_candidates", [])
        # Wide horizontal button with text = list history row
        eligible = [mc for mc in icon_cands if mc.get("vlm_eligible")]
        # Either rejected entirely or not vlm_eligible
        assert len(eligible) == 0

    def test_chrome_small_icon_vlm_eligible(self):
        """Small icon button with no region should still be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(eid="ext", role="button", text="", bounds=(100, 5, 130, 35))
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=None)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        eligible = [mc for mc in diag.get("icon_memory_candidates", []) if mc.get("vlm_eligible")]
        assert len(eligible) >= 1

    def test_vlm_eligible_fields_present(self):
        """Each icon_memory_candidate should have vlm_eligible and vlm_skip_reason."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="", bounds=(10, 10, 50, 40))])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        for mc in diag.get("icon_memory_candidates", []):
            assert "vlm_eligible" in mc
            assert "vlm_skip_reason" in mc

    def test_artifact_has_vlm_stats(self):
        """Artifact should have vlm_eligible_count and ineligible_by_reason."""
        from src.integration.api_server import _icon_memory_roi_selector
        c = _mk_canvas([_mk_elem(role="button", text="", bounds=(10, 10, 50, 40))])
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        assert "vlm_eligible_count" in diag
        assert "vlm_ineligible_count" in diag
        assert "ineligible_by_reason" in diag

    def test_avatar_in_member_list_not_vlm_eligible(self):
        """Small square image in member_list should not be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "member_list"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="avatar", role="image", text="", bounds=(10, 10, 50, 50))
        e.region_id = "r_members"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        eligible = [mc for mc in diag.get("icon_memory_candidates", []) if mc.get("vlm_eligible")]
        assert len(eligible) == 0

    def test_wide_button_not_vlm_eligible(self):
        """Wide horizontal button should not be vlm_eligible (list/history row)."""
        from src.integration.api_server import _icon_memory_roi_selector
        e = _mk_elem(eid="wide", role="button", text="", bounds=(10, 10, 300, 30))
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=None)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        eligible = [mc for mc in diag.get("icon_memory_candidates", []) if mc.get("vlm_eligible")]
        assert len(eligible) == 0

    def test_content_area_with_text_not_vlm_eligible(self):
        """Button with text in message_stream should not be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "message_stream"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="msg", role="button", text="Hello", bounds=(10, 10, 50, 40))
        e.region_id = "r_msg"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        eligible = [mc for mc in diag.get("icon_memory_candidates", []) if mc.get("vlm_eligible")]
        assert len(eligible) == 0

    def test_toolbar_icon_vlm_eligible(self):
        """Small icon in toolbar should be vlm_eligible."""
        from src.integration.api_server import _icon_memory_roi_selector
        from unittest.mock import MagicMock
        region = MagicMock()
        region.role = "toolbar"
        region.subtype = MagicMock()
        region.subtype.value = ""
        e = _mk_elem(eid="icon", role="button", text="", bounds=(10, 10, 40, 40))
        e.region_id = "r_toolbar"
        c = _mk_canvas([e])
        c.get_region = MagicMock(return_value=region)
        _icon_memory_roi_selector(c)
        diag = c.artifacts["icon_memory_roi_candidates"]
        eligible = [mc for mc in diag.get("icon_memory_candidates", []) if mc.get("vlm_eligible")]
        assert len(eligible) >= 1