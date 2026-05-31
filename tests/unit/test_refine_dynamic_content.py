"""Unit tests for RefineEngine dynamic content zone guard.

Verifies that keyword-based role inference does NOT misclassify text
in dynamic content areas (chat messages, document body, etc.) as
functional UI controls.
"""

import pytest

from src.canvas.refine_engine import RefineEngine
from src.perception.page_compiler_models import Candidate


@pytest.fixture
def engine():
    return RefineEngine()


def _make_candidate(
    element_id: str,
    text: str = "",
    name: str | None = None,
    control_type: str = "",
    provider_sources: list[str] | None = None,
    bounds: tuple[int, int, int, int] | None = None,
    region_id: str | None = None,
) -> Candidate:
    return Candidate(
        element_id=element_id,
        text=text,
        name=name,
        control_type=control_type,
        provider_sources=provider_sources or [],
        bounds=bounds,
        region_id=region_id,
    )


# ---------------------------------------------------------------------------
# Dynamic content zone: keyword must NOT trigger functional role inference
# ---------------------------------------------------------------------------


class TestDynamicContentZoneGuard:
    """Candidates in dynamic content zones must NOT get functional role labels."""

    def test_wechat_chat_body_text_file_not_labeled_as_file_button(
        self, engine: RefineEngine,
    ):
        """Chat body text '文件' must NOT be labeled as 文件按钮."""
        c = _make_candidate(
            "el_chat_1",
            text="文件",
            region_id="message_area",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "文件", (
            f"Dynamic content '文件' in message_area should not get role_label='文件', "
            f"got '{r.role_label}'"
        )
        assert "content.file" not in r.semantic_tags

    def test_wechat_chat_body_text_videonot_labeled_as_video_button(
        self, engine: RefineEngine,
    ):
        """Chat body text 'videoUrl' must NOT be labeled as 视频按钮."""
        c = _make_candidate(
            "el_chat_2",
            text="videoUrl",
            region_id="message_area",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "视频", (
            f"Dynamic content 'videoUrl' in message_area should not get role_label='视频', "
            f"got '{r.role_label}'"
        )
        assert "media.video" not in r.semantic_tags

    def test_chat_body_text_send_not_labeled_as_send_button(
        self, engine: RefineEngine,
    ):
        """Chat body text '发送' must NOT be labeled as 发送按钮."""
        c = _make_candidate(
            "el_chat_3",
            text="发送",
            region_id="chat_message_area",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "发送", (
            f"Dynamic content '发送' in chat_message_area should not get role_label='发送', "
            f"got '{r.role_label}'"
        )
        assert "action.send" not in r.semantic_tags

    def test_chat_body_text_delete_not_labeled_as_delete_button(
        self, engine: RefineEngine,
    ):
        """Chat body text '删除' in content_stream must NOT be labeled as 删除."""
        c = _make_candidate(
            "el_chat_4",
            text="删除这条消息",
            region_id="content_stream",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "删除", (
            f"Dynamic content in content_stream should not get role_label='删除', "
            f"got '{r.role_label}'"
        )
        assert "action.delete" not in r.semantic_tags

    def test_chat_body_text_confirm_not_labeled_as_confirm_button(
        self, engine: RefineEngine,
    ):
        """Chat body text '确认' in document_body must NOT be labeled as 确认."""
        c = _make_candidate(
            "el_chat_5",
            text="确认",
            region_id="document_body",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "确认", (
            f"Dynamic content in document_body should not get role_label='确认', "
            f"got '{r.role_label}'"
        )
        assert "action.confirm" not in r.semantic_tags

    def test_article_body_text_settings_not_labeled_as_settings_button(
        self, engine: RefineEngine,
    ):
        """Article body text '设置' must NOT be labeled as 设置."""
        c = _make_candidate(
            "el_article_1",
            text="设置",
            region_id="article_body",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "设置", (
            f"Dynamic content in article_body should not get role_label='设置', "
            f"got '{r.role_label}'"
        )
        assert "action.settings" not in r.semantic_tags

    # --- Dynamic content refine_status must be 'uncertain' ---

    def test_dynamic_content_refine_status_is_uncertain(
        self, engine: RefineEngine,
    ):
        """Text in dynamic content zones must have refine_status='uncertain'."""
        zones = [
            "message_area",
            "content_stream",
            "document_body",
            "chat_message_area",
            "right_panel_middle",
            "article_body",
            "code_block",
            "list_dynamic_content",
        ]
        for zone in zones:
            c = _make_candidate(
                f"el_dyn_{zone}",
                text="发送",
                region_id=zone,
                bounds=(100, 300, 400, 330),
            )
            output = engine.refine("canvas_1", [c])
            r = output.results[0]
            assert r.refine_status == "uncertain", (
                f"Text in zone '{zone}' should have refine_status='uncertain', "
                f"got '{r.refine_status}'"
            )

    # --- Dynamic content gets dynamic_text-like labels ---

    def test_dynamic_content_gets_dynamic_text_label(
        self, engine: RefineEngine,
    ):
        """Text in dynamic content zones must get a dynamic/message_content label."""
        c = _make_candidate(
            "el_dyn_label",
            text="你好世界",
            region_id="message_area",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label in (
            "dynamic_text",
            "message_content",
            "document_text",
            "code_text",
        ), (
            f"Dynamic content should get a dynamic label, got '{r.role_label}'"
        )


# ---------------------------------------------------------------------------
# UI control zones: keyword rules MUST still apply
# ---------------------------------------------------------------------------


class TestUIControlZoneKeywordRules:
    """Candidates in UI control zones must still get keyword-based role labels."""

    def test_actual_send_button_in_bottom_input_zone(
        self, engine: RefineEngine,
    ):
        """Actual send button in bottom_input zone is still correctly identified."""
        c = _make_candidate(
            "el_send_btn",
            text="发送",
            control_type="Button",
            region_id="bottom_input",
            bounds=(500, 700, 560, 730),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "发送", (
            f"Send button in bottom_input zone should be labeled '发送', "
            f"got '{r.role_label}'"
        )
        assert "action.send" in r.semantic_tags
        assert r.refine_status == "refined"

    def test_toolbar_file_button_still_works(self, engine: RefineEngine):
        """File button in toolbar zone is still correctly identified."""
        c = _make_candidate(
            "el_file_btn",
            text="文件",
            control_type="Button",
            region_id="toolbar",
            bounds=(10, 50, 70, 80),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "文件", (
            f"File button in toolbar should be labeled '文件', got '{r.role_label}'"
        )
        assert "content.file" in r.semantic_tags

    def test_sidebar_settings_still_works(self, engine: RefineEngine):
        """Settings in sidebar zone is still correctly identified."""
        c = _make_candidate(
            "el_settings",
            text="设置",
            region_id="sidebar",
            bounds=(10, 200, 80, 230),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "设置", (
            f"Settings in sidebar should be labeled '设置', got '{r.role_label}'"
        )
        assert "action.settings" in r.semantic_tags

    def test_title_bar_close_button_still_works(self, engine: RefineEngine):
        """Close button in title_bar zone is still correctly identified."""
        c = _make_candidate(
            "el_close_btn",
            text="关闭",
            control_type="Button",
            region_id="title_bar",
            bounds=(900, 0, 940, 30),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "关闭", (
            f"Close button in title_bar should be labeled '关闭', got '{r.role_label}'"
        )
        assert "window.close" in r.semantic_tags

    def test_menu_search_still_works(self, engine: RefineEngine):
        """Search in menu zone is still correctly identified."""
        c = _make_candidate(
            "el_search",
            text="搜索",
            region_id="menu",
            bounds=(10, 50, 200, 80),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "搜索", (
            f"Search in menu should be labeled '搜索', got '{r.role_label}'"
        )
        assert "input.search" in r.semantic_tags

    def test_button_area_video_still_works(self, engine: RefineEngine):
        """Video button in button_area zone is still correctly identified."""
        c = _make_candidate(
            "el_video_btn",
            text="视频",
            control_type="Button",
            region_id="button_area",
            bounds=(100, 600, 160, 630),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "视频", (
            f"Video button in button_area should be labeled '视频', got '{r.role_label}'"
        )
        assert "media.video" in r.semantic_tags


# ---------------------------------------------------------------------------
# Candidates with no region_id: keyword rules still apply (backward compat)
# ---------------------------------------------------------------------------


class TestNoRegionIdBackwardCompat:
    """Candidates without region_id should still get keyword-based inference."""

    def test_no_region_id_send_button_still_works(self, engine: RefineEngine):
        """Send button with no region_id still gets keyword match."""
        c = _make_candidate(
            "el_send_noregion",
            text="发送",
            control_type="Button",
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label == "发送"
        assert "action.send" in r.semantic_tags


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDynamicContentEdgeCases:
    """Edge cases for dynamic content zone guard."""

    def test_dynamic_content_with_uia_button_control_type_still_guarded(
        self, engine: RefineEngine,
    ):
        """Even if control_type is Button, dynamic zone text is guarded."""
        c = _make_candidate(
            "el_dyn_uia_btn",
            text="文件",
            control_type="Button",
            region_id="message_area",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        # In dynamic content zone, even Button control_type should not
        # get functional keyword labels
        assert r.role_label != "文件"
        assert "content.file" not in r.semantic_tags

    def test_dynamic_content_no_stable_key_id_assignment(
        self, engine: RefineEngine,
    ):
        """Dynamic content with refine_status=uncertain should not get stable_key_id.

        stable_key_id is only assigned to candidates with refine_status='refined'.
        This test verifies the refine engine marks dynamic content as 'uncertain',
        which prevents downstream stable_key_id assignment.
        """
        c = _make_candidate(
            "el_dyn_no_key",
            text="确认操作",
            region_id="message_area",
            bounds=(100, 300, 400, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        # Must be uncertain, not refined — downstream stable_key_id assignment
        # is gated on refine_status == 'refined'
        assert r.refine_status == "uncertain"
        assert r.role_confidence < 0.7

    def test_right_panel_middle_zone_guarded(self, engine: RefineEngine):
        """right_panel_middle is a dynamic content zone."""
        c = _make_candidate(
            "el_rpm",
            text="删除",
            region_id="right_panel_middle",
            bounds=(400, 200, 700, 230),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "删除"
        assert "action.delete" not in r.semantic_tags
        assert r.refine_status == "uncertain"

    def test_code_block_zone_guarded(self, engine: RefineEngine):
        """code_block is a dynamic content zone."""
        c = _make_candidate(
            "el_code",
            text="发送请求",
            region_id="code_block",
            bounds=(100, 300, 500, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.role_label != "发送"
        assert "action.send" not in r.semantic_tags

    def test_dynamic_zone_empty_text_not_guarded(self, engine: RefineEngine):
        """Candidates in dynamic zones with NO text should NOT trigger the zone guard.

        The guard is `if c.region_id in DYNAMIC_CONTENT_ZONES and text:`.
        Non-text elements (icons, images) in dynamic zones should fall through
        to normal heuristic inference.
        """
        c = _make_candidate(
            "el_dyn_icon",
            text="",  # empty text
            control_type="Image",
            region_id="message_area",
            bounds=(100, 300, 130, 330),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        # With empty text, the zone guard should NOT activate,
        # so refine_status should NOT be forced to "uncertain"
        # (it may be "uncertain" for other reasons, but not because of zone guard)
        assert r.refine_status != "uncertain" or r.role_label != "message_content"
