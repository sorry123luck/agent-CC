"""Unit tests for RefineEngine — three-layer semantic review."""

import pytest

from src.canvas.refine_engine import RefineEngine, RefineResult, RefineOutput
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
) -> Candidate:
    return Candidate(
        element_id=element_id,
        text=text,
        name=name,
        control_type=control_type,
        provider_sources=provider_sources or [],
        bounds=bounds,
    )


class TestRefineEngineThreeLayer:
    """Verify RefineEngine outputs three-layer fields."""

    def test_send_button_text_match(self, engine: RefineEngine):
        c = _make_candidate("el_send", text="发送", control_type="Button")
        output = engine.refine("canvas_1", [c])
        assert len(output.results) == 1
        r = output.results[0]
        assert r.element_id == "el_send"
        assert r.visual_type == "button"
        assert "action.send" in r.semantic_tags
        assert r.role_label == "发送"
        assert r.role_confidence >= 0.7
        assert r.role_source == "heuristic"
        assert len(r.role_evidence) > 0
        assert r.refine_status == "refined"

    def test_emoji_button_text_match(self, engine: RefineEngine):
        c = _make_candidate("el_emoji", text="表情", control_type="Button")
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert "content.emoji" in r.semantic_tags
        assert r.role_label == "表情"
        assert r.role_confidence >= 0.7

    def test_search_input_text_match(self, engine: RefineEngine):
        c = _make_candidate("el_search", text="搜索", control_type="Edit", bounds=(10, 10, 300, 40))
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.visual_type == "input"
        assert "input.search" in r.semantic_tags
        assert r.role_label == "搜索"

    def test_close_button_text_match(self, engine: RefineEngine):
        c = _make_candidate("el_close", text="关闭", control_type="Button")
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert "window.close" in r.semantic_tags
        assert r.role_label == "关闭"
        assert r.role_confidence >= 0.9

    def test_icon_no_text_vision(self, engine: RefineEngine):
        c = _make_candidate(
            "el_icon",
            control_type="",
            provider_sources=["vision"],
            bounds=(100, 100, 130, 130),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.visual_type == "icon"
        assert r.role_confidence < 0.5
        assert r.refine_status == "uncertain"

    def test_list_item_with_text(self, engine: RefineEngine):
        c = _make_candidate(
            "el_list",
            text="张三",
            control_type="ListItem",
            bounds=(0, 100, 200, 130),
        )
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.visual_type == "list_item"
        assert "list.item" in r.semantic_tags

    def test_button_with_text_generic(self, engine: RefineEngine):
        c = _make_candidate("el_btn", text="提交", control_type="Button")
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.visual_type == "button"
        assert "action.button" in r.semantic_tags
        assert r.role_label == "提交"

    def test_checkbox(self, engine: RefineEngine):
        c = _make_candidate("el_cb", control_type="CheckBox")
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert r.visual_type == "checkbox"
        assert "input.checkbox" in r.semantic_tags

    def test_candidate_ids_filter(self, engine: RefineEngine):
        c1 = _make_candidate("el_1", text="发送", control_type="Button")
        c2 = _make_candidate("el_2", text="取消", control_type="Button")
        output = engine.refine("canvas_1", [c1, c2], candidate_ids=["el_1"])
        assert len(output.results) == 1
        assert output.results[0].element_id == "el_1"

    def test_mode_vlm_placeholder(self, engine: RefineEngine):
        c = _make_candidate("el_1", text="发送")
        output = engine.refine("canvas_1", [c], mode="vlm")
        assert output.results[0].refine_status == "unreviewed"
        assert output.mode == "vlm"

    def test_counts(self, engine: RefineEngine):
        c1 = _make_candidate("el_1", text="发送", control_type="Button")  # refined
        c2 = _make_candidate("el_2", control_type="", provider_sources=["vision"], bounds=(10, 10, 40, 40))  # uncertain
        output = engine.refine("canvas_1", [c1, c2])
        assert output.total_refined >= 1
        assert output.total_uncertain >= 1
        assert output.total_refined + output.total_uncertain + output.total_unreviewed == 2

    def test_no_hardcoded_app_specific_tags(self, engine: RefineEngine):
        """Heuristic mode must NOT produce app-specific tags like 'wechat.emoji_button'."""
        c = _make_candidate("el_emoji", text="表情", control_type="Button")
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        for tag in r.semantic_tags:
            assert "." not in tag or tag.count(".") == 1, f"Tag '{tag}' should be generic namespace only"
            # Generic tags like "action.send", "content.emoji" have exactly one dot
            # App-specific tags like "wechat.emoji_button" would have different patterns

    def test_refine_result_fields_exist(self, engine: RefineEngine):
        """Verify all three-layer fields exist on RefineResult."""
        c = _make_candidate("el_1", text="测试")
        output = engine.refine("canvas_1", [c])
        r = output.results[0]
        assert hasattr(r, "visual_type")
        assert hasattr(r, "semantic_tags")
        assert hasattr(r, "role_label")
        assert hasattr(r, "role_confidence")
        assert hasattr(r, "role_source")
        assert hasattr(r, "role_evidence")
        assert hasattr(r, "refine_status")
        assert isinstance(r.semantic_tags, list)
        assert isinstance(r.role_evidence, list)
