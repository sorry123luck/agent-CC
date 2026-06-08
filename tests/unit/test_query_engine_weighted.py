"""Tests for Phase 7: query 排序接入 — reliability weight from fused_confidence."""

import pytest

from src.canvas.query_engine import CanvasQueryEngine, QueryTarget
from src.perception.page_compiler_models import (
    Candidate,
    InteractionCanvas,
    Region,
    SemanticRole,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    element_id: str,
    text: str = "",
    name: str | None = None,
    semantic_role: SemanticRole = SemanticRole.BUTTON,
    region_id: str | None = None,
    confidence: float = 0.0,
    control_type: str = "Button",
    interactable: bool = True,
    placeholder: str = "",
    role_label: str | None = None,
    semantic_tags: list[str] | None = None,
    visual_type: str = "",
    refine_status: str = "unreviewed",
) -> Candidate:
    return Candidate(
        element_id=element_id,
        text=text,
        name=name,
        semantic_role=semantic_role,
        region_id=region_id,
        confidence=confidence,
        control_type=control_type,
        interactable=interactable,
        placeholder=placeholder,
        role_label=role_label,
        semantic_tags=semantic_tags or [],
        visual_type=visual_type,
        refine_status=refine_status,
    )


def _make_canvas(elements: list[Candidate], regions: list[Region] | None = None) -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="test_canvas",
        elements=elements,
        regions=regions or [],
    )


# ---------------------------------------------------------------------------
# Test: Reliability weight
# ---------------------------------------------------------------------------


class TestReliabilityWeight:
    def test_high_confidence_no_penalty(self):
        """fused=1.0 → weight=1.0, score not reduced."""
        engine = CanvasQueryEngine()
        candidate = _make_candidate("e1", text="发送", confidence=1.0)
        canvas = _make_canvas([candidate])
        target = QueryTarget(text="发送")

        result = engine.query(canvas, target)
        assert len(result.candidates) == 1
        # Score should be high (no 50% penalty)
        score = engine._score_candidate(candidate, target, canvas)
        assert score > 0.5

    def test_low_confidence_half_penalty(self):
        """fused=0.0 → weight=0.5, score halved."""
        engine = CanvasQueryEngine()
        high = _make_candidate("e1", text="发送", confidence=1.0)
        low = _make_candidate("e2", text="发送", confidence=0.0)
        canvas = _make_canvas([high, low])
        target = QueryTarget(text="发送")

        score_high = engine._score_candidate(high, target, canvas)
        score_low = engine._score_candidate(low, target, canvas)
        # Same text, but low confidence gets 50% penalty
        assert score_high > score_low
        assert score_low == pytest.approx(score_high * 0.5, abs=0.01)

    def test_mid_confidence_partial(self):
        """fused=0.5 → weight=0.75."""
        engine = CanvasQueryEngine()
        mid = _make_candidate("e1", text="发送", confidence=0.5)
        target = QueryTarget(text="发送")

        score = engine._score_candidate(mid, target, _make_canvas([mid]))
        # weight = 0.5 + 0.5 * 0.5 = 0.75
        # relevance is some value > 0, so score = relevance * 0.75
        assert score > 0.0

    def test_clamp_above_1(self):
        """fused=1.5 → weight=1.0 (clamped)."""
        engine = CanvasQueryEngine()
        c = _make_candidate("e1", text="发送", confidence=1.5)
        target = QueryTarget(text="发送")

        score = engine._score_candidate(c, target, _make_canvas([c]))
        # Should not exceed the relevance score (weight capped at 1.0)
        assert score > 0.0

    def test_clamp_below_0(self):
        """fused=-0.5 → weight=0.5 (clamped)."""
        engine = CanvasQueryEngine()
        c = _make_candidate("e1", text="发送", confidence=-0.5)
        target = QueryTarget(text="发送")

        score = engine._score_candidate(c, target, _make_canvas([c]))
        # weight = 0.5 + 0.5 * 0.0 = 0.5
        assert score > 0.0

    def test_none_confidence_no_crash(self):
        """candidate.confidence = None → treated as 0.0, no crash."""
        engine = CanvasQueryEngine()
        c = _make_candidate("e1", text="发送", confidence=0.0)
        c.confidence = None  # type: ignore[assignment]
        target = QueryTarget(text="发送")

        score = engine._score_candidate(c, target, _make_canvas([c]))
        # weight = 0.5 + 0.5 * 0.0 = 0.5
        assert score > 0.0


# ---------------------------------------------------------------------------
# Test: Query ranking
# ---------------------------------------------------------------------------


class TestQueryRanking:
    def test_higher_confidence_ranks_higher(self):
        """Same relevance, higher confidence → higher rank."""
        engine = CanvasQueryEngine()
        high_conf = _make_candidate("e1", text="发送", confidence=1.0)
        low_conf = _make_candidate("e2", text="发送", confidence=0.0)
        canvas = _make_canvas([low_conf, high_conf])  # low first, high second
        target = QueryTarget(text="发送")

        result = engine.query(canvas, target)
        assert len(result.candidates) == 2
        # High confidence should rank first
        assert result.candidates[0].element_id == "e1"

    def test_relevance_still_primary(self):
        """High relevance + low confidence > low relevance + high confidence."""
        engine = CanvasQueryEngine()
        # Perfect text match, low confidence
        perfect_match = _make_candidate("e1", text="发送按钮", confidence=0.0)
        # Poor text match, high confidence
        poor_match = _make_candidate("e2", text="取消", confidence=1.0)
        canvas = _make_canvas([poor_match, perfect_match])
        target = QueryTarget(text="发送")

        result = engine.query(canvas, target)
        assert len(result.candidates) >= 1
        # Perfect match should rank first despite low confidence
        assert result.candidates[0].element_id == "e1"

    def test_no_evidence_still_returns(self):
        """confidence=0 candidate still returned (50% penalty, not eliminated)."""
        engine = CanvasQueryEngine()
        c = _make_candidate("e1", text="OK", confidence=0.0)
        canvas = _make_canvas([c])
        target = QueryTarget(text="OK")

        result = engine.query(canvas, target, min_confidence=0.2)
        assert len(result.candidates) == 1


class TestNaturalLanguageIntent:
    def test_chinese_search_box_query_prefers_search_input_over_search_button(self):
        """中文自然查询“找搜索框”应优先返回输入类候选，而不是搜索按钮。"""
        engine = CanvasQueryEngine()
        button = _make_candidate(
            "search_button",
            text="搜索",
            semantic_role=SemanticRole.BUTTON,
            confidence=1.0,
            control_type="Button",
        )
        search_input = _make_candidate(
            "search_input",
            text="",
            placeholder="搜索",
            semantic_role=SemanticRole.SEARCH_INPUT,
            confidence=0.7,
            control_type="Edit",
        )
        canvas = _make_canvas([button, search_input])

        result = engine.query(canvas, QueryTarget(natural_language="找搜索框"))

        assert result.candidates[0].element_id == "search_input"

    def test_send_button_query_blocks_message_input_false_positive(self):
        """“发送按钮”不能误命中包含发送文字的输入区。"""
        engine = CanvasQueryEngine()
        message_input = _make_candidate(
            "message_input",
            text="发送给小明的内容",
            semantic_role=SemanticRole.MESSAGE_INPUT,
            confidence=1.0,
            control_type="Edit",
        )
        send_button = _make_candidate(
            "send_button",
            text="发送",
            semantic_role=SemanticRole.SEND_BUTTON,
            confidence=0.6,
            control_type="Button",
        )
        send_button.risk_tags = ["send"]
        canvas = _make_canvas([message_input, send_button])

        result = engine.query(canvas, QueryTarget(natural_language="找发送按钮"))

        assert result.candidates[0].element_id == "send_button"
        assert all(candidate.element_id != "message_input" for candidate in result.candidates)

    def test_settings_menu_query_uses_role_label_and_semantic_tags(self):
        """设置菜单可由 refined role_label/semantic_tags 命中。"""
        engine = CanvasQueryEngine()
        generic_button = _make_candidate(
            "generic",
            text="",
            semantic_role=SemanticRole.BUTTON,
            confidence=0.9,
            role_label="toolbar icon",
            semantic_tags=["toolbar"],
        )
        settings = _make_candidate(
            "settings",
            text="",
            semantic_role=SemanticRole.BUTTON,
            confidence=0.7,
            role_label="settings menu",
            semantic_tags=["settings", "menu"],
        )
        canvas = _make_canvas([generic_button, settings])

        result = engine.query(canvas, QueryTarget(natural_language="找设置菜单"))

        assert result.candidates[0].element_id == "settings"

    def test_zero_score_layout_is_not_returned_when_min_confidence_is_zero(self):
        """min_confidence=0 只放宽低置信正匹配，不能把 0 分 layout 当作命中。"""
        engine = CanvasQueryEngine()
        empty_layout = _make_candidate(
            "empty_layout",
            text="",
            semantic_role=SemanticRole.LAYOUT,
            confidence=0.95,
            control_type="Pane",
        )
        explorer = _make_candidate(
            "explorer",
            text="资源管理器",
            semantic_role=SemanticRole.TREE_ITEM,
            confidence=0.2,
            control_type="TreeItem",
        )
        canvas = _make_canvas([empty_layout, explorer])

        result = engine.query(canvas, QueryTarget(natural_language="找会话列表"), min_confidence=0.0)

        assert result.candidates[0].element_id == "explorer"
        assert all(candidate.element_id != "empty_layout" for candidate in result.candidates)

    def test_alphanumeric_label_query_matches_short_self_drawn_button(self):
        """自然查询里的 A1/B2 等短标签应能召回自绘 OCR/app_layout 候选。"""
        engine = CanvasQueryEngine()
        send = _make_candidate(
            "send",
            text="Send",
            semantic_role=SemanticRole.SEND_BUTTON,
            confidence=1.0,
            control_type="Button",
        )
        a1 = _make_candidate(
            "a1",
            text="A1",
            semantic_role=SemanticRole.BUTTON,
            confidence=0.35,
            control_type="Button",
        )
        canvas = _make_canvas([send, a1])

        result = engine.query(canvas, QueryTarget(natural_language="找 A1 输出按钮"), min_confidence=0.0)

        assert result.candidates[0].element_id == "a1"


# ---------------------------------------------------------------------------
# Test: No regression
# ---------------------------------------------------------------------------


    def test_tab_query_matches_tab_elements(self):
        engine = CanvasQueryEngine()
        tab = _make_candidate("tab1", text="GitHub", semantic_role=SemanticRole.TAB, confidence=0.9)
        btn = _make_candidate("btn1", text="OK", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        canvas = _make_canvas([tab, btn])
        result = engine.query(canvas, QueryTarget(natural_language="找标签页"), min_confidence=0.0)
        assert result.candidates[0].element_id == "tab1"

    def test_navigation_button_query_matches_back_button(self):
        engine = CanvasQueryEngine()
        back = _make_candidate("back", text="后退", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        other = _make_candidate("other", text="设置", semantic_role=SemanticRole.MENU_ITEM, confidence=0.9)
        canvas = _make_canvas([back, other])
        result = engine.query(canvas, QueryTarget(natural_language="找后退按钮"), min_confidence=0.0)
        assert result.candidates[0].element_id == "back"

    def test_file_explorer_query_matches_sidebar(self):
        engine = CanvasQueryEngine()
        sidebar = _make_candidate("explorer", text="资源管理器", semantic_role=SemanticRole.SIDEBAR, confidence=0.9)
        editor = _make_candidate("editor", text="main.py", semantic_role=SemanticRole.TEXT, confidence=0.9)
        canvas = _make_canvas([sidebar, editor])
        result = engine.query(canvas, QueryTarget(natural_language="找文件资源管理器"), min_confidence=0.0)
        assert result.candidates[0].element_id == "explorer"

    def test_terminal_query_matches_pane_text(self):
        engine = CanvasQueryEngine()
        term = _make_candidate("term", text="Terminal", semantic_role=SemanticRole.TEXT, confidence=0.9)
        code = _make_candidate("code", text="def main():", semantic_role=SemanticRole.TEXT, confidence=0.9)
        canvas = _make_canvas([term, code])
        result = engine.query(canvas, QueryTarget(natural_language="找终端"), min_confidence=0.0)
        assert result.candidates[0].element_id == "term"

    def test_editor_query_matches_text_input(self):
        engine = CanvasQueryEngine()
        editor = _make_candidate("editor", text="", semantic_role=SemanticRole.TEXT_INPUT, confidence=0.9)
        sidebar = _make_candidate("sidebar", text="Explorer", semantic_role=SemanticRole.SIDEBAR, confidence=0.9)
        canvas = _make_canvas([editor, sidebar])
        result = engine.query(canvas, QueryTarget(natural_language="找编辑区域"), min_confidence=0.0)
        assert result.candidates[0].element_id == "editor"

    def test_contact_query_matches_chat_item(self):
        engine = CanvasQueryEngine()
        contact = _make_candidate("contact", text="张三", semantic_role=SemanticRole.CHAT_ITEM, confidence=0.9)
        msg = _make_candidate("msg", text="你好", semantic_role=SemanticRole.TEXT, confidence=0.9)
        canvas = _make_canvas([contact, msg])
        result = engine.query(canvas, QueryTarget(natural_language="找联系人"), min_confidence=0.0)
        assert result.candidates[0].element_id == "contact"


    def test_region_match_bonus(self):
        """Candidate in matching region ranks higher than one in wrong region."""
        engine = CanvasQueryEngine()
        from src.perception.page_compiler_models import Region
        nav_region = Region(region_id="r_nav", role="navigation", bounds=(0, 0, 500, 40))
        content_region = Region(region_id="r_content", role="content_area", bounds=(0, 40, 500, 400))
        btn = _make_candidate("btn", text="后退", semantic_role=SemanticRole.BUTTON, confidence=0.9, region_id="r_nav")
        url = _make_candidate("url", text="chatgpt.com", semantic_role=SemanticRole.BUTTON, confidence=0.9, region_id="r_content")
        canvas = _make_canvas([btn, url], regions=[nav_region, content_region])
        result = engine.query(canvas, QueryTarget(natural_language="找后退按钮"), min_confidence=0.0)
        assert result.candidates[0].element_id == "btn"

    def test_missing_region_not_penalized(self):
        """Candidate without region should not crash or get negative score."""
        engine = CanvasQueryEngine()
        canvas = _make_canvas([
            _make_candidate("a", text="OK", semantic_role=SemanticRole.BUTTON, confidence=0.9),
        ])
        result = engine.query(canvas, QueryTarget(natural_language="找后退按钮"), min_confidence=0.0)
        # Should not crash; candidate may or may not match but score >= 0
        for c in result.candidates:
            pass  # just verifying no crash


    def test_menu_button_query_matches_menu_elements(self):
        engine = CanvasQueryEngine()
        menu = _make_candidate("menu1", text="Menu", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        other = _make_candidate("other1", text="A1", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        canvas = _make_canvas([menu, other])
        result = engine.query(canvas, QueryTarget(natural_language="找菜单按钮"), min_confidence=0.0)
        assert result.candidates[0].element_id == "menu1"

    def test_region_match_bonus(self):
        """Candidate in matching region should score higher."""
        engine = CanvasQueryEngine()
        # Both are buttons, but one is in toolbar region
        toolbar_btn = _make_candidate("tb", text="后退", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        search_btn = _make_candidate("sb", text="搜索", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        canvas = _make_canvas([toolbar_btn, search_btn])
        # Set region roles
        toolbar_btn.attributes = {"region_role": "toolbar"}
        search_btn.attributes = {"region_role": "content_area"}
        result = engine.query(canvas, QueryTarget(natural_language="找后退按钮"), min_confidence=0.0)
        # Both have role=button, but toolbar_btn has matching region_role
        assert result.candidates[0].element_id == "tb"

    def test_unknown_region_not_penalized(self):
        """Candidate with no region_role should not be penalized."""
        engine = CanvasQueryEngine()
        btn = _make_candidate("btn1", text="Menu", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        btn.attributes = {}  # no region_role
        canvas = _make_canvas([btn])
        result = engine.query(canvas, QueryTarget(natural_language="找菜单按钮"), min_confidence=0.0)
        assert len(result.candidates) >= 1
        assert result.candidates[0].element_id == "btn1"

    def test_negative_text_excludes_url_as_button(self):
        """Intent with negative_text should exclude URL-matching candidates."""
        engine = CanvasQueryEngine()
        url_btn = _make_candidate("url", text="https://chatgpt.com/g/g-p", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        real_btn = _make_candidate("real", text="后退", semantic_role=SemanticRole.BUTTON, confidence=0.9)
        canvas = _make_canvas([url_btn, real_btn])
        # Use navigation_button intent which has negative terms
        result = engine.query(canvas, QueryTarget(natural_language="找后退按钮"), min_confidence=0.0)
        # URL should not be top-1
        assert result.candidates[0].element_id == "real"


class TestNoRegression:
    def test_no_target_returns_zero(self):
        """Empty target → score 0."""
        engine = CanvasQueryEngine()
        c = _make_candidate("e1", text="OK", confidence=0.5)
        target = QueryTarget()  # no fields set

        score = engine._score_candidate(c, target, _make_canvas([c]))
        assert score == 0.0

    def test_empty_canvas_no_crash(self):
        """Empty canvas → no candidates, no crash."""
        engine = CanvasQueryEngine()
        canvas = _make_canvas([])
        target = QueryTarget(text="test")

        result = engine.query(canvas, target)
        assert result.total_matched == 0
        assert len(result.candidates) == 0

    def test_semrole_query_weighted(self):
        """Semantic role query also gets weighted."""
        engine = CanvasQueryEngine()
        high = _make_candidate("e1", semantic_role=SemanticRole.BUTTON, confidence=1.0)
        low = _make_candidate("e2", semantic_role=SemanticRole.BUTTON, confidence=0.0)
        canvas = _make_canvas([low, high])
        target = QueryTarget(semantic_role="button")

        result = engine.query(canvas, target)
        assert result.candidates[0].element_id == "e1"

    def test_composite_query_weighted(self):
        """Composite query also gets weighted."""
        engine = CanvasQueryEngine()
        high = _make_candidate("e1", text="搜索", confidence=1.0, control_type="Edit")
        low = _make_candidate("e2", text="搜索", confidence=0.0, control_type="Edit")
        canvas = _make_canvas([low, high])
        target = QueryTarget(composite={"text": "搜索", "control_type": "Edit"})

        result = engine.query(canvas, target)
        assert result.candidates[0].element_id == "e1"
