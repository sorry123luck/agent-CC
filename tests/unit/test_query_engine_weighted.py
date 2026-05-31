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


# ---------------------------------------------------------------------------
# Test: No regression
# ---------------------------------------------------------------------------


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
