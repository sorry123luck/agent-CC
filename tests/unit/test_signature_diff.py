"""Tests for signature-based diff — 跨 canvas 元素追踪."""

import pytest

from src.canvas.diff_engine import CanvasDiffEngine
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    SemanticRole,
    WindowInfoSnapshot,
)


def _make_candidate(
    element_id: str,
    text: str = "",
    role: SemanticRole = SemanticRole.BUTTON,
    bounds: tuple[int, int, int, int] = (100, 100, 200, 150),
    stable_key_id: str | None = None,
) -> Candidate:
    return Candidate(
        element_id=element_id,
        text=text,
        semantic_role=role,
        bounds=bounds,
        stable_key_id=stable_key_id,
        provider_sources=["uia"],
        confidence=0.8,
    )


def _make_canvas(
    elements: list[Candidate],
    canvas_id: str = "canvas_1",
    window_size: tuple[int, int] = (896, 648),
) -> InteractionCanvas:
    w, h = window_size
    return InteractionCanvas(
        canvas_id=canvas_id,
        app=AppInfo(app_id="test_app"),
        window=WindowInfoSnapshot(hwnd=12345, rect_client=(0, 0, w, h)),
        elements=elements,
    )


# =============================================================================
# Test: 不同 element_id 但相同内容 → 正确 preserved（通过 role+IoU 兜底）
# =============================================================================


class TestRoleIoUMatch:
    def test_same_role_and_bounds_preserved(self):
        """不同 element_id，同 role + 高 IoU → preserved。"""
        before = _make_canvas([
            _make_candidate("vision_1", "发送", SemanticRole.SEND_BUTTON, (800, 590, 870, 625)),
        ], "canvas_before")

        after = _make_canvas([
            _make_candidate("vision_50", "发送", SemanticRole.SEND_BUTTON, (800, 590, 870, 625)),
        ], "canvas_after")

        engine = CanvasDiffEngine()
        changes = engine.diff(before, after)

        assert len(changes.preserved) == 1
        assert changes.preserved[0].text == "发送"


# =============================================================================
# Test: 不同 element_id 且不同内容 → 正确 added/removed
# =============================================================================


class TestAddedRemoved:
    def test_new_element_is_added(self):
        before = _make_canvas([
            _make_candidate("vision_1", "发送", bounds=(800, 590, 870, 625)),
        ])

        after = _make_canvas([
            _make_candidate("vision_1", "发送", bounds=(800, 590, 870, 625)),
            _make_candidate("vision_2", "123", bounds=(300, 500, 350, 530)),
        ])

        engine = CanvasDiffEngine()
        changes = engine.diff(before, after)

        assert len(changes.added) == 1
        assert changes.added[0].text == "123"

    def test_removed_element_is_removed(self):
        before = _make_canvas([
            _make_candidate("vision_1", "发送", bounds=(800, 590, 870, 625)),
            _make_candidate("vision_2", "123", bounds=(300, 500, 350, 530)),
        ])

        after = _make_canvas([
            _make_candidate("vision_1", "发送", bounds=(800, 590, 870, 625)),
        ])

        engine = CanvasDiffEngine()
        changes = engine.diff(before, after)

        assert len(changes.removed) == 1
        assert changes.removed[0].text == "123"


# =============================================================================
# Test: stable_key_id 精确匹配
# =============================================================================


class TestStableKeyMatch:
    def test_stable_key_id_exact_match(self):
        """即使 element_id 不同，stable_key_id 相同 → preserved。"""
        before = _make_canvas([
            _make_candidate("vision_1", "发送", stable_key_id="key-abc-123"),
        ])

        after = _make_canvas([
            _make_candidate("vision_50", "发送", stable_key_id="key-abc-123"),
        ])

        engine = CanvasDiffEngine()
        changes = engine.diff(before, after)

        assert len(changes.preserved) == 1
        assert len(changes.added) == 0
        assert len(changes.removed) == 0


# =============================================================================
# Test: 发送按钮跨 canvas 匹配（element_id 不同，text+role+IoU 匹配）
# =============================================================================


class TestSendButtonCrossCanvas:
    def test_send_button_preserved_across_canvas(self):
        """两次 observe 的发送按钮：element_id 不同，bounds 微变。"""
        before = _make_canvas([
            _make_candidate(
                "vision_54", "发送", SemanticRole.SEND_BUTTON,
                (811, 597, 869, 624),
            ),
            _make_candidate("vision_10", "文件", bounds=(10, 10, 60, 35)),
            _make_candidate("vision_20", "聊天", bounds=(200, 100, 400, 500)),
        ])

        after = _make_canvas([
            _make_candidate(
                "vision_47", "发送", SemanticRole.SEND_BUTTON,
                (809, 597, 869, 624),  # 微变 2px
            ),
            _make_candidate("vision_10", "文件", bounds=(10, 10, 60, 35)),
            _make_candidate("vision_20", "聊天", bounds=(200, 100, 400, 500)),
        ])

        engine = CanvasDiffEngine()
        changes = engine.diff(before, after)

        # 发送按钮应该被 preserved（通过 role+IoU 或 text 匹配）
        preserved_texts = [c.text for c in changes.preserved]
        assert "发送" in preserved_texts
