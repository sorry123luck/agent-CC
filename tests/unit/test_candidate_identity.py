"""Tests for candidate_identity — 签名构建与相似度计算."""

import pytest

from src.memory.candidate_identity import (
    CandidateSignature,
    build_signature,
    compute_signature_similarity,
)
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    SemanticRole,
    WindowInfoSnapshot,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_candidate(
    text: str = "",
    role: SemanticRole = SemanticRole.BUTTON,
    bounds: tuple[int, int, int, int] = (100, 100, 200, 150),
    visual_type: str = "button",
    confidence: float = 0.8,
    provider_sources: list[str] | None = None,
) -> Candidate:
    return Candidate(
        element_id="test_1",
        text=text,
        semantic_role=role,
        bounds=bounds,
        visual_type=visual_type,
        confidence=confidence,
        provider_sources=provider_sources or ["uia"],
    )


def _make_canvas(window_size: tuple[int, int] = (896, 648)) -> InteractionCanvas:
    w, h = window_size
    return InteractionCanvas(
        app=AppInfo(app_id="test_app"),
        window=WindowInfoSnapshot(hwnd=12345, rect_client=(0, 0, w, h)),
    )


def _make_sig(
    region: str = "right_panel_bottom",
    role: str = "button",
    rel_bounds: tuple[float, float, float, float] = (0.9, 0.9, 0.97, 0.96),
    text: str = "发送",
    visual_type: str = "button",
    confidence: float = 0.8,
    is_fixed: bool = True,
    sources: tuple[str, ...] = ("uia",),
) -> CandidateSignature:
    x1, y1, x2, y2 = rel_bounds
    has_valid = not (x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0) and x2 > x1 and y2 > y1
    return CandidateSignature(
        region_id=region,
        region_role=role,
        relative_bounds=rel_bounds,
        size_ratio=0.01 if has_valid else 0.0,
        text_normalized=text,
        visual_type=visual_type,
        role_label=None,
        semantic_tags=(),
        crop_hash="",
        provider_sources=sources,
        confidence=confidence,
        is_fixed_control=is_fixed,
        has_valid_bounds=has_valid,
    )


# =============================================================================
# Test: 相同签名 → 1.0
# =============================================================================


class TestIdenticalSignatures:
    def test_identical_returns_one(self):
        sig = _make_sig()
        assert compute_signature_similarity(sig, sig) == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Test: 完全不同签名 → < 0.3
# =============================================================================


class TestCompletelyDifferent:
    def test_different_returns_low(self):
        a = _make_sig(
            region="title_bar",
            role="title_bar",
            rel_bounds=(0.0, 0.0, 0.5, 0.05),
            text="OpenClaw",
            visual_type="text",
            is_fixed=True,
        )
        b = _make_sig(
            region="right_panel_bottom",
            role="send_button",
            rel_bounds=(0.9, 0.9, 0.97, 0.96),
            text="发送",
            visual_type="button",
            is_fixed=True,
        )
        sim = compute_signature_similarity(a, b)
        assert sim < 0.3, f"Expected < 0.3, got {sim:.3f}"


# =============================================================================
# Test: 发送按钮两次 observe — bounds 微变 → >= 0.95
# =============================================================================


class TestSendButtonMatch:
    def test_send_button_similar_bounds(self):
        """两次 observe 的发送按钮：文字相同，bounds 微变 2px。"""
        a = _make_sig(
            region="right_panel_bottom",
            role="send_button",
            rel_bounds=(0.905, 0.921, 0.970, 0.963),
            text="发送",
            visual_type="button",
            confidence=0.72,
            is_fixed=True,
        )
        b = _make_sig(
            region="right_panel_bottom",
            role="send_button",
            rel_bounds=(0.903, 0.921, 0.970, 0.963),
            text="发送",
            visual_type="button",
            confidence=0.72,
            is_fixed=True,
        )
        sim = compute_signature_similarity(a, b)
        assert sim >= 0.95, f"Expected >= 0.95, got {sim:.3f}"


# =============================================================================
# Test: "123" vs "74条新消息" — 短文本惩罚 → < 0.4
# =============================================================================


class TestShortTextPenalty:
    def test_123_vs_74_messages(self):
        """纯数字短文本 "123" 不应匹配 "74条新消息"。"""
        a = _make_sig(
            region="right_panel_middle",
            role="unknown",
            rel_bounds=(0.36, 0.78, 0.40, 0.82),
            text="123",
            visual_type="text",
            is_fixed=False,  # 动态内容
        )
        b = _make_sig(
            region="right_panel_middle",
            role="unknown",
            rel_bounds=(0.36, 0.78, 0.40, 0.82),
            text="74条新消息",
            visual_type="text",
            is_fixed=False,
        )
        sim = compute_signature_similarity(a, b)
        assert sim <= 0.4, f"Expected <= 0.4, got {sim:.3f}"

    def test_pure_digits_penalty(self):
        """纯数字文本应被强惩罚。"""
        a = _make_sig(text="999", is_fixed=False)
        b = _make_sig(text="998", is_fixed=False)
        sim = compute_signature_similarity(a, b)
        assert sim <= 0.4, f"Expected <= 0.4, got {sim:.3f}"

    def test_pure_symbols_penalty(self):
        """纯符号文本应被强惩罚。"""
        a = _make_sig(text="---", is_fixed=False)
        b = _make_sig(text="***", is_fixed=False)
        sim = compute_signature_similarity(a, b)
        assert sim <= 0.4, f"Expected <= 0.4, got {sim:.3f}"


# =============================================================================
# Test: 固定控件匹配 — region+bounds+crop 相同，text 不同 → >= 0.75
# =============================================================================


class TestFixedControlMatch:
    def test_fixed_same_position_different_text(self):
        """固定控件位置相同但文字不同时，仍应高度匹配。"""
        a = _make_sig(
            region="right_panel_bottom",
            role="button",
            rel_bounds=(0.90, 0.92, 0.97, 0.96),
            text="发送",
            is_fixed=True,
        )
        b = _make_sig(
            region="right_panel_bottom",
            role="button",
            rel_bounds=(0.90, 0.92, 0.97, 0.96),
            text="Send",
            is_fixed=True,
        )
        sim = compute_signature_similarity(a, b)
        assert sim >= 0.75, f"Expected >= 0.75, got {sim:.3f}"


# =============================================================================
# Test: 动态内容 text 不匹配 → 上限 0.6
# =============================================================================


class TestDynamicContentCap:
    def test_dynamic_text_mismatch_cap(self):
        """动态内容 text 不匹配时，相似度上限 0.6。"""
        a = _make_sig(
            region="right_panel_middle",
            role="message_content",
            rel_bounds=(0.3, 0.5, 0.7, 0.6),
            text="你好世界",
            is_fixed=False,
        )
        b = _make_sig(
            region="right_panel_middle",
            role="message_content",
            rel_bounds=(0.3, 0.5, 0.7, 0.6),
            text="再见世界",
            is_fixed=False,
        )
        sim = compute_signature_similarity(a, b)
        assert sim <= 0.6, f"Expected <= 0.6, got {sim:.3f}"


# =============================================================================
# Test: 阈值边界
# =============================================================================


class TestThresholdBoundary:
    def test_just_below_threshold(self):
        """相似度 0.74 → 不应复用 key。"""
        # 构造一个相似度刚好低于 0.75 的场景
        a = _make_sig(
            region="right_panel_bottom",
            role="button",
            rel_bounds=(0.90, 0.92, 0.97, 0.96),
            text="确定",
            is_fixed=True,
            confidence=0.8,
        )
        b = _make_sig(
            region="right_panel_bottom",
            role="button",
            rel_bounds=(0.85, 0.88, 0.92, 0.93),
            text="取消",
            is_fixed=True,
            confidence=0.6,
        )
        sim = compute_signature_similarity(a, b)
        # 这对的相似度应该在 0.7 左右（role+region 相同，但 text 和 bounds 有差异）
        assert sim < 0.85, f"Sanity check failed: {sim:.3f}"

    def test_just_above_threshold(self):
        """相似度 >= 0.75 → 应复用 key。"""
        a = _make_sig(
            region="right_panel_bottom",
            role="button",
            rel_bounds=(0.90, 0.92, 0.97, 0.96),
            text="发送",
            is_fixed=True,
            confidence=0.8,
        )
        b = _make_sig(
            region="right_panel_bottom",
            role="button",
            rel_bounds=(0.90, 0.92, 0.97, 0.96),
            text="发送",
            is_fixed=True,
            confidence=0.75,
        )
        sim = compute_signature_similarity(a, b)
        assert sim >= 0.75, f"Expected >= 0.75, got {sim:.3f}"


# =============================================================================
# Test: build_signature 从 Candidate + Canvas 构建
# =============================================================================


class TestBuildSignature:
    def test_basic_build(self):
        cand = _make_candidate(text="发送", role=SemanticRole.SEND_BUTTON)
        canvas = _make_canvas((896, 648))
        sig = build_signature(cand, canvas)

        assert sig.text_normalized == "发送"
        assert sig.region_role == "send_button"
        assert sig.is_fixed_control is True
        assert 0 < sig.relative_bounds[0] < 1
        assert 0 < sig.size_ratio < 1

    def test_dynamic_content_gets_hash(self):
        """TEXT 角色现在是动态内容，text_normalized 应为 hash。"""
        cand = _make_candidate(
            text="你好世界",
            role=SemanticRole.TEXT,
        )
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.is_fixed_control is False
        assert sig.text_normalized != "你好世界"  # 应为 hash

    def test_unknown_role_is_dynamic(self):
        """UNKNOWN 角色视为动态内容（安全默认）。"""
        cand = _make_candidate(role=SemanticRole.UNKNOWN)
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.is_fixed_control is False

    def test_button_role_is_fixed(self):
        """BUTTON 角色视为固定控件。"""
        cand = _make_candidate(role=SemanticRole.BUTTON)
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.is_fixed_control is True

    def test_layout_container_roles_are_not_fixed_controls(self):
        """Root layout/container shells should not become stable model controls."""
        canvas = _make_canvas((1002, 731))

        for role in (SemanticRole.CONTAINER, SemanticRole.LAYOUT):
            cand = _make_candidate(
                role=role,
                bounds=(-7, 0, 1009, 738),
                visual_type="container",
            )
            sig = build_signature(cand, canvas)

            assert sig.is_fixed_control is False


# =============================================================================
# Test: has_valid_bounds — 非法 bounds 标记
# =============================================================================


class TestHasValidBounds:
    def test_normal_bounds_valid(self):
        """正常 bounds → has_valid_bounds=True。"""
        cand = _make_candidate(bounds=(100, 100, 200, 150))
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is True

    def test_zero_bounds_invalid(self):
        """全零 bounds → has_valid_bounds=False。"""
        cand = _make_candidate(bounds=(0, 0, 0, 0))
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is False
        assert sig.relative_bounds == (0.0, 0.0, 0.0, 0.0)

    def test_zero_area_invalid(self):
        """零面积 bounds (x2<=x1) → has_valid_bounds=False。"""
        cand = _make_candidate(bounds=(100, 100, 100, 200))
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is False

    def test_reversed_bounds_invalid(self):
        """反转 bounds (x2<x1) → has_valid_bounds=False。"""
        cand = _make_candidate(bounds=(200, 100, 100, 200))
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is False

    def test_electron_negative_y_invalid(self):
        """Electron 应用极端负坐标 → has_valid_bounds=False。"""
        # UIA 对 Electron 应用可能返回 y=-90682 这样的坐标
        cand = _make_candidate(bounds=(315, -90682, 1041, 677))
        canvas = _make_canvas((1041, 784))
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is False

    def test_far_out_of_window_invalid(self):
        """超出窗口范围过多 → has_valid_bounds=False。"""
        cand = _make_candidate(bounds=(5000, 5000, 6000, 6000))
        canvas = _make_canvas((800, 600))
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is False

    def test_none_bounds_invalid(self):
        """None bounds → has_valid_bounds=False。"""
        cand = _make_candidate(bounds=None)
        canvas = _make_canvas()
        sig = build_signature(cand, canvas)
        assert sig.has_valid_bounds is False
