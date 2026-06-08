"""Tests for unified position band classification (Phase U0).

Two modes:
1. classify_position() — semantic position (classifier, fusion)
2. classify_grouping_band() — grouping band (grouper)

Proves consistency within each mode and documents the intentional difference.
"""

from src.perception.position_band import (
    PositionBand,
    classify_position,
    classify_position_str,
    classify_grouping_band,
)


class TestClassifyPosition:
    """Test semantic position classification."""

    def test_top_band(self):
        assert classify_position((0, 0, 800, 50), 800, 600) == PositionBand.TOP

    def test_bottom_band(self):
        assert classify_position((0, 550, 800, 600), 800, 600) == PositionBand.BOTTOM

    def test_left_band(self):
        assert classify_position((0, 100, 200, 500), 800, 600) == PositionBand.LEFT

    def test_right_band(self):
        assert classify_position((600, 100, 800, 500), 800, 600) == PositionBand.RIGHT

    def test_center_band(self):
        assert classify_position((200, 100, 600, 500), 800, 600) == PositionBand.CENTER

    def test_wide_left_is_center(self):
        """Wide region at left (width > 35%) → CENTER."""
        assert classify_position((0, 100, 400, 500), 800, 600) == PositionBand.CENTER

    def test_wide_right_is_center(self):
        """Wide region at right (width > 35%) → CENTER."""
        assert classify_position((400, 100, 800, 500), 800, 600) == PositionBand.CENTER

    def test_zero_window_returns_center(self):
        assert classify_position((0, 0, 100, 100), 0, 0) == PositionBand.CENTER

    def test_classify_position_str_returns_string(self):
        assert classify_position_str((0, 0, 800, 50), 800, 600) == "top"


class TestClassifyGroupingBand:
    """Test grouping-band classification."""

    def test_thin_top_bar_is_top(self):
        """Thin bar at top → top for grouping."""
        assert classify_grouping_band((0, 0, 800, 40), 800, 600) == "top"

    def test_bottom_bar_is_bottom(self):
        assert classify_grouping_band((0, 550, 800, 600), 800, 600) == "bottom"

    def test_left_narrow_is_left(self):
        assert classify_grouping_band((0, 100, 200, 500), 800, 600) == "left"

    def test_right_narrow_is_right(self):
        assert classify_grouping_band((600, 100, 800, 500), 800, 600) == "right"

    def test_center_is_center(self):
        assert classify_grouping_band((200, 100, 600, 500), 800, 600) == "center"

    def test_large_top_spanning_region_is_center(self):
        """Region starting at top but spanning past 15% → center for grouping.

        This is the key difference from classify_position():
        - classify_position((0,0,800,400), 800, 600) → TOP (rel_top=0 < 0.15)
        - classify_grouping_band((0,0,800,400), 800, 600) → CENTER (rel_bottom=0.67 > 0.15)

        The grouper must NOT group this with thin title bars.
        """
        bounds = (0, 0, 800, 400)
        # Semantic: TOP
        assert classify_position(bounds, 800, 600) == PositionBand.TOP
        # Grouping: CENTER
        assert classify_grouping_band(bounds, 800, 600) == "center"

    def test_thin_top_bar_both_modes_agree(self):
        """Thin bar at top → TOP in both modes."""
        bounds = (0, 0, 800, 40)
        assert classify_position(bounds, 800, 600) == PositionBand.TOP
        assert classify_grouping_band(bounds, 800, 600) == "top"


class TestCrossComponentConsistency:
    """Prove that semantic components (classifier, fusion) use the same band."""

    def _get_classifier_band(self, bounds, win_w, win_h):
        from src.perception.structural_classifier import compute_region_metrics
        m = compute_region_metrics(bounds, win_w, win_h)
        return m.position_band

    def _get_fusion_band(self, bounds, win_w, win_h):
        from src.perception.semantic_fusion import SemanticFusion
        left, top, right, bottom = bounds
        return SemanticFusion._classify_position(left, top, right, bottom, win_w, win_h)

    def test_top_consistent(self):
        bounds = (0, 0, 800, 50)
        assert self._get_classifier_band(bounds, 800, 600) == "top"
        assert self._get_fusion_band(bounds, 800, 600) == "top"

    def test_bottom_consistent(self):
        bounds = (0, 550, 800, 600)
        assert self._get_classifier_band(bounds, 800, 600) == "bottom"
        assert self._get_fusion_band(bounds, 800, 600) == "bottom"

    def test_left_consistent(self):
        bounds = (0, 100, 200, 500)
        assert self._get_classifier_band(bounds, 800, 600) == "left"
        assert self._get_fusion_band(bounds, 800, 600) == "left"

    def test_right_consistent(self):
        bounds = (600, 100, 800, 500)
        assert self._get_classifier_band(bounds, 800, 600) == "right"
        assert self._get_fusion_band(bounds, 800, 600) == "right"

    def test_center_consistent(self):
        bounds = (200, 100, 600, 500)
        assert self._get_classifier_band(bounds, 800, 600) == "center"
        assert self._get_fusion_band(bounds, 800, 600) == "center"
