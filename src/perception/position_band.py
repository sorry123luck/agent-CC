"""Unified position band classification (Phase U0).

Two classification modes sharing the same threshold constants:

1. classify_position() — semantic position
   "Where does this region sit in the window?"
   Used by StructuralClassifier and SemanticFusion for region typing.
   Uses rel_top for top-band check.

2. classify_grouping_band() — grouping band
   "Should this region be grouped with adjacent regions?"
   Used by RegionGrouper for spatial grouping.
   Uses rel_bottom for top-band check to avoid grouping large content
   regions that start near the top with actual top bars.

Example difference:
  bounds=(0, 0, 800, 400), window=800x600
  - classify_position()     → TOP   (region starts in top 15%)
  - classify_grouping_band() → CENTER (region extends past top 15%, is not a thin bar)
"""

from __future__ import annotations

from enum import Enum


class PositionBand(Enum):
    """Position band within a window."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


# ── Shared thresholds ──────────────────────────────────────────────────
_TOP_RATIO = 0.15         # top 15% of window height
_BOTTOM_RATIO = 0.85      # bottom 15% of window height (top > 85%)
_LEFT_WIDTH_RATIO = 0.05  # left edge < 5% of window width
_LEFT_MAX_WIDTH = 0.35    # AND region width < 35%
_RIGHT_WIDTH_RATIO = 0.65 # right edge > 65% of window width
_RIGHT_MAX_WIDTH = 0.35   # AND region width < 35%


def classify_position(
    bounds: tuple[int, int, int, int],
    window_width: int,
    window_height: int,
) -> PositionBand:
    """Semantic position classification.

    "Where does this region sit in the window?"
    Uses rel_top for top-band check.
    """
    if window_width <= 0 or window_height <= 0:
        return PositionBand.CENTER

    left, top, right, bottom = bounds
    region_w = max(right - left, 1)
    rel_top = top / window_height
    rel_bottom = bottom / window_height
    rel_left = left / window_width
    rel_right = right / window_width
    width_ratio = region_w / window_width

    if rel_left < _LEFT_WIDTH_RATIO and width_ratio < _LEFT_MAX_WIDTH:
        return PositionBand.LEFT
    if rel_right > _RIGHT_WIDTH_RATIO and width_ratio < _RIGHT_MAX_WIDTH:
        return PositionBand.RIGHT
    if rel_top < _TOP_RATIO:
        return PositionBand.TOP
    if rel_bottom > _BOTTOM_RATIO:
        return PositionBand.BOTTOM
    return PositionBand.CENTER


def classify_position_str(
    bounds: tuple[int, int, int, int],
    window_width: int,
    window_height: int,
) -> str:
    """String-returning variant for components that use string bands."""
    return classify_position(bounds, window_width, window_height).value


def classify_grouping_band(
    bounds: tuple[int, int, int, int],
    window_width: int,
    window_height: int,
) -> str:
    """Grouping-band classification for spatial adjacency grouping.

    "Should this region be grouped with adjacent regions?"
    Uses rel_bottom for top-band check: a region is only "top" for grouping
    purposes if its BOTTOM edge is within the top band. This prevents large
    content regions that start near the top from being grouped with actual
    thin top bars.

    Example:
      bounds=(0, 0, 800, 400), window=800x600
      rel_bottom = 400/600 = 0.67 > 0.15 → NOT top for grouping
      → "center" (grouped with content, not with title bar)
    """
    if window_width <= 0 or window_height <= 0:
        return "center"

    left, top, right, bottom = bounds
    region_w = max(right - left, 1)
    rel_top = top / window_height
    rel_bottom = bottom / window_height
    rel_left = left / window_width
    rel_right = right / window_width
    width_ratio = region_w / window_width

    if rel_left < _LEFT_WIDTH_RATIO and width_ratio < _LEFT_MAX_WIDTH:
        return "left"
    if rel_right > _RIGHT_WIDTH_RATIO and width_ratio < _RIGHT_MAX_WIDTH:
        return "right"
    # Key difference: use rel_bottom, not rel_top
    if rel_bottom <= _TOP_RATIO:
        return "top"
    if rel_top >= _BOTTOM_RATIO:
        return "bottom"
    return "center"
