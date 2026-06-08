"""Structural Classifier — combination-evidence based region classification (Phase R3.1).

Computes RegionMetrics for each region and classifies structure_type using
multi-signal combination rules. Does NOT use single keywords to determine
structure_type. OCR keywords are weak evidence only.

This module is read-only — it does not modify regions, elements, or canvas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.perception.structured_region import StructureType


# =============================================================================
# Region Metrics
# =============================================================================


@dataclass
class RegionMetrics:
    """Quantitative metrics for a single region."""

    # Position
    position_band: str = "center"  # "top", "bottom", "left", "right", "center"
    rel_top: float = 0.0
    rel_bottom: float = 0.0
    rel_left: float = 0.0
    rel_right: float = 0.0

    # Geometry
    area_ratio: float = 0.0
    aspect_ratio: float = 0.0
    height_ratio: float = 0.0
    width_ratio: float = 0.0

    # OCR
    ocr_count: int = 0
    short_text_count: int = 0  # text <= 8 chars
    text_density: float = 0.0  # ocr_count / area_ratio
    horizontal_text_row_score: float = 0.0  # 0-1, how horizontally aligned
    vertical_list_score: float = 0.0  # 0-1, how vertically stacked

    # UIA controls
    uia_button_count: int = 0
    uia_edit_count: int = 0
    uia_list_count: int = 0
    uia_image_count: int = 0
    uia_total_count: int = 0

    # Vision
    vision_candidate_count: int = 0
    icon_like_count: int = 0  # small vision candidates

    # Fusion
    fusion_label: str | None = None
    fusion_confidence: float = 0.0

    # Weak keyword hints (NOT used for classification, only for evidence)
    keyword_hints: list[str] = field(default_factory=list)


def compute_region_metrics(
    bounds: tuple[int, int, int, int],
    window_width: int,
    window_height: int,
    raw_elements: list[dict[str, Any]] | None = None,
    ocr_blocks: list[dict[str, Any]] | None = None,
    vision_candidates: list[dict[str, Any]] | None = None,
    fusion_result: dict[str, Any] | None = None,
) -> RegionMetrics:
    """Compute comprehensive metrics for a region."""
    rl, rt, rr, rb = bounds
    region_w = max(rr - rl, 1)
    region_h = max(rb - rt, 1)

    m = RegionMetrics()

    if window_width <= 0 or window_height <= 0:
        return m

    win_area = window_width * window_height
    region_area = region_w * region_h

    # Geometry (compute before position_band check)
    m.area_ratio = region_area / win_area
    m.aspect_ratio = region_w / region_h
    m.height_ratio = region_h / window_height
    m.width_ratio = region_w / window_width

    # Position (unified via position_band module)
    m.rel_top = rt / window_height
    m.rel_bottom = rb / window_height
    m.rel_left = rl / window_width
    m.rel_right = rr / window_width

    from src.perception.position_band import classify_position_str
    m.position_band = classify_position_str(bounds, window_width, window_height)

    # OCR metrics
    region_ocr = _collect_ocr_in_region(bounds, ocr_blocks)
    m.ocr_count = len(region_ocr)
    m.short_text_count = sum(1 for b in region_ocr if len(b.get("text", "")) <= 8)
    if m.area_ratio > 0:
        m.text_density = m.ocr_count / m.area_ratio
    m.horizontal_text_row_score = _compute_horizontal_row_score(region_ocr, bounds)
    m.vertical_list_score = _compute_vertical_list_score(region_ocr, bounds)

    # UIA control metrics
    region_elements = _collect_elements_in_region(bounds, raw_elements)
    for elem in region_elements:
        ct = (elem.get("control_type") or "").lower()
        m.uia_total_count += 1
        if "button" in ct:
            m.uia_button_count += 1
        if "edit" in ct:
            m.uia_edit_count += 1
        if "list" in ct or "listitem" in ct:
            m.uia_list_count += 1
        if "image" in ct:
            m.uia_image_count += 1

    # Vision metrics
    region_vision = _collect_items_in_region(bounds, vision_candidates)
    m.vision_candidate_count = len(region_vision)
    m.icon_like_count = sum(
        1 for v in region_vision
        if _is_icon_like(v, window_width, window_height)
    )

    # Fusion
    if fusion_result:
        m.fusion_label = fusion_result.get("semantic_label")
        m.fusion_confidence = float(fusion_result.get("confidence", 0.0))

    # Weak keyword hints (for evidence only, NOT for classification)
    m.keyword_hints = _extract_keyword_hints(region_ocr, region_elements)

    return m


# =============================================================================
# Structural Classifier
# =============================================================================


class StructuralClassifier:
    """Classify region structure_type using combination evidence.

    Rules use multiple signals together. No single keyword or metric
    can determine structure_type alone.
    """

    def classify(self, m: RegionMetrics) -> tuple[StructureType, float, str]:
        """Classify structure type from metrics.

        Returns (structure_type, confidence, reason).
        """
        # Priority order: specific patterns first, generic last

        # 1. Status region: bottom band + wide + (short texts OR low content)
        if self._is_status_region(m):
            return StructureType.STATUS_REGION, 0.6, "bottom_wide_short_texts"

        # 2. Input region: bottom area + edit control + button
        if self._is_input_region(m):
            return StructureType.INPUT_REGION, 0.65, "bottom_edit_button"

        # 3. Top bar: top band + wide + horizontal text row
        if self._is_top_bar(m):
            return StructureType.TOP_BAR, 0.6, "top_wide_horizontal_texts"

        # 4. Toolbar: top area + buttons/icons/horizontal text row
        if self._is_toolbar(m):
            return StructureType.TOOLBAR, 0.55, "top_buttons_icons_or_text_row"

        # 5. Side rail: left/right + tall + vertical repeated items
        if self._is_side_rail(m):
            return StructureType.SIDE_RAIL, 0.55, "side_tall_vertical_items"

        # 6. Content stream: dynamic message/chat/list flow
        if self._is_content_stream(m):
            return StructureType.CONTENT_STREAM, 0.5, "dynamic_content_flow"

        # 7. List region: vertical item pattern + list controls
        if self._is_list_region(m):
            return StructureType.LIST_REGION, 0.5, "vertical_items_list_controls"

        # 8. Media control bar: bottom + media-specific evidence
        if self._is_media_control_bar(m):
            return StructureType.MEDIA_CONTROL_BAR, 0.5, "bottom_media_controls"

        # 9. Control strip: bottom + generic buttons (not media-specific)
        if self._is_control_strip(m):
            return StructureType.CONTROL_STRIP, 0.45, "bottom_generic_controls"

        # 10. Document region: large area + text dense + static content
        if self._is_document_region(m):
            return StructureType.DOCUMENT_REGION, 0.5, "large_text_dense_static"

        # 11. Canvas region: large area + low text density + vision candidates
        if self._is_canvas_region(m):
            return StructureType.CANVAS_REGION, 0.45, "large_low_text_vision"

        return StructureType.UNKNOWN_STRUCTURED, 0.0, "no_match"

    # --- Classification rules (combination evidence only) ---

    def _is_status_region(self, m: RegionMetrics) -> bool:
        """Bottom band + wide + short texts + small area."""
        return (
            m.position_band == "bottom"
            and m.aspect_ratio > 3.0
            and m.height_ratio < 0.12
            and m.short_text_count >= 1
            and m.uia_button_count <= 2
        )

    def _is_input_region(self, m: RegionMetrics) -> bool:
        """Bottom area + edit control present + button nearby."""
        return (
            m.position_band == "bottom"
            and m.uia_edit_count >= 1
            and m.uia_button_count >= 1
            and m.height_ratio < 0.30
        )

    def _is_top_bar(self, m: RegionMetrics) -> bool:
        """Top band + wide + horizontal text alignment."""
        return (
            m.position_band == "top"
            and m.aspect_ratio > 2.0
            and m.height_ratio < 0.18
            and m.horizontal_text_row_score > 0.4
        )

    def _is_toolbar(self, m: RegionMetrics) -> bool:
        """Top area + buttons/icons/horizontal text row + moderate width."""
        return (
            m.position_band == "top"
            and m.aspect_ratio > 2.0
            and m.height_ratio < 0.15
            and (m.uia_button_count >= 2 or m.icon_like_count >= 3 or m.horizontal_text_row_score > 0.4)
        )

    def _is_side_rail(self, m: RegionMetrics) -> bool:
        """Left/right + tall + vertical repeated items + actual content."""
        return (
            m.position_band in ("left", "right")
            and m.height_ratio > 0.40
            and m.aspect_ratio < 0.6
            and (m.uia_list_count >= 2 or m.vertical_list_score > 0.5)
            and m.ocr_count >= 3  # Must have actual content
        )

    def _is_content_stream(self, m: RegionMetrics) -> bool:
        """Dynamic content flow: messages, chat, timeline, list stream."""
        return (
            m.area_ratio > 0.15
            and m.vertical_list_score > 0.5
            and m.ocr_count >= 5
            and m.uia_button_count <= 1  # Not a control-heavy area
        )

    def _is_list_region(self, m: RegionMetrics) -> bool:
        """Vertical item pattern + list controls + moderate area.

        Must have actual list controls (not just vertical text layout).
        Must not look like a document/editor with continuous text flow.
        EditControl presence is NOT a veto (search boxes in lists are fine).
        """
        if m.vertical_list_score <= 0.5:
            return False
        if m.uia_list_count < 3:
            return False
        if m.area_ratio <= 0.05:
            return False
        if m.uia_button_count > 3:  # Not a toolbar-heavy area
            return False
        # Exclude document/editor-like regions:
        # Large Edit/Document control + high text density + continuous text
        if self._looks_like_document_editor(m):
            return False
        return True

    def _looks_like_document_editor(self, m: RegionMetrics) -> bool:
        """Check if region looks like a document/editor (continuous text flow).

        Distinguishes from list regions where items are discrete rows.
        """
        # Signal 1: High text density with many OCR blocks → continuous document
        if m.text_density > 10.0 and m.ocr_count >= 10:
            return True
        # Signal 2: Very high OCR count relative to area → paragraphs, not items
        if m.ocr_count >= 15 and m.area_ratio > 0.10:
            return True
        # Signal 3: Large area + high text density + low list controls → document
        if m.area_ratio > 0.15 and m.text_density > 5.0 and m.uia_list_count < 5:
            return True
        return False

    def _is_media_control_bar(self, m: RegionMetrics) -> bool:
        """Bottom area + dense media controls (many icons + buttons, very low text)."""
        return (
            m.position_band == "bottom"
            and m.height_ratio < 0.20
            and m.icon_like_count >= 4  # Dense icon layout
            and m.uia_button_count >= 3  # Multiple control buttons
            and m.ocr_count <= 2  # Very low text (media controls are icon-heavy)
        )

    def _is_control_strip(self, m: RegionMetrics) -> bool:
        """Bottom area + generic buttons/icons (not media-specific).

        Must NOT be a text-heavy area (file lists, content blocks, etc.)
        """
        if m.position_band != "bottom":
            return False
        if m.height_ratio >= 0.20:
            return False
        if not (m.uia_button_count >= 2 or m.icon_like_count >= 2):
            return False
        if m.ocr_count > 3:  # Tight: controls have little text
            return False
        if m.vertical_list_score >= 0.4:  # Not a file/content list
            return False
        # If there's OCR, mostly short labels
        if m.ocr_count > 0 and m.short_text_count < m.ocr_count * 0.3:
            return False
        return True

    def _is_document_region(self, m: RegionMetrics) -> bool:
        """Large area + text dense + static content (not dynamic stream)."""
        return (
            m.area_ratio > 0.20
            and m.text_density > 3.0
            and m.uia_button_count <= 2
            and m.uia_edit_count == 0
            and m.vertical_list_score < 0.4  # Exclude dynamic streams
        )

    def _is_canvas_region(self, m: RegionMetrics) -> bool:
        """Large area + low text density + vision candidates present."""
        return (
            m.area_ratio > 0.15
            and m.text_density < 2.0
            and m.vision_candidate_count >= 5
            and m.uia_edit_count == 0
        )


# =============================================================================
# Helper functions
# =============================================================================


def _get_bounds(element: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bounds = element.get("bounding_rect") or element.get("bounds") or element.get("bbox")
    if bounds and len(bounds) >= 4:
        return (bounds[0], bounds[1], bounds[2], bounds[3])
    return None


def _collect_ocr_in_region(
    region_bounds: tuple[int, int, int, int],
    ocr_blocks: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not ocr_blocks:
        return []
    rl, rt, rr, rb = region_bounds
    result = []
    for block in ocr_blocks:
        bbox = block.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        bl, bt, br, bb = bbox
        cx, cy = (bl + br) // 2, (bt + bb) // 2
        if rl <= cx <= rr and rt <= cy <= rb:
            result.append(block)
    return result


def _collect_elements_in_region(
    region_bounds: tuple[int, int, int, int],
    raw_elements: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not raw_elements:
        return []
    rl, rt, rr, rb = region_bounds
    result = []
    for elem in raw_elements:
        bounds = _get_bounds(elem)
        if not bounds:
            continue
        el, et, er, eb = bounds
        cx, cy = (el + er) // 2, (et + eb) // 2
        if rl <= cx <= rr and rt <= cy <= rb:
            result.append(elem)
    return result


def _collect_items_in_region(
    region_bounds: tuple[int, int, int, int],
    items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    rl, rt, rr, rb = region_bounds
    result = []
    for item in items:
        bounds = _get_bounds(item)
        if not bounds:
            continue
        bl, bt, br, bb = bounds
        cx, cy = (bl + br) // 2, (bt + bb) // 2
        if rl <= cx <= rr and rt <= cy <= rb:
            result.append(item)
    return result


def _compute_horizontal_row_score(
    ocr_blocks: list[dict[str, Any]],
    region_bounds: tuple[int, int, int, int],
) -> float:
    """Score how horizontally aligned OCR blocks are (0-1).

    High score = blocks aligned in a horizontal row (like a menu/toolbar).
    """
    if len(ocr_blocks) < 2:
        return 0.0

    _, rt, _, rb = region_bounds
    region_h = max(rb - rt, 1)

    centers_y = []
    for block in ocr_blocks:
        bbox = block.get("bbox", [])
        if len(bbox) >= 4:
            centers_y.append((bbox[1] + bbox[3]) / 2)

    if len(centers_y) < 2:
        return 0.0

    mean_y = sum(centers_y) / len(centers_y)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in centers_y) / len(centers_y))

    # Low std relative to region height = horizontally aligned
    ratio = std_y / region_h
    if ratio < 0.05:
        return 1.0
    if ratio < 0.10:
        return 0.7
    if ratio < 0.20:
        return 0.4
    return 0.0


def _compute_vertical_list_score(
    ocr_blocks: list[dict[str, Any]],
    region_bounds: tuple[int, int, int, int],
) -> float:
    """Score how vertically stacked OCR blocks are (0-1).

    High score = blocks stacked vertically with regular spacing (like a list).
    """
    if len(ocr_blocks) < 3:
        return 0.0

    rl, _, rr, _ = region_bounds
    region_w = max(rr - rl, 1)

    centers_x = []
    centers_y = []
    for block in ocr_blocks:
        bbox = block.get("bbox", [])
        if len(bbox) >= 4:
            centers_x.append((bbox[0] + bbox[2]) / 2)
            centers_y.append((bbox[1] + bbox[3]) / 2)

    if len(centers_x) < 3:
        return 0.0

    # Check x alignment (should be similar for vertical list)
    mean_x = sum(centers_x) / len(centers_x)
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in centers_x) / len(centers_x))
    x_aligned = std_x / region_w < 0.25

    # Check y spacing (should be roughly regular)
    sorted_y = sorted(centers_y)
    gaps = [sorted_y[i + 1] - sorted_y[i] for i in range(len(sorted_y) - 1)]
    if not gaps:
        return 0.0
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return 0.0
    gap_std = math.sqrt(sum((g - mean_gap) ** 2 for g in gaps) / len(gaps))
    regular_spacing = gap_std / mean_gap < 0.6

    if x_aligned and regular_spacing:
        return 0.8
    if x_aligned:
        return 0.5
    if regular_spacing:
        return 0.3
    return 0.0


def _is_icon_like(
    vision_candidate: dict[str, Any],
    window_width: int,
    window_height: int,
) -> bool:
    """Check if a vision candidate is icon-like (small, roughly square)."""
    bounds = _get_bounds(vision_candidate)
    if not bounds:
        return False
    bl, bt, br, bb = bounds
    w = br - bl
    h = bb - bt
    if w <= 0 or h <= 0:
        return False
    # Small: less than 5% of window in each dimension
    if w > window_width * 0.05 or h > window_height * 0.05:
        return False
    # Roughly square: aspect ratio < 3
    aspect = max(w, h) / max(min(w, h), 1)
    return aspect < 3.0


def _extract_keyword_hints(
    ocr_blocks: list[dict[str, Any]],
    elements: list[dict[str, Any]],
) -> list[str]:
    """Extract weak keyword hints for evidence (NOT for classification)."""
    hints: list[str] = []

    # OCR text patterns
    for block in ocr_blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        lower = text.lower()
        if any(k in lower for k in ("menu", "菜单", "文件", "编辑", "查看")):
            hints.append(f"menu_keyword:{text[:15]}")
        if any(k in lower for k in ("搜索", "search", "查找")):
            hints.append(f"search_keyword:{text[:15]}")
        if "%" in text or "像素" in text or "pixel" in lower:
            hints.append(f"status_keyword:{text[:15]}")

    # UIA control patterns
    for elem in elements:
        ct = (elem.get("control_type") or "").lower()
        if "toolbar" in ct:
            hints.append("uia_toolbar_control")
        if "statusbar" in ct:
            hints.append("uia_statusbar_control")
        if "menubar" in ct or "menu" in ct:
            hints.append("uia_menu_control")

    return hints[:10]  # limit
