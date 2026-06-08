"""Tests for StructuralClassifier and RegionMetrics (Phase R3.1)."""

import math

from src.perception.structural_classifier import (
    RegionMetrics,
    StructuralClassifier,
    compute_region_metrics,
    _compute_horizontal_row_score,
    _compute_vertical_list_score,
    _is_icon_like,
)
from src.perception.structured_region import StructureType


# =============================================================================
# RegionMetrics tests
# =============================================================================


class TestRegionMetrics:
    def test_default_values(self):
        m = RegionMetrics()
        assert m.position_band == "center"
        assert m.area_ratio == 0.0
        assert m.ocr_count == 0
        assert m.uia_button_count == 0

    def test_compute_basic(self):
        m = compute_region_metrics(
            bounds=(0, 0, 800, 60),
            window_width=800,
            window_height=600,
        )
        assert m.position_band == "top"
        assert m.area_ratio > 0
        assert m.aspect_ratio > 10
        assert m.height_ratio < 0.15

    def test_compute_with_ocr(self):
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 10, 50, 30)},
            {"text": "编辑", "bbox": (60, 10, 100, 30)},
        ]
        m = compute_region_metrics(
            bounds=(0, 0, 200, 100),
            window_width=800,
            window_height=600,
            ocr_blocks=ocr_blocks,
        )
        assert m.ocr_count == 2
        assert m.short_text_count == 2

    def test_compute_with_elements(self):
        raw_elements = [
            {"element_id": "e1", "control_type": "ButtonControl", "bounding_rect": (10, 10, 50, 50)},
            {"element_id": "e2", "control_type": "EditControl", "bounding_rect": (60, 10, 100, 50)},
        ]
        m = compute_region_metrics(
            bounds=(0, 0, 200, 100),
            window_width=800,
            window_height=600,
            raw_elements=raw_elements,
        )
        assert m.uia_button_count == 1
        assert m.uia_edit_count == 1
        assert m.uia_total_count == 2

    def test_compute_with_vision(self):
        vision_candidates = [
            {"bounding_rect": (10, 10, 30, 30)},
            {"bounding_rect": (500, 500, 600, 600)},  # outside
        ]
        m = compute_region_metrics(
            bounds=(0, 0, 200, 100),
            window_width=800,
            window_height=600,
            vision_candidates=vision_candidates,
        )
        assert m.vision_candidate_count == 1

    def test_position_band_top(self):
        m = compute_region_metrics((0, 0, 800, 60), 800, 600)
        assert m.position_band == "top"

    def test_position_band_bottom(self):
        m = compute_region_metrics((0, 540, 800, 600), 800, 600)
        assert m.position_band == "bottom"

    def test_position_band_left(self):
        m = compute_region_metrics((0, 100, 200, 500), 800, 600)
        assert m.position_band == "left"

    def test_position_band_right(self):
        m = compute_region_metrics((600, 100, 800, 500), 800, 600)
        assert m.position_band == "right"

    def test_position_band_center(self):
        m = compute_region_metrics((200, 100, 600, 500), 800, 600)
        assert m.position_band == "center"

    def test_keyword_hints_menu(self):
        ocr_blocks = [{"text": "文件", "bbox": (10, 10, 50, 30)}]
        m = compute_region_metrics(
            (0, 0, 200, 100), 800, 600, ocr_blocks=ocr_blocks,
        )
        assert any("menu_keyword" in h for h in m.keyword_hints)

    def test_keyword_hints_status(self):
        ocr_blocks = [{"text": "100%", "bbox": (10, 10, 50, 30)}]
        m = compute_region_metrics(
            (0, 0, 200, 100), 800, 600, ocr_blocks=ocr_blocks,
        )
        assert any("status_keyword" in h for h in m.keyword_hints)


# =============================================================================
# Horizontal row / vertical list score tests
# =============================================================================


class TestLayoutScores:
    def test_horizontal_row_aligned(self):
        blocks = [
            {"bbox": (10, 20, 50, 40)},
            {"bbox": (60, 22, 100, 42)},
            {"bbox": (110, 18, 150, 38)},
        ]
        score = _compute_horizontal_row_score(blocks, (0, 0, 200, 100))
        assert score >= 0.7

    def test_horizontal_row_scattered(self):
        blocks = [
            {"bbox": (10, 10, 50, 30)},
            {"bbox": (60, 80, 100, 100)},
            {"bbox": (110, 150, 150, 170)},
        ]
        score = _compute_horizontal_row_score(blocks, (0, 0, 200, 200))
        assert score <= 0.3

    def test_vertical_list_aligned(self):
        blocks = [
            {"bbox": (20, 10, 100, 30)},
            {"bbox": (22, 50, 102, 70)},
            {"bbox": (21, 90, 101, 110)},
            {"bbox": (23, 130, 103, 150)},
        ]
        score = _compute_vertical_list_score(blocks, (0, 0, 200, 200))
        assert score >= 0.5

    def test_vertical_list_too_few(self):
        blocks = [
            {"bbox": (10, 10, 50, 30)},
            {"bbox": (10, 50, 50, 70)},
        ]
        score = _compute_vertical_list_score(blocks, (0, 0, 200, 200))
        assert score == 0.0


class TestIconLike:
    def test_small_square_is_icon(self):
        assert _is_icon_like({"bounding_rect": (10, 10, 30, 30)}, 800, 600) is True

    def test_large_rect_not_icon(self):
        assert _is_icon_like({"bounding_rect": (10, 10, 200, 200)}, 800, 600) is False

    def test_wide_strip_not_icon(self):
        assert _is_icon_like({"bounding_rect": (10, 10, 200, 20)}, 800, 600) is False


# =============================================================================
# StructuralClassifier tests
# =============================================================================


class TestStructuralClassifier:
    def setup_method(self):
        self.clf = StructuralClassifier()

    def test_status_region(self):
        m = RegionMetrics(
            position_band="bottom",
            aspect_ratio=10.0,
            height_ratio=0.05,
            short_text_count=2,
            uia_button_count=0,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.STATUS_REGION
        assert conf > 0

    def test_input_region(self):
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.15,
            uia_edit_count=1,
            uia_button_count=1,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.INPUT_REGION

    def test_top_bar(self):
        m = RegionMetrics(
            position_band="top",
            aspect_ratio=8.0,
            height_ratio=0.08,
            horizontal_text_row_score=0.6,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.TOP_BAR

    def test_toolbar(self):
        m = RegionMetrics(
            position_band="top",
            aspect_ratio=4.0,
            height_ratio=0.10,
            uia_button_count=3,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.TOOLBAR

    def test_side_rail(self):
        m = RegionMetrics(
            position_band="left",
            height_ratio=0.70,
            aspect_ratio=0.3,
            vertical_list_score=0.6,
            uia_list_count=2,
            ocr_count=5,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.SIDE_RAIL

    def test_side_rail_requires_vertical_evidence(self):
        """Side rail must have vertical repeated items, not just left+narrow."""
        m = RegionMetrics(
            position_band="left",
            height_ratio=0.70,
            aspect_ratio=0.3,
            vertical_list_score=0.2,  # Low vertical score
            uia_list_count=0,
            ocr_count=2,
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.SIDE_RAIL

    def test_side_rail_requires_content(self):
        """Side rail must have actual OCR content."""
        m = RegionMetrics(
            position_band="left",
            height_ratio=0.70,
            aspect_ratio=0.3,
            vertical_list_score=0.6,
            uia_list_count=2,
            ocr_count=1,  # Too few
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.SIDE_RAIL

    def test_list_region(self):
        m = RegionMetrics(
            vertical_list_score=0.7,
            uia_list_count=3,
            area_ratio=0.15,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.LIST_REGION

    def test_content_stream(self):
        """Dynamic message/chat flow → CONTENT_STREAM."""
        m = RegionMetrics(
            area_ratio=0.25,
            vertical_list_score=0.6,
            ocr_count=10,
            uia_button_count=0,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.CONTENT_STREAM

    def test_content_stream_requires_flow_pattern(self):
        """Content stream must have vertical list pattern + many OCR blocks."""
        m = RegionMetrics(
            area_ratio=0.25,
            vertical_list_score=0.3,  # Low flow pattern
            ocr_count=10,
            uia_button_count=0,
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.CONTENT_STREAM

    def test_document_region(self):
        m = RegionMetrics(
            area_ratio=0.40,
            text_density=8.0,
            uia_button_count=0,
            uia_edit_count=0,
            vertical_list_score=0.2,  # Static, not flow
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.DOCUMENT_REGION

    def test_document_region_excludes_dynamic_stream(self):
        """Dynamic stream with high vertical_list_score should NOT be document."""
        m = RegionMetrics(
            area_ratio=0.40,
            text_density=8.0,
            uia_button_count=0,
            uia_edit_count=0,
            vertical_list_score=0.6,  # High flow → should be content_stream
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.DOCUMENT_REGION

    def test_canvas_region(self):
        m = RegionMetrics(
            area_ratio=0.30,
            text_density=0.5,
            vision_candidate_count=10,
            uia_edit_count=0,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.CANVAS_REGION

    def test_media_control_bar(self):
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            icon_like_count=4,
            uia_button_count=3,
            ocr_count=1,  # Low text
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.MEDIA_CONTROL_BAR

    def test_media_control_bar_requires_icons(self):
        """Media control bar must have actual icons, not just buttons."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            icon_like_count=1,  # Too few icons
            uia_button_count=3,
            ocr_count=1,
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.MEDIA_CONTROL_BAR

    def test_media_control_bar_requires_low_text(self):
        """Media control bar must have low text count."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            icon_like_count=4,
            uia_button_count=3,
            ocr_count=5,  # Too much text for media controls
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.MEDIA_CONTROL_BAR

    def test_control_strip(self):
        """Bottom area with buttons but not media-specific → CONTROL_STRIP."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            icon_like_count=2,
            uia_button_count=3,
            ocr_count=2,  # Low text, but not as dense as media_control_bar
            short_text_count=2,
        )
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.CONTROL_STRIP

    def test_unknown_when_no_signals(self):
        m = RegionMetrics()
        st, conf, reason = self.clf.classify(m)
        assert st == StructureType.UNKNOWN_STRUCTURED
        assert conf == 0.0

    def test_single_menu_keyword_does_not_classify(self):
        """A single '文件' OCR text should NOT determine menu_bar."""
        m = RegionMetrics(
            position_band="center",
            ocr_count=1,
            short_text_count=1,
            keyword_hints=["menu_keyword:文件"],
        )
        st, conf, reason = self.clf.classify(m)
        assert st not in (StructureType.TOP_BAR, StructureType.TOOLBAR)

    def test_status_keyword_does_not_make_media_control(self):
        """Status keyword (100%) should NOT make media_control_bar."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.05,
            aspect_ratio=10.0,
            short_text_count=2,
            uia_button_count=0,
            keyword_hints=["status_keyword:100%"],
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.MEDIA_CONTROL_BAR

    def test_single_status_keyword_does_not_classify(self):
        """A single '100%' OCR text should NOT determine status_region."""
        m = RegionMetrics(
            position_band="center",
            ocr_count=1,
            short_text_count=1,
            keyword_hints=["status_keyword:100%"],
        )
        st, conf, reason = self.clf.classify(m)
        assert st != StructureType.STATUS_REGION

    def test_priority_order(self):
        """Status region check should take priority over toolbar when both match."""
        m = RegionMetrics(
            position_band="bottom",
            aspect_ratio=10.0,
            height_ratio=0.05,
            short_text_count=2,
            uia_button_count=0,
        )
        st, _, _ = self.clf.classify(m)
        assert st == StructureType.STATUS_REGION

    # --- R3.2.3 Regression Tests ---

    def test_control_strip_rejects_content_text(self):
        """Bottom area with long OCR text (content/title) → NOT control_strip."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            aspect_ratio=1.4,
            ocr_count=1,
            short_text_count=0,  # Long text
            icon_like_count=2,
            uia_button_count=0,
            vertical_list_score=0.0,
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.CONTROL_STRIP

    def test_control_strip_rejects_file_list(self):
        """Bottom area with vertical file list → NOT control_strip."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            ocr_count=6,
            short_text_count=0,
            icon_like_count=3,
            uia_button_count=1,
            vertical_list_score=0.8,  # File list pattern
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.CONTROL_STRIP

    def test_control_strip_rejects_high_ocr(self):
        """Bottom area with many OCR blocks → NOT control_strip."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            ocr_count=5,  # Too many
            short_text_count=2,
            icon_like_count=3,
            uia_button_count=2,
            vertical_list_score=0.0,
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.CONTROL_STRIP

    def test_control_strip_accepts_real_controls(self):
        """Bottom area with few short labels + icons → control_strip."""
        m = RegionMetrics(
            position_band="bottom",
            height_ratio=0.10,
            ocr_count=2,
            short_text_count=2,  # All short
            icon_like_count=4,
            uia_button_count=2,
            vertical_list_score=0.0,
        )
        st, _, _ = self.clf.classify(m)
        assert st == StructureType.CONTROL_STRIP

    def test_list_region_rejects_mixed_toolbar_content(self):
        """Area with many buttons + list controls → NOT list_region (mixed)."""
        m = RegionMetrics(
            vertical_list_score=0.6,
            uia_list_count=5,
            uia_button_count=5,  # Too many buttons for pure list
            area_ratio=0.10,
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.LIST_REGION

    def test_list_region_requires_strong_list_evidence(self):
        """Area with weak list evidence → NOT list_region."""
        m = RegionMetrics(
            vertical_list_score=0.6,
            uia_list_count=2,  # Too few list controls
            uia_button_count=0,
            area_ratio=0.10,
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.LIST_REGION

    def test_content_stream_requires_flow_pattern(self):
        """Area with high vertical score but few OCR → NOT content_stream."""
        m = RegionMetrics(
            area_ratio=0.20,
            vertical_list_score=0.6,
            ocr_count=3,  # Too few
            uia_button_count=0,
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.CONTENT_STREAM

    # --- R3.2.4: list_region vs document/editor ---

    def test_document_editor_not_list_region(self):
        """Multiline document/editor content → NOT list_region.

        High text density + many OCR blocks + large area = document, not list.
        """
        m = RegionMetrics(
            vertical_list_score=0.7,  # High (code lines look vertical)
            uia_list_count=5,
            uia_button_count=0,
            area_ratio=0.20,
            ocr_count=20,  # Many lines of code/text
            text_density=15.0,  # High text density
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.LIST_REGION

    def test_list_with_search_box_is_list_region(self):
        """List with EditControl (search box) → still list_region.

        EditControl presence is NOT a veto. Only document-like density excludes.
        """
        m = RegionMetrics(
            vertical_list_score=0.6,
            uia_list_count=4,
            uia_edit_count=1,  # Search box
            uia_button_count=0,
            area_ratio=0.15,
            ocr_count=8,
            text_density=3.0,  # Moderate, not document-like
        )
        st, _, _ = self.clf.classify(m)
        assert st == StructureType.LIST_REGION

    def test_large_editor_area_not_list_region(self):
        """Large area with high text density + moderate list controls → document."""
        m = RegionMetrics(
            vertical_list_score=0.6,
            uia_list_count=4,  # Some list controls (line numbers?)
            uia_button_count=0,
            area_ratio=0.25,  # Large
            ocr_count=12,
            text_density=6.0,  # High
        )
        st, _, _ = self.clf.classify(m)
        assert st != StructureType.LIST_REGION
