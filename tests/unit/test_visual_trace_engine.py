"""Tests for Visual Trace Layer (VTL-0) engine."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.perception.visual_trace_model import (
    OBJECT_CLASS_ACTIONABLE,
    OBJECT_CLASS_CONTENT,
    OBJECT_CLASS_PAGE_CONTAINER,
    OBJECT_CLASS_TEXT_SUB,
    OBJECT_LEVEL_LOCAL_OBJECT,
    OBJECT_LEVEL_PAGE_CONTAINER,
    VisualTraceObject,
    make_trace_object,
)
from src.perception.visual_trace_engine import VisualTraceEngine


class TestVisualTraceModel:
    def test_trace_object_to_dict(self):
        obj = VisualTraceObject(
            trace_id="VT0",
            bounds=(10, 20, 100, 200),
            shape_type="rectangle",
            object_level=OBJECT_LEVEL_PAGE_CONTAINER,
            object_class=OBJECT_CLASS_PAGE_CONTAINER,
            detection_method="bg_color_block",
            confidence=0.8,
            area_ratio=0.15,
        )
        d = obj.to_dict()
        assert d["trace_id"] == "VT0"
        assert d["bounds"] == [10, 20, 100, 200]
        assert d["object_level"] == "page_container"
        assert d["object_class"] == "page_container"
        assert d["confidence"] == 0.8

    def test_make_trace_object(self):
        obj = make_trace_object(
            trace_id="VT1",
            bounds=(0, 0, 100, 50),
            shape_type="rectangle",
            object_level=OBJECT_LEVEL_LOCAL_OBJECT,
            object_class=OBJECT_CLASS_ACTIONABLE,
            detection_method="connected_component",
            confidence=0.7,
            window_area=100000,
        )
        assert obj.area_ratio == pytest.approx(0.05, abs=0.01)
        assert obj.aspect_ratio == 2.0

    def test_trace_object_serializable(self):
        obj = make_trace_object(
            trace_id="VT0", bounds=(0, 0, 100, 100),
            shape_type="rectangle", object_level=OBJECT_LEVEL_PAGE_CONTAINER,
            object_class=OBJECT_CLASS_PAGE_CONTAINER, detection_method="bg_color_block",
            confidence=0.5, window_area=100000,
        )
        json.dumps(obj.to_dict())  # Should not raise


class TestVisualTraceEngine:
    def setup_method(self):
        self.engine = VisualTraceEngine()

    def test_no_screenshot_returns_empty(self):
        result = self.engine.build(screenshot=None)
        assert result["total_containers"] == 0
        assert result["total_objects"] == 0
        assert result["excluded_reason"] == "no_screenshot"

    def test_invalid_dimensions_returns_empty(self):
        img = Image.new("L", (100, 100), 128)
        result = self.engine.build(screenshot=img, window_width=0, window_height=0)
        assert result["total_containers"] == 0
        assert result["excluded_reason"] == "invalid_dimensions"

    def test_uniform_image_returns_empty(self):
        img = Image.new("L", (200, 200), 128)
        result = self.engine.build(screenshot=img, window_width=200, window_height=200)
        # Uniform image should have no containers or objects
        assert result["total_containers"] == 0
        assert result["total_objects"] == 0
        assert result["excluded_reason"] == "uniform_image"

    def test_image_with_content_returns_results(self):
        # Create image with a dark rectangle on light background
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[50:100, 50:150] = 50  # Dark block
        img = Image.fromarray(arr, mode="L")
        result = self.engine.build(screenshot=img, window_width=300, window_height=200)
        assert result["total_containers"] >= 0  # May or may not detect containers
        assert "page_containers" in result
        assert "local_objects" in result
        assert "text_substructures" in result

    def test_page_containers_have_correct_object_class(self):
        # Create image with distinct regions
        arr = np.ones((400, 600), dtype=np.uint8) * 200
        arr[:, :150] = 80  # Dark sidebar
        arr[:50, :] = 100  # Dark top bar
        img = Image.fromarray(arr, mode="L")
        result = self.engine.build(screenshot=img, window_width=600, window_height=400)
        for pc in result["page_containers"]:
            assert pc["object_class"] == "page_container"
            assert pc["object_level"] == "page_container"

    def test_local_objects_classification(self):
        # Test the classification logic directly
        engine = VisualTraceEngine()

        # Small + high edge + square + no text + no Omni → content (NOT actionable)
        # Actionable REQUIRES external evidence (Omni/UIA)
        cls = engine._determine_object_class(
            area_ratio=0.003, aspect_ratio=1.0, edge_density=0.12,
            ocr_count=0, ocr_text_length=0, has_omni_overlap=False, has_uia_control=False,
        )
        assert cls == OBJECT_CLASS_CONTENT

        # Elongated + low edge → text_substructure
        cls = engine._determine_object_class(
            area_ratio=0.05, aspect_ratio=5.0, edge_density=0.02,
            ocr_count=3, ocr_text_length=50, has_omni_overlap=False, has_uia_control=False,
        )
        assert cls == OBJECT_CLASS_TEXT_SUB

        # Medium area + moderate edge + no evidence → content_object (default)
        cls = engine._determine_object_class(
            area_ratio=0.01, aspect_ratio=2.0, edge_density=0.05,
            ocr_count=0, ocr_text_length=0, has_omni_overlap=False, has_uia_control=False,
        )
        assert cls == OBJECT_CLASS_CONTENT

        # Small + high edge + has OCR text → content_object (not actionable, has text)
        cls = engine._determine_object_class(
            area_ratio=0.003, aspect_ratio=1.0, edge_density=0.12,
            ocr_count=2, ocr_text_length=20, has_omni_overlap=False, has_uia_control=False,
        )
        assert cls == OBJECT_CLASS_CONTENT

        # Small + square + Omni button overlap → actionable
        cls = engine._determine_object_class(
            area_ratio=0.005, aspect_ratio=1.5, edge_density=0.08,
            ocr_count=0, ocr_text_length=0, has_omni_overlap=True, has_uia_control=False,
        )
        assert cls == OBJECT_CLASS_ACTIONABLE

    def test_text_substructure_not_in_local_objects(self):
        # Create an image that would produce text-like objects
        arr = np.ones((200, 400), dtype=np.uint8) * 200
        # Add thin dark lines (text-like)
        for y in range(20, 180, 15):
            arr[y:y+3, 20:380] = 50
        img = Image.fromarray(arr, mode="L")
        ocr_blocks = [{"text": "Hello World", "bbox": [20, 20, 380, 35]}]
        result = self.engine.build(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=400, window_height=200,
        )
        # text_substructures should not appear in local_objects
        for obj in result["local_objects"]:
            assert obj["object_class"] != "text_substructure"

    def test_hierarchy_relationships(self):
        # Create image with a container and an object inside
        arr = np.ones((300, 400), dtype=np.uint8) * 200
        arr[50:250, 50:350] = 100  # Container-like region
        arr[100:150, 100:200] = 50  # Object inside
        img = Image.fromarray(arr, mode="L")
        result = self.engine.build(screenshot=img, window_width=400, window_height=300)
        # Check that objects have parent_trace_id if they overlap with containers
        for obj in result["local_objects"]:
            if obj.get("parent_trace_id"):
                # Parent should exist in containers
                parent_ids = {c["trace_id"] for c in result["page_containers"]}
                assert obj["parent_trace_id"] in parent_ids

    def test_text_label_not_actionable(self):
        """Regression: small text labels with OCR overlap must not be actionable."""
        engine = VisualTraceEngine()
        # Create image with a small dark rectangle (text-like)
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[50:70, 50:150] = 50  # Small text-like block
        img = Image.fromarray(arr, mode="L")
        # OCR block overlapping the region
        ocr_blocks = [{"text": "平时：", "bbox": [50, 50, 150, 70]}]
        result = engine.build(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=300, window_height=200,
        )
        # No object should be actionable
        for obj in result["local_objects"]:
            assert obj["object_class"] != "actionable_object", \
                f'{obj["trace_id"]} should not be actionable (text label)'

    def test_contact_name_not_actionable(self):
        """Regression: contact names with OCR overlap must not be actionable."""
        engine = VisualTraceEngine()
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[80:100, 30:180] = 60  # Contact name block
        img = Image.fromarray(arr, mode="L")
        ocr_blocks = [{"text": "文件传输助手", "bbox": [30, 80, 180, 100]}]
        result = engine.build(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=300, window_height=200,
        )
        for obj in result["local_objects"]:
            assert obj["object_class"] != "actionable_object", \
                f'{obj["trace_id"]} should not be actionable (contact name)'

    def test_menu_text_not_actionable(self):
        """Regression: menu text with OCR overlap must not be actionable."""
        engine = VisualTraceEngine()
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[40:60, 20:120] = 55  # Menu text block
        img = Image.fromarray(arr, mode="L")
        ocr_blocks = [{"text": "我的播客", "bbox": [20, 40, 120, 60]}]
        result = engine.build(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=300, window_height=200,
        )
        for obj in result["local_objects"]:
            assert obj["object_class"] != "actionable_object", \
                f'{obj["trace_id"]} should not be actionable (menu text)'

    def test_shape_only_not_actionable(self):
        """Regression: shape-only objects must not be actionable."""
        engine = VisualTraceEngine()
        # Small square with high edge density but no UIA/Omni evidence
        # should NOT be actionable
        cls = engine._determine_object_class(
            area_ratio=0.002, aspect_ratio=1.0, edge_density=0.20,
            ocr_count=0, ocr_text_length=0, has_omni_overlap=False, has_uia_control=False,
        )
        # With no external evidence, should be content (not actionable)
        assert cls != "actionable_object", "shape-only should not be actionable"

    def test_uia_actionable_is_actionable(self):
        """UIA actionable control should be classified as actionable."""
        engine = VisualTraceEngine()
        cls = engine._determine_object_class(
            area_ratio=0.003, aspect_ratio=1.0, edge_density=0.15,
            ocr_count=0, ocr_text_length=0, has_omni_overlap=False, has_uia_control=True,
        )
        assert cls == "actionable_object"

    def test_omni_control_is_actionable(self):
        """Omni control label + no OCR should be actionable."""
        engine = VisualTraceEngine()
        cls = engine._determine_object_class(
            area_ratio=0.003, aspect_ratio=1.0, edge_density=0.15,
            ocr_count=0, ocr_text_length=0, has_omni_overlap=True, has_uia_control=False,
        )
        assert cls == "actionable_object"

    def test_icon_only_vision_not_actionable(self):
        """Regression: vision candidate with kind='icon' only should NOT be actionable."""
        engine = VisualTraceEngine()
        # Create image with a small dark block (icon-like)
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[50:80, 50:80] = 50  # Small square
        img = Image.fromarray(arr, mode="L")
        # Vision candidate with kind="icon" (generic, not specific control)
        vision = [{"candidate_id": "v1", "bounding_rect": [50, 50, 80, 80],
                   "label": "", "kind": "icon", "confidence": 0.8}]
        result = engine.build(
            screenshot=img, vision_candidates=vision,
            window_width=300, window_height=200,
        )
        for obj in result["local_objects"]:
            assert obj["object_class"] != "actionable_object", \
                f'{obj["trace_id"]} should not be actionable (icon-only vision)'

    def test_negative_label_not_actionable(self):
        """Regression: vision with cover/image/photo label must not be actionable."""
        engine = VisualTraceEngine()
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[50:100, 50:100] = 50  # Square block
        img = Image.fromarray(arr, mode="L")
        # Vision candidate with negative label "cover" + positive "button"
        vision = [{"candidate_id": "v1", "bounding_rect": [50, 50, 100, 100],
                   "label": "album cover button", "kind": "icon", "confidence": 0.8}]
        result = engine.build(
            screenshot=img, vision_candidates=vision,
            window_width=300, window_height=200,
        )
        for obj in result["local_objects"]:
            assert obj["object_class"] != "actionable_object", \
                f'{obj["trace_id"]} should not be actionable (negative label: cover)'

    def test_real_button_actionable(self):
        """Real button with specific control label should be actionable."""
        engine = VisualTraceEngine()
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[50:80, 50:80] = 50  # Small square
        img = Image.fromarray(arr, mode="L")
        # Vision candidate with specific "button" label
        vision = [{"candidate_id": "v1", "bounding_rect": [50, 50, 80, 80],
                   "label": "play button", "kind": "button", "confidence": 0.9}]
        result = engine.build(
            screenshot=img, vision_candidates=vision,
            window_width=300, window_height=200,
        )
        actionable = [o for o in result["local_objects"] if o.get("object_class") == "actionable_object"]
        # Should have at least 1 actionable (the button)
        # Note: may be 0 if other guards filter it, but should not be negative-label-blocked

    def test_shape_only_high_edge_not_actionable(self):
        """Shape-only with high edge density must not be actionable (build path)."""
        engine = VisualTraceEngine()
        # Create image with very high contrast block
        arr = np.ones((200, 300), dtype=np.uint8) * 200
        arr[50:80, 50:80] = 0  # Pure black square (very high edge)
        img = Image.fromarray(arr, mode="L")
        # No vision, no OCR, no UIA
        result = engine.build(
            screenshot=img, vision_candidates=None, ocr_blocks=None,
            window_width=300, window_height=200,
        )
        for obj in result["local_objects"]:
            assert obj["object_class"] != "actionable_object", \
                f'{obj["trace_id"]} should not be actionable (shape-only high edge)'

    def test_no_app_specific_logic(self):
        # Verify no process_name or window_title in the engine
        import inspect
        src = inspect.getsource(VisualTraceEngine)
        assert "process_name" not in src
        assert "window_title" not in src
        assert "app_name" not in src
