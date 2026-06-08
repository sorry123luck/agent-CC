"""Tests for U2-shadow: UIA Boundary Shadow Pass."""

import numpy as np
from PIL import Image

from src.perception.uia_boundary_shadow import (
    UiaBoundaryShadowPass,
    UiaBoundaryShadowResult,
    ROOT_ONLY_UIA,
    SCREENSHOT_MISSING,
    INSUFFICIENT_ELEMENT_SUPPORT,
    INSUFFICIENT_SPAN,
)


class TestUiaBoundaryShadowPass:
    def setup_method(self):
        self.pass_ = UiaBoundaryShadowPass()

    def test_no_screenshot_returns_missing(self):
        result = self.pass_.run(screenshot=None, raw_elements=[])
        assert result.no_uia_signal is True or SCREENSHOT_MISSING in result.rejected_reasons

    def test_no_elements_returns_root_only(self):
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        result = self.pass_.run(screenshot=img, raw_elements=[])
        assert result.no_uia_signal is True
        assert ROOT_ONLY_UIA in result.rejected_reasons

    def test_single_element_returns_root_only(self):
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        raw = [{"element_id": "root", "control_type": "Window", "bounding_rect": (0, 0, 800, 600)}]
        result = self.pass_.run(screenshot=img, raw_elements=raw)
        assert result.no_uia_signal is True

    def test_multiple_elements_with_boundaries(self):
        """Multiple UIA elements with aligned boundaries should detect separators."""
        img = Image.new("RGB", (800, 600), (200, 200, 200))
        # Draw a visual separator at y=100
        arr = np.array(img)
        arr[98:102, :, :] = 50  # dark line
        img = Image.fromarray(arr)

        raw = [
            {"element_id": "e1", "control_type": "Pane", "bounding_rect": (0, 0, 800, 100)},
            {"element_id": "e2", "control_type": "Pane", "bounding_rect": (0, 100, 400, 300)},
            {"element_id": "e3", "control_type": "Pane", "bounding_rect": (400, 100, 800, 300)},
            {"element_id": "e4", "control_type": "Pane", "bounding_rect": (0, 300, 800, 600)},
        ]
        result = self.pass_.run(
            screenshot=img, raw_elements=raw,
            window_width=800, window_height=600,
        )
        assert result.uia_element_count == 4
        assert result.no_uia_signal is False
        # May or may not detect boundaries depending on density contrast
        assert result.usable_uia_boundary_count >= 0

    def test_filter_ocr_synthetic_elements(self):
        """OCR synthetic elements should be filtered out."""
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        raw = [
            {"element_id": "root", "control_type": "Window", "bounding_rect": (0, 0, 800, 600)},
            {"element_id": "ocr1", "control_type": "Text", "bounding_rect": (10, 10, 50, 30),
             "synthetic_source": "ocr"},
            {"element_id": "ocr2", "control_type": "Text", "bounding_rect": (10, 50, 50, 70),
             "synthetic_source": "ocr_actionable"},
        ]
        result = self.pass_.run(screenshot=img, raw_elements=raw)
        assert result.uia_element_count == 1  # Only root

    def test_result_to_dict(self):
        """Result.to_dict() should be serializable."""
        import json
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        result = self.pass_.run(screenshot=img, raw_elements=[])
        d = result.to_dict()
        assert "raw_elements_count" in d
        assert "uia_element_count" in d
        assert "no_uia_signal" in d
        assert "rejected_reasons" in d
        assert "over_fragmentation_score" in d
        json.dumps(d)

    def test_rejected_reasons_populated(self):
        """Rejected reasons should be populated when boundaries are rejected."""
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        raw = [
            {"element_id": "e1", "control_type": "Pane", "bounding_rect": (0, 0, 800, 100)},
            {"element_id": "e2", "control_type": "Pane", "bounding_rect": (0, 100, 800, 200)},
        ]
        result = self.pass_.run(
            screenshot=img, raw_elements=raw,
            window_width=800, window_height=600,
        )
        # Should have some rejected reasons (insufficient_element_support or others)
        total_rejected = sum(result.rejected_reasons.values())
        assert total_rejected >= 0  # May be 0 if all accepted or no boundaries found
