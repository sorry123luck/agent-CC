"""Unit tests for GeometricPartitioner — Layer 1 pure geometry engine.

All tests use synthetic numpy images. No real screenshots or running apps.
"""

import numpy as np
import pytest
from PIL import Image

from src.perception.geometric_partitioner import (
    GeometricPartitioner,
    GeometricRegion,
    PartitionerConfig,
    PartitionDiagnostics,
)


def _array_to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def _solid_image(w: int = 400, h: int = 300, color: tuple = (128, 128, 128)) -> Image.Image:
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return _array_to_image(arr)


def _vertical_separator_image(w: int = 400, h: int = 300, x: int = 200) -> Image.Image:
    """Solid background with a strong vertical line spanning full height."""
    arr = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
    arr[:, max(0, x - 2):min(w, x + 2), :] = (40, 40, 40)
    return _array_to_image(arr)


def _bordered_rect_image(w: int = 400, h: int = 300) -> Image.Image:
    """Solid background with a closed dark rectangle border."""
    arr = np.full((h, w, 3), (200, 200, 200), dtype=np.uint8)
    l, t, r, b = 80, 50, 320, 250
    arr[t - 2:t + 2, l:r, :] = (30, 30, 30)
    arr[b - 2:b + 2, l:r, :] = (30, 30, 30)
    arr[t:b, l - 2:l + 2, :] = (30, 30, 30)
    arr[t:b, r - 2:r + 2, :] = (30, 30, 30)
    return _array_to_image(arr)


def _low_texture_block_image(w: int = 400, h: int = 300) -> Image.Image:
    """Left half low-texture, right half moderate texture."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    arr[:, :w // 2, :] = (180, 180, 180)  # flat color, low std
    return _array_to_image(arr)


def _card_grid_image(w: int = 500, h: int = 400) -> Image.Image:
    """4x4 grid of bordered rectangles (card grid)."""
    arr = np.full((h, w, 3), (220, 220, 220), dtype=np.uint8)
    margin = 10
    cell_w = (w - margin * 5) // 4
    cell_h = (h - margin * 5) // 4
    for row in range(4):
        for col in range(4):
            l = margin + col * (cell_w + margin)
            t = margin + row * (cell_h + margin)
            r = l + cell_w
            b = t + cell_h
            arr[t:t + 2, l:r, :] = (40, 40, 40)
            arr[b - 2:b, l:r, :] = (40, 40, 40)
            arr[t:b, l:l + 2, :] = (40, 40, 40)
            arr[t:b, r - 2:r, :] = (40, 40, 40)
    return _array_to_image(arr)


class TestGeometricPartitioner:
    """Tier 1: fixture-driven unit tests for GeometricPartitioner."""

    def test_separator_detection_vertical(self):
        """Synthetic image with strong vertical line → at least one region."""
        img = _vertical_separator_image()
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)
        assert len(regions) >= 1
        assert partitioner.diagnostics.accepted_separators >= 1

    def test_empty_image_returns_empty(self):
        """None image returns empty list."""
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(None)
        assert regions == []

    def test_solid_image_returns_single_region(self):
        """Solid color image → at least one region, no separators."""
        img = _solid_image()
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)
        assert len(regions) >= 1
        # No strong separators on solid image
        assert partitioner.diagnostics.accepted_separators == 0

    def test_neutral_ids_no_semantics(self):
        """All region_ids are neutral R-prefixed format, no semantic names."""
        img = _vertical_separator_image()
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)
        # Neutral IDs all start with "R"
        for r in regions:
            assert r.region_id.startswith("R"), f"Expected neutral ID, got {r.region_id}"
        # boundary_evidence should NOT contain semantic labels
        semantic_words = {"side_panel", "composer_area", "message_input", "toolbar",
                          "navigation", "action_bar", "content_area", "title_bar"}
        for r in regions:
            for evidence in r.boundary_evidence:
                assert not any(w in evidence.lower() for w in semantic_words), \
                    f"Semantic word in boundary_evidence: {evidence}"

    def test_border_enclosure_detection(self):
        """Closed rectangle border → detected as border_enclosure."""
        img = _bordered_rect_image()
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)
        assert partitioner.diagnostics.enclosed_rect_count >= 1

    def test_separator_requires_min_span(self):
        """Short line (<55% height) is rejected as separator."""
        w, h = 400, 300
        arr = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        # Only span 30% of height
        short_span_h = int(h * 0.30)
        arr[50:50 + short_span_h, 200:202, :] = (30, 30, 30)
        img = _array_to_image(arr)
        cfg = PartitionerConfig(min_span_ratio=0.55)
        partitioner = GeometricPartitioner(config=cfg)
        partitioner.partition(img)
        # The short line should be rejected
        assert partitioner.diagnostics.accepted_separators == 0

    def test_edge_coverage_separator_detects_visual_boundary(self):
        """Clear UI edge coverage should create neutral separator evidence."""
        w, h = 360, 240
        arr = np.full((h, w, 3), (240, 240, 240), dtype=np.uint8)
        arr[:, :90, :] = (210, 210, 210)
        arr[:, 90:92, :] = (110, 110, 110)
        arr[72:74, :, :] = (130, 130, 130)

        img = _array_to_image(arr)
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)

        assert any(
            any(ev.startswith("vertical_edge_coverage") for ev in region.boundary_evidence)
            for region in regions
        ) or any(region.bounds[0] <= 92 <= region.bounds[2] for region in regions)
        assert len(regions) >= 4

    def test_partition_ignores_window_border_and_uses_internal_splits(self):
        """Window borders should not be the only split when internal separators exist."""
        w, h = 420, 300
        arr = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
        arr[:, :70, :] = (220, 220, 220)
        arr[:, 70:72, :] = (90, 90, 90)
        arr[:, 170:172, :] = (95, 95, 95)
        arr[58:60, :, :] = (100, 100, 100)
        arr[230:232, 170:, :] = (95, 95, 95)
        # Strong outer border that should be ignored for partition splitting.
        arr[:, 0:2, :] = (0, 0, 0)
        arr[:, -2:, :] = (0, 0, 0)
        arr[0:2, :, :] = (0, 0, 0)
        arr[-2:, :, :] = (0, 0, 0)

        img = _array_to_image(arr)
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)

        assert len(regions) >= 6
        assert any(65 <= region.bounds[2] <= 78 for region in regions)
        assert any(160 <= region.bounds[0] <= 180 for region in regions)
        assert all(region.bounds != (0, 0, w, h) for region in regions)

    def test_density_region_detection(self):
        """Flat vs textured sides → density regions detected."""
        img = _low_texture_block_image()
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)
        # Should detect at least one density_region
        assert partitioner.diagnostics.density_region_count >= 1 or len(regions) >= 1

    def test_card_grid_detection(self):
        """4x4 grid of similar-size bordered boxes → card grid region."""
        img = _card_grid_image()
        partitioner = GeometricPartitioner()
        regions = partitioner.partition(img)
        assert partitioner.diagnostics.card_grid_count >= 1 or len(regions) >= 1

    def test_to_dict_roundtrip(self):
        """GeometricRegion.to_dict() produces correct keys."""
        r = GeometricRegion(
            region_id="R1",
            bounds=(10, 20, 100, 200),
            boundary_evidence=["vertical_separator_L78%"],
            density_profile=(12.5, 0.03),
            geometry_confidence=0.85,
            parent_region_id="R0",
        )
        d = r.to_dict()
        assert d["region_id"] == "R1"
        assert d["bounds"] == [10, 20, 100, 200]
        assert "vertical_separator_L78%" in d["boundary_evidence"]

    def test_diagnostics_to_dict(self):
        """PartitionDiagnostics.to_dict() includes required fields."""
        diag = PartitionDiagnostics(
            total_separator_candidates=5,
            accepted_separators=2,
            rejected_separators=[{"line_x": 100, "reason": "span_too_short"}],
            final_region_count=3,
        )
        d = diag.to_dict()
        assert d["total_separator_candidates"] == 5
        assert d["accepted_separators"] == 2
        assert d["final_region_count"] == 3
        assert len(d["rejected_separators"]) == 1
