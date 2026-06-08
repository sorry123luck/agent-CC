"""Tests for RegionGrouper (Phase R3.2)."""

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.region_grouper import RegionGrouper


class TestRegionGrouper:
    def setup_method(self):
        self.grouper = RegionGrouper()

    def test_empty_regions_returns_empty(self):
        groups = self.grouper.group([], 800, 600)
        assert groups == []

    def test_single_region_returns_single_group(self):
        regions = [GeometricRegion(region_id="R0", bounds=(0, 0, 800, 60))]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1
        assert groups[0].child_region_ids == ["R0"]

    def test_adjacent_top_regions_grouped(self):
        """Two adjacent regions in the top band should be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 400, 50)),
            GeometricRegion(region_id="R1", bounds=(400, 0, 800, 50)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1
        assert set(groups[0].child_region_ids) == {"R0", "R1"}
        assert groups[0].band == "top"

    def test_adjacent_left_regions_grouped(self):
        """Two adjacent regions in the left band should be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 200, 300)),
            GeometricRegion(region_id="R1", bounds=(0, 300, 200, 600)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1
        assert set(groups[0].child_region_ids) == {"R0", "R1"}

    def test_top_and_bottom_not_merged(self):
        """Top band and bottom band regions should NOT be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 800, 50)),
            GeometricRegion(region_id="R1", bounds=(0, 550, 800, 600)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 2

    def test_left_and_right_not_merged(self):
        """Left band and right band regions should NOT be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 200, 600)),
            GeometricRegion(region_id="R1", bounds=(600, 0, 800, 600)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 2

    def test_separated_regions_not_merged(self):
        """Regions with a gap > threshold should NOT be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 350, 50)),
            GeometricRegion(region_id="R1", bounds=(450, 0, 800, 50)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 2

    def test_center_regions_with_horizontal_alignment(self):
        """Center regions in the same horizontal row should be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(100, 200, 300, 250)),
            GeometricRegion(region_id="R1", bounds=(310, 200, 500, 250)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1

    def test_center_regions_vertical_alignment(self):
        """Center regions in the same vertical column should be grouped."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(100, 100, 200, 200)),
            GeometricRegion(region_id="R1", bounds=(100, 210, 200, 300)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1

    def test_separator_prevents_grouping(self):
        """Regions with separator evidence should not be grouped with each other."""
        regions = [
            GeometricRegion(
                region_id="R0", bounds=(0, 0, 400, 50),
                boundary_evidence=["vertical_separator_X50%"],
            ),
            GeometricRegion(
                region_id="R1", bounds=(400, 0, 800, 50),
                boundary_evidence=["vertical_separator_X50%"],
            ),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 2

    def test_three_adjacent_top_regions(self):
        """Three adjacent top regions should form one group."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 260, 50)),
            GeometricRegion(region_id="R1", bounds=(260, 0, 530, 50)),
            GeometricRegion(region_id="R2", bounds=(530, 0, 800, 50)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1
        assert len(groups[0].child_region_ids) == 3

    def test_group_bounds_are_merged(self):
        """Group bounds should encompass all child regions."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(10, 5, 400, 50)),
            GeometricRegion(region_id="R1", bounds=(400, 5, 790, 50)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) == 1
        l, t, r, b = groups[0].bounds
        assert l == 10
        assert t == 5
        assert r == 790
        assert b == 50

    def test_group_to_dict(self):
        """Group.to_dict() should be serializable."""
        import json
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 400, 50)),
            GeometricRegion(region_id="R1", bounds=(400, 0, 800, 50)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        json.dumps(groups[0].to_dict())

    def test_mixed_bands_complex(self):
        """Complex case: top bar, side rail, content, status bar."""
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 400, 40)),
            GeometricRegion(region_id="R1", bounds=(400, 0, 800, 40)),
            GeometricRegion(region_id="R2", bounds=(0, 40, 200, 300)),
            GeometricRegion(region_id="R3", bounds=(0, 300, 200, 560)),
            GeometricRegion(region_id="R4", bounds=(200, 40, 800, 560)),
            GeometricRegion(region_id="R5", bounds=(0, 560, 800, 600)),
        ]
        groups = self.grouper.group(regions, 800, 600)
        assert len(groups) >= 3
        top_groups = [g for g in groups if g.band == "top"]
        assert len(top_groups) >= 1
        assert len(top_groups[0].child_region_ids) >= 2
        left_groups = [g for g in groups if g.band == "left"]
        assert len(left_groups) >= 1


class TestRegionGrouperIntegration:
    """Test grouping integrated into overlay builder."""

    def test_overlay_has_groups_field(self):
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        builder = StructuredRegionOverlayBuilder()
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 400, 50)),
            GeometricRegion(region_id="R1", bounds=(400, 0, 800, 50)),
        ]
        overlay = builder.build(geometric_regions=regions, window_width=800, window_height=600)
        assert hasattr(overlay, "groups")
        assert overlay.total_groups >= 1
        assert len(overlay.groups) >= 1

    def test_top_regions_with_horizontal_ocr_classified_as_top_bar(self):
        """Adjacent top regions + horizontal OCR text row → TOP_BAR or TOOLBAR."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.structured_region import StructureType
        builder = StructuredRegionOverlayBuilder()
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 260, 40)),
            GeometricRegion(region_id="R1", bounds=(260, 0, 530, 40)),
            GeometricRegion(region_id="R2", bounds=(530, 0, 800, 40)),
        ]
        ocr_blocks = [
            {"text": "文件", "bbox": (20, 10, 60, 32)},
            {"text": "编辑", "bbox": (80, 11, 120, 33)},
            {"text": "查看", "bbox": (140, 10, 180, 32)},
            {"text": "帮助", "bbox": (200, 12, 240, 34)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
        )
        assert len(overlay.groups) >= 1
        top_groups = [g for g in overlay.groups if g.structure_type in (
            StructureType.TOP_BAR, StructureType.TOOLBAR,
        )]
        assert len(top_groups) >= 1, f"Expected TOP_BAR or TOOLBAR, got {[g.structure_type.value for g in overlay.groups]}"

    def test_bottom_button_group_classified_as_control_strip(self):
        """Bottom region with buttons/icons → CONTROL_STRIP (or MEDIA_CONTROL_BAR if dense)."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.structured_region import StructureType
        builder = StructuredRegionOverlayBuilder()
        # rel_bottom = 600/600 = 1.0 > 0.85 → "bottom"
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 520, 800, 600)),
        ]
        raw_elements = [
            {"element_id": "play", "control_type": "ButtonControl", "bounding_rect": (300, 540, 340, 580)},
            {"element_id": "pause", "control_type": "ButtonControl", "bounding_rect": (350, 540, 390, 580)},
            {"element_id": "stop", "control_type": "ButtonControl", "bounding_rect": (400, 540, 440, 580)},
        ]
        # Use small icon-like vision candidates (20x20 < 5% of 600=30)
        vision_candidates = [
            {"bounding_rect": (305, 545, 325, 565)},
            {"bounding_rect": (355, 545, 375, 565)},
            {"bounding_rect": (405, 545, 425, 565)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            raw_elements=raw_elements,
            vision_candidates=vision_candidates,
            window_width=800,
            window_height=600,
        )
        assert len(overlay.groups) >= 1
        bottom_groups = [g for g in overlay.groups if g.structure_type in (
            StructureType.MEDIA_CONTROL_BAR, StructureType.CONTROL_STRIP,
            StructureType.INPUT_REGION, StructureType.STATUS_REGION,
        )]
        assert len(bottom_groups) >= 1, f"Expected MEDIA/CONTROL_STRIP/INPUT/STATUS, got {[g.structure_type.value for g in overlay.groups]}"

    def test_left_vertical_ocr_classified_as_side_rail_or_list(self):
        """Left region with vertical OCR pattern → SIDE_RAIL or LIST_REGION."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.structured_region import StructureType
        builder = StructuredRegionOverlayBuilder()
        # rel_top = 100/600 = 0.167 > 0.15 → not "top"
        # rel_left = 0/800 = 0.0 < 0.05, width_ratio = 200/800 = 0.25 < 0.35 → "left"
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 100, 200, 500)),
        ]
        ocr_blocks = [
            {"text": "联系人A", "bbox": (20, 120, 180, 145)},
            {"text": "联系人B", "bbox": (20, 160, 180, 185)},
            {"text": "联系人C", "bbox": (20, 200, 180, 225)},
            {"text": "联系人D", "bbox": (20, 240, 180, 265)},
            {"text": "联系人E", "bbox": (20, 280, 180, 305)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
        )
        assert len(overlay.groups) >= 1
        side_groups = [g for g in overlay.groups if g.structure_type in (
            StructureType.SIDE_RAIL, StructureType.LIST_REGION,
        )]
        assert len(side_groups) >= 1, f"Expected SIDE_RAIL or LIST, got {[g.structure_type.value for g in overlay.groups]}"

    def test_top_and_center_not_merged(self):
        """Top band region should NOT merge with center content region."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        builder = StructuredRegionOverlayBuilder()
        regions = [
            GeometricRegion(region_id="R0", bounds=(0, 0, 800, 40)),
            GeometricRegion(region_id="R1", bounds=(0, 40, 800, 600)),
        ]
        overlay = builder.build(geometric_regions=regions, window_width=800, window_height=600)
        assert overlay.total_groups >= 2

    def test_separator_prevents_grouping_in_overlay(self):
        """Regions with separator evidence should not be grouped in overlay."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        builder = StructuredRegionOverlayBuilder()
        regions = [
            GeometricRegion(
                region_id="R0", bounds=(0, 0, 400, 50),
                boundary_evidence=["vertical_separator_X50%"],
            ),
            GeometricRegion(
                region_id="R1", bounds=(400, 0, 800, 50),
                boundary_evidence=["vertical_separator_X50%"],
            ),
        ]
        overlay = builder.build(geometric_regions=regions, window_width=800, window_height=600)
        assert overlay.total_groups >= 2
