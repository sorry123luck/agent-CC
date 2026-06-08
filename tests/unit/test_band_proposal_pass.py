"""Tests for BandProposalPass (Phase R3.3)."""

import numpy as np
from PIL import Image

from src.perception.band_proposal_pass import BandProposalPass


class TestBandProposalPass:
    def setup_method(self):
        self.pass_ = BandProposalPass()

    def test_empty_screenshot_returns_empty(self):
        diag = self.pass_.detect(screenshot=None)
        assert diag.top_bands_found == 0
        assert diag.bottom_bands_found == 0

    def test_no_ocr_no_elements_returns_empty(self):
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        diag = self.pass_.detect(screenshot=img, window_width=800, window_height=600)
        assert diag.top_bands_found == 0
        assert diag.bottom_bands_found == 0

    def test_top_ocr_creates_top_band(self):
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
            {"text": "编辑", "bbox": (60, 5, 100, 25)},
            {"text": "查看", "bbox": (110, 5, 150, 25)},
        ]
        diag = self.pass_.detect(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=800, window_height=600,
        )
        assert diag.top_bands_found >= 1
        assert diag.proposals[0].band_type == "top_band"
        assert diag.proposals[0].bounds[1] == 0  # starts at y=0

    def test_bottom_ocr_creates_bottom_band(self):
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "100%", "bbox": (700, 580, 750, 600)},
            {"text": "UTF-8", "bbox": (600, 580, 650, 600)},
        ]
        diag = self.pass_.detect(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=800, window_height=600,
        )
        assert diag.bottom_bands_found >= 1
        assert diag.proposals[0].band_type == "bottom_band"
        assert diag.proposals[0].bounds[3] == 600  # ends at window bottom

    def test_top_band_stack_multiple_bands(self):
        """Multiple OCR rows at top should create multiple bands."""
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        # Draw a horizontal line at y=30 to create a boundary
        arr = np.array(img)
        arr[29:31, :, :] = 0  # black line
        img = Image.fromarray(arr)

        ocr_blocks = [
            {"text": "Title", "bbox": (10, 5, 100, 25)},  # row 1
            {"text": "Menu", "bbox": (10, 35, 100, 55)},  # row 2
        ]
        diag = self.pass_.detect(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=800, window_height=600,
        )
        # Should detect at least 1 top band
        assert diag.top_bands_found >= 1

    def test_no_false_band_from_content_text(self):
        """Content text in middle of window should NOT create band."""
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "Hello World", "bbox": (100, 300, 300, 320)},  # center
        ]
        diag = self.pass_.detect(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=800, window_height=600,
        )
        assert diag.top_bands_found == 0
        assert diag.bottom_bands_found == 0

    def test_band_proposal_to_dict(self):
        """BandProposal.to_dict() should be serializable."""
        import json
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        diag = self.pass_.detect(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=800, window_height=600,
        )
        if diag.proposals:
            json.dumps(diag.proposals[0].to_dict())

    def test_uia_elements_create_top_band(self):
        """UIA elements at top should create band even without OCR."""
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        raw_elements = [
            {"element_id": "menu", "control_type": "MenuBar", "bounding_rect": (0, 0, 800, 30)},
        ]
        diag = self.pass_.detect(
            screenshot=img, raw_elements=raw_elements,
            window_width=800, window_height=600,
        )
        assert diag.top_bands_found >= 1

    def test_band_height_within_limits(self):
        """Band height should not exceed MAX_TOP_BAND_RATIO."""
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        # OCR at y=0-150 (25% of window, exceeds 15% limit)
        ocr_blocks = [
            {"text": "text", "bbox": (10, 5, 100, 25)},
            {"text": "text2", "bbox": (10, 50, 100, 70)},
            {"text": "text3", "bbox": (10, 100, 100, 120)},
            {"text": "text4", "bbox": (10, 140, 100, 160)},
        ]
        diag = self.pass_.detect(
            screenshot=img, ocr_blocks=ocr_blocks,
            window_width=800, window_height=600,
        )
        for band in diag.proposals:
            if band.band_type == "top_band":
                assert band.height_ratio <= 0.15


class TestBandProposalIntegration:
    """Test band proposals integrated into overlay builder."""

    def test_overlay_has_band_proposals(self):
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(0, 0, 800, 600))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        assert hasattr(overlay, "band_proposals")
        assert hasattr(overlay, "total_band_proposals")
        assert hasattr(overlay, "band_proposal_audit")

    def test_band_proposals_separate_from_groups(self):
        """Band proposals should NOT affect existing groups."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from src.perception.structured_region import StructureType
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(0, 0, 800, 600))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        # Groups should not be affected by band proposals
        assert overlay.total_groups >= 1
        # Band proposals should be separate
        assert overlay.total_band_proposals >= 0

    def test_overlay_to_dict_contains_band_fields(self):
        """overlay.to_dict() must include band_proposals and band_proposal_audit."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image
        import json

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(0, 0, 800, 600))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        d = overlay.to_dict()
        assert "band_proposals" in d
        assert "band_proposal_audit" in d
        assert "total_band_proposals" in d
        # Must be JSON serializable
        json.dumps(d)

    def test_uia_only_top_menu_not_fp(self):
        """UIA-only top band (menu/toolbar) should NOT be FP."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(0, 0, 800, 600))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        # UIA elements at top, no OCR
        raw_elements = [
            {"element_id": "menu", "control_type": "MenuBar", "bounding_rect": (0, 0, 800, 30)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            raw_elements=raw_elements,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        audit = overlay.band_proposal_audit
        # UIA-only band should be TP or AMB, not FP
        for detail in audit.get("details", []):
            if detail.get("uia_count", 0) >= 1:
                assert detail["verdict"] != "FP", f"UIA-only band {detail['band_id']} should not be FP"

    def test_empty_band_is_fp(self):
        """Band with no OCR, no UIA, no boundary → FP."""
        from src.perception.band_proposal_pass import BandProposalPass, BandProposal
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder

        builder = StructuredRegionOverlayBuilder()
        # Create an empty band proposal
        empty_band = BandProposal(
            band_id="TB_EMPTY",
            band_type="top_band",
            bounds=(0, 0, 800, 20),
            height_px=20,
            height_ratio=0.03,
            detection_method="ocr_density",
            evidence=[],  # No evidence at all
            confidence=0.0,
        )
        audit = builder._audit_band_proposals([empty_band], [], 800, 600)
        assert audit["fp"] == 1
        assert audit["details"][0]["verdict"] == "FP"
        assert audit["details"][0]["reason"] == "no_evidence"

    def test_audit_details_include_all_fields(self):
        """Audit details must include ocr_count, uia_count, detection_method, bounds."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(0, 0, 800, 600))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        audit = overlay.band_proposal_audit
        for detail in audit.get("details", []):
            assert "ocr_count" in detail
            assert "uia_count" in detail
            assert "detection_method" in detail
            assert "bounds" in detail

    def test_over_fragmentation_marks_consecutive_thin_as_amb(self):
        """4+ consecutive thin top bands (<=20px) → marked AMB."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.band_proposal_pass import BandProposal

        builder = StructuredRegionOverlayBuilder()
        bands = []
        for i in range(5):
            bands.append(BandProposal(
                band_id=f"TB{i}",
                band_type="top_band",
                bounds=(0, i * 15, 800, (i + 1) * 15),
                height_px=15,
                height_ratio=0.025,
                detection_method="ocr_density",
                evidence=[f"ocr_count=2", f"boundary_y={(i+1)*15}"],
                confidence=0.5,
            ))
        audit = builder._audit_band_proposals(bands, [], 800, 600)
        over_fragmented = audit.get("over_fragmented", 0)
        assert over_fragmented >= 1
        over_frag_details = [d for d in audit["details"] if "over_fragmented" in d.get("reason", "")]
        assert len(over_frag_details) >= 1

    def test_single_top_menu_band_stays_tp(self):
        """Single top band with content → stays TP."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.band_proposal_pass import BandProposal

        builder = StructuredRegionOverlayBuilder()
        bands = [
            BandProposal(band_id="TB0", band_type="top_band",
                         bounds=(0, 0, 800, 40), height_px=40, height_ratio=0.067,
                         detection_method="ocr_density", evidence=["ocr_count=4"], confidence=0.5),
        ]
        audit = builder._audit_band_proposals(bands, [], 800, 600)
        assert audit["tp"] == 1
        assert audit["over_fragmented"] == 0

    def test_bottom_status_band_stays_tp(self):
        """Bottom band with content → stays TP."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.band_proposal_pass import BandProposal

        builder = StructuredRegionOverlayBuilder()
        bands = [
            BandProposal(band_id="BB0", band_type="bottom_band",
                         bounds=(0, 570, 800, 600), height_px=30, height_ratio=0.05,
                         detection_method="ocr_density", evidence=["ocr_count=2"], confidence=0.5),
        ]
        audit = builder._audit_band_proposals(bands, [], 800, 600)
        assert audit["tp"] == 1
        assert audit["over_fragmented"] == 0

    def test_audit_returns_over_fragmented_count(self):
        """Audit dict should include over_fragmented key."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.band_proposal_pass import BandProposal

        builder = StructuredRegionOverlayBuilder()
        bands = [
            BandProposal(
                band_id="TB0", band_type="top_band",
                bounds=(0, 0, 800, 15), height_px=15, height_ratio=0.025,
                detection_method="ocr_density", evidence=["ocr_count=1"], confidence=0.5,
            )
        ]
        audit = builder._audit_band_proposals(bands, [], 800, 600)
        assert "over_fragmented" in audit

    def test_shadow_combined_count_equals_original_plus_injected(self):
        """combined_region_count == original_region_count + injected_band_count."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
            {"text": "编辑", "bbox": (60, 5, 100, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        assert sc["combined_region_count"] == sc["original_region_count"] + sc["injected_band_count"]

    def test_shadow_no_tp_bands_combined_equals_original(self):
        """When no TP bands, combined_* equals original_*."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        # No OCR → no band proposals → no TP bands
        overlay = builder.build(
            geometric_regions=regions,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        assert sc["injected_band_count"] == 0
        assert sc["combined_region_count"] == sc["original_region_count"]
        assert sc["combined_unknown_count"] == sc["original_unknown_count"]

    def test_shadow_does_not_affect_original_regions(self):
        """Shadow overlay should NOT modify original regions or groups."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        # Original regions should be unchanged
        assert overlay.total_regions == 1
        assert overlay.regions[0].region_id == "R0"

    def test_shadow_skipped_amb_fp_counts(self):
        """Shadow comparison should report skipped AMB/FP band counts."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        assert "skipped_amb_band_count" in sc
        assert "skipped_fp_band_count" in sc
        # injected + skipped_amb + skipped_fp should equal total band proposals
        total_bands = overlay.total_band_proposals
        assert sc["injected_band_count"] + sc["skipped_amb_band_count"] + sc["skipped_fp_band_count"] == total_bands

    def test_injected_band_details_exist_and_match_count(self):
        """injected_band_details should exist and its length == injected_band_count."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
            {"text": "编辑", "bbox": (60, 5, 100, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        details = sc.get("injected_band_details", [])
        assert len(details) == sc.get("injected_band_count", 0), \
            f"details count {len(details)} != injected_band_count {sc.get('injected_band_count', 0)}"

    def test_injected_band_detail_has_required_fields(self):
        """Each injected_band_detail must have all required fields."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        required = {"band_id", "band_type", "bounds", "height_px", "detection_method",
                     "audit_reason", "resulting_structure_type", "roi_eligible",
                     "classification_reason", "confidence"}
        for d in sc.get("injected_band_details", []):
            for field in required:
                assert field in d, f"Missing field '{field}' in injected_band_detail"

    def test_shadow_has_useful_noisy_fields(self):
        """shadow_comparison should include useful/noisy classification."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
            {"text": "编辑", "bbox": (60, 5, 100, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        assert "useful_injected_band_count" in sc
        assert "noisy_injected_band_count" in sc
        assert "useful_injected_band_details" in sc
        assert "noisy_injected_band_details" in sc
        # useful + noisy should equal injected
        assert sc["useful_injected_band_count"] + sc["noisy_injected_band_count"] == sc["injected_band_count"]

    def test_useful_noisy_details_match_counts(self):
        """useful/noisy detail list lengths should match counts."""
        from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
        from src.perception.geometric_partitioner import GeometricRegion
        from PIL import Image

        builder = StructuredRegionOverlayBuilder()
        regions = [GeometricRegion(region_id="R0", bounds=(200, 100, 600, 500))]
        img = Image.new("RGB", (800, 600), (255, 255, 255))
        ocr_blocks = [
            {"text": "文件", "bbox": (10, 5, 50, 25)},
        ]
        overlay = builder.build(
            geometric_regions=regions,
            ocr_blocks=ocr_blocks,
            window_width=800,
            window_height=600,
            screenshot=img,
        )
        sc = overlay.shadow_comparison
        assert len(sc.get("useful_injected_band_details", [])) == sc.get("useful_injected_band_count", 0)
        assert len(sc.get("noisy_injected_band_details", [])) == sc.get("noisy_injected_band_count", 0)
