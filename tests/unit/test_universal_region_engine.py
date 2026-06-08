"""Tests for UniversalRegionEngine P0: Region Evidence Model."""

import json
from unittest.mock import MagicMock

import pytest

from src.perception.region_evidence_model import (
    BoundaryEvidence,
    PageRegion,
    RegionSemanticHint,
    RegionSubstructure,
    SKELETON_SOURCE_BAND_PROPOSAL,
    SKELETON_SOURCE_FULL_WINDOW,
    SKELETON_SOURCE_GEOMETRIC,
    SKELETON_SOURCE_STRUCTURED_GROUP,
    SKELETON_SOURCE_U4_CANDIDATE,
    compute_evidence_sources,
    compute_evidence_summary,
)
from src.perception.universal_region_engine import UniversalRegionEngine


class TestRegionEvidenceModel:
    def test_page_region_to_dict(self):
        pr = PageRegion(
            region_id="PR0",
            bounds=(0, 0, 100, 100),
            region_type="unknown",
            confidence=0.0,
            skeleton_source=SKELETON_SOURCE_GEOMETRIC,
            evidence_sources=["ocr", "uia"],
            area_ratio=0.5,
        )
        d = pr.to_dict()
        assert d["region_id"] == "PR0"
        assert d["region_type"] == "unknown"
        assert d["confidence"] == 0.0
        assert d["skeleton_source"] == "geometric_region"
        assert d["evidence_sources"] == ["ocr", "uia"]
        assert d["area_ratio"] == 0.5

    def test_substructure_to_dict(self):
        s = RegionSubstructure(
            sub_id="S0",
            sub_type="text_row",
            bounds=(10, 10, 90, 30),
            evidence_sources=["ocr"],
            confidence=0.7,
            text="hello",
        )
        d = s.to_dict()
        assert d["sub_id"] == "S0"
        assert d["sub_type"] == "text_row"
        assert d["text"] == "hello"

    def test_boundary_evidence_to_dict(self):
        b = BoundaryEvidence(
            boundary_id="B0",
            direction="horizontal",
            position=100,
            evidence_type="whitespace_gap",
            strength=0.8,
            source="pixel",
        )
        d = b.to_dict()
        assert d["boundary_id"] == "B0"
        assert d["direction"] == "horizontal"
        assert d["position"] == 100

    def test_semantic_hint_to_dict(self):
        h = RegionSemanticHint(
            hint_id="H0",
            source="roi_vlm",
            label="toolbar",
            confidence=0.9,
        )
        d = h.to_dict()
        assert d["hint_id"] == "H0"
        assert d["label"] == "toolbar"

    def test_compute_evidence_sources(self):
        assert compute_evidence_sources(5, 3, 2, True) == ["ocr", "vision", "uia", "pixel"]
        assert compute_evidence_sources(0, 0, 0, False) == []
        assert compute_evidence_sources(1, 0, 0, False) == ["ocr"]
        assert compute_evidence_sources(0, 1, 0, False) == ["vision"]

    def test_compute_evidence_summary(self):
        s = compute_evidence_summary(5, 3, 2)
        assert s == {"ocr_count": 5, "vision_count": 3, "uia_count": 2}

    def test_page_region_serializable(self):
        pr = PageRegion(
            region_id="PR0",
            bounds=(0, 0, 100, 100),
            region_type="unknown",
            confidence=0.0,
            skeleton_source=SKELETON_SOURCE_GEOMETRIC,
            evidence_sources=["ocr"],
            substructures=[RegionSubstructure("S0", "text_row", (10, 10, 90, 30), ["ocr"], 0.7)],
            boundary_evidence=[BoundaryEvidence("B0", "horizontal", 50, "whitespace_gap", 0.8, "pixel")],
            semantic_hints=[],
            area_ratio=0.5,
            evidence_summary={"ocr_count": 1, "vision_count": 0, "uia_count": 0},
        )
        d = pr.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert "PR0" in json_str
        assert "text_row" in json_str


class TestUniversalRegionEngine:
    def setup_method(self):
        self.engine = UniversalRegionEngine()

    def test_empty_input_returns_full_window_fallback(self):
        result = self.engine.build(
            geometric_regions=[],
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 1
        pr = result["page_regions"][0]
        assert pr["skeleton_source"] == SKELETON_SOURCE_FULL_WINDOW
        assert pr["bounds"] == [0, 0, 800, 600]
        assert pr["region_type"] == "unknown"

    def test_zero_window_size_returns_empty_with_reason(self):
        result = self.engine.build(
            geometric_regions=[],
            window_width=0,
            window_height=0,
        )
        assert result["total_regions"] == 0
        assert result["page_regions"] == []
        assert result["excluded_reason"] == "zero_window_size"

    def test_single_geometric_region_becomes_page_region(self):
        gr = MagicMock()
        gr.bounds = (10, 20, 300, 400)
        result = self.engine.build(
            geometric_regions=[gr],
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 1
        pr = result["page_regions"][0]
        assert pr["skeleton_source"] == SKELETON_SOURCE_GEOMETRIC
        assert pr["bounds"] == [10, 20, 300, 400]

    def test_multiple_geometric_regions_become_page_regions(self):
        gr1 = MagicMock()
        gr1.bounds = (0, 0, 400, 300)
        gr2 = MagicMock()
        gr2.bounds = (400, 0, 800, 300)
        result = self.engine.build(
            geometric_regions=[gr1, gr2],
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 2
        assert result["page_regions"][0]["skeleton_source"] == SKELETON_SOURCE_GEOMETRIC
        assert result["page_regions"][1]["skeleton_source"] == SKELETON_SOURCE_GEOMETRIC

    def test_structured_groups_as_skeleton_source(self):
        groups = [
            {"bounds": [0, 0, 400, 300]},
            {"bounds": [400, 0, 800, 300]},
        ]
        result = self.engine.build(
            geometric_regions=[],
            structured_overlay={"groups": groups},
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 2
        assert result["page_regions"][0]["skeleton_source"] == SKELETON_SOURCE_STRUCTURED_GROUP

    def test_band_proposals_as_skeleton_source(self):
        bands = [
            {"bounds": [0, 0, 800, 50]},
            {"bounds": [0, 550, 800, 600]},
        ]
        result = self.engine.build(
            geometric_regions=[],
            band_proposals=bands,
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 2
        assert result["page_regions"][0]["skeleton_source"] == SKELETON_SOURCE_BAND_PROPOSAL

    def test_u4_candidates_as_optional_skeleton_source(self):
        u4 = {
            "candidate_regions": [
                {"bounds": [0, 0, 400, 300], "ocr_count": 5, "vision_count": 3, "uia_count": 2},
            ],
        }
        result = self.engine.build(
            geometric_regions=[],
            u4_layout=u4,
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 1
        pr = result["page_regions"][0]
        assert pr["skeleton_source"] == SKELETON_SOURCE_U4_CANDIDATE
        assert pr["evidence_sources"] == ["ocr", "vision", "uia"]

    def test_u4_not_available_still_works(self):
        result = self.engine.build(
            geometric_regions=[],
            u4_layout=None,
            window_width=800,
            window_height=600,
        )
        assert result["total_regions"] == 1
        assert result["page_regions"][0]["skeleton_source"] == SKELETON_SOURCE_FULL_WINDOW

    def test_substructures_from_ocr_vision_uia(self):
        gr = MagicMock()
        gr.bounds = (0, 0, 800, 600)
        ocr = [{"text": "Hello", "bbox": [10, 10, 100, 30]}]
        vis = [{"label": "button", "bounding_rect": [200, 200, 250, 250]}]
        uia = [{"control_type": "Button", "bounding_rect": [300, 300, 400, 350]}]
        result = self.engine.build(
            geometric_regions=[gr],
            ocr_blocks=ocr,
            vision_candidates=vis,
            raw_elements=uia,
            window_width=800,
            window_height=600,
        )
        pr = result["page_regions"][0]
        assert len(pr["substructures"]) == 3
        types = {s["sub_type"] for s in pr["substructures"]}
        assert "text_row" in types or "list_item_like" in types
        assert "icon_candidate" in types or "control_candidate" in types

    def test_skeleton_source_field(self):
        gr = MagicMock()
        gr.bounds = (0, 0, 400, 300)
        result = self.engine.build(geometric_regions=[gr], window_width=800, window_height=600)
        pr = result["page_regions"][0]
        assert "skeleton_source" in pr
        assert pr["skeleton_source"] == SKELETON_SOURCE_GEOMETRIC

    def test_evidence_sources_field(self):
        gr = MagicMock()
        gr.bounds = (0, 0, 400, 300)
        result = self.engine.build(geometric_regions=[gr], window_width=800, window_height=600)
        pr = result["page_regions"][0]
        assert "evidence_sources" in pr
        assert isinstance(pr["evidence_sources"], list)

    def test_region_type_always_unknown_in_p0(self):
        gr = MagicMock()
        gr.bounds = (0, 0, 400, 300)
        result = self.engine.build(geometric_regions=[gr], window_width=800, window_height=600)
        for pr in result["page_regions"]:
            assert pr["region_type"] == "unknown"

    def test_semantic_hints_empty_in_p0(self):
        gr = MagicMock()
        gr.bounds = (0, 0, 400, 300)
        result = self.engine.build(geometric_regions=[gr], window_width=800, window_height=600)
        for pr in result["page_regions"]:
            assert pr["semantic_hints"] == []
