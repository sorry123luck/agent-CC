"""Tests for evidence_fusion — ConfidenceProfile 计算。"""

import pytest

from src.memory.evidence_fusion import (
    ConfidenceProfile,
    uia_structure_score,
    compute_from_evidence,
)


class TestConfidenceProfile:
    def test_empty_evidence_returns_zero(self):
        p = ConfidenceProfile()
        assert p.fused_confidence == 0.0

    def test_all_dimensions_max(self):
        p = ConfidenceProfile(
            spatial_confidence=1.0,
            semantic_confidence=1.0,
            text_confidence=1.0,
            structure_confidence=1.0,
            visual_anchor_confidence=1.0,
            memory_confidence=1.0,
            action_confidence=1.0,
        )
        assert p.fused_confidence == 1.0

    def test_spatial_only(self):
        p = ConfidenceProfile(spatial_confidence=1.0)
        assert abs(p.fused_confidence - 0.24) < 0.001

    def test_semantic_only(self):
        p = ConfidenceProfile(semantic_confidence=1.0)
        assert abs(p.fused_confidence - 0.24) < 0.001

    def test_conflict_penalty_reduces(self):
        p = ConfidenceProfile(spatial_confidence=1.0, conflict_penalty=0.1)
        assert abs(p.fused_confidence - 0.14) < 0.001

    def test_fused_clamped_at_1(self):
        p = ConfidenceProfile(
            spatial_confidence=1.0,
            semantic_confidence=1.0,
            conflict_penalty=-0.5,
        )
        assert p.fused_confidence <= 1.0

    def test_fused_clamped_at_0(self):
        p = ConfidenceProfile(conflict_penalty=1.0)
        assert p.fused_confidence >= 0.0

    def test_to_dict_includes_fused(self):
        p = ConfidenceProfile(spatial_confidence=0.5)
        d = p.to_dict()
        assert "fused_confidence" in d
        assert d["spatial_confidence"] == 0.5


class TestUIAStructureScore:
    def test_core_control(self):
        score = uia_structure_score("Button", True, True, True, True, width=100, height=30)
        assert score > 0.85

    def test_container_control(self):
        score = uia_structure_score("Pane", True, True, True, True, width=400, height=300)
        assert 0.5 < score < 0.7

    def test_text_control(self):
        score = uia_structure_score("Text", True, True, True, True, width=200, height=20)
        assert 0.65 < score < 0.85

    def test_unknown_control(self):
        score = uia_structure_score("Unknown", False, False, True, True, width=50, height=50)
        assert score < 0.4

    def test_invalid_bounds_returns_zero(self):
        assert uia_structure_score("Button", True, True, True, False, width=100, height=30) == 0.0

    def test_invisible_returns_zero(self):
        assert uia_structure_score("Button", True, True, True, True, visible=False, width=100, height=30) == 0.0

    def test_tiny_size_returns_zero(self):
        assert uia_structure_score("Button", True, True, True, True, width=1, height=1) == 0.0

    def test_disabled_reduces(self):
        enabled = uia_structure_score("Button", True, True, True, True, width=100, height=30)
        disabled = uia_structure_score("Button", True, True, False, True, width=100, height=30)
        assert disabled < enabled

    def test_automation_id_boosts(self):
        with_id = uia_structure_score("Button", True, True, True, True, width=100, height=30)
        without_id = uia_structure_score("Button", False, True, True, True, width=100, height=30)
        assert with_id > without_id

    def test_clamped_at_1(self):
        score = uia_structure_score("Button", True, True, True, True, width=100, height=30)
        assert score <= 1.0


class TestComputeFromEvidence:
    def test_empty_summary(self):
        p = compute_from_evidence({})
        assert p.fused_confidence == 0.0
        assert p.source_count == 0
        assert p.source_diversity == 0

    def test_single_source(self):
        summary = {"uia": {"count": 1, "best_confidence": 0.9, "last_seen": "2026-01-01"}}
        p = compute_from_evidence(summary, uia_quality=0.8)
        assert p.spatial_confidence == 0.9
        assert p.source_diversity == 1

    def test_multi_source_diversity(self):
        summary = {
            "uia": {"count": 2, "best_confidence": 0.8, "last_seen": "2026-01-01"},
            "vlm": {"count": 1, "best_confidence": 0.7, "last_seen": "2026-01-01"},
            "ocr": {"count": 3, "best_confidence": 0.6, "last_seen": "2026-01-01"},
        }
        p = compute_from_evidence(summary)
        assert p.source_diversity == 3
        assert p.source_count == 6
        assert p.text_confidence == 0.8  # max of all

    def test_spatial_uses_max_of_uia_omni(self):
        summary = {
            "uia": {"count": 1, "best_confidence": 0.6, "last_seen": "2026-01-01"},
            "omni": {"count": 1, "best_confidence": 0.8, "last_seen": "2026-01-01"},
        }
        p = compute_from_evidence(summary)
        assert p.spatial_confidence == 0.8

    def test_with_all_params(self):
        summary = {"uia": {"count": 1, "best_confidence": 0.9, "last_seen": "2026-01-01"}}
        p = compute_from_evidence(
            summary,
            uia_quality=0.85,
            visual_anchor_confidence=0.95,
            memory_confidence=0.7,
            action_confidence=0.6,
            conflict_penalty=0.05,
        )
        assert p.visual_anchor_confidence == 0.95
        assert p.memory_confidence == 0.7
        assert p.action_confidence == 0.6
        assert p.conflict_penalty == 0.05
        assert p.structure_confidence == 0.85

    def test_action_confidence_from_feedback_success(self):
        """Feedback with success → action_confidence = 1.0 from best_action_score."""
        summary = {
            "feedback": {"count": 1, "best_confidence": 0.0, "last_seen": "2026-01-01", "best_action_score": 1.0},
        }
        p = compute_from_evidence(summary)
        assert p.action_confidence == 1.0

    def test_action_confidence_from_feedback_failure(self):
        """Feedback with failure → action_confidence = 0.0 from best_action_score."""
        summary = {
            "feedback": {"count": 1, "best_confidence": 0.0, "last_seen": "2026-01-01", "best_action_score": 0.0},
        }
        p = compute_from_evidence(summary)
        assert p.action_confidence == 0.0

    def test_action_confidence_explicit_overrides_feedback(self):
        """Explicit action_confidence param takes precedence over feedback best_action_score."""
        summary = {
            "feedback": {"count": 1, "best_confidence": 0.0, "last_seen": "2026-01-01", "best_action_score": 1.0},
        }
        p = compute_from_evidence(summary, action_confidence=0.3)
        assert p.action_confidence == 0.3

    def test_action_confidence_feedback_with_observe_sources(self):
        """Feedback + observe sources combined → action from feedback, spatial from uia."""
        summary = {
            "uia": {"count": 2, "best_confidence": 0.85, "last_seen": "2026-01-01", "best_action_score": None},
            "feedback": {"count": 1, "best_confidence": 0.0, "last_seen": "2026-01-01", "best_action_score": 1.0},
        }
        p = compute_from_evidence(summary)
        assert p.spatial_confidence == 0.85
        assert p.action_confidence == 1.0
        assert p.source_diversity == 2
