"""Tests for EvidenceFusionEngine — ConfidenceProfile 融合集成测试。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.memory.evidence_store import EvidenceStore
from src.memory.evidence_collector import EvidenceCollector
from src.memory.evidence_fusion_engine import EvidenceFusionEngine
from src.memory.confidence_profile_store import ConfidenceProfileStore
from src.memory.evidence_fusion import ConfidenceProfile, uia_structure_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def evidence_store():
    return EvidenceStore()


@pytest.fixture
def profile_store():
    return ConfidenceProfileStore()


@pytest.fixture
def fusion_engine(evidence_store, profile_store):
    return EvidenceFusionEngine(
        evidence_store=evidence_store,
        profile_store=profile_store,
    )


class _FakeElementState:
    def __init__(self, visible=True, enabled=True):
        self.visible = visible
        self.enabled = enabled


class _FakeCandidate:
    def __init__(
        self,
        element_id,
        provider_sources=None,
        stable_key_id=None,
        bounds=None,
        text="",
        semantic_role=None,
        control_type="",
        confidence=0.0,
        region_id=None,
        name=None,
        attributes=None,
        state=None,
    ):
        self.element_id = element_id
        self.provider_sources = provider_sources or []
        self.stable_key_id = stable_key_id
        self.bounds = bounds
        self.text = text
        self.semantic_role = semantic_role
        self.control_type = control_type
        self.confidence = confidence
        self.region_id = region_id
        self.name = name
        self.attributes = attributes or {}
        self.state = state or _FakeElementState()


class _FakeApp:
    def __init__(self, attributes=None):
        self.app_id = "test_app"
        self.attributes = attributes or {}


class _FakeCanvas:
    def __init__(
        self,
        canvas_id="c1",
        elements=None,
        regions=None,
        artifacts=None,
        page_model_id="pm1",
        state_template_id="st1",
        app=None,
    ):
        self.canvas_id = canvas_id
        self.elements = elements or []
        self.regions = regions or []
        self.artifacts = artifacts or {}
        self.page_model_id = page_model_id
        self.state_template_id = state_template_id
        self.app = app or _FakeApp()


# ---------------------------------------------------------------------------
# Helper: seed evidence into DB
# ---------------------------------------------------------------------------

def _seed_evidence(db, canvas, elements, *, vlm_used=False):
    """Use EvidenceCollector to seed evidence for the given canvas."""
    collector = EvidenceCollector()
    with db.session() as session:
        collector.collect(session, canvas, vlm_used=vlm_used)


# ---------------------------------------------------------------------------
# Core fusion tests
# ---------------------------------------------------------------------------

class TestFusionUpdatesConfidence:
    def test_fusion_updates_element_confidence(self, db, fusion_engine):
        """有 evidence → confidence 更新为 fused_confidence。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk1",
            bounds=(0, 0, 100, 30),
            control_type="Button",
            confidence=0.5,
        )
        canvas = _FakeCanvas(elements=[elem])

        # Seed evidence
        _seed_evidence(db, canvas, [elem])

        # Run fusion
        with db.session() as session:
            count = fusion_engine.apply_to_canvas(session, canvas)

        assert count == 1
        # Confidence should be updated from 0.5 to fused value
        assert elem.confidence != 0.5
        assert 0.0 <= elem.confidence <= 1.0

    def test_fusion_no_stable_key_skipped(self, db, fusion_engine):
        """无 stable_key_id → 跳过。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id=None,
            confidence=0.5,
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        with db.session() as session:
            count = fusion_engine.apply_to_canvas(session, canvas)

        assert count == 0
        assert elem.confidence == 0.5  # unchanged

    def test_fusion_no_evidence_skipped(self, db, fusion_engine):
        """有 key 无 evidence → 跳过。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_no_evidence",
            confidence=0.5,
        )
        canvas = _FakeCanvas(elements=[elem])
        # Don't seed evidence — element has stable_key_id but no evidence in DB

        with db.session() as session:
            count = fusion_engine.apply_to_canvas(session, canvas)

        assert count == 0
        assert elem.confidence == 0.5  # unchanged


# ---------------------------------------------------------------------------
# DB persistence tests
# ---------------------------------------------------------------------------

class TestProfilePersistence:
    def test_fusion_writes_profile_to_db(self, db, fusion_engine, profile_store):
        """candidate_confidence_profiles 表有记录。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_persist",
            bounds=(0, 0, 100, 30),
            control_type="Button",
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas)

        with db.session() as session:
            profile = profile_store.get_latest(session, "sk_persist")

        assert profile is not None
        assert profile["stable_key_id"] == "sk_persist"
        assert profile["formula_version"] == "v1_dim_weighted"
        assert 0.0 <= profile["fused_confidence"] <= 1.0
        assert profile["canvas_id"] == "c1"

    def test_fusion_writes_to_attributes(self, db, fusion_engine):
        """element.attributes["confidence_profile"] 存在。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_attrs",
            bounds=(0, 0, 100, 30),
            control_type="Button",
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas)

        assert "confidence_profile" in elem.attributes
        profile_dict = elem.attributes["confidence_profile"]
        assert "fused_confidence" in profile_dict
        assert "spatial_confidence" in profile_dict


# ---------------------------------------------------------------------------
# UIA quality tests
# ---------------------------------------------------------------------------

class TestUIAQuality:
    def test_fusion_uia_quality_button(self, db, fusion_engine):
        """Button + bounds + enabled → structure_score > 0。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_button",
            bounds=(10, 20, 100, 50),
            control_type="Button",
            name="OK",
            state=_FakeElementState(visible=True, enabled=True),
            attributes={"automation_id": "btn_ok"},
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas)

        profile_dict = elem.attributes["confidence_profile"]
        # Button (core=0.85) × has_automation_id(1.10) × has_name(1.05) = 0.982
        assert profile_dict["structure_confidence"] > 0.8

    def test_fusion_uia_quality_unknown_type(self, db, fusion_engine):
        """Unknown control_type → low structure_score。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_unknown",
            bounds=(10, 20, 100, 50),
            control_type="UnknownWidget",
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas)

        profile_dict = elem.attributes["confidence_profile"]
        # Unknown base=0.30, no modifiers
        assert profile_dict["structure_confidence"] == pytest.approx(0.30, abs=0.01)


# ---------------------------------------------------------------------------
# Multi-source tests
# ---------------------------------------------------------------------------

class TestMultiSource:
    def test_fusion_multi_source_higher(self, db, fusion_engine):
        """uia + vlm 两条 evidence → fused > 单条。"""
        # Single source
        elem_single = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_single",
            bounds=(10, 20, 100, 50),
            control_type="Button",
        )
        canvas_single = _FakeCanvas(
            canvas_id="c_single",
            elements=[elem_single],
        )
        _seed_evidence(db, canvas_single, [elem_single])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas_single)
        single_confidence = elem_single.confidence

        # Multi source
        elem_multi = _FakeCandidate(
            "e2",
            provider_sources=["uia", "vlm"],
            stable_key_id="sk_multi",
            bounds=(10, 20, 100, 50),
            control_type="Button",
        )
        canvas_multi = _FakeCanvas(
            canvas_id="c_multi",
            elements=[elem_multi],
        )
        _seed_evidence(db, canvas_multi, [elem_multi])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas_multi)

        # Multi-source should have higher or equal confidence
        # because semantic dimension gets VLM confidence added
        assert elem_multi.confidence >= single_confidence

    def test_fusion_profile_dimensions(self, db, fusion_engine):
        """各维度值与 evidence summary 对应。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_dims",
            bounds=(10, 20, 100, 50),
            control_type="Edit",
            name="Search",
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas)

        profile_dict = elem.attributes["confidence_profile"]
        # spatial = max(uia best_confidence) — from raw_confidence
        assert profile_dict["spatial_confidence"] >= 0.0
        # semantic = vlm best — no vlm evidence → 0.0
        assert profile_dict["semantic_confidence"] == 0.0
        # text = max(ocr, uia, vlm) — uia contributes
        assert profile_dict["text_confidence"] >= 0.0
        # structure = uia_structure_score
        assert profile_dict["structure_confidence"] > 0.0
        # Phase 3: these are all 0.0
        assert profile_dict["visual_anchor_confidence"] == 0.0
        assert profile_dict["memory_confidence"] == 0.0
        assert profile_dict["action_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Canvas immutability
# ---------------------------------------------------------------------------

class TestCanvasImmutability:
    def test_fusion_canvas_format_unchanged(self, db, fusion_engine):
        """融合前后 canvas 结构不变（除 confidence 和 attributes 更新）。"""
        elem = _FakeCandidate(
            "e1",
            provider_sources=["uia"],
            stable_key_id="sk_imm",
            bounds=(10, 20, 100, 50),
            control_type="Button",
            text="OK",
        )
        canvas = _FakeCanvas(elements=[elem])
        _seed_evidence(db, canvas, [elem])

        elem_id_before = elem.element_id
        text_before = elem.text
        bounds_before = elem.bounds
        control_type_before = elem.control_type

        with db.session() as session:
            fusion_engine.apply_to_canvas(session, canvas)

        # Unchanged fields
        assert elem.element_id == elem_id_before
        assert elem.text == text_before
        assert elem.bounds == bounds_before
        assert elem.control_type == control_type_before

    def test_apply_to_canvas_returns_count(self, db, fusion_engine):
        """返回值 = 处理的 element 数。"""
        elems = [
            _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="sk_a",
                           bounds=(0, 0, 100, 30), control_type="Button"),
            _FakeCandidate("e2", provider_sources=["uia"], stable_key_id="sk_b",
                           bounds=(0, 0, 100, 30), control_type="Edit"),
            _FakeCandidate("e3", provider_sources=["uia"], stable_key_id=None),
        ]
        canvas = _FakeCanvas(elements=elems)
        # Seed evidence for e1 and e2 only (e3 has no stable_key_id)
        _seed_evidence(db, canvas, elems[:2])

        with db.session() as session:
            count = fusion_engine.apply_to_canvas(session, canvas)

        assert count == 2  # e1 and e2 processed, e3 skipped


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_canvas(self, db, fusion_engine):
        """空 canvas → 返回 0，不崩溃。"""
        canvas = _FakeCanvas(elements=[])
        with db.session() as session:
            count = fusion_engine.apply_to_canvas(session, canvas)
        assert count == 0

    def test_partial_failure_still_fuses_others(self, db):
        """单个 element 失败不影响其余 element 融合。"""
        class _BrokenStore(EvidenceStore):
            """aggregate_summary raises for sk_fail."""
            def aggregate_summary(self, session, stable_key_id):
                if stable_key_id == "sk_fail":
                    raise RuntimeError("simulated failure")
                return super().aggregate_summary(session, stable_key_id)

        engine = EvidenceFusionEngine(evidence_store=_BrokenStore())

        elem_ok = _FakeCandidate(
            "e1", provider_sources=["uia"], stable_key_id="sk_ok",
            bounds=(0, 0, 100, 30), control_type="Button",
        )
        elem_fail = _FakeCandidate(
            "e2", provider_sources=["uia"], stable_key_id="sk_fail",
            bounds=(0, 0, 100, 30), control_type="Button",
        )
        canvas = _FakeCanvas(elements=[elem_ok, elem_fail])

        # Seed evidence for both
        collector = EvidenceCollector()
        with db.session() as session:
            collector.collect(session, canvas)

        with db.session() as session:
            count = engine.apply_to_canvas(session, canvas)

        # e1 fused, e2 failed → count=1
        assert count == 1
        assert "confidence_profile" in elem_ok.attributes
