"""Tests for evidence_collector — observe 管线 evidence 收集。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.memory.evidence_store import EvidenceStore
from src.memory.evidence_collector import EvidenceCollector
from src.perception.page_compiler_models import SemanticRole


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
def store():
    return EvidenceStore()


@pytest.fixture
def collector(store):
    return EvidenceCollector(evidence_store=store)


class _FakeRegion:
    """Minimal Region stand-in for testing."""

    def __init__(self, region_id, role="unknown", bounds=None, attributes=None):
        self.region_id = region_id
        self.role = role
        self.bounds = bounds
        self.attributes = attributes or {}


class _FakeCandidate:
    """Minimal Candidate stand-in for testing."""

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


class _FakeApp:
    def __init__(self, attributes=None):
        self.app_id = "test_app"
        self.attributes = attributes or {}


class _FakeCanvas:
    """Minimal InteractionCanvas stand-in for testing."""

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
# Candidate-level evidence
# ---------------------------------------------------------------------------

class TestCandidateEvidence:
    def test_single_source(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas)
        assert count == 1
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            assert len(evs) == 1
            assert evs[0]["provider"] == "uia"
            assert evs[0]["evidence_scope"] == "candidate"
            assert evs[0]["evidence_event"] == "observe"

    def test_multi_source(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia", "ocr"], stable_key_id="k1"),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas)
        assert count == 2
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            providers = {e["provider"] for e in evs}
            assert providers == {"uia", "ocr"}

    def test_multiple_elements(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
                _FakeCandidate("e2", provider_sources=["ocr"], stable_key_id="k2"),
                _FakeCandidate("e3", provider_sources=["vlm"], stable_key_id="k3"),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas)
        assert count == 3

    def test_skips_empty_provider_sources(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=[], stable_key_id="k1"),
                _FakeCandidate("e2", provider_sources=["uia"], stable_key_id="k2"),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas)
        assert count == 1

    def test_skips_none_provider_sources(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=None, stable_key_id="k1"),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas)
        assert count == 0


# ---------------------------------------------------------------------------
# Region-level evidence
# ---------------------------------------------------------------------------

class TestRegionEvidence:
    def test_vlm_region_evidence(self, db, collector):
        canvas = _FakeCanvas(
            regions=[
                _FakeRegion("r1", attributes={"vlm_purpose": "chat_area", "source": "vlm"}),
                _FakeRegion("r2", attributes={"source": "uia"}),  # not VLM
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=True)
        assert count == 1
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            region_evs = [e for e in evs if e["evidence_scope"] == "region"]
            assert len(region_evs) == 1
            assert region_evs[0]["region_id"] == "r1"
            assert region_evs[0]["provider"] == "vlm"

    def test_no_region_evidence_when_vlm_not_used(self, db, collector):
        canvas = _FakeCanvas(
            regions=[
                _FakeRegion("r1", attributes={"vlm_purpose": "chat_area"}),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=False)
        assert count == 0

    def test_region_without_vlm_attrs_skipped(self, db, collector):
        canvas = _FakeCanvas(
            regions=[
                _FakeRegion("r1", attributes={"source": "uia"}),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=True)
        assert count == 0


# ---------------------------------------------------------------------------
# Page-level evidence
# ---------------------------------------------------------------------------

class TestPageEvidence:
    def test_page_evidence_with_vlm_app_identity(self, db, collector):
        canvas = _FakeCanvas(
            app=_FakeApp(attributes={"vlm_app_name": "wechat"}),
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=True)
        assert count == 1
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            page_evs = [e for e in evs if e["evidence_scope"] == "page"]
            assert len(page_evs) == 1
            assert page_evs[0]["provider"] == "vlm"
            assert page_evs[0]["evidence_event"] == "vlm_semantic"

    def test_no_page_evidence_without_vlm_app(self, db, collector):
        canvas = _FakeCanvas(
            app=_FakeApp(attributes={}),
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=True)
        assert count == 0

    def test_no_page_evidence_when_vlm_not_used(self, db, collector):
        canvas = _FakeCanvas(
            app=_FakeApp(attributes={"vlm_app_name": "wechat"}),
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=False)
        assert count == 0


# ---------------------------------------------------------------------------
# Integration: all levels combined
# ---------------------------------------------------------------------------

class TestCombinedCollection:
    def test_all_levels(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
                _FakeCandidate("e2", provider_sources=["vlm"], stable_key_id="k2"),
            ],
            regions=[
                _FakeRegion("r1", attributes={"vlm_purpose": "chat_area"}),
            ],
            app=_FakeApp(attributes={"vlm_app_name": "wechat"}),
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=True)
        # 2 candidate + 1 region + 1 page = 4
        assert count == 4

    def test_returns_count_matches_actual(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia", "ocr"], stable_key_id="k1"),
                _FakeCandidate("e2", provider_sources=["vlm"], stable_key_id="k2"),
            ],
        )
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=False)
        assert count == 3
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            assert len(evs) == 3


# ---------------------------------------------------------------------------
# Evidence content verification
# ---------------------------------------------------------------------------

class TestEvidenceContent:
    def test_content_matches_element(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate(
                    "e1",
                    provider_sources=["uia"],
                    stable_key_id="k1",
                    bounds=(10, 20, 100, 50),
                    text="OK",
                    control_type="Button",
                    confidence=0.85,
                    region_id="r1",
                ),
            ],
        )
        with db.session() as session:
            collector.collect(session, canvas)
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            ev = evs[0]
            assert ev["element_id"] == "e1"
            assert ev["stable_key_id"] == "k1"
            assert "[10, 20, 100, 50]" in ev["bounds_json"]
            assert ev["text"] == "OK"
            assert ev["control_type"] == "Button"
            assert ev["raw_confidence"] == 0.85
            assert ev["region_id"] == "r1"
            assert ev["page_model_id"] == "pm1"
            assert ev["state_template_id"] == "st1"

    def test_stable_key_id_present(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="sk_abc"),
            ],
        )
        with db.session() as session:
            collector.collect(session, canvas)
        with db.session() as session:
            evs = EvidenceStore().list_by_stable_key(session, "sk_abc")
            assert len(evs) == 1

    def test_stable_key_id_none(self, db, collector):
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id=None),
            ],
        )
        with db.session() as session:
            collector.collect(session, canvas)
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            assert len(evs) == 1
            assert evs[0]["stable_key_id"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_exception_does_not_propagate(self, db):
        """Evidence write failure should be caught, not raised."""
        class _BrokenStore:
            def create_evidence(self, session, **kwargs):
                raise RuntimeError("db down")

        broken_collector = EvidenceCollector(evidence_store=_BrokenStore())
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
            ],
        )
        with db.session() as session:
            count = broken_collector.collect(session, canvas)
        assert count == 0  # exception caught, count stays 0

    def test_partial_failure_still_collects_others(self, db):
        """If one element fails, others should still be collected."""
        class _PartialStore:
            def __init__(self):
                self._real = EvidenceStore()
                self._fail_ids = {"e2"}

            def create_evidence(self, session, **kwargs):
                if kwargs.get("element_id") in self._fail_ids:
                    raise RuntimeError("simulated failure")
                return self._real.create_evidence(session, **kwargs)

        partial_collector = EvidenceCollector(evidence_store=_PartialStore())
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
                _FakeCandidate("e2", provider_sources=["uia"], stable_key_id="k2"),
                _FakeCandidate("e3", provider_sources=["uia"], stable_key_id="k3"),
            ],
        )
        with db.session() as session:
            count = partial_collector.collect(session, canvas)
        assert count == 2  # e1 and e3 succeed, e2 fails


# ---------------------------------------------------------------------------
# Canvas immutability
# ---------------------------------------------------------------------------

class TestCanvasImmutability:
    def test_canvas_format_unchanged(self, db, collector):
        elements = [
            _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
            _FakeCandidate("e2", provider_sources=["vlm"], stable_key_id="k2"),
        ]
        regions = [
            _FakeRegion("r1", attributes={"vlm_purpose": "chat_area"}),
        ]
        artifacts = {"some_key": "some_value"}
        canvas = _FakeCanvas(elements=elements, regions=regions, artifacts=artifacts)

        # Snapshot before
        elem_ids_before = [e.element_id for e in canvas.elements]
        region_ids_before = [r.region_id for r in canvas.regions]
        artifact_keys_before = set(canvas.artifacts.keys())

        with db.session() as session:
            collector.collect(session, canvas, vlm_used=True)

        # Verify canvas unchanged
        assert [e.element_id for e in canvas.elements] == elem_ids_before
        assert [r.region_id for r in canvas.regions] == region_ids_before
        assert set(canvas.artifacts.keys()) == artifact_keys_before


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_semantic_role_enum_extracted(self, db, collector):
        """SemanticRole enum value should be stored as string."""
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate(
                    "e1",
                    provider_sources=["uia"],
                    stable_key_id="k1",
                    semantic_role=SemanticRole.SEND_BUTTON,
                ),
            ],
        )
        with db.session() as session:
            collector.collect(session, canvas)
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            assert evs[0]["semantic_role"] == "send_button"

    def test_semantic_role_string_passthrough(self, db, collector):
        """Plain string semantic_role should pass through unchanged."""
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate(
                    "e1",
                    provider_sources=["uia"],
                    stable_key_id="k1",
                    semantic_role="custom_role",
                ),
            ],
        )
        with db.session() as session:
            collector.collect(session, canvas)
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            assert evs[0]["semantic_role"] == "custom_role"

    def test_none_page_model_id(self, db, collector):
        """page_model_id=None should not cause errors."""
        canvas = _FakeCanvas(
            elements=[
                _FakeCandidate("e1", provider_sources=["uia"], stable_key_id="k1"),
            ],
            page_model_id=None,
            state_template_id=None,
        )
        with db.session() as session:
            count = collector.collect(session, canvas)
        assert count == 1
        with db.session() as session:
            evs = EvidenceStore().list_by_canvas(session, "c1")
            assert evs[0]["page_model_id"] is None
            assert evs[0]["state_template_id"] is None

    def test_empty_canvas(self, db, collector):
        """Canvas with no elements and no regions should produce 0 evidence."""
        canvas = _FakeCanvas(elements=[], regions=[])
        with db.session() as session:
            count = collector.collect(session, canvas, vlm_used=True)
        assert count == 0
