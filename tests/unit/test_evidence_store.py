"""Tests for evidence_store — 候选证据 CRUD。"""

import pytest

from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.memory.evidence_store import EvidenceStore


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def store():
    return EvidenceStore()


class TestEvidenceCreateAndGet:
    def test_create_and_get(self, db, store):
        with db.session() as session:
            eid = store.create_evidence(
                session,
                provider="uia",
                evidence_scope="candidate",
                evidence_event="observe",
                canvas_id="c1",
                element_id="e1",
                stable_key_id="k1",
                text="OK",
                raw_confidence=0.9,
            )
        with db.session() as session:
            ev = store.get_evidence(session, eid)
            assert ev is not None
            assert ev["provider"] == "uia"
            assert ev["evidence_scope"] == "candidate"
            assert ev["text"] == "OK"

    def test_create_region_evidence(self, db, store):
        with db.session() as session:
            eid = store.create_evidence(
                session,
                provider="vlm",
                evidence_scope="region",
                evidence_event="vlm_semantic",
                canvas_id="c1",
                region_id="r1",
            )
        with db.session() as session:
            ev = store.get_evidence(session, eid)
            assert ev["region_id"] == "r1"

    def test_create_memory_replay_no_canvas(self, db, store):
        with db.session() as session:
            eid = store.create_evidence(
                session,
                provider="memory",
                evidence_scope="candidate",
                evidence_event="memory_replay",
                stable_key_id="k1",
            )
        with db.session() as session:
            ev = store.get_evidence(session, eid)
            assert ev["canvas_id"] is None
            assert ev["stable_key_id"] == "k1"

    def test_create_action_feedback(self, db, store):
        with db.session() as session:
            eid = store.create_evidence(
                session,
                provider="feedback",
                evidence_scope="action",
                evidence_event="feedback",
                feedback_outcome="success",
                action_type="click",
                stable_key_id="k1",
            )
        with db.session() as session:
            ev = store.get_evidence(session, eid)
            assert ev["feedback_outcome"] == "success"

    def test_create_action_override(self, db, store):
        with db.session() as session:
            eid = store.create_evidence(
                session,
                provider="manual",
                evidence_scope="action",
                evidence_event="override",
                stable_key_id="k1",
            )
        with db.session() as session:
            ev = store.get_evidence(session, eid)
            assert ev["provider"] == "manual"


class TestEvidenceValidation:
    def test_invalid_scope_event_raises(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="Invalid scope/event"):
                store.create_evidence(
                    session,
                    provider="uia",
                    evidence_scope="candidate",
                    evidence_event="feedback",
                )

    def test_invalid_provider_raises(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="Invalid provider"):
                store.create_evidence(
                    session,
                    provider="feedback",
                    evidence_scope="candidate",
                    evidence_event="observe",
                    canvas_id="c1",
                    element_id="e1",
                )

    def test_missing_canvas_id_raises(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="requires canvas_id"):
                store.create_evidence(
                    session,
                    provider="uia",
                    evidence_scope="candidate",
                    evidence_event="observe",
                    element_id="e1",
                )

    def test_memory_replay_requires_stable_key_id(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="requires stable_key_id"):
                store.create_evidence(
                    session,
                    provider="memory",
                    evidence_scope="candidate",
                    evidence_event="memory_replay",
                )

    def test_action_feedback_requires_outcome_and_action_type(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="requires feedback_outcome"):
                store.create_evidence(
                    session,
                    provider="feedback",
                    evidence_scope="action",
                    evidence_event="feedback",
                    stable_key_id="k1",
                )

    def test_region_vlm_semantic_requires_region_id(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="requires canvas_id and region_id"):
                store.create_evidence(
                    session,
                    provider="vlm",
                    evidence_scope="region",
                    evidence_event="vlm_semantic",
                    canvas_id="c1",
                )

    def test_page_requires_canvas_id(self, db, store):
        with db.session() as session:
            with pytest.raises(ValueError, match="requires canvas_id"):
                store.create_evidence(
                    session,
                    provider="vlm",
                    evidence_scope="page",
                    evidence_event="vlm_semantic",
                )


class TestEvidenceQueries:
    def test_list_by_stable_key(self, db, store):
        with db.session() as session:
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1", stable_key_id="k1")
            store.create_evidence(session, provider="vlm", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1", stable_key_id="k1")
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e2", stable_key_id="k2")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1")
            assert len(results) == 2

    def test_list_by_stable_key_filter_provider(self, db, store):
        with db.session() as session:
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1", stable_key_id="k1")
            store.create_evidence(session, provider="vlm", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1", stable_key_id="k1")
        with db.session() as session:
            results = store.list_by_stable_key(session, "k1", provider="vlm")
            assert len(results) == 1
            assert results[0]["provider"] == "vlm"

    def test_list_by_canvas(self, db, store):
        with db.session() as session:
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1")
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c2", element_id="e2")
        with db.session() as session:
            results = store.list_by_canvas(session, "c1")
            assert len(results) == 1

    def test_aggregate_summary(self, db, store):
        with db.session() as session:
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1",
                                  stable_key_id="k1", raw_confidence=0.8)
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c2", element_id="e1",
                                  stable_key_id="k1", raw_confidence=0.9)
            store.create_evidence(session, provider="vlm", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1",
                                  stable_key_id="k1", raw_confidence=0.7)
        with db.session() as session:
            summary = store.aggregate_summary(session, "k1")
            assert summary["uia"]["count"] == 2
            assert summary["uia"]["best_confidence"] == 0.9
            assert summary["vlm"]["count"] == 1
            assert summary["vlm"]["best_confidence"] == 0.7

    def test_aggregate_summary_includes_best_action_score(self, db, store):
        """aggregate_summary returns best_action_score per provider."""
        with db.session() as session:
            store.create_evidence(session, provider="feedback", evidence_scope="action",
                                  evidence_event="feedback", stable_key_id="k_fb",
                                  feedback_outcome="success", action_type="click",
                                  action_score=1.0)
            store.create_evidence(session, provider="feedback", evidence_scope="action",
                                  evidence_event="feedback", stable_key_id="k_fb",
                                  feedback_outcome="fail", action_type="click",
                                  action_score=0.0)
        with db.session() as session:
            summary = store.aggregate_summary(session, "k_fb")
            assert "feedback" in summary
            assert summary["feedback"]["best_action_score"] == 1.0
            assert summary["feedback"]["count"] == 2

    def test_aggregate_summary_best_action_score_none_when_no_feedback(self, db, store):
        """aggregate_summary best_action_score is None for non-feedback providers."""
        with db.session() as session:
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1",
                                  stable_key_id="k_no_fb", raw_confidence=0.8)
        with db.session() as session:
            summary = store.aggregate_summary(session, "k_no_fb")
            assert summary["uia"]["best_action_score"] is None

    def test_delete_by_canvas(self, db, store):
        with db.session() as session:
            store.create_evidence(session, provider="uia", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1")
            store.create_evidence(session, provider="vlm", evidence_scope="candidate",
                                  evidence_event="observe", canvas_id="c1", element_id="e1")
        with db.session() as session:
            count = store.delete_by_canvas(session, "c1")
            assert count == 2
            results = store.list_by_canvas(session, "c1")
            assert len(results) == 0
