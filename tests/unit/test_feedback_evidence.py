"""Tests for Phase 8: feedback 回流 — feedback writes candidate_evidence."""

import pytest

from src.memory.evidence_fusion import compute_from_evidence
from src.memory.evidence_store import EvidenceStore
from src.memory.feedback_manager import FeedbackManager
from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.storage.schema import CandidateEvidenceRecord


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


# ---------------------------------------------------------------------------
# Test: Feedback writes evidence
# ---------------------------------------------------------------------------


class TestFeedbackWritesEvidence:
    def test_success_feedback_writes_evidence(self, db, evidence_store):
        """success feedback → candidate_evidence record with outcome=success."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id="c1",
                candidate_key="sk1",
                feedback_type="success",
            )

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk1",
            ).all()

        assert len(records) == 1
        rec = records[0]
        assert rec.provider == "feedback"
        assert rec.evidence_scope == "action"
        assert rec.evidence_event == "feedback"
        assert rec.feedback_outcome == "success"
        assert rec.action_score == 1.0

    def test_failure_feedback_writes_evidence(self, db, evidence_store):
        """failure feedback → outcome=fail, action_score=0.0."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id="c1",
                candidate_key="sk2",
                feedback_type="failure",
            )

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk2",
            ).all()

        assert len(records) == 1
        assert records[0].feedback_outcome == "fail"
        assert records[0].action_score == 0.0

    def test_unclear_feedback_writes_evidence(self, db, evidence_store):
        """unclear feedback → outcome=partial."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id="c1",
                candidate_key="sk3",
                feedback_type="unclear",
            )

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk3",
            ).all()

        assert len(records) == 1
        assert records[0].feedback_outcome == "partial"

    def test_action_type_from_detail(self, db, evidence_store):
        """detail.action_type → evidence.action_type."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id="c1",
                candidate_key="sk4",
                feedback_type="success",
                detail={"action_type": "type_text"},
            )

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk4",
            ).all()

        assert len(records) == 1
        assert records[0].action_type == "type_text"

    def test_action_type_default_click(self, db, evidence_store):
        """No detail → action_type defaults to 'click'."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id="c1",
                candidate_key="sk5",
                feedback_type="success",
            )

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk5",
            ).all()

        assert len(records) == 1
        assert records[0].action_type == "click"

    def test_canvas_id_preserved(self, db, evidence_store):
        """canvas_id is written to evidence."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id="my_canvas_123",
                candidate_key="sk6",
                feedback_type="success",
            )

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk6",
            ).all()

        assert len(records) == 1
        assert records[0].canvas_id == "my_canvas_123"

    def test_multiple_feedbacks_accumulate(self, db, evidence_store):
        """Multiple feedbacks for same key → multiple evidence records."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk7", "success")
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk7", "failure")
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk7", "success")

        with db.session() as session:
            records = session.query(CandidateEvidenceRecord).filter_by(
                stable_key_id="sk7",
            ).all()

        assert len(records) == 3


# ---------------------------------------------------------------------------
# Test: Evidence failure doesn't break feedback
# ---------------------------------------------------------------------------


class TestFeedbackEvidenceErrorIsolation:
    def test_evidence_write_failure_no_crash(self, db):
        """Evidence write failure → feedback still recorded."""
        from unittest.mock import patch

        with db.session() as session:
            manager = FeedbackManager(session)
            # Patch EvidenceStore.create_evidence to raise
            with patch(
                "src.memory.evidence_store.EvidenceStore.create_evidence",
                side_effect=RuntimeError("DB error"),
            ):
                manager.record_feedback(
                    canvas_id="c1",
                    candidate_key="sk_err",
                    feedback_type="success",
                )

        # FeedbackRecord should still be written
        from src.storage.schema import FeedbackRecord

        with db.session() as session:
            records = session.query(FeedbackRecord).filter_by(
                candidate_key="sk_err",
            ).all()
        assert len(records) == 1
        assert records[0].feedback_type == "success"


# ---------------------------------------------------------------------------
# Test: Existing feedback behavior preserved
# ---------------------------------------------------------------------------


class TestFeedbackNoRegression:
    def test_feedback_record_created(self, db):
        """FeedbackRecord is still written."""
        from src.storage.schema import FeedbackRecord

        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_nr", "success")

        with db.session() as session:
            records = session.query(FeedbackRecord).filter_by(
                candidate_key="sk_nr",
            ).all()
        assert len(records) == 1

    def test_candidate_history_updated(self, db):
        """CandidateHistory is still updated."""
        from src.storage.schema import CandidateHistory

        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_hist", "success")

        with db.session() as session:
            history = session.query(CandidateHistory).filter_by(
                candidate_key="sk_hist",
            ).first()
        assert history is not None
        assert history.use_count == 1
        assert history.success_count == 1

    def test_get_stats_still_works(self, db):
        """get_stats still works after Phase 8 changes."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_stats", "success")
            manager.record_feedback("c1", "sk_stats", "failure")

        with db.session() as session:
            manager = FeedbackManager(session)
            stats = manager.get_stats("sk_stats")

        assert stats.total_feedback == 2
        assert stats.success_count == 1
        assert stats.failure_count == 1


# ---------------------------------------------------------------------------
# Test: action_confidence data flow (feedback → evidence → fusion)
# ---------------------------------------------------------------------------


class TestFeedbackActionConfidenceFlow:
    def test_success_feedback_action_confidence_positive(self, db, evidence_store):
        """success feedback → aggregate_summary → compute_from_evidence → action_confidence > 0."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_ac1", "success")

        with db.session() as session:
            summary = evidence_store.aggregate_summary(session, "sk_ac1")

        profile = compute_from_evidence(summary)
        assert profile.action_confidence == 1.0

    def test_failure_feedback_action_confidence_zero(self, db, evidence_store):
        """failure feedback → aggregate_summary → compute_from_evidence → action_confidence == 0."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_ac2", "failure")

        with db.session() as session:
            summary = evidence_store.aggregate_summary(session, "sk_ac2")

        profile = compute_from_evidence(summary)
        assert profile.action_confidence == 0.0

    def test_mixed_feedback_action_confidence_uses_best(self, db, evidence_store):
        """Multiple feedbacks → best_action_score = 1.0 (from success)."""
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_ac3", "failure")
        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_ac3", "success")

        with db.session() as session:
            summary = evidence_store.aggregate_summary(session, "sk_ac3")

        profile = compute_from_evidence(summary)
        assert profile.action_confidence == 1.0

    def test_no_feedback_action_confidence_zero(self, db, evidence_store):
        """No feedback evidence → action_confidence = 0.0."""
        with db.session() as session:
            evidence_store.create_evidence(
                session, provider="uia", evidence_scope="candidate",
                evidence_event="observe", canvas_id="c1", element_id="e1",
                stable_key_id="sk_ac4", raw_confidence=0.8,
            )

        with db.session() as session:
            summary = evidence_store.aggregate_summary(session, "sk_ac4")

        profile = compute_from_evidence(summary)
        assert profile.action_confidence == 0.0

    def test_feedback_plus_observe_fused_confidence(self, db, evidence_store):
        """Feedback + UIA evidence → fused_confidence reflects both spatial and action."""
        with db.session() as session:
            evidence_store.create_evidence(
                session, provider="uia", evidence_scope="candidate",
                evidence_event="observe", canvas_id="c1", element_id="e1",
                stable_key_id="sk_ac5", raw_confidence=0.9,
            )

        with db.session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback("c1", "sk_ac5", "success")

        with db.session() as session:
            summary = evidence_store.aggregate_summary(session, "sk_ac5")

        profile = compute_from_evidence(summary, uia_quality=0.85)
        # spatial = 0.9 (from uia), action = 1.0 (from feedback success)
        # fused = 0.24*0.9 + 0.06*1.0 + 0.09*0.85 = 0.216 + 0.06 + 0.0765 = 0.3525
        assert profile.spatial_confidence == 0.9
        assert profile.action_confidence == 1.0
        assert profile.fused_confidence > 0.3
