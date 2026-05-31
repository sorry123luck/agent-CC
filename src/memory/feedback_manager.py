"""Feedback Manager — record and query Agent feedback on candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.storage.schema import CandidateHistory, FeedbackRecord

_logger = logging.getLogger(__name__)

# Map feedback_type to evidence feedback_outcome
_OUTCOME_MAP: dict[str, str] = {
    "success": "success",
    "failure": "fail",
    "unclear": "partial",
}


@dataclass
class FeedbackStats:
    """Feedback statistics for a candidate."""
    candidate_key: str
    total_feedback: int
    success_count: int
    failure_count: int
    unclear_count: int
    success_rate: float
    last_feedback_at: datetime | None


class FeedbackManager:
    """Record and query Agent feedback on candidates."""

    def __init__(self, session: Session):
        self._session = session

    def record_feedback(
        self,
        canvas_id: str,
        candidate_key: str,
        feedback_type: str,
        detail: dict | None = None,
    ) -> None:
        """Record Agent feedback and write candidate_evidence."""
        record = FeedbackRecord(
            canvas_id=canvas_id,
            candidate_key=candidate_key,
            feedback_type=feedback_type,
            detail_json=detail,
        )
        self._session.add(record)

        # Update candidate history
        history = (
            self._session.query(CandidateHistory)
            .filter_by(candidate_key=candidate_key)
            .first()
        )
        if history:
            history.use_count += 1
            if feedback_type == "success":
                history.success_count += 1
            history.last_seen_at = datetime.now(timezone.utc)
        else:
            history = CandidateHistory(
                canvas_id=canvas_id,
                candidate_key=candidate_key,
                use_count=1,
                success_count=1 if feedback_type == "success" else 0,
                last_seen_at=datetime.now(timezone.utc),
            )
            self._session.add(history)

        # Phase 8: write candidate_evidence
        try:
            from src.memory.evidence_store import EvidenceStore

            outcome = _OUTCOME_MAP.get(feedback_type, "partial")
            action_type = "click"  # default
            if detail and isinstance(detail, dict):
                action_type = detail.get("action_type", "click")

            ev_store = EvidenceStore()
            ev_store.create_evidence(
                self._session,
                provider="feedback",
                evidence_scope="action",
                evidence_event="feedback",
                stable_key_id=candidate_key,
                canvas_id=canvas_id,
                feedback_outcome=outcome,
                action_type=action_type,
                action_score=1.0 if outcome == "success" else 0.0,
            )
        except Exception as exc:
            _logger.warning("Failed to write feedback evidence: %s", exc)

        self._session.commit()

    def get_stats(self, candidate_key: str) -> FeedbackStats:
        """Get feedback statistics for a candidate."""
        records = (
            self._session.query(FeedbackRecord)
            .filter_by(candidate_key=candidate_key)
            .all()
        )

        success = sum(1 for r in records if r.feedback_type == "success")
        failure = sum(1 for r in records if r.feedback_type == "failure")
        unclear = sum(1 for r in records if r.feedback_type == "unclear")
        total = len(records)

        last_at = None
        if records:
            last_at = max(r.created_at for r in records if r.created_at)

        return FeedbackStats(
            candidate_key=candidate_key,
            total_feedback=total,
            success_count=success,
            failure_count=failure,
            unclear_count=unclear,
            success_rate=success / max(total, 1),
            last_feedback_at=last_at,
        )

    def update_confidence(
        self,
        candidate_key: str,
        feedback: str,
        base_confidence: float = 0.5,
    ) -> float:
        """Calculate updated confidence based on feedback history."""
        stats = self.get_stats(candidate_key)
        if stats.total_feedback == 0:
            return base_confidence

        # Weight recent success rate
        success_rate = stats.success_rate
        # Blend: 70% historical success rate, 30% base
        updated = 0.7 * success_rate + 0.3 * base_confidence
        return max(0.0, min(1.0, updated))
