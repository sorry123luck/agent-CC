"""
ConfidenceProfileStore — 候选置信度快照 CRUD。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.storage.schema import CandidateConfidenceProfileRecord

_CCP_ALLOWED_CREATE_FIELDS = frozenset({
    "canvas_id", "page_model_id", "state_template_id",
    "spatial_confidence", "semantic_confidence", "text_confidence",
    "structure_confidence", "visual_anchor_confidence", "memory_confidence",
    "action_confidence", "conflict_penalty", "source_count", "source_diversity",
    "evidence_ids", "extra_metadata",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: CandidateConfidenceProfileRecord) -> dict:
    return {
        "profile_id": row.profile_id,
        "stable_key_id": row.stable_key_id,
        "canvas_id": row.canvas_id,
        "page_model_id": row.page_model_id,
        "state_template_id": row.state_template_id,
        "spatial_confidence": row.spatial_confidence,
        "semantic_confidence": row.semantic_confidence,
        "text_confidence": row.text_confidence,
        "structure_confidence": row.structure_confidence,
        "visual_anchor_confidence": row.visual_anchor_confidence,
        "memory_confidence": row.memory_confidence,
        "action_confidence": row.action_confidence,
        "conflict_penalty": row.conflict_penalty,
        "fused_confidence": row.fused_confidence,
        "formula_version": row.formula_version,
        "source_count": row.source_count,
        "source_diversity": row.source_diversity,
        "evidence_ids": row.evidence_ids,
        "computed_at": row.computed_at,
        "metadata": row.extra_metadata,
    }


class ConfidenceProfileStore:
    """候选置信度快照存储。"""

    def create_profile(
        self,
        session,
        stable_key_id: str,
        fused_confidence: float,
        formula_version: str,
        **kwargs,
    ) -> str:
        # Map API-level 'metadata' to ORM column 'extra_metadata'
        if "metadata" in kwargs:
            kwargs["extra_metadata"] = kwargs.pop("metadata")
        filtered = {k: v for k, v in kwargs.items() if k in _CCP_ALLOWED_CREATE_FIELDS}
        profile_id = uuid.uuid4().hex
        row = CandidateConfidenceProfileRecord(
            profile_id=profile_id,
            stable_key_id=stable_key_id,
            fused_confidence=fused_confidence,
            formula_version=formula_version,
            computed_at=_now_iso(),
            **filtered,
        )
        session.add(row)
        session.flush()
        return profile_id

    def get_profile(self, session, profile_id: str) -> dict | None:
        row = session.query(CandidateConfidenceProfileRecord).filter_by(profile_id=profile_id).first()
        return _row_to_dict(row) if row else None

    def get_latest(self, session, stable_key_id: str) -> dict | None:
        row = (
            session.query(CandidateConfidenceProfileRecord)
            .filter_by(stable_key_id=stable_key_id)
            .order_by(CandidateConfidenceProfileRecord.computed_at.desc())
            .first()
        )
        return _row_to_dict(row) if row else None

    def list_by_stable_key(self, session, stable_key_id: str, limit: int = 20) -> list[dict]:
        rows = (
            session.query(CandidateConfidenceProfileRecord)
            .filter_by(stable_key_id=stable_key_id)
            .order_by(CandidateConfidenceProfileRecord.computed_at.desc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(r) for r in rows]

    def list_by_canvas(self, session, canvas_id: str) -> list[dict]:
        rows = (
            session.query(CandidateConfidenceProfileRecord)
            .filter_by(canvas_id=canvas_id)
            .order_by(CandidateConfidenceProfileRecord.computed_at.desc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]
