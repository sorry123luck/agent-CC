"""
EvidenceStore — 候选证据 CRUD（只追加，不修改）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func

from src.storage.schema import CandidateEvidenceRecord

_CE_ALLOWED_CREATE_FIELDS = frozenset({
    "canvas_id", "element_id", "stable_key_id", "provider_element_id",
    "region_id", "page_model_id", "state_template_id", "bounds_json",
    "relative_bounds_json", "text", "semantic_role", "control_type",
    "raw_confidence", "spatial_score", "semantic_score", "text_score",
    "structure_score", "visual_score", "action_score", "memory_score",
    "feedback_outcome", "action_type", "error_code", "match_method",
    "match_confidence", "raw_data_ref", "extra_metadata",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# scope/event/provider 校验矩阵
_VALID_COMBINATIONS: dict[tuple[str, str], set[str]] = {
    ("candidate", "observe"): {"uia", "omni", "ocr", "vlm", "dom"},
    ("candidate", "vlm_semantic"): {"vlm"},
    ("candidate", "override"): {"manual", "agent"},
    ("candidate", "memory_replay"): {"memory"},
    ("region", "observe"): {"uia", "omni", "vlm", "dom"},
    ("region", "vlm_semantic"): {"vlm"},
    ("page", "observe"): {"uia", "omni", "vlm", "dom"},
    ("page", "vlm_semantic"): {"vlm"},
    ("action", "feedback"): {"feedback"},
    ("action", "override"): {"manual", "agent"},
}


def _validate_evidence(scope: str, event: str, provider: str, **kwargs) -> None:
    key = (scope, event)
    if key not in _VALID_COMBINATIONS:
        raise ValueError(f"Invalid scope/event combination: {scope}/{event}")
    if provider not in _VALID_COMBINATIONS[key]:
        raise ValueError(f"Invalid provider '{provider}' for {scope}/{event}")

    if scope == "candidate" and event != "memory_replay":
        if not kwargs.get("canvas_id") or not kwargs.get("element_id"):
            raise ValueError(f"{scope}/{event} requires canvas_id and element_id")

    if scope == "candidate" and event == "memory_replay":
        if not kwargs.get("stable_key_id"):
            raise ValueError("memory_replay requires stable_key_id")

    if scope == "region" and event == "vlm_semantic":
        if not kwargs.get("canvas_id") or not kwargs.get("region_id"):
            raise ValueError("region/vlm_semantic requires canvas_id and region_id")

    if scope == "page":
        if not kwargs.get("canvas_id"):
            raise ValueError(f"{scope}/{event} requires canvas_id")

    if scope == "action" and event == "feedback":
        if not kwargs.get("feedback_outcome") or not kwargs.get("action_type"):
            raise ValueError("action/feedback requires feedback_outcome and action_type")
        if not kwargs.get("stable_key_id") and not kwargs.get("element_id"):
            raise ValueError("action/feedback requires stable_key_id or element_id")

    if scope == "action" and event == "override":
        if not kwargs.get("stable_key_id") and not kwargs.get("element_id"):
            raise ValueError("action/override requires stable_key_id or element_id")


def _row_to_dict(row: CandidateEvidenceRecord) -> dict:
    return {
        "evidence_id": row.evidence_id,
        "canvas_id": row.canvas_id,
        "element_id": row.element_id,
        "stable_key_id": row.stable_key_id,
        "provider": row.provider,
        "provider_element_id": row.provider_element_id,
        "evidence_scope": row.evidence_scope,
        "evidence_event": row.evidence_event,
        "region_id": row.region_id,
        "page_model_id": row.page_model_id,
        "state_template_id": row.state_template_id,
        "bounds_json": row.bounds_json,
        "relative_bounds_json": row.relative_bounds_json,
        "text": row.text,
        "semantic_role": row.semantic_role,
        "control_type": row.control_type,
        "raw_confidence": row.raw_confidence,
        "spatial_score": row.spatial_score,
        "semantic_score": row.semantic_score,
        "text_score": row.text_score,
        "structure_score": row.structure_score,
        "visual_score": row.visual_score,
        "action_score": row.action_score,
        "memory_score": row.memory_score,
        "feedback_outcome": row.feedback_outcome,
        "action_type": row.action_type,
        "error_code": row.error_code,
        "match_method": row.match_method,
        "match_confidence": row.match_confidence,
        "raw_data_ref": row.raw_data_ref,
        "created_at": row.created_at,
        "metadata": row.extra_metadata,
    }


class EvidenceStore:
    """候选证据存储。"""

    def create_evidence(
        self,
        session,
        provider: str,
        evidence_scope: str,
        evidence_event: str,
        **kwargs,
    ) -> str:
        # Map API-level 'metadata' to ORM column 'extra_metadata'
        if "metadata" in kwargs:
            kwargs["extra_metadata"] = kwargs.pop("metadata")
        _validate_evidence(evidence_scope, evidence_event, provider, **kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in _CE_ALLOWED_CREATE_FIELDS}

        evidence_id = uuid.uuid4().hex
        row = CandidateEvidenceRecord(
            evidence_id=evidence_id,
            provider=provider,
            evidence_scope=evidence_scope,
            evidence_event=evidence_event,
            created_at=_now_iso(),
            **filtered,
        )
        session.add(row)
        session.flush()
        return evidence_id

    def get_evidence(self, session, evidence_id: str) -> dict | None:
        row = session.query(CandidateEvidenceRecord).filter_by(evidence_id=evidence_id).first()
        return _row_to_dict(row) if row else None

    def list_by_stable_key(
        self,
        session,
        stable_key_id: str,
        provider: str | None = None,
        evidence_event: str | None = None,
    ) -> list[dict]:
        q = session.query(CandidateEvidenceRecord).filter_by(stable_key_id=stable_key_id)
        if provider:
            q = q.filter_by(provider=provider)
        if evidence_event:
            q = q.filter_by(evidence_event=evidence_event)
        q = q.order_by(CandidateEvidenceRecord.created_at.desc())
        return [_row_to_dict(r) for r in q.all()]

    def list_by_canvas(self, session, canvas_id: str) -> list[dict]:
        rows = (
            session.query(CandidateEvidenceRecord)
            .filter_by(canvas_id=canvas_id)
            .order_by(CandidateEvidenceRecord.created_at.desc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]

    def list_by_page(
        self,
        session,
        page_model_id: str,
        state_template_id: str | None = None,
    ) -> list[dict]:
        q = session.query(CandidateEvidenceRecord).filter_by(page_model_id=page_model_id)
        if state_template_id:
            q = q.filter_by(state_template_id=state_template_id)
        q = q.order_by(CandidateEvidenceRecord.created_at.desc())
        return [_row_to_dict(r) for r in q.all()]

    def aggregate_summary(self, session, stable_key_id: str) -> dict:
        # SQL-level aggregation: avoids loading all rows into memory
        rows = (
            session.query(
                CandidateEvidenceRecord.provider,
                func.count(CandidateEvidenceRecord.evidence_id).label("cnt"),
                func.max(CandidateEvidenceRecord.raw_confidence).label("best_conf"),
                func.max(CandidateEvidenceRecord.created_at).label("last_seen"),
                func.max(CandidateEvidenceRecord.action_score).label("best_action"),
            )
            .filter_by(stable_key_id=stable_key_id)
            .group_by(CandidateEvidenceRecord.provider)
            .all()
        )
        return {
            row.provider: {
                "count": row.cnt,
                "best_confidence": row.best_conf or 0.0,
                "last_seen": row.last_seen,
                "best_action_score": row.best_action,
            }
            for row in rows
        }

    def delete_by_canvas(self, session, canvas_id: str) -> int:
        count = (
            session.query(CandidateEvidenceRecord)
            .filter_by(canvas_id=canvas_id)
            .delete()
        )
        session.flush()
        return count
