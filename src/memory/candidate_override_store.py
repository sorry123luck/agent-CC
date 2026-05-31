"""Persistent candidate override store.

Manual/agent/VLM corrections are stored as an overlay keyed by a scoped
candidate identity. Raw perception records stay unchanged; callers merge this
layer when they need effective candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from uuid import uuid4

from src.storage.schema import CandidateOverrideRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_list(value: list[float] | list[int] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _load_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        loaded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, list) else None


@dataclass(frozen=True)
class CandidateOverrideData:
    override_id: str
    scope_key: str
    app_id: str | None = None
    page_model_id: str | None = None
    state_template_id: str | None = None
    canvas_id: str | None = None
    element_id: str | None = None
    stable_key_id: str | None = None
    label: str | None = None
    semantic_role: str | None = None
    visual_type: str | None = None
    region_id: str | None = None
    kind: str | None = None
    relative_bounds: list[float] | None = None
    absolute_bounds: list[float] | None = None
    source: str = "manual"
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""


class CandidateOverrideStore:
    """SQLite-backed candidate correction overlay."""

    def upsert(
        self,
        session,
        *,
        scope_key: str,
        app_id: str | None = None,
        page_model_id: str | None = None,
        state_template_id: str | None = None,
        canvas_id: str | None = None,
        element_id: str | None = None,
        stable_key_id: str | None = None,
        label: str | None = None,
        semantic_role: str | None = None,
        visual_type: str | None = None,
        region_id: str | None = None,
        kind: str | None = None,
        relative_bounds: list[float] | None = None,
        absolute_bounds: list[float] | None = None,
        source: str = "manual",
        status: str = "active",
    ) -> CandidateOverrideData:
        now = _now()
        row = (
            session.query(CandidateOverrideRecord)
            .filter(CandidateOverrideRecord.scope_key == scope_key)
            .one_or_none()
        )
        if row is None:
            row = CandidateOverrideRecord(
                override_id=str(uuid4()),
                scope_key=scope_key,
                created_at=now,
                updated_at=now,
            )
            session.add(row)

        row.app_id = app_id
        row.page_model_id = page_model_id
        row.state_template_id = state_template_id
        row.canvas_id = canvas_id
        row.element_id = element_id
        row.stable_key_id = stable_key_id
        row.label = label
        row.semantic_role = semantic_role
        row.visual_type = visual_type
        row.region_id = region_id
        row.kind = kind
        row.relative_bounds = _dump_list(relative_bounds)
        row.absolute_bounds = _dump_list(absolute_bounds)
        row.source = source or "manual"
        row.status = status or "active"
        row.updated_at = now
        session.flush()
        return self._row_to_data(row)

    def list(
        self,
        session,
        *,
        state_template_id: str | None = None,
        canvas_id: str | None = None,
        page_model_id: str | None = None,
    ) -> list[CandidateOverrideData]:
        query = session.query(CandidateOverrideRecord)
        filters = []
        if state_template_id:
            filters.append(CandidateOverrideRecord.state_template_id == state_template_id)
        if canvas_id:
            filters.append(CandidateOverrideRecord.canvas_id == canvas_id)
        if page_model_id:
            filters.append(CandidateOverrideRecord.page_model_id == page_model_id)
        if filters:
            from sqlalchemy import or_

            query = query.filter(or_(*filters))
        return [self._row_to_data(row) for row in query.order_by(CandidateOverrideRecord.updated_at.desc()).all()]

    def delete(self, session, scope_key: str) -> bool:
        row = (
            session.query(CandidateOverrideRecord)
            .filter(CandidateOverrideRecord.scope_key == scope_key)
            .one_or_none()
        )
        if row is None:
            return False
        session.delete(row)
        session.flush()
        return True

    def _row_to_data(self, row: CandidateOverrideRecord) -> CandidateOverrideData:
        return CandidateOverrideData(
            override_id=row.override_id,
            scope_key=row.scope_key,
            app_id=row.app_id,
            page_model_id=row.page_model_id,
            state_template_id=row.state_template_id,
            canvas_id=row.canvas_id,
            element_id=row.element_id,
            stable_key_id=row.stable_key_id,
            label=row.label,
            semantic_role=row.semantic_role,
            visual_type=row.visual_type,
            region_id=row.region_id,
            kind=row.kind,
            relative_bounds=_load_list(row.relative_bounds),
            absolute_bounds=_load_list(row.absolute_bounds),
            source=row.source or "manual",
            status=row.status or "active",
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
