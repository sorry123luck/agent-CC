"""Persistence for derived AppShell / Region / Element model templates."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from src.storage.schema import AppShellTemplateRecord, ElementTemplateRecord, RegionTemplateRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelTemplateStore:
    """Stores derived model workbench summaries as durable template snapshots."""

    def upsert_workbench_summary(
        self,
        session,
        *,
        app_id: str,
        page_model_id: str | None,
        state_template_id: str | None,
        display_name: str | None,
        surface_type: str | None,
        page_class: str,
        state_label: str | None,
        workbench: dict[str, Any],
    ) -> dict[str, int]:
        now = _now()
        shell_count = self._upsert_shell(
            session,
            app_id=app_id,
            page_model_id=page_model_id,
            state_template_id=state_template_id,
            display_name=display_name,
            surface_type=surface_type,
            page_class=page_class,
            state_label=state_label,
            shell=workbench.get("app_shell") or {},
            now=now,
        )
        region_count = self._upsert_regions(
            session,
            app_id=app_id,
            page_model_id=page_model_id,
            state_template_id=state_template_id,
            regions=list(workbench.get("region_templates") or []),
            now=now,
        )
        element_count = self._upsert_elements(
            session,
            app_id=app_id,
            page_model_id=page_model_id,
            state_template_id=state_template_id,
            elements=list(workbench.get("element_templates") or []),
            now=now,
        )
        return {"app_shell": shell_count, "regions": region_count, "elements": element_count}

    def list_region_templates(self, session, state_template_id: str) -> list[dict[str, Any]]:
        rows = (
            session.query(RegionTemplateRecord)
            .filter(RegionTemplateRecord.state_template_id == state_template_id)
            .order_by(RegionTemplateRecord.region_id.asc())
            .all()
        )
        return [_json_row(row.template_json) for row in rows]

    def get_app_shell(self, session, state_template_id: str) -> dict[str, Any] | None:
        row = (
            session.query(AppShellTemplateRecord)
            .filter(AppShellTemplateRecord.state_template_id == state_template_id)
            .first()
        )
        if row is None:
            return None
        return _json_row(row.shell_json)

    def list_element_templates(self, session, state_template_id: str) -> list[dict[str, Any]]:
        rows = (
            session.query(ElementTemplateRecord)
            .filter(ElementTemplateRecord.state_template_id == state_template_id)
            .order_by(ElementTemplateRecord.region_id.asc(), ElementTemplateRecord.label.asc())
            .all()
        )
        return [_json_row(row.template_json) for row in rows]

    def _upsert_shell(
        self,
        session,
        *,
        app_id: str,
        page_model_id: str | None,
        state_template_id: str | None,
        display_name: str | None,
        surface_type: str | None,
        page_class: str,
        state_label: str | None,
        shell: dict[str, Any],
        now: str,
    ) -> int:
        row = (
            session.query(AppShellTemplateRecord)
            .filter(
                AppShellTemplateRecord.app_id == app_id,
                AppShellTemplateRecord.state_template_id == state_template_id,
            )
            .first()
        )
        if row is None:
            row = AppShellTemplateRecord(shell_id=str(uuid.uuid4()), app_id=app_id, state_template_id=state_template_id)
            session.add(row)
        row.page_model_id = page_model_id
        row.display_name = display_name
        row.surface_type = surface_type
        row.page_class = page_class
        row.state_label = state_label
        row.shell_json = json.dumps(shell, ensure_ascii=False)
        row.source = "derived"
        row.updated_at = now
        return 1

    def _upsert_regions(
        self,
        session,
        *,
        app_id: str,
        page_model_id: str | None,
        state_template_id: str | None,
        regions: list[dict[str, Any]],
        now: str,
    ) -> int:
        count = 0
        for idx, region in enumerate(regions):
            region_id = str(region.get("region_id") or f"region_{idx}")
            row = (
                session.query(RegionTemplateRecord)
                .filter(
                    RegionTemplateRecord.state_template_id == state_template_id,
                    RegionTemplateRecord.region_id == region_id,
                )
                .first()
            )
            if row is None:
                row = RegionTemplateRecord(
                    region_template_id=str(uuid.uuid4()),
                    app_id=app_id,
                    state_template_id=state_template_id,
                    region_id=region_id,
                )
                session.add(row)
            row.page_model_id = page_model_id
            row.role = str(region.get("role") or "unknown")
            row.purpose = str(region.get("purpose") or "")
            row.bounds_json = json.dumps(region.get("bounds") or [])
            row.candidate_count = int(region.get("candidate_count") or 0)
            row.source = str(region.get("source") or "derived")
            row.template_json = json.dumps(region, ensure_ascii=False)
            row.updated_at = now
            count += 1
        return count

    def _upsert_elements(
        self,
        session,
        *,
        app_id: str,
        page_model_id: str | None,
        state_template_id: str | None,
        elements: list[dict[str, Any]],
        now: str,
    ) -> int:
        count = 0
        for idx, element in enumerate(elements):
            key_id = str(element.get("key_id") or f"element_{idx}")
            row = (
                session.query(ElementTemplateRecord)
                .filter(
                    ElementTemplateRecord.state_template_id == state_template_id,
                    ElementTemplateRecord.key_id == key_id,
                )
                .first()
            )
            if row is None:
                row = ElementTemplateRecord(
                    element_template_id=str(uuid.uuid4()),
                    app_id=app_id,
                    state_template_id=state_template_id,
                    key_id=key_id,
                )
                session.add(row)
            row.page_model_id = page_model_id
            row.role = str(element.get("role") or "unknown")
            row.label = str(element.get("label") or "")
            row.region_id = str(element.get("region_id") or "")
            row.kind = str(element.get("kind") or "")
            row.actionability = str(element.get("actionability") or "review")
            row.confidence = float(element.get("confidence") or 0.0)
            row.source = str(element.get("source") or "derived")
            row.template_json = json.dumps(element, ensure_ascii=False)
            row.updated_at = now
            count += 1
        return count


def _json_row(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}
