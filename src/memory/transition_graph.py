"""Transition Graph Manager — record and query historical page state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.storage.schema import ControlStateTransition, TransitionEdge


@dataclass
class TransitionInfo:
    """Information about a historical transition."""
    from_page_class: str
    to_page_class: str
    trigger_action: str | None
    trigger_candidate_key: str | None
    observe_count: int
    success_count: int
    success_rate: float
    last_observed_at: datetime | None
    drifted: bool


@dataclass
class ControlTransitionInfo:
    candidate_key: str
    stable_key_id: str | None
    from_page_class: str
    to_page_class: str
    action_type: str | None
    observe_count: int
    success_count: int
    failure_count: int
    success_rate: float
    last_observed_at: datetime | None
    canvas_id_before: str | None = None
    canvas_id_after: str | None = None


class TransitionGraphManager:
    """Record and query historical page state transitions.

    This is NOT a prediction engine or auto-execution planner.
    It only records what has been observed historically.
    """

    def __init__(self, session: Session):
        self._session = session

    def record_transition(
        self,
        from_class: str,
        to_class: str,
        action: str | None = None,
        candidate_key: str | None = None,
        success: bool = True,
    ) -> None:
        """Record one observed transition."""
        existing = (
            self._session.query(TransitionEdge)
            .filter_by(
                from_page_class=from_class,
                to_page_class=to_class,
                trigger_action=action,
            )
            .first()
        )

        now = datetime.now(timezone.utc)
        if existing:
            existing.observe_count += 1
            if success:
                existing.success_count += 1
            existing.last_observed_at = now
        else:
            edge = TransitionEdge(
                from_page_class=from_class,
                to_page_class=to_class,
                trigger_action=action,
                trigger_candidate_key=candidate_key,
                observe_count=1,
                success_count=1 if success else 0,
                last_observed_at=now,
            )
            self._session.add(edge)

        if candidate_key:
            self.record_control_transition(
                candidate_key=candidate_key,
                from_class=from_class,
                to_class=to_class,
                action_type=action,
                success=success,
                commit=False,
            )

        self._session.commit()

    def record_control_transition(
        self,
        *,
        candidate_key: str,
        from_class: str,
        to_class: str,
        action_type: str | None = None,
        stable_key_id: str | None = None,
        canvas_id_before: str | None = None,
        canvas_id_after: str | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> None:
        """Record one control → state transition association."""
        import json

        row = (
            self._session.query(ControlStateTransition)
            .filter_by(
                candidate_key=candidate_key,
                from_page_class=from_class,
                to_page_class=to_class,
                action_type=action_type,
            )
            .first()
        )
        now = datetime.now(timezone.utc)
        if row is None:
            row = ControlStateTransition(
                candidate_key=candidate_key,
                stable_key_id=stable_key_id,
                from_page_class=from_class,
                to_page_class=to_class,
                action_type=action_type,
                observe_count=0,
                success_count=0,
                failure_count=0,
            )
            self._session.add(row)
        row.stable_key_id = stable_key_id or row.stable_key_id
        row.canvas_id_before = canvas_id_before or row.canvas_id_before
        row.canvas_id_after = canvas_id_after or row.canvas_id_after
        row.observe_count = (row.observe_count or 0) + 1
        if success:
            row.success_count = (row.success_count or 0) + 1
        else:
            row.failure_count = (row.failure_count or 0) + 1
        row.last_observed_at = now
        if metadata:
            row.metadata_json = json.dumps(metadata, ensure_ascii=False)
        if commit:
            self._session.commit()

    def get_transitions(self, page_class: str) -> list[TransitionInfo]:
        """Get all historical transitions from a given page class."""
        edges = (
            self._session.query(TransitionEdge)
            .filter_by(from_page_class=page_class)
            .all()
        )
        return [self._edge_to_info(e) for e in edges]

    def get_candidate_experience(self, candidate_key: str) -> list[TransitionInfo]:
        """Get transitions triggered by a specific candidate."""
        edges = (
            self._session.query(TransitionEdge)
            .filter_by(trigger_candidate_key=candidate_key)
            .all()
        )
        return [self._edge_to_info(e) for e in edges]

    def get_control_transitions(self, candidate_key: str) -> list[ControlTransitionInfo]:
        rows = (
            self._session.query(ControlStateTransition)
            .filter_by(candidate_key=candidate_key)
            .order_by(ControlStateTransition.last_observed_at.desc())
            .all()
        )
        return [self._control_edge_to_info(row) for row in rows]

    def get_control_transitions_from_state(self, page_class: str) -> list[ControlTransitionInfo]:
        rows = (
            self._session.query(ControlStateTransition)
            .filter_by(from_page_class=page_class)
            .order_by(ControlStateTransition.last_observed_at.desc())
            .all()
        )
        return [self._control_edge_to_info(row) for row in rows]

    def mark_drifted(
        self,
        from_class: str,
        to_class: str,
        action: str | None = None,
    ) -> bool:
        """Mark a transition as drifted. Returns True if found and updated."""
        edge = (
            self._session.query(TransitionEdge)
            .filter_by(
                from_page_class=from_class,
                to_page_class=to_class,
                trigger_action=action,
            )
            .first()
        )
        if edge:
            edge.drifted = True
            edge.drift_detected_at = datetime.now(timezone.utc)
            self._session.commit()
            return True
        return False

    def _edge_to_info(self, edge: TransitionEdge) -> TransitionInfo:
        return TransitionInfo(
            from_page_class=edge.from_page_class,
            to_page_class=edge.to_page_class,
            trigger_action=edge.trigger_action,
            trigger_candidate_key=edge.trigger_candidate_key,
            observe_count=edge.observe_count,
            success_count=edge.success_count,
            success_rate=edge.success_count / max(edge.observe_count, 1),
            last_observed_at=edge.last_observed_at,
            drifted=edge.drifted,
        )

    def _control_edge_to_info(self, edge: ControlStateTransition) -> ControlTransitionInfo:
        observe_count = edge.observe_count or 0
        success_count = edge.success_count or 0
        return ControlTransitionInfo(
            candidate_key=edge.candidate_key,
            stable_key_id=edge.stable_key_id,
            from_page_class=edge.from_page_class,
            to_page_class=edge.to_page_class,
            action_type=edge.action_type,
            observe_count=observe_count,
            success_count=success_count,
            failure_count=edge.failure_count or 0,
            success_rate=success_count / max(observe_count, 1),
            last_observed_at=edge.last_observed_at,
            canvas_id_before=edge.canvas_id_before,
            canvas_id_after=edge.canvas_id_after,
        )

    def record_vlm_transitions(
        self,
        transitions: list[dict[str, Any]],
        page_model_id: str,
    ) -> int:
        """VLM 推断转场 → transition_edges 表。

        使用 "vlm:{trigger}" 作为 action 前缀区分 VLM 推断的转场。
        transitions 来自 PageSemanticModel.transitions，每项包含
        from_state, trigger, to_state, confidence。

        Returns:
            写入/更新的转场数
        """
        count = 0
        for t in transitions:
            from_state = t.get("from_state", "")
            to_state = t.get("to_state", "")
            trigger = t.get("trigger", "")
            if not from_state or not to_state:
                continue
            action = f"vlm:{trigger}" if trigger else "vlm:inferred"
            self.record_transition(
                from_class=from_state,
                to_class=to_state,
                action=action,
                success=True,
            )
            count += 1
        return count
