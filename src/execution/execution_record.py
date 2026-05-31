"""Formal execution-record helpers shared by runtime and tooling."""

from __future__ import annotations

from typing import Any


def build_execution_record(outcome: Any) -> dict[str, Any]:
    """Build a stable canonical execution record from an ActionOutcome-like payload."""
    if hasattr(outcome, "to_dict"):
        outcome = outcome.to_dict()
    if not isinstance(outcome, dict):
        return {"error": "unsupported_action_outcome"}

    verification_details = dict(outcome.get("verification_details") or {})
    recovery = verification_details.get("recovery") or {}
    recovery_execution = verification_details.get("recovery_execution") or {}
    runtime_trace = verification_details.get("runtime_trace") or {}

    return {
        "success": outcome.get("success"),
        "action": outcome.get("action"),
        "message": outcome.get("message"),
        "attempts": outcome.get("attempts"),
        "strategy": outcome.get("strategy"),
        "error": outcome.get("error"),
        "verification_status": verification_details.get("status"),
        "verification_policy": verification_details.get("policy"),
        "verification_element_id": verification_details.get("verification_element_id"),
        "risk_level": verification_details.get("risk_level"),
        "policy_mode": verification_details.get("policy_mode"),
        "policy_source": verification_details.get("policy_source"),
        "reason": verification_details.get("reason"),
        "recovery_mode": recovery_execution.get("mode"),
        "original_element_id": recovery_execution.get("original"),
        "resolved_element_id": recovery_execution.get("resolved"),
        "terminal_reason": recovery.get("reason"),
        "terminal_strategy": recovery.get("terminal_strategy") or recovery.get("strategy"),
        "next_suggestion": recovery.get("next_suggestion"),
        "attempt_trace": recovery.get("attempt_trace"),
        "runtime_trace": runtime_trace,
        "decision_record": verification_details.get("decision_record"),
        "decision_candidate": verification_details.get("decision_candidate"),
        "decision_candidate_source": verification_details.get("decision_candidate_source"),
        "decision_candidate_kind": verification_details.get("decision_candidate_kind"),
        "decision_candidate_bbox": verification_details.get("decision_candidate_bbox"),
        "decision_candidate_origin": verification_details.get("decision_candidate_origin"),
        "recrop_request": verification_details.get("recrop_request"),
        "decision_loop": verification_details.get("decision_loop"),
        "decision_round_count": len(verification_details.get("decision_loop") or []),
        "typed_text": verification_details.get("typed_text"),
        "used_bbox_fallback": bool(
            dict(verification_details.get("decision_candidate") or {})
            .get("attributes", {})
            .get("used_bbox_fallback")
        ),
    }
