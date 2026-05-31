from src.execution.action_executor import ActionResult
from src.execution.failure_recovery import FailureRecovery, LocatorCandidate, RecoveryStrategy


def test_execute_locator_fallback_chain_uses_second_candidate_after_failure():
    recovery = FailureRecovery()
    calls: list[str] = []

    result = recovery.execute_locator_fallback_chain(
        candidates=[
            LocatorCandidate(
                name="uia",
                strategy=RecoveryStrategy.FALLBACK_ELEMENT,
                action_fn=lambda: calls.append("uia") or ActionResult(
                    success=False,
                    action="click",
                    details="uia failed",
                    error="uia_failed",
                ),
            ),
            LocatorCandidate(
                name="vision",
                strategy=RecoveryStrategy.FALLBACK_COORDINATE,
                action_fn=lambda: calls.append("vision") or ActionResult(
                    success=True,
                    action="click",
                    details="vision ok",
                ),
            ),
        ]
    )

    assert result.recovered is True
    assert result.strategy_used == RecoveryStrategy.FALLBACK_COORDINATE
    assert result.attempts == 2
    assert calls == ["uia", "vision"]


def test_plan_locator_candidates_prefers_lower_risk_and_cost_with_same_rank():
    recovery = FailureRecovery()

    candidates = recovery.plan_locator_candidates(
        [
            LocatorCandidate(
                name="vision_bbox",
                preferred_rank=2,
                risk_tier=2,
                cost_score=0.5,
                durability_score=0.4,
                confidence=0.6,
                action_fn=lambda: ActionResult(success=True, action="click", details="vision"),
            ),
            LocatorCandidate(
                name="relative",
                preferred_rank=2,
                risk_tier=1,
                cost_score=0.2,
                durability_score=0.7,
                confidence=0.8,
                action_fn=lambda: ActionResult(success=True, action="click", details="relative"),
            ),
        ]
    )

    assert [candidate.name for candidate in candidates] == ["relative", "vision_bbox"]


def test_plan_locator_candidates_prefers_anchor_and_scroll_context_when_cost_equal():
    recovery = FailureRecovery()

    candidates = recovery.plan_locator_candidates(
        [
            LocatorCandidate(
                name="viewport_relative",
                preferred_rank=2,
                risk_tier=1,
                cost_score=0.2,
                durability_score=0.6,
                confidence=0.7,
                anchor_count=2,
                scroll_context=True,
                content_group_id="cg_viewport",
                action_fn=lambda: ActionResult(success=True, action="click", details="viewport"),
            ),
            LocatorCandidate(
                name="plain_relative",
                preferred_rank=2,
                risk_tier=1,
                cost_score=0.2,
                durability_score=0.6,
                confidence=0.7,
                anchor_count=0,
                scroll_context=False,
                content_group_id=None,
                action_fn=lambda: ActionResult(success=True, action="click", details="plain"),
            ),
        ]
    )

    assert [candidate.name for candidate in candidates] == ["viewport_relative", "plain_relative"]


def test_execute_locator_fallback_chain_marks_viewport_recovery_when_all_candidates_fail():
    recovery = FailureRecovery()

    result = recovery.execute_locator_fallback_chain(
        candidates=[
            LocatorCandidate(
                name="viewport_relative",
                strategy=RecoveryStrategy.VIEWPORT_RECOVERY,
                scroll_context=True,
                region_role="viewport",
                action_fn=lambda: ActionResult(success=False, action="click", details="miss"),
            )
        ]
    )

    assert result.recovered is False
    assert result.strategy_used == RecoveryStrategy.VIEWPORT_RECOVERY
    assert result.error == "locator_chain_failed_viewport_recovery_required"
    assert result.details["reason"] == "viewport_context_requires_recovery"
    assert result.details["escalation_path"] == ["viewport_relative:viewport_recovery"]
    assert result.details["next_suggestion"]["action"] == "scroll_and_recapture"


def test_plan_locator_candidates_prefers_interaction_hint_and_structure_evidence():
    recovery = FailureRecovery()

    candidates = recovery.plan_locator_candidates(
        [
            LocatorCandidate(
                name="relative_plain",
                preferred_rank=1,
                risk_tier=1,
                cost_score=0.2,
                durability_score=0.5,
                confidence=0.6,
                action_fn=lambda: ActionResult(success=True, action="click", details="plain"),
            ),
            LocatorCandidate(
                name="relative_guided",
                preferred_rank=1,
                risk_tier=1,
                cost_score=0.2,
                durability_score=0.5,
                confidence=0.6,
                interaction_hints={"preferred_action": "click"},
                structure_evidence_score=0.85,
                action_fn=lambda: ActionResult(success=True, action="click", details="guided"),
            ),
        ]
    )

    assert [candidate.name for candidate in candidates] == ["relative_guided", "relative_plain"]


def test_execute_locator_fallback_chain_uses_candidate_recovery_fn_after_failure():
    recovery = FailureRecovery()
    calls: list[str] = []

    result = recovery.execute_locator_fallback_chain(
        candidates=[
            LocatorCandidate(
                name="viewport_relative",
                strategy=RecoveryStrategy.VIEWPORT_RECOVERY,
                scroll_context=True,
                region_role="viewport",
                action_fn=lambda: calls.append("action") or ActionResult(success=False, action="click", details="miss"),
                recovery_fn=lambda: calls.append("recover") or ActionResult(success=True, action="click", details="after_scroll"),
            )
        ]
    )

    assert result.recovered is True
    assert result.strategy_used == RecoveryStrategy.VIEWPORT_RECOVERY
    assert result.attempts == 2
    assert calls == ["action", "recover"]
    assert result.details["candidate"] == "viewport_relative"
    assert result.details["attempt_trace"][0]["stage"] == "action"
    assert result.details["attempt_trace"][1]["stage"] == "recovery"


def test_execute_locator_fallback_chain_records_verify_failure_before_terminal_escalation():
    recovery = FailureRecovery()

    verify_calls = {"count": 0}

    def verify_fn():
        verify_calls["count"] += 1
        return type("VerifyResult", (), {"verified": False})()

    result = recovery.execute_locator_fallback_chain(
        candidates=[
            LocatorCandidate(
                name="grid_relative",
                strategy=RecoveryStrategy.VIEWPORT_RECOVERY,
                scroll_context=True,
                region_role="viewport",
                action_fn=lambda: ActionResult(success=True, action="click", details="hit"),
            )
        ],
        verify_fn=verify_fn,
    )

    assert result.recovered is False
    assert result.strategy_used == RecoveryStrategy.VIEWPORT_RECOVERY
    assert verify_calls["count"] == 1
    assert any(item["stage"] == "verify" for item in result.details["attempt_trace"])
    assert result.details["next_suggestion"]["error"] == "locator_chain_failed_viewport_recovery_required"
    assert result.details["next_suggestion"]["strategy"] == "viewport_recovery"


def test_execute_locator_fallback_chain_passes_candidate_and_result_to_verify_fn():
    recovery = FailureRecovery()
    captured: dict[str, str] = {}

    def verify_fn(candidate, result):
        captured["candidate"] = candidate.name
        captured["details"] = result.details
        return type("VerifyResult", (), {"verified": True, "message": "ok", "details": {}})()

    result = recovery.execute_locator_fallback_chain(
        candidates=[
            LocatorCandidate(
                name="dom",
                strategy=RecoveryStrategy.FALLBACK_ELEMENT,
                action_fn=lambda: ActionResult(success=True, action="click", details="dom ok"),
            )
        ],
        verify_fn=verify_fn,
    )

    assert result.recovered is True
    assert captured == {"candidate": "dom", "details": "dom ok"}


def test_plan_locator_candidates_prefers_viewport_then_anchor_then_replan():
    recovery = FailureRecovery()

    candidates = recovery.plan_locator_candidates(
        [
            LocatorCandidate(
                name="replan_candidate",
                preferred_rank=0,
                risk_tier=0,
                cost_score=0.1,
                durability_score=0.8,
                confidence=0.8,
                interaction_hints={"expected_effect": "open_dialog"},
                structure_evidence_score=0.85,
                replan_hint=True,
                action_fn=lambda: ActionResult(success=True, action="click", details="replan"),
            ),
            LocatorCandidate(
                name="anchor_candidate",
                preferred_rank=0,
                risk_tier=0,
                cost_score=0.1,
                durability_score=0.8,
                confidence=0.8,
                anchor_count=2,
                content_group_id="cg_action_bar",
                region_role="action_bar",
                action_fn=lambda: ActionResult(success=True, action="click", details="anchor"),
            ),
            LocatorCandidate(
                name="viewport_candidate",
                preferred_rank=0,
                risk_tier=0,
                cost_score=0.1,
                durability_score=0.8,
                confidence=0.8,
                scroll_context=True,
                viewport_index=1,
                region_role="viewport",
                action_fn=lambda: ActionResult(success=True, action="click", details="viewport"),
            ),
        ]
    )

    assert [candidate.name for candidate in candidates] == [
        "viewport_candidate",
        "anchor_candidate",
        "replan_candidate",
    ]
