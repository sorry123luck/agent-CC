"""Failure recovery and locator-chain planning for the Pro runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.execution.action_executor import ActionExecutor, ActionResult
from src.execution.action_verifier import ActionVerifier, VerificationResult


class RecoveryStrategy(Enum):
    """Recovery strategies used by the runtime."""

    RETRY = "retry"
    FALLBACK_COORDINATE = "fallback_coordinate"
    FALLBACK_ELEMENT = "fallback_element"
    VIEWPORT_RECOVERY = "viewport_recovery"
    ANCHOR_RELOCATE = "anchor_relocate"
    REPLAN = "replan"
    ABORT = "abort"


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_attempts: int = 3
    initial_delay: float = 0.3
    backoff_multiplier: float = 2.0
    max_delay: float = 5.0


@dataclass
class FallbackConfig:
    """Fallback options for legacy coordinate flows."""

    fallback_coordinates: list[tuple[int, int]] = field(default_factory=list)
    fallback_automation_ids: list[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Final recovery result."""

    recovered: bool
    strategy_used: RecoveryStrategy
    attempts: int
    final_result: ActionResult | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


@dataclass
class LocatorCandidate:
    """One runtime candidate in the locator-first recovery chain."""

    name: str
    action_fn: Callable[[], ActionResult]
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    preferred_rank: int = 999
    risk_tier: int = 9
    cost_score: float = 1.0
    durability_score: float = 0.0
    confidence: float = 0.0
    anchor_count: int = 0
    scroll_context: bool = False
    content_group_id: str | None = None
    region_role: str | None = None
    interaction_hints: dict[str, Any] = field(default_factory=dict)
    structure_evidence_score: float = 0.0
    viewport_index: int | None = None
    replan_hint: bool = False
    recovery_fn: Callable[[], ActionResult] | None = None


class FailureRecovery:
    """Retries actions and executes locator fallback chains."""

    def __init__(
        self,
        executor: ActionExecutor | None = None,
        verifier: ActionVerifier | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._executor = executor or ActionExecutor()
        self._verifier = verifier or ActionVerifier()
        self._retry_config = retry_config or RetryConfig()

    def execute_with_recovery(
        self,
        action_fn: Callable[[], ActionResult],
        verify_fn: Callable[..., VerificationResult] | None = None,
        fallback_config: FallbackConfig | None = None,
    ) -> RecoveryResult:
        """Execute an action with basic retry and legacy fallbacks."""
        attempts = 0
        last_result: ActionResult | None = None

        while attempts < self._retry_config.max_attempts:
            attempts += 1
            last_result = action_fn()
            if not last_result.success:
                time.sleep(self._get_delay(attempts))
                continue

            if verify_fn is not None and not verify_fn().verified:
                time.sleep(self._get_delay(attempts))
                continue

            return RecoveryResult(
                recovered=True,
                strategy_used=RecoveryStrategy.RETRY,
                attempts=attempts,
                final_result=last_result,
            )

        if fallback_config and fallback_config.fallback_automation_ids:
            for _automation_id in fallback_config.fallback_automation_ids:
                attempts += 1
                last_result = action_fn()
                if not last_result.success:
                    continue
                if verify_fn is not None and not verify_fn().verified:
                    continue
                return RecoveryResult(
                    recovered=True,
                    strategy_used=RecoveryStrategy.FALLBACK_ELEMENT,
                    attempts=attempts,
                    final_result=last_result,
                )

        return RecoveryResult(
            recovered=False,
            strategy_used=RecoveryStrategy.ABORT,
            attempts=attempts,
            final_result=last_result,
            error="all_recovery_strategies_failed",
        )

    def click_with_recovery(
        self,
        hwnd: int,
        x: int,
        y: int,
        verify_fn: Callable[..., VerificationResult] | None = None,
        fallback_config: FallbackConfig | None = None,
    ) -> RecoveryResult:
        """Click a point with retry and optional coordinate/UIA fallbacks."""
        attempts = 0
        last_result: ActionResult | None = None

        while attempts < self._retry_config.max_attempts:
            attempts += 1
            last_result = self._executor.click(hwnd, x, y)
            if not last_result.success:
                time.sleep(self._get_delay(attempts))
                continue

            if verify_fn is not None and not verify_fn().verified:
                time.sleep(self._get_delay(attempts))
                continue

            return RecoveryResult(
                recovered=True,
                strategy_used=RecoveryStrategy.RETRY,
                attempts=attempts,
                final_result=last_result,
            )

        if fallback_config and fallback_config.fallback_coordinates:
            for fx, fy in fallback_config.fallback_coordinates:
                attempts += 1
                last_result = self._executor.click(hwnd, fx, fy)
                if not last_result.success:
                    continue
                if verify_fn is not None and not verify_fn().verified:
                    continue
                return RecoveryResult(
                    recovered=True,
                    strategy_used=RecoveryStrategy.FALLBACK_COORDINATE,
                    attempts=attempts,
                    final_result=last_result,
                )

        if fallback_config and fallback_config.fallback_automation_ids:
            for automation_id in fallback_config.fallback_automation_ids:
                attempts += 1
                last_result = self._executor.invoke_uia_element(hwnd, automation_id=automation_id)
                if not last_result.success:
                    continue
                if verify_fn is not None and not verify_fn().verified:
                    continue
                return RecoveryResult(
                    recovered=True,
                    strategy_used=RecoveryStrategy.FALLBACK_ELEMENT,
                    attempts=attempts,
                    final_result=last_result,
                )

        return RecoveryResult(
            recovered=False,
            strategy_used=RecoveryStrategy.ABORT,
            attempts=attempts,
            final_result=last_result,
            error="all_recovery_strategies_failed",
        )

    def execute_locator_fallback_chain(
        self,
        candidates: list[LocatorCandidate],
        verify_fn: Callable[..., VerificationResult] | None = None,
    ) -> RecoveryResult:
        """Try runtime candidates in policy order until one verifies."""
        attempts = 0
        last_result: ActionResult | None = None
        planned_candidates = self.plan_locator_candidates(candidates)
        escalation_path = [self._strategy_label(candidate) for candidate in planned_candidates]
        attempt_trace: list[dict[str, Any]] = []

        for candidate in planned_candidates:
            attempts += 1
            last_result = candidate.action_fn()
            attempt_trace.append(
                {
                    "candidate": candidate.name,
                    "stage": "action",
                    "strategy": candidate.strategy.value,
                    "success": last_result.success if last_result is not None else False,
                    "error": last_result.error if last_result is not None else None,
                }
            )
            if not last_result.success and candidate.recovery_fn is not None:
                attempts += 1
                last_result = candidate.recovery_fn()
                attempt_trace.append(
                    {
                        "candidate": candidate.name,
                        "stage": "recovery",
                        "strategy": candidate.strategy.value,
                        "success": last_result.success if last_result is not None else False,
                        "error": last_result.error if last_result is not None else None,
                    }
                )
            if not last_result.success:
                continue

            verification_result = self._run_verify_fn(verify_fn, candidate, last_result)
            if verification_result is not None and not verification_result.verified:
                attempt_trace.append(
                    {
                        "candidate": candidate.name,
                        "stage": "verify",
                        "strategy": candidate.strategy.value,
                        "success": False,
                        "error": "verification_failed",
                        "verification_message": getattr(verification_result, "message", None),
                        "verification_details": getattr(verification_result, "details", None),
                    }
                )
                if candidate.recovery_fn is not None:
                    attempts += 1
                    last_result = candidate.recovery_fn()
                    attempt_trace.append(
                        {
                            "candidate": candidate.name,
                            "stage": "recovery_after_verify",
                            "strategy": candidate.strategy.value,
                            "success": last_result.success if last_result is not None else False,
                            "error": last_result.error if last_result is not None else None,
                        }
                    )
                    if not last_result.success:
                        continue
                    verification_result = self._run_verify_fn(verify_fn, candidate, last_result)
                    if verification_result is not None and not verification_result.verified:
                        attempt_trace.append(
                            {
                                "candidate": candidate.name,
                                "stage": "verify_after_recovery",
                                "strategy": candidate.strategy.value,
                                "success": False,
                                "error": "verification_failed",
                                "verification_message": getattr(verification_result, "message", None),
                                "verification_details": getattr(verification_result, "details", None),
                            }
                        )
                        continue
                else:
                    continue

            if last_result is None or not last_result.success:
                continue

            return RecoveryResult(
                recovered=True,
                strategy_used=candidate.strategy,
                attempts=attempts,
                final_result=last_result,
                details={
                    "candidate": candidate.name,
                    "strategy": candidate.strategy.value,
                    "escalation_path": escalation_path,
                    "reason": "candidate_verified",
                    "attempt_trace": attempt_trace,
                    "next_suggestion": None,
                },
            )

        terminal_strategy = self._select_terminal_strategy(planned_candidates)
        terminal_error = self._build_terminal_error(planned_candidates)
        terminal_reason = self._build_terminal_reason(planned_candidates)
        return RecoveryResult(
            recovered=False,
            strategy_used=terminal_strategy,
            attempts=attempts,
            final_result=last_result,
            error=terminal_error,
            details={
                "escalation_path": escalation_path,
                "terminal_strategy": terminal_strategy.value,
                "reason": terminal_reason,
                "attempt_trace": attempt_trace,
                "next_suggestion": self._build_next_suggestion(terminal_strategy, terminal_error, terminal_reason),
            },
        )

    def plan_locator_candidates(
        self,
        candidates: list[LocatorCandidate],
    ) -> list[LocatorCandidate]:
        """Sort locator candidates using runtime evidence and recovery policy."""
        return sorted(
            candidates,
            key=lambda candidate: (
                self._recovery_stage(candidate),
                candidate.preferred_rank,
                candidate.risk_tier,
                candidate.cost_score,
                0 if self._prefers_viewport_recovery(candidate) else 1,
                0 if self._prefers_anchor_relocation(candidate) else 1,
                -candidate.anchor_count,
                0 if candidate.scroll_context else 1,
                0 if candidate.content_group_id else 1,
                0 if self._has_interaction_preference(candidate) else 1,
                -candidate.structure_evidence_score,
                -candidate.durability_score,
                -candidate.confidence,
                candidate.viewport_index if candidate.viewport_index is not None else 999,
                candidate.name,
            ),
        )

    def _recovery_stage(self, candidate: LocatorCandidate) -> int:
        if self._prefers_viewport_recovery(candidate):
            return 0
        if self._prefers_anchor_relocation(candidate):
            return 1
        if self._prefers_replan(candidate):
            return 2
        return 3

    def _prefers_viewport_recovery(self, candidate: LocatorCandidate) -> bool:
        if not candidate.scroll_context:
            return False
        if candidate.viewport_index is not None and candidate.viewport_index > 0:
            return True
        if candidate.structure_evidence_score >= 0.7 and candidate.region_role in {"grid_table", "viewport"}:
            return True
        if candidate.region_role in {"viewport", "detail_panel", "list_panel", "message_stream", "dialog_body"}:
            return True
        return bool(candidate.interaction_hints.get("scroll_into_view"))

    def _prefers_anchor_relocation(self, candidate: LocatorCandidate) -> bool:
        if candidate.anchor_count <= 0:
            return False
        if candidate.interaction_hints.get("anchor_relocate"):
            return True
        if candidate.content_group_id in {"cg_action_bar", "cg_toolbar", "cg_filter_bar", "cg_composer"}:
            return True
        return candidate.region_role in {"action_bar", "toolbar", "composer_area", "filter_bar"}

    def _prefers_replan(self, candidate: LocatorCandidate) -> bool:
        if candidate.replan_hint:
            return True
        expected_effect = str(candidate.interaction_hints.get("expected_effect") or "").lower()
        if expected_effect in {"navigate", "open_dialog", "open_panel", "send_message"}:
            return True
        if candidate.structure_evidence_score >= 0.8 and candidate.region_role in {
            "dialog_body",
            "viewport",
            "detail_panel",
        }:
            return True
        return False

    def _has_interaction_preference(self, candidate: LocatorCandidate) -> bool:
        return bool(
            candidate.interaction_hints.get("preferred_action")
            or candidate.interaction_hints.get("expected_effect")
            or candidate.interaction_hints.get("region_role")
        )

    def _select_terminal_strategy(
        self,
        candidates: list[LocatorCandidate],
    ) -> RecoveryStrategy:
        if any(self._prefers_viewport_recovery(candidate) for candidate in candidates):
            return RecoveryStrategy.VIEWPORT_RECOVERY
        if any(self._prefers_anchor_relocation(candidate) for candidate in candidates):
            return RecoveryStrategy.ANCHOR_RELOCATE
        if any(self._prefers_replan(candidate) for candidate in candidates):
            return RecoveryStrategy.REPLAN
        return RecoveryStrategy.ABORT

    def _build_terminal_error(
        self,
        candidates: list[LocatorCandidate],
    ) -> str:
        if any(self._prefers_viewport_recovery(candidate) for candidate in candidates):
            return "locator_chain_failed_viewport_recovery_required"
        if any(self._prefers_anchor_relocation(candidate) for candidate in candidates):
            return "locator_chain_failed_anchor_relocation_required"
        if any(self._prefers_replan(candidate) for candidate in candidates):
            return "locator_chain_failed_replan_required"
        return "all_locator_candidates_failed"

    def _build_terminal_reason(
        self,
        candidates: list[LocatorCandidate],
    ) -> str:
        if any(self._prefers_viewport_recovery(candidate) for candidate in candidates):
            return "viewport_context_requires_recovery"
        if any(self._prefers_anchor_relocation(candidate) for candidate in candidates):
            return "anchor_context_requires_relocation"
        if any(self._prefers_replan(candidate) for candidate in candidates):
            return "semantic_context_requires_replan"
        return "no_additional_recovery_path"

    def _strategy_label(self, candidate: LocatorCandidate) -> str:
        return f"{candidate.name}:{candidate.strategy.value}"

    def _run_verify_fn(
        self,
        verify_fn: Callable[..., VerificationResult] | None,
        candidate: LocatorCandidate,
        result: ActionResult | None,
    ) -> VerificationResult | None:
        if verify_fn is None:
            return None
        try:
            return verify_fn(candidate, result)
        except TypeError:
            return verify_fn()

    def _build_next_suggestion(
        self,
        strategy: RecoveryStrategy,
        error: str,
        reason: str,
    ) -> dict[str, str]:
        suggestion_map = {
            RecoveryStrategy.VIEWPORT_RECOVERY: "scroll_and_recapture",
            RecoveryStrategy.ANCHOR_RELOCATE: "anchor_relocate_and_retarget",
            RecoveryStrategy.REPLAN: "semantic_replan_on_refreshed_snapshot",
            RecoveryStrategy.ABORT: "manual_review_required",
        }
        return {
            "strategy": strategy.value,
            "action": suggestion_map.get(strategy, "manual_review_required"),
            "error": error,
            "reason": reason,
        }

    def _get_delay(self, attempts: int) -> float:
        delay = self._retry_config.initial_delay * (
            self._retry_config.backoff_multiplier ** (attempts - 1)
        )
        return min(delay, self._retry_config.max_delay)
