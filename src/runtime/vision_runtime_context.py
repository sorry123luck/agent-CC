"""VisionRuntimeContext — encapsulates budget/fallback/timing recording.

This module eliminates the repetitive recording code in
``PerceptionService.analyze()``.  Each provider call site uses a
``ProviderRunContext`` (via ``VisionRuntimeContext.provider()``) which
automatically handles budget allocation, fallback tracking, and timing.

Usage in ``analyze()``::

    ctx = VisionRuntimeContext(fallback_policy=self._fallback_policy)

    with ctx.provider("uia", timeout=8.0) as run:
        try:
            uia_elements = uia_client.find_all()
            run.record(success=True)
        except Exception as exc:
            uia_elements = []
            run.record(success=False, error=str(exc))
        run.set_details({"element_count": len(uia_elements)})

    # After all providers:
    _sub_timing.update(ctx.timing_summary())
    uia_provider_details = ctx.get_details("uia")
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from src.runtime.fallback_policy import FallbackPolicy, ProviderResult
from src.runtime.timeout_budget import TimeoutBudget, DEFAULT_OBSERVE_BUDGET_SECONDS


@dataclass
class _ProviderRunState:
    """Mutable state for a single provider run."""

    provider_id: str
    started: float = 0.0
    elapsed: float = 0.0
    success: bool = False
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    recorded: bool = False


class ProviderRunContext:
    """Context manager for a single provider invocation.

    Handles budget.allocate/record/skip, fallback.record_result,
    and timing automatically.
    """

    def __init__(
        self,
        provider_id: str,
        budget: TimeoutBudget,
        fallback: FallbackPolicy,
        timeout: float,
    ) -> None:
        self._provider_id = provider_id
        self._budget = budget
        self._fallback = fallback
        self._timeout = timeout
        self._state = _ProviderRunState(provider_id=provider_id)

    def should_skip(self) -> tuple[bool, str]:
        """Check budget + fallback for skip conditions.

        Returns ``(True, reason)`` if provider should be skipped.
        """
        # Check fallback circuit breaker first
        fb_skip, fb_reason = self._fallback.should_skip(self._provider_id)
        if fb_skip:
            return True, fb_reason

        # Check budget
        avail = self._budget.allocate(self._provider_id, self._timeout)
        if avail <= 0:
            return True, "budget_exhausted"

        return False, ""

    def skip(self, reason: str) -> None:
        """Record that this provider was skipped."""
        self._state.skipped = True
        self._state.skip_reason = reason
        self._budget.skip(self._provider_id, reason)
        self._state.recorded = True

    def record(self, success: bool, error: str | None = None, elapsed: float | None = None) -> None:
        """Record provider result. Updates budget and fallback automatically."""
        if self._state.recorded:
            return  # Already recorded (skip or previous record)

        actual = elapsed if elapsed is not None else (time.perf_counter() - self._state.started)
        self._state.elapsed = actual
        self._state.success = success
        self._state.error = error
        self._state.recorded = True

        self._budget.record(self._provider_id, actual)
        self._fallback.record_result(ProviderResult(
            provider_id=self._provider_id,
            success=success,
            elapsed_seconds=actual,
            error=error,
        ))

    def set_details(self, details: dict[str, Any]) -> None:
        """Merge additional details into this provider's details dict."""
        self._state.details.update(details)

    def __enter__(self) -> ProviderRunContext:
        self._state.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Only auto-record on unhandled exception (provider crashed).
        # Do NOT auto-record success — fast_mode, should_call_vlm=False,
        # screenshot=None etc. leave the provider "not called", which
        # must NOT count as success or failure in fallback tracking.
        if not self._state.recorded and exc_val is not None:
            self.record(success=False, error=str(exc_val))


class VisionRuntimeContext:
    """Holds budget/fallback and creates ProviderRunContext instances.

    Usage::

        ctx = VisionRuntimeContext(fallback_policy=self._fallback_policy)

        with ctx.provider("uia", timeout=8.0) as run:
            ...
            run.record(success=True)
            run.set_details({"mode": "full"})

        with ctx.provider("omniparser", timeout=90.0) as run:
            skip, reason = run.should_skip()
            if skip:
                run.skip(reason)
            else:
                ...

        # Get all provider details
        details = ctx.get_details("uia")  # {"provider": "uia", "success": True, ...}
        all_details = ctx.all_provider_details()
    """

    def __init__(
        self,
        fallback_policy: FallbackPolicy,
        budget_seconds: float = DEFAULT_OBSERVE_BUDGET_SECONDS,
    ) -> None:
        self._budget = TimeoutBudget(total_seconds=budget_seconds)
        self._fallback = fallback_policy
        self._runs: dict[str, _ProviderRunState] = {}

    @contextmanager
    def provider(self, provider_id: str, timeout: float = 30.0) -> Generator[ProviderRunContext, None, None]:
        """Create a ProviderRunContext for *provider_id*."""
        run = ProviderRunContext(
            provider_id=provider_id,
            budget=self._budget,
            fallback=self._fallback,
            timeout=timeout,
        )
        try:
            with run:
                yield run
        finally:
            self._runs[provider_id] = run._state

    def get_details(self, provider_id: str) -> dict[str, Any]:
        """Return the details dict for *provider_id*."""
        state = self._runs.get(provider_id)
        if state is None:
            return {}
        details: dict[str, Any] = {
            "provider": provider_id,
            "success": state.success,
            "elapsed_seconds": round(state.elapsed, 3),
        }
        if state.error:
            details["error"] = state.error
        if state.skipped:
            details["skipped"] = True
            details["skip_reason"] = state.skip_reason
        details.update(state.details)
        return details

    def all_provider_details(self) -> dict[str, dict[str, Any]]:
        """Return details for all providers that have been run."""
        return {pid: self.get_details(pid) for pid in self._runs}

    def timing_summary(self) -> dict[str, Any]:
        """Return budget + fallback + per-provider timing summary."""
        return {
            "budget": self._budget.summary(),
            "fallback": self._fallback.summary(),
        }
