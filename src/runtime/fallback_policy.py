"""FallbackPolicy — cross-observe provider failure tracking and circuit breaker.

This module tracks provider success/failure across multiple observe() calls.
When a provider fails consecutively, the circuit breaker opens and the
provider is skipped for a cooldown period.

Usage::

    # In PerceptionService.__init__:
    self._fallback_policy = FallbackPolicy()

    # In analyze(), before calling a provider:
    skip, reason = self._fallback_policy.should_skip("omniparser")
    if skip:
        budget.skip("omniparser", reason)
    else:
        # ... run OmniParser ...
        self._fallback_policy.record_result(ProviderResult(...))

This module does NOT call any provider — it only tracks state.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


# Default thresholds
_DEFAULT_MAX_FAILURES: int = 3
_DEFAULT_COOLDOWN_SECONDS: float = 60.0


@dataclass
class ProviderResult:
    """Result of a single provider call (for recording)."""

    provider_id: str
    success: bool
    elapsed_seconds: float = 0.0
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class FallbackRule:
    """A rule describing when to skip a provider."""

    provider_id: str
    max_failures: int = _DEFAULT_MAX_FAILURES
    cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS


@dataclass
class _ProviderState:
    """Internal mutable state for a single provider."""

    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    last_error: str | None = None
    last_elapsed: float = 0.0
    total_calls: int = 0
    total_failures: int = 0


class FallbackPolicy:
    """Cross-observe provider failure tracking and circuit breaker.

    Thread-safe: all state mutations are protected by a lock, so concurrent
    observe() calls from different threads won't corrupt failure counts.

    Does NOT call any provider — only tracks state and advises callers.
    """

    def __init__(self, rules: list[FallbackRule] | None = None) -> None:
        self._rules = {r.provider_id: r for r in (rules or self._default_rules())}
        self._states: dict[str, _ProviderState] = {}
        self._lock = threading.RLock()

    def should_skip(self, provider_id: str) -> tuple[bool, str]:
        """Check whether *provider_id* should be skipped.

        Returns ``(True, reason)`` if circuit is open (too many recent failures).
        Returns ``(False, "")`` if the provider may be called.
        """
        rule = self._rules.get(provider_id)
        if rule is None:
            return False, ""

        with self._lock:
            state = self._states.get(provider_id)
            if state is None:
                return False, ""

            now = time.monotonic()
            if state.consecutive_failures >= rule.max_failures:
                if now < state.circuit_open_until:
                    remaining = state.circuit_open_until - now
                    return True, f"circuit_open_failures={state.consecutive_failures}_recover_in_{remaining:.0f}s"
                # Cooldown expired — allow retry
                state.consecutive_failures = 0

            return False, ""

    def record_result(self, result: ProviderResult) -> None:
        """Record the outcome of a provider call.

        Success resets the consecutive failure count.
        Failure increments it and may open the circuit breaker.
        Skipped results (budget_exhausted, circuit_open) are recorded
        for diagnostics but do NOT count as failures or refresh cooldown.
        """
        rule = self._rules.get(result.provider_id)
        if rule is None:
            return

        with self._lock:
            state = self._states.setdefault(result.provider_id, _ProviderState())

            # Skipped results: record for diagnostics only, don't count as failure
            if result.skipped:
                return

            state.total_calls += 1
            state.last_elapsed = result.elapsed_seconds

            if result.success:
                state.consecutive_failures = 0
                state.last_error = None
                state.circuit_open_until = 0.0
            else:
                state.consecutive_failures += 1
                state.total_failures += 1
                state.last_error = result.error
                if state.consecutive_failures >= rule.max_failures:
                    state.circuit_open_until = time.monotonic() + rule.cooldown_seconds

    def get_state(self, provider_id: str) -> dict[str, Any]:
        """Return current state for a provider (for diagnostics)."""
        with self._lock:
            state = self._states.get(provider_id)
            if state is None:
                return {"provider_id": provider_id, "calls": 0}
            rule = self._rules.get(provider_id)
            return {
                "provider_id": provider_id,
                "consecutive_failures": state.consecutive_failures,
                "total_calls": state.total_calls,
                "total_failures": state.total_failures,
                "last_error": state.last_error,
                "last_elapsed": state.last_elapsed,
                "circuit_open": state.consecutive_failures >= (rule.max_failures if rule else 3),
                "circuit_open_until": state.circuit_open_until,
            }

    def summary(self) -> dict[str, Any]:
        """Return serialisable summary of all tracked providers."""
        with self._lock:
            return {
                pid: self.get_state(pid)
                for pid in set(list(self._rules.keys()) + list(self._states.keys()))
            }

    @staticmethod
    def _default_rules() -> list[FallbackRule]:
        """Default rules for known providers."""
        return [
            FallbackRule(provider_id="omniparser", max_failures=3, cooldown_seconds=60),
            FallbackRule(provider_id="vlm", max_failures=3, cooldown_seconds=60),
        ]
