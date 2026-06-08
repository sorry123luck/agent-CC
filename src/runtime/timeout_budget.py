"""TimeoutBudget — cooperative per-observe provider timeout tracking.

This module does NOT enforce hard timeouts or cancel running providers.
It tracks elapsed time and tells callers whether the remaining budget
is sufficient for the next provider.  Individual provider timeouts
(UIA 8s, OmniParser 90s, etc.) are handled by the providers themselves.

Usage in ``PerceptionService.analyze()``::

    budget = TimeoutBudget(total_seconds=DEFAULT_OBSERVE_BUDGET)
    budget.allocate("uia", 8.0)
    # ... run UIA ...
    budget.record("uia", elapsed)

    if budget.remaining() <= 0:
        budget.skip("ocr", "budget_exhausted")
    else:
        budget.allocate("ocr", 30.0)
        # ... run OCR ...
        budget.record("ocr", elapsed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Default total budget for one observe() call.
# Configurable via PerceptionService constructor or constant override.
DEFAULT_OBSERVE_BUDGET_SECONDS: float = 30.0


@dataclass
class ProviderTiming:
    """Timing record for a single provider within one observe."""

    provider_id: str
    allocated: float = 0.0
    actual: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


class TimeoutBudget:
    """Cooperative per-observe timeout budget.

    Tracks how much time has been spent and how much remains.
    Does NOT cancel running providers — only advises callers
    whether to start the next one.
    """

    def __init__(self, total_seconds: float = DEFAULT_OBSERVE_BUDGET_SECONDS) -> None:
        import time

        self._total = max(0.1, float(total_seconds))
        self._started = time.perf_counter()
        self._timings: dict[str, ProviderTiming] = {}
        self._recorded_total: float = 0.0

    @property
    def total(self) -> float:
        return self._total

    def remaining(self) -> float:
        """Return remaining budget in seconds (may be negative if over budget)."""
        import time

        return self._total - (time.perf_counter() - self._started)

    def allocate(self, provider_id: str, requested: float) -> float:
        """Return the number of seconds this provider may use.

        Returns ``min(requested, remaining())``, or 0 if budget exhausted.
        Does NOT start a timer — just records the allocation intent.
        """
        avail = max(0.0, min(float(requested), self.remaining()))
        self._timings[provider_id] = ProviderTiming(
            provider_id=provider_id,
            allocated=avail,
        )
        return avail

    def record(self, provider_id: str, actual: float) -> None:
        """Record actual elapsed time for a provider."""
        entry = self._timings.get(provider_id)
        if entry is None:
            entry = ProviderTiming(provider_id=provider_id)
            self._timings[provider_id] = entry
        entry.actual = float(actual)
        self._recorded_total += float(actual)

    def skip(self, provider_id: str, reason: str) -> None:
        """Record that a provider was skipped."""
        self._timings[provider_id] = ProviderTiming(
            provider_id=provider_id,
            skipped=True,
            skip_reason=reason,
        )

    def summary(self) -> dict[str, Any]:
        """Return a serialisable summary of all provider timings."""
        import time

        wall = time.perf_counter() - self._started
        return {
            "total_budget_seconds": self._total,
            "wall_elapsed_seconds": round(wall, 3),
            "recorded_total_seconds": round(self._recorded_total, 3),
            "remaining_seconds": round(self._total - wall, 3),
            "providers": {
                pid: {
                    "allocated": round(t.allocated, 3),
                    "actual": round(t.actual, 3),
                    "skipped": t.skipped,
                    "skip_reason": t.skip_reason,
                }
                for pid, t in self._timings.items()
            },
        }
