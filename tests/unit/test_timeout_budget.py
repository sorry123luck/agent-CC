"""Unit tests for TimeoutBudget."""

from __future__ import annotations

import time

from src.runtime.timeout_budget import TimeoutBudget, DEFAULT_OBSERVE_BUDGET_SECONDS


class TestAllocate:
    def test_returns_requested_when_budget_sufficient(self):
        budget = TimeoutBudget(total_seconds=30.0)
        avail = budget.allocate("uia", 8.0)
        assert avail == 8.0

    def test_returns_limited_when_budget_low(self):
        budget = TimeoutBudget(total_seconds=5.0)
        avail = budget.allocate("uia", 8.0)
        assert avail <= 5.0
        assert avail > 0

    def test_returns_zero_when_budget_exhausted(self):
        budget = TimeoutBudget(total_seconds=0.0)
        # total is clamped to 0.1, but remaining() will be < 0 almost immediately
        # Force the internal total to 0 for a clean test
        budget._total = 0.0
        avail = budget.allocate("ocr", 10.0)
        assert avail == 0.0


class TestRecord:
    def test_tracks_actual_elapsed(self):
        budget = TimeoutBudget(total_seconds=30.0)
        budget.allocate("uia", 8.0)
        budget.record("uia", 2.5)
        summary = budget.summary()
        assert summary["providers"]["uia"]["actual"] == 2.5

    def test_record_without_allocate(self):
        budget = TimeoutBudget(total_seconds=30.0)
        budget.record("omniparser", 5.0)
        summary = budget.summary()
        assert summary["providers"]["omniparser"]["actual"] == 5.0


class TestSkip:
    def test_records_skip_reason(self):
        budget = TimeoutBudget(total_seconds=30.0)
        budget.skip("vlm", "budget_exhausted")
        summary = budget.summary()
        assert summary["providers"]["vlm"]["skipped"] is True
        assert summary["providers"]["vlm"]["skip_reason"] == "budget_exhausted"


class TestRemaining:
    def test_decreases_after_time(self):
        budget = TimeoutBudget(total_seconds=10.0)
        before = budget.remaining()
        time.sleep(0.05)
        after = budget.remaining()
        assert after < before

    def test_is_positive_initially(self):
        budget = TimeoutBudget(total_seconds=30.0)
        assert budget.remaining() > 25.0


class TestSummary:
    def test_returns_all_providers(self):
        budget = TimeoutBudget(total_seconds=30.0)
        budget.allocate("uia", 8.0)
        budget.record("uia", 2.0)
        budget.skip("ocr", "test")
        summary = budget.summary()
        assert "uia" in summary["providers"]
        assert "ocr" in summary["providers"]
        assert summary["total_budget_seconds"] == 30.0


class TestDefaultBudget:
    def test_default_is_30s(self):
        assert DEFAULT_OBSERVE_BUDGET_SECONDS == 30.0

    def test_configurable(self):
        budget = TimeoutBudget(total_seconds=60.0)
        assert budget.total == 60.0
