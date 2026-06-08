"""Unit tests for FallbackPolicy — cross-observe failure tracking."""

from __future__ import annotations

import time

from src.runtime.fallback_policy import FallbackPolicy, FallbackRule, ProviderResult


class TestShouldSkip:
    def test_returns_false_initially(self):
        policy = FallbackPolicy()
        skip, reason = policy.should_skip("omniparser")
        assert skip is False
        assert reason == ""

    def test_returns_false_for_unknown_provider(self):
        policy = FallbackPolicy()
        skip, reason = policy.should_skip("nonexistent")
        assert skip is False


class TestRecordFailure:
    def test_increments_consecutive_failures(self):
        policy = FallbackPolicy([FallbackRule("test", max_failures=3, cooldown_seconds=60)])
        for _ in range(2):
            policy.record_result(ProviderResult("test", success=False))
        state = policy.get_state("test")
        assert state["consecutive_failures"] == 2

    def test_opens_circuit_after_max_failures(self):
        policy = FallbackPolicy([FallbackRule("test", max_failures=3, cooldown_seconds=60)])
        for _ in range(3):
            policy.record_result(ProviderResult("test", success=False))
        skip, reason = policy.should_skip("test")
        assert skip is True
        assert "circuit_open" in reason

    def test_tracks_total_failures(self):
        policy = FallbackPolicy([FallbackRule("test", max_failures=10, cooldown_seconds=60)])
        for _ in range(5):
            policy.record_result(ProviderResult("test", success=False))
        state = policy.get_state("test")
        assert state["total_failures"] == 5


class TestRecordSuccess:
    def test_resets_consecutive_failures(self):
        policy = FallbackPolicy([FallbackRule("test", max_failures=3, cooldown_seconds=60)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=True))
        state = policy.get_state("test")
        assert state["consecutive_failures"] == 0

    def test_success_after_failures_allows_retry(self):
        policy = FallbackPolicy([FallbackRule("test", max_failures=3, cooldown_seconds=60)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=True))
        skip, _ = policy.should_skip("test")
        assert skip is False


class TestCrossObserve:
    def test_failure_persists_across_should_skip_calls(self):
        """Consecutive failures persist — not reset between observe calls."""
        policy = FallbackPolicy([FallbackRule("test", max_failures=3, cooldown_seconds=60)])
        # Simulate 3 separate observe calls, each failing once
        for _ in range(3):
            skip, _ = policy.should_skip("test")
            assert skip is False  # Not yet at threshold
            policy.record_result(ProviderResult("test", success=False))
        # Now should be skipped
        skip, reason = policy.should_skip("test")
        assert skip is True

    def test_success_resets_across_observe_calls(self):
        """Success in one observe resets failure count for next observe."""
        policy = FallbackPolicy([FallbackRule("test", max_failures=3, cooldown_seconds=60)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        # One success resets
        policy.record_result(ProviderResult("test", success=True))
        # Need 3 more failures to open circuit
        for _ in range(2):
            policy.record_result(ProviderResult("test", success=False))
        skip, _ = policy.should_skip("test")
        assert skip is False  # Only 2 consecutive failures


class TestCooldown:
    def test_circuit_recovers_after_cooldown(self):
        policy = FallbackPolicy([FallbackRule("test", max_failures=2, cooldown_seconds=0.1)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        skip, _ = policy.should_skip("test")
        assert skip is True
        time.sleep(0.15)
        skip, _ = policy.should_skip("test")
        assert skip is False


class TestSummary:
    def test_returns_all_tracked_providers(self):
        policy = FallbackPolicy([
            FallbackRule("a", max_failures=3, cooldown_seconds=60),
            FallbackRule("b", max_failures=3, cooldown_seconds=60),
        ])
        policy.record_result(ProviderResult("a", success=True))
        policy.record_result(ProviderResult("b", success=False))
        summary = policy.summary()
        assert "a" in summary
        assert "b" in summary
        assert summary["a"]["consecutive_failures"] == 0
        assert summary["b"]["consecutive_failures"] == 1


class TestSkippedNotCountingAsFailure:
    def test_skipped_result_does_not_increment_failure_count(self):
        """A skipped result (circuit_open/budget_exhausted) must NOT count as a failure."""
        policy = FallbackPolicy([FallbackRule("test", max_failures=2, cooldown_seconds=60)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        # Circuit is now open
        skip, _ = policy.should_skip("test")
        assert skip is True
        # Simulate 10 skipped observations
        for _ in range(10):
            policy.record_result(ProviderResult("test", success=False, skipped=True))
        state = policy.get_state("test")
        # Failure count should still be 2, not 12
        assert state["consecutive_failures"] == 2
        assert state["total_failures"] == 2

    def test_skipped_does_not_extend_cooldown(self):
        """Skipping during circuit_open must not push circuit_open_until further."""
        policy = FallbackPolicy([FallbackRule("test", max_failures=2, cooldown_seconds=0.5)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        skip, _ = policy.should_skip("test")
        assert skip is True
        # Record the cooldown deadline
        state_before = policy.get_state("test")
        deadline_before = state_before["circuit_open_until"]
        # Simulate skipped observations
        for _ in range(5):
            policy.record_result(ProviderResult("test", success=False, skipped=True))
        state_after = policy.get_state("test")
        # Deadline must not have changed
        assert state_after["circuit_open_until"] == deadline_before

    def test_cooldown_expires_and_allows_retry(self):
        """After cooldown, should_skip returns False even with prior failures."""
        policy = FallbackPolicy([FallbackRule("test", max_failures=2, cooldown_seconds=0.1)])
        policy.record_result(ProviderResult("test", success=False))
        policy.record_result(ProviderResult("test", success=False))
        skip, _ = policy.should_skip("test")
        assert skip is True
        import time
        time.sleep(0.15)
        skip, _ = policy.should_skip("test")
        assert skip is False
        # After cooldown, consecutive_failures is reset
        state = policy.get_state("test")
        assert state["consecutive_failures"] == 0


class TestDeferredNotCountingAsFailure:
    """Providers that are deferred (fast_mode, no screenshot, etc.)
    must NOT be recorded as failures because they were never called."""

    def test_fast_mode_does_not_record_failure(self):
        """When provider is deferred due to fast_mode, no record_result is called.
        This test verifies the FallbackPolicy contract: only called providers
        should have record_result invoked."""
        policy = FallbackPolicy([FallbackRule("omniparser", max_failures=2, cooldown_seconds=60)])
        # Simulate: fast_mode means provider is never called, so no record_result
        # After many fast_mode observes, failure count should still be 0
        for _ in range(20):
            skip, _ = policy.should_skip("omniparser")
            assert skip is False
            # No record_result call — provider was deferred
        state = policy.get_state("omniparser")
        # When no record_result was ever called, get_state returns {"calls": 0}
        assert state.get("consecutive_failures", 0) == 0
        assert state.get("total_calls", 0) == 0

    def test_no_screenshot_does_not_record_failure(self):
        """When provider is not called due to no screenshot, no record."""
        policy = FallbackPolicy([FallbackRule("omniparser", max_failures=2, cooldown_seconds=60)])
        # Simulate: no screenshot, provider never called
        for _ in range(20):
            skip, _ = policy.should_skip("omniparser")
            assert skip is False
        state = policy.get_state("omniparser")
        assert state.get("total_calls", 0) == 0

    def test_called_then_deferred_does_not_accumulate(self):
        """One real failure + many deferrals = still 1 failure, not circuit open."""
        policy = FallbackPolicy([FallbackRule("omniparser", max_failures=3, cooldown_seconds=60)])
        # One real call that fails
        policy.record_result(ProviderResult("omniparser", success=False, error="connection_refused"))
        # Many deferred observes (fast_mode)
        for _ in range(20):
            skip, _ = policy.should_skip("omniparser")
            assert skip is False  # 1 failure < 3 threshold
            # No record_result — provider deferred
        state = policy.get_state("omniparser")
        assert state["consecutive_failures"] == 1
        assert state["total_failures"] == 1


class TestVlmFallback:
    def test_vlm_consecutive_failures_opens_circuit(self):
        policy = FallbackPolicy([FallbackRule("vlm", max_failures=3, cooldown_seconds=60)])
        for _ in range(3):
            policy.record_result(ProviderResult("vlm", success=False, error="timeout"))
        skip, reason = policy.should_skip("vlm")
        assert skip is True
        assert "circuit_open" in reason

    def test_vlm_success_resets_failure_count(self):
        policy = FallbackPolicy([FallbackRule("vlm", max_failures=3, cooldown_seconds=60)])
        policy.record_result(ProviderResult("vlm", success=False))
        policy.record_result(ProviderResult("vlm", success=False))
        policy.record_result(ProviderResult("vlm", success=True))
        state = policy.get_state("vlm")
        assert state["consecutive_failures"] == 0

    def test_vlm_budget_exhausted_skip_not_counted(self):
        """budget_exhausted skip for VLM should not increase failure count."""
        policy = FallbackPolicy([FallbackRule("vlm", max_failures=2, cooldown_seconds=60)])
        policy.record_result(ProviderResult("vlm", success=False))
        # One more real failure would open circuit, but we skip instead
        policy.record_result(ProviderResult("vlm", success=False, skipped=True))
        state = policy.get_state("vlm")
        # Only 1 real failure, not 2
        assert state["consecutive_failures"] == 1
        skip, _ = policy.should_skip("vlm")
        assert skip is False


class TestThreadSafety:
    def test_concurrent_record_does_not_crash(self):
        """Concurrent record_result calls from multiple threads don't corrupt state."""
        import threading

        policy = FallbackPolicy([FallbackRule("test", max_failures=1000, cooldown_seconds=60)])
        errors = []

        def _record(n):
            try:
                for _ in range(n):
                    policy.record_result(ProviderResult("test", success=False))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_record, args=(100,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        state = policy.get_state("test")
        assert state["total_failures"] == 400
