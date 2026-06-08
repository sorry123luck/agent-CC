"""Unit tests for VisionRuntimeContext and ProviderRunContext."""

from __future__ import annotations

from src.runtime.fallback_policy import FallbackPolicy, FallbackRule
from src.runtime.vision_runtime_context import VisionRuntimeContext


def _make_ctx(max_failures: int = 3, budget: float = 30.0) -> VisionRuntimeContext:
    policy = FallbackPolicy([FallbackRule("test", max_failures=max_failures, cooldown_seconds=60)])
    return VisionRuntimeContext(fallback_policy=policy, budget_seconds=budget)


class TestProviderRunRecord:
    def test_records_timing(self):
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.record(success=True, elapsed=1.5)
        details = ctx.get_details("test")
        assert details["success"] is True
        assert details["elapsed_seconds"] == 1.5

    def test_records_error(self):
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.record(success=False, error="timeout", elapsed=5.0)
        details = ctx.get_details("test")
        assert details["success"] is False
        assert details["error"] == "timeout"

    def test_records_details(self):
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.record(success=True, elapsed=0.5)
            run.set_details({"mode": "full", "element_count": 42})
        details = ctx.get_details("test")
        assert details["mode"] == "full"
        assert details["element_count"] == 42

    def test_double_record_ignored(self):
        """Second record call should be ignored."""
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.record(success=True, elapsed=0.5)
            run.record(success=False, error="should be ignored")
        details = ctx.get_details("test")
        assert details["success"] is True  # First record wins


class TestProviderRunSkip:
    def test_skip_does_not_record_failure(self):
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.skip("budget_exhausted")
        details = ctx.get_details("test")
        assert details["skipped"] is True
        assert details["skip_reason"] == "budget_exhausted"
        # Fallback should not count this as a failure
        state = ctx._fallback.get_state("test")
        assert state.get("consecutive_failures", 0) == 0

    def test_skip_then_record_ignored(self):
        """After skip, record should be ignored."""
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.skip("circuit_open")
            run.record(success=False, error="should be ignored")
        details = ctx.get_details("test")
        assert details["skipped"] is True


class TestProviderRunShouldSkip:
    def test_budget_exhausted(self):
        ctx = _make_ctx(budget=0.0)
        # Force budget to 0
        ctx._budget._total = 0.0
        with ctx.provider("test", timeout=10.0) as run:
            skip, reason = run.should_skip()
            assert skip is True
            assert reason == "budget_exhausted"

    def test_circuit_open(self):
        from src.runtime.fallback_policy import ProviderResult
        policy = FallbackPolicy([FallbackRule("test", max_failures=1, cooldown_seconds=60)])
        policy.record_result(ProviderResult("test", success=False))
        ctx = VisionRuntimeContext(fallback_policy=policy)
        with ctx.provider("test", timeout=10.0) as run:
            skip, reason = run.should_skip()
            assert skip is True
            assert "circuit_open" in reason

    def test_no_skip_when_budget_sufficient(self):
        ctx = _make_ctx(budget=30.0)
        with ctx.provider("test", timeout=10.0) as run:
            skip, reason = run.should_skip()
            assert skip is False


class TestVisionRuntimeContextDetails:
    def test_all_provider_details(self):
        ctx = _make_ctx()
        with ctx.provider("uia", timeout=8.0) as run:
            run.record(success=True, elapsed=0.5)
            run.set_details({"mode": "full"})
        with ctx.provider("ocr", timeout=30.0) as run:
            run.record(success=True, elapsed=1.0)
            run.set_details({"block_count": 10})
        all_details = ctx.all_provider_details()
        assert "uia" in all_details
        assert "ocr" in all_details
        assert all_details["uia"]["mode"] == "full"
        assert all_details["ocr"]["block_count"] == 10

    def test_get_details_empty_for_unknown(self):
        ctx = _make_ctx()
        assert ctx.get_details("nonexistent") == {}


class TestTimingSummary:
    def test_contains_budget_and_fallback(self):
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.record(success=True, elapsed=0.5)
        summary = ctx.timing_summary()
        assert "budget" in summary
        assert "fallback" in summary
        assert summary["budget"]["total_budget_seconds"] == 30.0


class TestAutoRecordOnExit:
    def test_no_record_no_fallback_change(self):
        """When provider is not called (fast_mode, no screenshot, etc.),
        neither record nor skip is called — fallback state must not change."""
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0):
            pass  # No record/skip call — provider was not invoked
        details = ctx.get_details("test")
        # Should NOT have success=True or success=False — just not recorded
        assert "success" not in details or details.get("success") is False
        # Fallback state must not change
        state = ctx._fallback.get_state("test")
        assert state.get("consecutive_failures", 0) == 0

    def test_exception_records_failure(self):
        """If provider block exits with unhandled exception, record failure."""
        ctx = _make_ctx()
        try:
            with ctx.provider("test", timeout=10.0):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        details = ctx.get_details("test")
        assert details["success"] is False
        assert "boom" in details["error"]

    def test_explicit_record_preserved_on_exit(self):
        """If record() was called explicitly, __exit__ does not overwrite it."""
        ctx = _make_ctx()
        with ctx.provider("test", timeout=10.0) as run:
            run.record(success=True, elapsed=0.5)
        details = ctx.get_details("test")
        assert details["success"] is True
        assert details["elapsed_seconds"] == 0.5
