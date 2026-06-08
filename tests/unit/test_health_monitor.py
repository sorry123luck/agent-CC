"""Unit tests for HealthMonitor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.runtime.provider_metadata import ProviderMetadata
from src.runtime.provider_registry import ProviderRegistry
from src.runtime.health_monitor import HealthMonitor, HealthResult, RuntimeHealthSnapshot


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with an empty registry."""
    reg = ProviderRegistry()
    reg.clear()
    yield
    reg.clear()


def _make_meta(provider_id: str, health_check=None, **kwargs) -> ProviderMetadata:
    defaults = dict(
        provider_id=provider_id,
        provider_type="perception",
        mode="in_process",
        capabilities=frozenset(),
        default_timeout_seconds=5.0,
    )
    defaults.update(kwargs)
    if health_check is not None:
        defaults["health_check"] = health_check
    return ProviderMetadata(**defaults)


class TestCheckOne:
    def test_returns_ok_when_health_check_passes(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("a", health_check=lambda: {"ok": True}))
        monitor = HealthMonitor(reg)
        result = monitor.check_one("a")
        assert result.ok is True
        assert result.provider_id == "a"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    def test_returns_fail_when_health_check_fails(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("a", health_check=lambda: {"ok": False, "error": "down"}))
        monitor = HealthMonitor(reg)
        result = monitor.check_one("a")
        assert result.ok is False
        assert result.error == "down"

    def test_returns_fail_when_health_check_raises(self):
        def bad_check():
            raise RuntimeError("boom")

        reg = ProviderRegistry()
        reg.register(_make_meta("a", health_check=bad_check))
        monitor = HealthMonitor(reg)
        result = monitor.check_one("a")
        assert result.ok is False
        assert "boom" in result.error

    def test_returns_ok_when_no_health_check_defined(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("a"))
        monitor = HealthMonitor(reg)
        result = monitor.check_one("a")
        assert result.ok is True
        assert result.details.get("note") == "no_health_check_defined"

    def test_returns_fail_when_not_registered(self):
        reg = ProviderRegistry()
        monitor = HealthMonitor(reg)
        result = monitor.check_one("ghost")
        assert result.ok is False
        assert result.error == "provider_not_registered"


class TestCheckAll:
    def test_returns_all_providers(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("a", health_check=lambda: {"ok": True}))
        reg.register(_make_meta("b", health_check=lambda: {"ok": False, "error": "x"}))
        reg.register(_make_meta("c"))
        monitor = HealthMonitor(reg)
        snapshot = monitor.check_all()
        assert snapshot.summary["total"] == 3
        assert snapshot.summary["healthy"] == 2  # a (ok) + c (no check = ok)
        assert snapshot.summary["unhealthy"] == 1
        assert len(snapshot.providers) == 3
        assert snapshot.timestamp is not None

    def test_empty_registry(self):
        reg = ProviderRegistry()
        monitor = HealthMonitor(reg)
        snapshot = monitor.check_all()
        assert snapshot.summary == {"total": 0, "healthy": 0, "unhealthy": 0}
        assert snapshot.providers == []
