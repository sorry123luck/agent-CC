"""Unit tests for ProviderRegistry."""

from __future__ import annotations

import pytest

from src.runtime.provider_metadata import ProviderMetadata
from src.runtime.provider_registry import ProviderRegistry, get_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with an empty registry."""
    reg = ProviderRegistry()
    reg.clear()
    yield
    reg.clear()


def _make_meta(provider_id: str = "test", **kwargs) -> ProviderMetadata:
    defaults = dict(
        provider_id=provider_id,
        provider_type="perception",
        mode="in_process",
        capabilities=frozenset({"test"}),
        default_timeout_seconds=5.0,
    )
    defaults.update(kwargs)
    return ProviderMetadata(**defaults)


class TestRegisterAndGet:
    def test_register_and_get(self):
        reg = ProviderRegistry()
        meta = _make_meta("uia")
        reg.register(meta)
        assert reg.get("uia") is meta

    def test_get_unknown_returns_none(self):
        reg = ProviderRegistry()
        assert reg.get("nonexistent") is None

    def test_register_duplicate_overwrites(self):
        reg = ProviderRegistry()
        old = _make_meta("uia", default_timeout_seconds=5.0)
        new = _make_meta("uia", default_timeout_seconds=8.0)
        reg.register(old)
        reg.register(new)
        assert reg.get("uia").default_timeout_seconds == 8.0

    def test_len(self):
        reg = ProviderRegistry()
        assert len(reg) == 0
        reg.register(_make_meta("a"))
        reg.register(_make_meta("b"))
        assert len(reg) == 2


class TestListAll:
    def test_list_all_returns_registered(self):
        reg = ProviderRegistry()
        a = _make_meta("a")
        b = _make_meta("b")
        reg.register(a)
        reg.register(b)
        result = reg.list_all()
        assert result == [a, b]

    def test_list_all_empty(self):
        reg = ProviderRegistry()
        assert reg.list_all() == []


class TestListByType:
    def test_filters_correctly(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("uia", provider_type="perception"))
        reg.register(_make_meta("ocr", provider_type="ocr"))
        reg.register(_make_meta("v", provider_type="vision"))
        assert [m.provider_id for m in reg.list_by_type("perception")] == ["uia"]
        assert [m.provider_id for m in reg.list_by_type("ocr")] == ["ocr"]

    def test_empty_type(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("uia", provider_type="perception"))
        assert reg.list_by_type("vlm") == []


class TestProviderIds:
    def test_returns_ids(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("a"))
        reg.register(_make_meta("b"))
        assert reg.provider_ids() == ["a", "b"]


class TestUpdateEndpoint:
    def test_updates_endpoint(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("omni", endpoint="http://old:8001/"))
        reg.update_endpoint("omni", "http://new:9000/")
        assert reg.get("omni").endpoint == "http://new:9000/"

    def test_update_nonexistent_is_noop(self):
        reg = ProviderRegistry()
        reg.update_endpoint("ghost", "http://x:1/")
        assert reg.get("ghost") is None

    def test_update_preserves_other_fields(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("omni", provider_type="vision", model_dependency="YOLO"))
        reg.update_endpoint("omni", "http://new:9000/")
        meta = reg.get("omni")
        assert meta.provider_type == "vision"
        assert meta.model_dependency == "YOLO"


class TestSingleton:
    def test_singleton_identity(self):
        a = ProviderRegistry()
        b = ProviderRegistry()
        assert a is b

    def test_get_registry_returns_singleton(self):
        assert get_registry() is ProviderRegistry()


class TestClear:
    def test_clear_removes_all(self):
        reg = ProviderRegistry()
        reg.register(_make_meta("a"))
        reg.register(_make_meta("b"))
        reg.clear()
        assert len(reg) == 0
        assert reg.list_all() == []
