"""E2E tests for /api/v1/runtime/health endpoint."""

from __future__ import annotations

from unittest.mock import patch


def _mock_omniparser_health():
    return {
        "provider": "omniparser",
        "enabled": True,
        "endpoint": "http://127.0.0.1:8001/parse/",
        "ok": False,
        "probe": {"ok": False, "error": "mock_probe_fail"},
    }


def test_runtime_health_endpoint_returns_snapshot(client):
    """The runtime health endpoint returns a well-formed snapshot."""
    resp = client.get("/api/v1/runtime/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "timestamp" in data
    assert "summary" in data
    assert "providers" in data
    assert "metadata" in data
    assert isinstance(data["providers"], list)
    assert isinstance(data["metadata"], list)


def test_runtime_health_includes_registered_providers(client):
    """After startup, the registry should contain the standard providers."""
    resp = client.get("/api/v1/runtime/health")
    data = resp.json()
    provider_ids = [p["provider_id"] for p in data["providers"]]
    # At minimum these should be registered
    assert "uia" in provider_ids
    assert "screenshot" in provider_ids
    assert "paddleocr" in provider_ids
    assert "omniparser" in provider_ids


def test_runtime_health_provider_entry_has_required_fields(client):
    """Each provider entry has the required fields."""
    resp = client.get("/api/v1/runtime/health")
    data = resp.json()
    for entry in data["providers"]:
        assert "provider_id" in entry
        assert "ok" in entry
        assert isinstance(entry["ok"], bool)


def test_runtime_health_metadata_separated_from_health(client):
    """Metadata and health are separate sections."""
    resp = client.get("/api/v1/runtime/health")
    data = resp.json()

    # Metadata has static descriptors
    for meta in data["metadata"]:
        assert "provider_id" in meta
        assert "provider_type" in meta
        assert "mode" in meta
        assert "capabilities" in meta
        assert "default_timeout_seconds" in meta

    # Health has dynamic probe results
    for health in data["providers"]:
        assert "provider_id" in health
        assert "ok" in health

    # Both sections cover the same providers
    meta_ids = {m["provider_id"] for m in data["metadata"]}
    health_ids = {h["provider_id"] for h in data["providers"]}
    assert meta_ids == health_ids


def test_runtime_health_metadata_has_config_values(client):
    """Metadata reflects actual config values, not just defaults."""
    resp = client.get("/api/v1/runtime/health")
    data = resp.json()

    ocr_meta = next(m for m in data["metadata"] if m["provider_id"] == "paddleocr")
    # worker_mode should come from config (default "persistent")
    assert ocr_meta["mode"] == "subprocess"
    assert ocr_meta["provider_type"] == "ocr"

    omni_meta = next(m for m in data["metadata"] if m["provider_id"] == "omniparser")
    assert omni_meta["provider_type"] == "vision"
    assert "endpoint" in omni_meta


def test_runtime_health_does_not_break_old_health(client):
    """The old /health endpoint still works."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_runtime_health_does_not_break_vision_health(client):
    """The old /api/v1/providers/vision/health endpoint still works."""
    with patch(
        "src.perception.providers.remote_vision_provider.OmniParserRemoteVisionProvider.health_status",
        return_value=_mock_omniparser_health(),
    ):
        resp = client.get("/api/v1/providers/vision/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "omniparser"


def test_runtime_health_no_mock_vlm_registered(client):
    """vlm_mock is NOT registered — only real VLM providers appear."""
    resp = client.get("/api/v1/runtime/health")
    data = resp.json()
    provider_ids = [p["provider_id"] for p in data["providers"]]
    assert "vlm_mock" not in provider_ids
