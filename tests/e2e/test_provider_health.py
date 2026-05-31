"""E2E tests for provider health endpoints."""

from __future__ import annotations

from unittest.mock import patch


def test_vision_provider_health_endpoint(client):
    with patch(
        "src.perception.providers.remote_vision_provider.OmniParserRemoteVisionProvider.health_status",
        return_value={
            "provider": "omniparser",
            "enabled": True,
            "endpoint": "http://127.0.0.1:8001/parse/",
            "probe_endpoint": "http://127.0.0.1:8001/probe/",
            "ok": False,
            "probe": {"ok": False, "http_status": 404, "error": "vision_probe_http_404"},
        },
    ):
        resp = client.get("/api/v1/providers/vision/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "omniparser"
    assert data["ok"] is False
    assert data["probe"]["error"] == "vision_probe_http_404"
