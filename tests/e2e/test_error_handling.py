"""E2E tests for error handling across all endpoints."""

from __future__ import annotations


def test_query_missing_canvas_id(client):
    """Query without canvas_id returns 422 validation error."""
    resp = client.post(
        "/api/v1/query",
        json={"target": {"text": "test"}},
    )
    assert resp.status_code == 422


def test_query_missing_target(client):
    """Query without target returns 422 validation error."""
    resp = client.post(
        "/api/v1/query",
        json={"canvas_id": "some_id"},
    )
    assert resp.status_code == 422


def test_diff_missing_hwnd(client):
    """Diff without hwnd returns 422 validation error."""
    resp = client.post(
        "/api/v1/diff",
        json={"previous_canvas_id": "some_id"},
    )
    assert resp.status_code == 422


def test_diff_missing_previous_canvas_id(client):
    """Diff without previous_canvas_id returns 422 validation error."""
    resp = client.post(
        "/api/v1/diff",
        json={"hwnd": 12345},
    )
    assert resp.status_code == 422


def test_remember_missing_canvas_id(client):
    """Remember without canvas_id returns 422 validation error."""
    resp = client.post(
        "/api/v1/remember",
        json={"confirm": True},
    )
    assert resp.status_code == 422


def test_feedback_missing_fields(client):
    """Feedback without required fields returns 422 validation error."""
    resp = client.post(
        "/api/v1/feedback",
        json={"canvas_id": "some_id"},
    )
    assert resp.status_code == 422


def test_act_missing_candidate_id(client):
    """Act without candidate_id returns 422 validation error."""
    resp = client.post(
        "/api/v1/act",
        json={"action": "click"},
    )
    assert resp.status_code == 422


def test_act_missing_action(client):
    """Act without action returns 422 validation error."""
    resp = client.post(
        "/api/v1/act",
        json={"candidate_id": "elem_0"},
    )
    assert resp.status_code == 422


def test_root_endpoint(client):
    """Root endpoint returns service info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "OpenClaw Desktop Agent API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Health endpoint returns healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_capabilities_endpoint(client):
    """Capabilities endpoint returns supported features."""
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "1.0"
    assert len(data["supported_actions"]) > 0
    assert "read" in data["supported_actions"]
    assert "inspect" in data["supported_actions"]
    assert len(data["providers"]) > 0
    assert data["supported_canvas_region_apis"] == [
        "GET /api/v1/canvases/{canvas_id}/operability",
        "POST /api/v1/canvases/{canvas_id}/read-region",
        "POST /api/v1/canvases/{canvas_id}/scroll-region",
    ]
    assert data["supported_memory_apis"] == [
        "GET /api/v1/transitions",
        "GET /api/v1/control-transitions",
        "POST /api/v1/control-transitions",
    ]
    action_policy = data["action_policy"]
    assert action_policy["read"]["action_level"] == "read_only"
    assert action_policy["read"]["executes_desktop_input"] is False
    assert action_policy["read"]["requires_execute_confirmed"] is False
    assert action_policy["click"]["action_level"] == "controlled"
    assert action_policy["click"]["requires_execute_confirmed"] is True
    assert action_policy["click"]["verification"] == ["observe_after", "diff_after", "readback_after"]
    assert action_policy["scroll"]["action_level"] == "controlled"
    assert action_policy["scroll"]["requires_execute_confirmed"] is True
    assert action_policy["type_text"]["action_level"] == "blocked_by_default"
    assert action_policy["type_text"]["requires_safe_to_type"] is True
    assert action_policy["type_text"]["verification"] == ["state_probe", "readback_expected_text"]
    assert action_policy["send"]["action_level"] == "blocked_by_default"
    assert action_policy["send"]["requires_expected_text"] is True
    assert action_policy["send"]["verification"] == ["state_probe", "readback_expected_text"]
    assert action_policy["hotkey"]["action_level"] == "review"
    assert action_policy["key_press"]["enabled_for_execution"] is False


def test_observe_empty_body(client):
    """Observe with empty JSON body works (all fields optional, uses foreground window)."""
    resp = client.post("/api/v1/observe", json={})
    # Without hwnd, it tries foreground window — may succeed or 404 if none found
    assert resp.status_code in (200, 404)
