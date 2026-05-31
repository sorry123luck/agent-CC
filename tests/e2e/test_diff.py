"""E2E tests for POST /api/v1/diff."""

from __future__ import annotations


def _observe_first(client, mock_observe):
    """Helper: run observe and return the canvas_id."""
    resp = client.post("/api/v1/observe", json={"hwnd": 12345})
    assert resp.status_code == 200
    return resp.json()["canvas_id"]


def test_diff_returns_changes(client, mock_observe_varying):
    """Diff between two observations returns added/removed/preserved."""
    # First observe
    resp1 = client.post("/api/v1/observe", json={"hwnd": 12345})
    assert resp1.status_code == 200
    first_canvas_id = resp1.json()["canvas_id"]

    # Diff (triggers second observe internally)
    resp2 = client.post(
        "/api/v1/diff",
        json={
            "hwnd": 12345,
            "previous_canvas_id": first_canvas_id,
        },
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert "new_canvas_id" in data
    assert data["new_canvas_id"] != first_canvas_id
    assert "added" in data
    assert "removed" in data
    assert "preserved" in data
    assert "page_changed" in data
    assert "summary" in data


def test_diff_new_canvas_cached(client, mock_observe_varying):
    """Diff caches the new canvas for subsequent queries."""
    resp1 = client.post("/api/v1/observe", json={"hwnd": 12345})
    first_canvas_id = resp1.json()["canvas_id"]

    resp2 = client.post(
        "/api/v1/diff",
        json={"hwnd": 12345, "previous_canvas_id": first_canvas_id},
    )
    new_canvas_id = resp2.json()["new_canvas_id"]

    # New canvas should be queryable
    resp3 = client.post(
        "/api/v1/query",
        json={"canvas_id": new_canvas_id, "target": {"text": "Element"}},
    )
    assert resp3.status_code == 200


def test_diff_invalid_previous_canvas(client, mock_observe):
    """Diff with invalid previous_canvas_id returns 404."""
    resp = client.post(
        "/api/v1/diff",
        json={
            "hwnd": 12345,
            "previous_canvas_id": "nonexistent",
        },
    )
    assert resp.status_code == 404


def test_diff_preserves_page_info(client, mock_observe_varying):
    """Diff response includes page_changed and page_class_changed flags."""
    resp1 = client.post("/api/v1/observe", json={"hwnd": 12345})
    first_canvas_id = resp1.json()["canvas_id"]

    resp2 = client.post(
        "/api/v1/diff",
        json={"hwnd": 12345, "previous_canvas_id": first_canvas_id},
    )
    data = resp2.json()
    assert isinstance(data["page_changed"], bool)
    assert isinstance(data["page_class_changed"], bool)
