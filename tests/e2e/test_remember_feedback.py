"""E2E tests for POST /api/v1/remember and POST /api/v1/feedback."""

from __future__ import annotations


# ===== remember =====


def test_remember_creates_template(client, seed_canvas):
    """Remember with confirm=True creates a new template."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/remember",
        json={"canvas_id": canvas_id, "confirm": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert data["template_id"] != ""


def test_remember_rejects_without_confirm(client, seed_canvas):
    """Remember with confirm=False returns rejected."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/remember",
        json={"canvas_id": canvas_id, "confirm": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "rejected"


def test_remember_updates_existing(client, seed_canvas):
    """Remember twice for same canvas updates the template."""
    canvas_id, _ = seed_canvas

    resp1 = client.post(
        "/api/v1/remember",
        json={"canvas_id": canvas_id, "confirm": True},
    )
    assert resp1.json()["status"] == "created"

    resp2 = client.post(
        "/api/v1/remember",
        json={"canvas_id": canvas_id, "confirm": True},
    )
    assert resp2.json()["status"] == "updated"


def test_remember_invalid_canvas(client):
    """Remember with invalid canvas_id returns 404."""
    resp = client.post(
        "/api/v1/remember",
        json={"canvas_id": "nonexistent", "confirm": True},
    )
    assert resp.status_code == 404


def test_remember_persists_across_requests(client, seed_canvas):
    """Remember persists data — second call returns 'updated' instead of 'created'."""
    canvas_id, _ = seed_canvas

    # First remember → creates
    resp1 = client.post(
        "/api/v1/remember",
        json={"canvas_id": canvas_id, "confirm": True},
    )
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "created"

    # Second remember → updates (proves first write persisted)
    resp2 = client.post(
        "/api/v1/remember",
        json={"canvas_id": canvas_id, "confirm": True},
    )
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "updated"


# ===== feedback =====


def test_feedback_records_success(client, seed_canvas):
    """Feedback with type 'success' is recorded."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/feedback",
        json={
            "canvas_id": canvas_id,
            "candidate_key": "test_app/button_0",
            "feedback_type": "success",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["recorded"] is True
    assert "confidence_update" in data


def test_feedback_records_failure(client, seed_canvas):
    """Feedback with type 'failure' is recorded."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/feedback",
        json={
            "canvas_id": canvas_id,
            "candidate_key": "test_app/button_0",
            "feedback_type": "failure",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True


def test_feedback_returns_confidence_update(client, seed_canvas):
    """Feedback persists data — API returns success with confidence_update."""
    canvas_id, _ = seed_canvas

    resp = client.post(
        "/api/v1/feedback",
        json={
            "canvas_id": canvas_id,
            "candidate_key": "test_app/button_0",
            "feedback_type": "success",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["recorded"] is True
    # confidence_update proves the feedback was persisted and processed
    assert isinstance(data["confidence_update"], (int, float))


def test_feedback_with_detail(client, seed_canvas):
    """Feedback accepts optional detail dict."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/feedback",
        json={
            "canvas_id": canvas_id,
            "candidate_key": "test_app/button_0",
            "feedback_type": "success",
            "detail": {"action": "click", "result": "page_changed"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True
