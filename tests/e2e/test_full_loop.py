"""E2E test: full protocol loop observe → query → diff → remember → feedback."""

from __future__ import annotations


def test_full_protocol_loop(client, mock_observe_varying):
    """Agent can complete the full protocol loop via HTTP."""
    # Step 1: Observe
    resp = client.post("/api/v1/observe", json={"hwnd": 12345})
    assert resp.status_code == 200
    canvas_id = resp.json()["canvas_id"]
    assert canvas_id != ""

    # Step 2: Query
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"text": "Element"},
            "max_results": 5,
        },
    )
    assert resp.status_code == 200
    query_data = resp.json()
    assert query_data["total_matched"] >= 1
    candidate = query_data["candidates"][0]
    assert candidate["element_id"] != ""

    # Step 3: Diff (triggers re-observe)
    resp = client.post(
        "/api/v1/diff",
        json={"hwnd": 12345, "previous_canvas_id": canvas_id},
    )
    assert resp.status_code == 200
    diff_data = resp.json()
    new_canvas_id = diff_data["new_canvas_id"]
    assert new_canvas_id != canvas_id

    # Step 4: Remember
    resp = client.post(
        "/api/v1/remember",
        json={"canvas_id": new_canvas_id, "confirm": True},
    )
    assert resp.status_code == 200
    remember_data = resp.json()
    assert remember_data["status"] == "created"
    assert remember_data["template_id"] != ""

    # Step 5: Feedback
    resp = client.post(
        "/api/v1/feedback",
        json={
            "canvas_id": new_canvas_id,
            "candidate_key": f"test_app/{candidate['semantic_role']}",
            "feedback_type": "success",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True


def test_query_after_diff_uses_new_canvas(client, mock_observe_varying):
    """After diff, the new canvas can be queried."""
    # Observe
    resp = client.post("/api/v1/observe", json={"hwnd": 12345})
    canvas_id = resp.json()["canvas_id"]

    # Diff
    resp = client.post(
        "/api/v1/diff",
        json={"hwnd": 12345, "previous_canvas_id": canvas_id},
    )
    new_canvas_id = resp.json()["new_canvas_id"]

    # Query new canvas
    resp = client.post(
        "/api/v1/query",
        json={"canvas_id": new_canvas_id, "target": {"text": "Element"}},
    )
    assert resp.status_code == 200
    assert resp.json()["total_matched"] >= 1


def test_remember_then_capabilities(client, mock_observe):
    """After remember, capabilities endpoint still works."""
    resp = client.post("/api/v1/observe", json={"hwnd": 12345})
    canvas_id = resp.json()["canvas_id"]

    client.post("/api/v1/remember", json={"canvas_id": canvas_id, "confirm": True})

    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    assert "observe" in resp.json()["supported_query_types"] or len(resp.json()["supported_query_types"]) > 0
