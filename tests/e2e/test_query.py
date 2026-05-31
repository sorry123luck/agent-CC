"""E2E tests for POST /api/v1/query."""

from __future__ import annotations


def test_query_by_text(client, seed_canvas):
    """Query by text returns matching candidates."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"text": "Element 0"},
            "max_results": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1
    assert any(c["text"] == "Element 0" for c in data["candidates"])


def test_query_by_semantic_role(client, seed_canvas):
    """Query by semantic_role returns matching candidates."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"semantic_role": "button"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1


def test_query_no_match(client, seed_canvas):
    """Query with non-matching text returns empty candidates when min_confidence filters weak matches."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"text": "completely_unrelated_zzz"},
            "min_confidence": 0.5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] == 0
    assert len(data["candidates"]) == 0


def test_query_max_results(client, seed_canvas):
    """Query respects max_results limit."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"text": "Element"},
            "max_results": 1,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["candidates"]) <= 1


def test_query_candidate_fields(client, seed_canvas):
    """Query returns candidates with all required fields."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={"canvas_id": canvas_id, "target": {"text": "Element"}},
    )
    assert resp.status_code == 200
    candidates = resp.json()["candidates"]
    assert len(candidates) > 0
    c = candidates[0]
    assert "element_id" in c
    assert "semantic_role" in c
    assert "text" in c
    assert "confidence" in c
    assert "confidence_level" in c
    assert "risk_level" in c
    assert "provider_sources" in c


def test_query_by_natural_language(client, seed_canvas):
    """Query by natural_language returns matching candidates."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"natural_language": "button element"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1


def test_query_by_composite(client, seed_canvas):
    """Query by composite filter returns matching candidates."""
    canvas_id, _ = seed_canvas
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"composite": {"interactable": True}},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1


def test_query_invalid_canvas_id(client):
    """Query with invalid canvas_id returns 404."""
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": "nonexistent_canvas_id",
            "target": {"text": "test"},
        },
    )
    assert resp.status_code == 404


def test_query_vlm_only_candidate_preserves_provider_sources_and_confidence(client, seed_canvas_with_vlm_element):
    """VLM-only candidates are queryable and preserve provider_sources=['vlm'] and real confidence."""
    canvas_id, canvas, vlm_element_id = seed_canvas_with_vlm_element
    resp = client.post(
        "/api/v1/query",
        json={
            "canvas_id": canvas_id,
            "target": {"text": "VLM Send"},
            "max_results": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1

    vlm_candidate = next((c for c in data["candidates"] if c["element_id"] == vlm_element_id), None)
    assert vlm_candidate is not None, f"VLM element {vlm_element_id} not found in query results"
    assert vlm_candidate["provider_sources"] == ["vlm"]
    assert vlm_candidate["confidence"] == 0.85
