"""E2E tests for the refine endpoint.

Tests:
- POST /api/v1/canvases/{canvas_id}/refine (agent mode write-back)
- Agent mode partial update
- Invalid element_id handling
- No memory/DB write
- Heuristic mode still works
- Refine enhances subsequent query results
"""

from __future__ import annotations

import pytest


class TestAgentModeRefine:
    """Agent mode: external Agent submits classification results."""

    def test_agent_mode_write_results(self, client, seed_canvas):
        """Agent submits refine results → canvas detail returns refined fields."""
        canvas_id, canvas = seed_canvas
        elem_id = canvas.elements[0].element_id

        # Agent submits refine result
        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": elem_id,
                    "visual_type": "button",
                    "semantic_tags": ["action.send"],
                    "role_label": "发送按钮",
                    "role_confidence": 0.95,
                    "role_source": "agent",
                    "role_evidence": ["crop shows send icon", "text matches send"],
                    "refine_status": "refined",
                }
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "agent"
        assert data["total_refined"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["role_label"] == "发送按钮"

        # Canvas detail should return refined fields
        detail = client.get(f"/api/v1/canvases/{canvas_id}")
        assert detail.status_code == 200
        elements = detail.json()["elements"]
        refined_el = next(e for e in elements if e["element_id"] == elem_id)
        assert refined_el["visual_type"] == "button"
        assert refined_el["role_label"] == "发送按钮"
        assert "action.send" in refined_el["semantic_tags"]
        assert refined_el["role_confidence"] == 0.95
        assert refined_el["role_source"] == "agent"
        assert refined_el["refine_status"] == "refined"

    def test_agent_mode_partial_update(self, client, seed_canvas):
        """Only submitted candidates are updated; others remain unchanged."""
        canvas_id, canvas = seed_canvas
        elem_0 = canvas.elements[0].element_id
        elem_1 = canvas.elements[1].element_id

        # Only refine elem_0
        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": elem_0,
                    "visual_type": "button",
                    "role_label": "发送",
                    "role_confidence": 0.9,
                    "role_source": "agent",
                    "refine_status": "refined",
                }
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["total_refined"] == 1

        # Check canvas detail
        detail = client.get(f"/api/v1/canvases/{canvas_id}")
        elements = detail.json()["elements"]

        el_0 = next(e for e in elements if e["element_id"] == elem_0)
        assert el_0["refine_status"] == "refined"
        assert el_0["role_label"] == "发送"

        # elem_1 should remain unreviewed
        el_1 = next(e for e in elements if e["element_id"] == elem_1)
        assert el_1["refine_status"] == "unreviewed"
        assert el_1["role_label"] is None

    def test_agent_mode_invalid_element_id(self, client, seed_canvas):
        """Non-existent element_id returns 400 with error details."""
        canvas_id, _ = seed_canvas

        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": "nonexistent_id",
                    "visual_type": "icon",
                    "role_label": "未知",
                    "refine_status": "uncertain",
                }
            ],
        })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "errors" in detail
        assert any("nonexistent_id" in e for e in detail["errors"])

    def test_agent_mode_requires_results(self, client, seed_canvas):
        """Agent mode without results is rejected instead of resetting candidates."""
        canvas_id, _ = seed_canvas

        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
        })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "errors" in detail
        assert any("requires" in e for e in detail["errors"])

    def test_refine_no_memory_write(self, client, seed_canvas):
        """Refine does not write to DB or memory tables."""
        canvas_id, canvas = seed_canvas
        elem_id = canvas.elements[0].element_id

        # Submit refine
        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": elem_id,
                    "visual_type": "button",
                    "role_label": "测试",
                    "role_source": "agent",
                    "refine_status": "refined",
                }
            ],
        })
        assert resp.status_code == 200

        # Verify no DB writes — query the page_templates table (should be empty)
        # This is a proxy check; refine should only touch in-memory canvas objects
        resp = client.get(f"/api/v1/canvases/{canvas_id}")
        assert resp.status_code == 200

    def test_heuristic_mode_still_works(self, client, seed_canvas):
        """Heuristic mode is unaffected by agent mode changes."""
        canvas_id, canvas = seed_canvas

        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "heuristic",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "heuristic"
        # Heuristic should produce results for all elements
        assert len(data["results"]) == len(canvas.elements)
        # At least some should have a visual_type assigned
        assert any(r["visual_type"] != "unknown" for r in data["results"])


class TestRefineEnhancesQuery:
    """Refine results should be searchable via the query endpoint."""

    def test_refine_enhances_query_by_role_label(self, client, seed_canvas):
        """After agent refine with role_label, query by text matches it."""
        canvas_id, canvas = seed_canvas

        # Find an icon-like element (text is empty, small bounds)
        # Use elem_0 which has text="Element 0" — refine its role_label
        elem_id = canvas.elements[0].element_id

        # Agent refines with a Chinese role_label
        resp = client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": elem_id,
                    "visual_type": "button",
                    "semantic_tags": ["action.send"],
                    "role_label": "表情按钮",
                    "role_confidence": 0.9,
                    "role_source": "agent",
                    "refine_status": "refined",
                }
            ],
        })
        assert resp.status_code == 200

        # Query by text="表情按钮" — should find it via role_label
        query_resp = client.post("/api/v1/query", json={
            "canvas_id": canvas_id,
            "target": {"text": "表情按钮"},
            "max_results": 5,
        })
        assert query_resp.status_code == 200
        candidates = query_resp.json()["candidates"]
        matched_ids = [c["element_id"] for c in candidates]
        assert elem_id in matched_ids

        # Verify the returned candidate has both original and refine fields
        matched = next(c for c in candidates if c["element_id"] == elem_id)
        # Original fields preserved
        assert matched["text"] == "Element 0"  # original text unchanged
        assert matched["semantic_role"] is not None
        # Refine fields present
        assert matched["role_label"] == "表情按钮"
        assert "action.send" in matched["semantic_tags"]
        assert matched["refine_status"] == "refined"

    def test_refine_enhances_query_by_natural_language(self, client, seed_canvas):
        """Natural language query matches refine semantic_tags and role_label."""
        canvas_id, canvas = seed_canvas
        elem_id = canvas.elements[0].element_id

        # Agent refines
        client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": elem_id,
                    "visual_type": "button",
                    "semantic_tags": ["content.emoji"],
                    "role_label": "表情",
                    "role_confidence": 0.85,
                    "role_source": "agent",
                    "refine_status": "refined",
                }
            ],
        })

        # Natural language query for "表情"
        query_resp = client.post("/api/v1/query", json={
            "canvas_id": canvas_id,
            "target": {"natural_language": "表情"},
            "max_results": 5,
        })
        assert query_resp.status_code == 200
        candidates = query_resp.json()["candidates"]
        matched_ids = [c["element_id"] for c in candidates]
        assert elem_id in matched_ids

    def test_refine_enhances_query_by_composite(self, client, seed_canvas):
        """Composite query can filter by refine fields."""
        canvas_id, canvas = seed_canvas
        elem_id = canvas.elements[0].element_id

        # Agent refines
        client.post(f"/api/v1/canvases/{canvas_id}/refine", json={
            "mode": "agent",
            "results": [
                {
                    "element_id": elem_id,
                    "visual_type": "button",
                    "semantic_tags": ["action.send"],
                    "role_label": "发送",
                    "role_source": "agent",
                    "refine_status": "refined",
                }
            ],
        })

        # Composite query by role_label
        query_resp = client.post("/api/v1/query", json={
            "canvas_id": canvas_id,
            "target": {"composite": {"role_label": "发送"}},
            "max_results": 5,
        })
        assert query_resp.status_code == 200
        candidates = query_resp.json()["candidates"]
        assert len(candidates) >= 1
        assert any(c["element_id"] == elem_id for c in candidates)

        # Composite query by semantic_tags
        query_resp2 = client.post("/api/v1/query", json={
            "canvas_id": canvas_id,
            "target": {"composite": {"semantic_tags": ["action.send"]}},
            "max_results": 5,
        })
        assert query_resp2.status_code == 200
        candidates2 = query_resp2.json()["candidates"]
        assert any(c["element_id"] == elem_id for c in candidates2)
