from src.memory.evidence_store import EvidenceStore


def test_manual_candidate_override_writes_evidence(client):
    payload = {
        "scope_key": "canvas:snap_a:elem_1",
        "app_id": "test_app",
        "page_model_id": "pm_1",
        "state_template_id": "st_1",
        "canvas_id": "snap_a",
        "element_id": "elem_1",
        "stable_key_id": "stable_1",
        "label": "Search",
        "semantic_role": "search_input",
        "visual_type": "input",
        "region_id": "content",
        "kind": "fixed",
        "absolute_bounds": [10, 20, 110, 60],
        "source": "manual",
        "status": "active",
    }

    response = client.put("/api/v1/candidate-overrides", json=payload)

    assert response.status_code == 200
    from src.storage.db import Session
    with Session() as session:
        evidence = EvidenceStore().list_by_canvas(session, "snap_a")

    assert len(evidence) == 1
    row = evidence[0]
    assert row["provider"] == "manual"
    assert row["evidence_scope"] == "candidate"
    assert row["evidence_event"] == "override"
    assert row["element_id"] == "elem_1"
    assert row["stable_key_id"] == "stable_1"
    assert row["semantic_role"] == "search_input"
    assert row["control_type"] == "input"
    assert row["raw_confidence"] == 1.0


def test_vlm_candidate_override_does_not_write_manual_override_evidence(client):
    payload = {
        "scope_key": "canvas:snap_a:elem_1",
        "canvas_id": "snap_a",
        "element_id": "elem_1",
        "semantic_role": "search_input",
        "source": "vlm_semantic",
        "status": "active",
    }

    response = client.put("/api/v1/candidate-overrides", json=payload)

    assert response.status_code == 200
    from src.storage.db import Session
    with Session() as session:
        evidence = EvidenceStore().list_by_canvas(session, "snap_a")

    assert evidence == []
