"""Control-state transition API tests."""

from __future__ import annotations


def test_record_and_query_control_transition(client):
    payload = {
        "candidate_key": "stable:pm:key_1",
        "stable_key_id": "key_1",
        "from_page_class": "wechat/chat",
        "to_page_class": "wechat/settings",
        "action_type": "click",
        "canvas_id_before": "canvas_a",
        "canvas_id_after": "canvas_b",
        "success": True,
    }
    resp = client.post("/api/v1/control-transitions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidate_key"] == "stable:pm:key_1"
    assert data["to_page_class"] == "wechat/settings"
    assert data["success_rate"] == 1.0

    by_candidate = client.get("/api/v1/control-transitions?candidate_key=stable%3Apm%3Akey_1")
    assert by_candidate.status_code == 200
    assert by_candidate.json()["total"] == 1

    by_page = client.get("/api/v1/control-transitions?page_class=wechat%2Fchat")
    assert by_page.status_code == 200
    assert by_page.json()["transitions"][0]["candidate_key"] == "stable:pm:key_1"


def test_control_transition_requires_filter(client):
    resp = client.get("/api/v1/control-transitions")
    assert resp.status_code == 400
