import uuid

import sqlalchemy


def _create_state(session, *, page_class: str = "test_app/main/default") -> tuple[str, str]:
    pm_id = str(uuid.uuid4())
    st_id = str(uuid.uuid4())
    now = "2026-01-01T00:00:00"
    session.execute(
        sqlalchemy.text(
            """INSERT INTO page_models
               (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                created_at, last_seen_at, observe_count, state_count)
               VALUES (:pm, 'test_app', 'test_app/main', 'TestApp', 'native_uia',
                       :now, :now, 1, 1)"""
        ),
        {"pm": pm_id, "now": now},
    )
    session.execute(
        sqlalchemy.text(
            """INSERT INTO state_templates
               (state_template_id, page_model_id, app_id, page_class, state_label,
                layout_fingerprint, state_signature, created_at, last_seen_at,
                verify_count, snapshot_count, fixed_element_count, total_element_count)
               VALUES (:st, :pm, 'test_app', :page_class, 'main',
                       '{}', 'sig', :now, :now, 1, 0, 0, 0)"""
        ),
        {"st": st_id, "pm": pm_id, "page_class": page_class, "now": now},
    )
    return pm_id, st_id


def test_transition_graph_query_by_page_class(client):
    from src.memory.transition_graph import TransitionGraphManager
    from src.storage.db import Session

    with Session() as session:
        manager = TransitionGraphManager(session)
        manager.record_transition("page_a", "page_b", action="click", candidate_key="key_1", success=True)
        manager.record_transition("page_a", "page_b", action="click", candidate_key="key_1", success=False)

    response = client.get("/api/v1/transitions", params={"page_class": "page_a"})

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    edge = data["transitions"][0]
    assert edge["from_page_class"] == "page_a"
    assert edge["to_page_class"] == "page_b"
    assert edge["trigger_action"] == "click"
    assert edge["observe_count"] == 2
    assert edge["success_count"] == 1
    assert edge["success_rate"] == 0.5


def test_transition_graph_query_by_state_template(client):
    from src.memory.transition_graph import TransitionGraphManager
    from src.storage.db import Session

    with Session() as session:
        _, st_id = _create_state(session, page_class="test_app/main/default")
        TransitionGraphManager(session).record_transition(
            "test_app/main/default",
            "test_app/settings/default",
            action="open_settings",
            candidate_key="settings_key",
            success=True,
        )

    response = client.get(f"/api/v1/state-templates/{st_id}/transitions")

    assert response.status_code == 200
    data = response.json()
    assert data["page_class"] == "test_app/main/default"
    assert data["total"] == 1
    assert data["transitions"][0]["trigger_candidate_key"] == "settings_key"


def test_transition_graph_requires_filter(client):
    response = client.get("/api/v1/transitions")

    assert response.status_code == 400
