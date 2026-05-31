"""E2E tests for VLM Semantic Modeler integration.

Covers:
- VLM override affects CanvasDetail
- VirtualModel uses VLM data (priority: manual > VLM > persistent > canvas)
- manual override beats VLM override
- shared_regions don't cross app boundaries
"""

from __future__ import annotations

import json
import uuid

import sqlalchemy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vlm_model_dict(
    app_name: str = "TestApp",
    state_label: str = "main",
    controls: list[dict] | None = None,
    regions: list[dict] | None = None,
) -> dict:
    """Create a minimal VLM parsed_model dict."""
    return {
        "app_identity": {"app_name": app_name, "display_name": app_name},
        "page_state": {
            "page_class": f"{app_name.lower()}/main",
            "state_label": state_label,
            "state_flags": [],
        },
        "regions": regions or [
            {"region_id": "chat_area", "role": "content_area", "bounds": [0, 0, 800, 600], "purpose": "Chat"},
        ],
        "fixed_controls": controls or [
            {
                "control_id": "vlm_send_btn", "semantic_role": "send_button",
                "control_type": "button", "bounds": [700, 550, 780, 590],
                "text": "Send", "interactable": True, "confidence": 0.9,
                "region_id": "chat_area", "visual_type": "button",
            }
        ],
        "candidate_corrections": [],
        "dynamic_zones": [],
        "transitions": [],
    }


def _seed_vlm_response(
    session,
    *,
    page_model_id: str = "",
    state_template_id: str = "",
    parsed_model: dict | None = None,
) -> str:
    """Insert a vlm_responses row. Returns response_id."""
    response_id = str(uuid.uuid4())
    model = parsed_model or _make_vlm_model_dict()
    session.execute(
        sqlalchemy.text(
            """INSERT INTO vlm_responses
               (response_id, cache_key, screenshot_hash, candidate_hash,
                prompt_version, schema_version, provider_name, model_name,
                parsed_model, token_input, token_output, cost_usd, latency_ms,
                status, created_at, page_model_id, state_template_id)
               VALUES (:rid, :ck, :sh, :ch, '1.0', '1.0', 'test', 'test',
                       :pm, 0, 0, 0.0, 0, 'success', datetime('now'), :pm_id, :st_id)"""
        ),
        {
            "rid": response_id,
            "ck": f"cache_{response_id[:8]}",
            "sh": f"ss_{response_id[:8]}",
            "ch": f"cd_{response_id[:8]}",
            "pm": json.dumps(model),
            "pm_id": page_model_id or None,
            "st_id": state_template_id or None,
        },
    )
    return response_id


def _create_page_model_and_state(session, *, app_id="test_app", label_suffix=""):
    """Create a page_model + state_template pair via raw SQL. Returns (pm_id, st_id)."""
    pm_id = str(uuid.uuid4())
    st_id = str(uuid.uuid4())
    now = "2026-01-01T00:00:00"

    session.execute(
        sqlalchemy.text(
            """INSERT INTO page_models
               (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                layout_signature, created_at, last_seen_at, observe_count, state_count)
               VALUES (:pm_id, :app_id, :pcp, :dn, :st, :ls, :now, :now, 1, 1)"""
        ),
        {"pm_id": pm_id, "app_id": app_id, "pcp": f"{app_id}/main",
         "dn": f"Test {app_id}", "st": "native_uia", "ls": f"sig_{label_suffix}", "now": now},
    )
    session.execute(
        sqlalchemy.text(
            """INSERT INTO state_templates
               (state_template_id, page_model_id, app_id, page_class, state_signature,
                layout_fingerprint, state_label, fixed_element_count, total_element_count,
                created_at, last_seen_at, verify_count, snapshot_count)
               VALUES (:st_id, :pm_id, :app_id, :pc, :sig, :fp, :sl, 1, 3, :now, :now, 1, 0)"""
        ),
        {
            "st_id": st_id, "pm_id": pm_id, "app_id": app_id,
            "pc": f"{app_id}/main/default", "sig": f"sig_{label_suffix}",
            "fp": json.dumps({"fixed_roles": ["button"], "region_structure": [], "state_flags": [],
                              "element_count_bucket": "0-5", "input_state": "unknown",
                              "has_dialog": False, "has_menu": False}),
            "sl": "main", "now": now,
        },
    )
    return pm_id, st_id


# ---------------------------------------------------------------------------
# Tests: CanvasDetail + VLM overrides
# ---------------------------------------------------------------------------

class TestCanvasDetailVLMOverride:

    def test_vlm_override_changes_semantic_role(self, client):
        """VLM candidate override changes element's semantic_role via apply_candidate_overrides."""
        from src.memory.candidate_override_apply import apply_candidate_overrides
        from src.memory.candidate_override_store import CandidateOverrideData
        from src.canvas.canvas_cache import get_canvas_cache
        from tests.e2e.conftest import _make_test_canvas

        cache = get_canvas_cache()
        canvas = _make_test_canvas()
        cache.put(canvas)
        elem = canvas.elements[0]

        # Create override data directly (bypasses DB session isolation issues)
        override = CandidateOverrideData(
            override_id="test_override_001",
            scope_key=f"canvas:{canvas.canvas_id}:{elem.element_id}",
            app_id="test_app",
            canvas_id=canvas.canvas_id,
            element_id=elem.element_id,
            semantic_role="submit_button",
            source="vlm_semantic",
            status="active",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        effective = apply_candidate_overrides(canvas, overrides=[override])
        overridden = next(e for e in effective.elements if e.element_id == elem.element_id)
        role_str = (
            overridden.semantic_role.value
            if hasattr(overridden.semantic_role, "value")
            else str(overridden.semantic_role)
        )
        assert role_str == "submit_button"

    def test_canvas_detail_shows_vlm_element(self, client):
        from src.canvas.canvas_cache import get_canvas_cache
        from tests.e2e.conftest import _make_test_canvas
        from src.perception.page_compiler_models import Candidate, ElementState, SemanticRole, ConfidenceLevel, RiskLevel

        cache = get_canvas_cache()
        canvas = _make_test_canvas(element_count=2)
        # Add a VLM-sourced element
        vlm_elem = Candidate(
            element_id="vlm_only_btn",
            region_id="content",
            semantic_role=SemanticRole.UNKNOWN,
            control_type="button",
            bounds=(300, 200, 400, 240),
            text="VLM Button",
            name=None,
            interactable=True,
            state=ElementState(enabled=True, visible=True),
            provider_sources=["vlm"],
            confidence=0.85,
            confidence_level=ConfidenceLevel.MEDIUM,
            risk_level=RiskLevel.L0,
            attributes={"vision_only": True, "source": "vlm"},
        )
        canvas.elements.append(vlm_elem)
        cache.put(canvas)

        resp = client.get(f"/api/v1/canvases/{canvas.canvas_id}")
        assert resp.status_code == 200
        data = resp.json()
        element_ids = [e["element_id"] for e in data["elements"]]
        assert "vlm_only_btn" in element_ids


# ---------------------------------------------------------------------------
# Tests: VirtualModel + VLM priority
# ---------------------------------------------------------------------------

class TestVirtualModelVLMPriority:

    def test_virtual_model_does_not_import_fixed_controls_from_other_state(self, client):
        """Same-app VLM history must not leak fixed_controls across state templates."""
        from src.storage.db import Session

        login_model = _make_vlm_model_dict(controls=[
            {
                "control_id": "login_btn", "semantic_role": "login_button",
                "control_type": "button", "bounds": [100, 100, 180, 140],
                "text": "登录", "interactable": True, "confidence": 0.9,
                "region_id": "main", "visual_type": "button",
            }
        ])
        main_model = _make_vlm_model_dict(controls=[
            {
                "control_id": "send_btn", "semantic_role": "send_button",
                "control_type": "button", "bounds": [700, 550, 780, 590],
                "text": "发送", "interactable": True, "confidence": 0.9,
                "region_id": "main", "visual_type": "button",
            }
        ])

        with Session() as session:
            pm_id, login_st = _create_page_model_and_state(session, app_id="same_app", label_suffix="login")
            _, main_st = _create_page_model_and_state(session, app_id="same_app", label_suffix="main")
            # Re-parent second state to same PageModel to simulate one app/page family.
            session.execute(
                sqlalchemy.text("UPDATE state_templates SET page_model_id = :pm WHERE state_template_id = :st"),
                {"pm": pm_id, "st": main_st},
            )
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=login_st, parsed_model=login_model)
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=main_st, parsed_model=main_model)
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{main_st}/virtual-model")
        assert resp.status_code == 200
        texts = [c["canonical_text"] for c in resp.json()["candidates"]]
        assert "发送" in texts
        assert "登录" not in texts

    def test_virtual_model_includes_vlm_semantic_fields(self, client):
        from src.storage.db import Session

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="vlf")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id)
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()
        assert "vlm_semantic_model" in data
        assert "regions" in data
        assert data["vlm_semantic_model"] is not None

    def test_virtual_model_includes_workbench_summary_fields(self, client):
        from src.storage.db import Session

        model = _make_vlm_model_dict()
        model["visible_items"] = [
            {"item_id": "selected_chat", "matched_candidate_ids": [], "selected": True}
        ]
        model["missing_suggestions"] = [
            {
                "suggestion_id": "m1",
                "type": "possible_missing_control",
                "rough_area": [100, 100, 160, 160],
                "actionability": "safe",
            }
        ]

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="wb")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id, parsed_model=model)
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app_shell"]["app_id"] == "test_app"
        assert data["model_layers"]["candidate_count"] >= 1
        assert data["region_templates"][0]["region_id"] == "chat_area"
        assert data["element_templates"][0]["actionability"] == "review"
        assert data["visible_items"][0]["item_id"] == "selected_chat"
        assert data["missing_suggestions"][0]["actionability"] == "review"

        templates = client.get(f"/api/v1/state-templates/{st_id}/model-templates")
        assert templates.status_code == 200
        persisted = templates.json()
        assert persisted["app_shell"]["app_id"] == "test_app"
        assert persisted["region_templates"][0]["region_id"] == "chat_area"
        assert persisted["element_templates"][0]["actionability"] == "review"

    def test_manual_override_beats_vlm_in_virtual_model(self, client):
        from src.storage.db import Session
        from src.memory.candidate_override_store import CandidateOverrideStore

        store = CandidateOverrideStore()
        stable_key = "stable_key_manual_vs_vlm"

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="mvv")
            # VLM override
            store.upsert(
                session,
                scope_key=f"stable:{pm_id}:{stable_key}",
                app_id="test_app", page_model_id=pm_id, state_template_id=st_id,
                stable_key_id=stable_key, semantic_role="vlm_detected_role",
                source="vlm_semantic", status="active",
            )
            # Manual override (higher priority)
            store.upsert(
                session,
                scope_key=f"manual_override:{stable_key}",
                app_id="test_app", page_model_id=pm_id, state_template_id=st_id,
                stable_key_id=stable_key, semantic_role="manual_corrected_role",
                source="manual", status="active",
            )
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()
        matching = [c for c in data["candidates"] if c["key_id"] == stable_key]
        assert len(matching) > 0
        assert matching[0]["canonical_role"] == "manual_corrected_role"

    def test_vlm_override_applied_when_no_manual(self, client):
        from src.storage.db import Session
        from src.memory.candidate_override_store import CandidateOverrideStore

        store = CandidateOverrideStore()
        stable_key = "stable_key_vlm_only"

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="vlo")
            store.upsert(
                session,
                scope_key=f"stable:{pm_id}:{stable_key}",
                app_id="test_app", page_model_id=pm_id, state_template_id=st_id,
                stable_key_id=stable_key, semantic_role="vlm_detected_role",
                source="vlm_semantic", status="active",
            )
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()
        matching = [c for c in data["candidates"] if c["key_id"] == stable_key]
        assert len(matching) > 0
        assert matching[0]["canonical_role"] == "vlm_detected_role"


# ---------------------------------------------------------------------------
# Tests: shared_regions isolation
# ---------------------------------------------------------------------------

class TestSharedRegionsIsolation:

    def test_shared_regions_filter_by_app_id(self, client):
        from src.storage.db import Session
        from src.perception.vlm_semantic_bridge import VLMSemanticBridge

        bridge = VLMSemanticBridge()

        model_a = _make_vlm_model_dict(regions=[
            {"region_id": "toolbar_a", "role": "toolbar", "bounds": [0, 0, 800, 50], "purpose": "Top"},
        ])
        model_b = _make_vlm_model_dict(app_name="OtherApp", regions=[
            {"region_id": "toolbar_b", "role": "toolbar", "bounds": [0, 0, 800, 50], "purpose": "Other"},
        ])

        with Session() as session:
            pm_a_id, _ = _create_page_model_and_state(session, app_id="app_a", label_suffix="sa")
            pm_b_id, _ = _create_page_model_and_state(session, app_id="app_b", label_suffix="sb")
            for _ in range(3):
                _seed_vlm_response(session, page_model_id=pm_a_id, parsed_model=model_a)
            _seed_vlm_response(session, page_model_id=pm_b_id, parsed_model=model_b)
            session.commit()

        result = bridge.merge_shared_regions(
            "app_a", current_regions=[], current_dynamic_zone_ids=set(),
            db_session_factory=Session,
        )
        for sr in result:
            assert sr.get("region_id") != "toolbar_b"

    def test_dynamic_zones_excluded_from_shared(self, client):
        from src.storage.db import Session
        from src.perception.vlm_semantic_bridge import VLMSemanticBridge

        bridge = VLMSemanticBridge()
        model = _make_vlm_model_dict(regions=[
            {"region_id": "static_area", "role": "content_area", "bounds": [0, 50, 800, 600], "purpose": "Content"},
            {"region_id": "chat_feed", "role": "dynamic_content", "bounds": [50, 100, 750, 500], "purpose": "Chat"},
        ])

        with Session() as session:
            pm_id, _ = _create_page_model_and_state(session, label_suffix="dz")
            for _ in range(3):
                _seed_vlm_response(session, page_model_id=pm_id, parsed_model=model)
            session.commit()

        result = bridge.merge_shared_regions(
            "test_app", current_regions=[], current_dynamic_zone_ids={"chat_feed"},
            db_session_factory=Session,
        )
        for sr in result:
            assert sr.get("region_id") != "chat_feed"


# ---------------------------------------------------------------------------
# Tests: Invariant 2 — display_name identity
# ---------------------------------------------------------------------------

class TestAppIdentityDisplayName:

    def test_display_name_in_vlm_response_roundtrip(self, client):
        """display_name in VLM app_identity survives DB roundtrip and shows in virtual-model."""
        from src.storage.db import Session

        model = _make_vlm_model_dict(app_name="WeChat")
        # Inject display_name into the VLM model dict
        model["app_identity"]["display_name"] = "微信"

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, app_id="wechat", label_suffix="dn")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id, parsed_model=model)
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()
        # The vlm_semantic_model should contain display_name
        assert data.get("vlm_semantic_model") is not None
        ai = data["vlm_semantic_model"].get("app_identity", {})
        assert ai.get("display_name") == "微信"


# ---------------------------------------------------------------------------
# Tests: Invariant 3 — coordinate bounds conversion
# ---------------------------------------------------------------------------

class TestVirtualModelBoundsConversion:

    def test_vlm_pixel_bounds_converted_to_relative(self, client):
        """VLM fixed_controls pixel bounds must be converted to 0-1 relative bounds."""
        from src.storage.db import Session
        from src.canvas.canvas_cache import get_canvas_cache
        from tests.e2e.conftest import _make_test_canvas

        # Seed a canvas with known window dimensions (800x600)
        cache = get_canvas_cache()
        canvas = _make_test_canvas()
        # Set window rect so canvas_w=800, canvas_h=600
        canvas.window.rect_client = (0, 0, 800, 600)
        cache.put(canvas)

        # VLM control at pixel bounds [100, 100, 200, 200]
        # Expected relative: [100/800, 100/600, 200/800, 200/600] = [0.125, 0.1667, 0.25, 0.3333]
        model = _make_vlm_model_dict(controls=[
            {
                "control_id": "vlm_btn", "semantic_role": "button",
                "control_type": "button", "bounds": [100, 100, 200, 200],
                "text": "Test", "interactable": True, "confidence": 0.9,
                "region_id": "chat_area", "visual_type": "button",
            }
        ])

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="bc")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id, parsed_model=model)
            # Create snapshot linking canvas to this state
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO canvas_snapshots
                       (snapshot_id, canvas_id, page_model_id, state_template_id, captured_at,
                        element_count, has_screenshot)
                       VALUES (:sid, :cid, :pm, :st, datetime('now'), 3, 0)"""
                ),
                {"sid": str(uuid.uuid4()), "cid": canvas.canvas_id, "pm": pm_id, "st": st_id},
            )
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()

        vlm_candidates = [c for c in data["candidates"] if c.get("source") == "vlm"]
        assert len(vlm_candidates) > 0
        c = vlm_candidates[0]
        rel = c["relative_bounds"]
        # Should be in [0, 1] range, NOT raw pixel values
        assert all(0.0 <= v <= 1.0 for v in rel), f"Expected 0-1 relative bounds, got {rel}"
        # Check actual values: [100/800, 100/600, 200/800, 200/600]
        assert abs(rel[0] - 100 / 800) < 0.01
        assert abs(rel[1] - 100 / 600) < 0.01
        assert abs(rel[2] - 200 / 800) < 0.01
        assert abs(rel[3] - 200 / 600) < 0.01

    def test_vlm_bounds_use_persisted_image_size_without_canvas_cache(self, client):
        """VLM relative bounds remain valid after CanvasCache is unavailable."""
        from src.storage.db import Session

        model = _make_vlm_model_dict(controls=[
            {
                "control_id": "persisted_bounds_btn", "semantic_role": "button",
                "control_type": "button", "bounds": [100, 100, 200, 200],
                "text": "Persisted", "interactable": True, "confidence": 0.9,
                "region_id": "chat_area", "visual_type": "button",
            }
        ])
        model["image_size"] = [800, 600]

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="pb")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id, parsed_model=model)
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()
        vlm_candidates = [c for c in data["candidates"] if c.get("source") == "vlm"]
        assert vlm_candidates
        rel = vlm_candidates[0]["relative_bounds"]
        assert rel != [0.0, 0.0, 0.0, 0.0]
        assert abs(rel[0] - 0.125) < 0.01
        assert abs(rel[1] - (100 / 600)) < 0.01


# ---------------------------------------------------------------------------
# Tests: Invariant 4 — priority consistency (manual > VLM, no duplicates)
# ---------------------------------------------------------------------------

class TestVirtualModelPriorityConsistency:

    def test_manual_override_covers_vlm_control_no_duplicate(self, client):
        """When manual override targets same element as VLM control, only one candidate appears."""
        from src.storage.db import Session
        from src.memory.candidate_override_store import CandidateOverrideStore

        store = CandidateOverrideStore()
        stable_key = "stable_key_overlap_test"

        # VLM model has a control
        model = _make_vlm_model_dict(controls=[
            {
                "control_id": stable_key, "semantic_role": "send_button",
                "control_type": "button", "bounds": [700, 550, 780, 590],
                "text": "Send", "interactable": True, "confidence": 0.9,
                "region_id": "chat_area", "visual_type": "button",
            }
        ])

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="po")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id, parsed_model=model)
            # Manual override for the same stable_key
            store.upsert(
                session,
                scope_key=f"stable:{pm_id}:{stable_key}",
                app_id="test_app", page_model_id=pm_id, state_template_id=st_id,
                stable_key_id=stable_key, semantic_role="manual_corrected_role",
                source="manual", status="active",
            )
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        data = resp.json()

        # Should have exactly one candidate with this key, not duplicates
        matching = [c for c in data["candidates"] if c["key_id"] == stable_key]
        assert len(matching) == 1, f"Expected 1 candidate, got {len(matching)}: {[c['source'] for c in matching]}"
        # Manual override should win
        assert matching[0]["canonical_role"] == "manual_corrected_role"
        assert matching[0]["source"] == "manual"

    def test_manual_override_covers_vlm_stable_alias_without_canvas_cache(self, client):
        """Persisted stable_key_id alias lets manual override suppress VLM control without CanvasCache."""
        from src.storage.db import Session
        from src.memory.candidate_override_store import CandidateOverrideStore

        store = CandidateOverrideStore()
        stable_key = "stable_key_alias_no_cache"
        model = _make_vlm_model_dict(controls=[
            {
                "control_id": "vlm_send_control", "stable_key_id": stable_key,
                "aliases": ["vlm_send_control", stable_key],
                "semantic_role": "send_button", "control_type": "button",
                "bounds": [700, 550, 780, 590], "text": "Send",
                "interactable": True, "confidence": 0.9,
                "region_id": "chat_area", "visual_type": "button",
            }
        ])
        model["control_aliases"] = {"vlm_send_control": stable_key}
        model["image_size"] = [800, 600]

        with Session() as session:
            pm_id, st_id = _create_page_model_and_state(session, label_suffix="anc")
            _seed_vlm_response(session, page_model_id=pm_id, state_template_id=st_id, parsed_model=model)
            store.upsert(
                session,
                scope_key=f"stable:{pm_id}:{stable_key}",
                app_id="test_app", page_model_id=pm_id, state_template_id=st_id,
                stable_key_id=stable_key, semantic_role="manual_role",
                source="manual", status="active",
            )
            session.commit()

        resp = client.get(f"/api/v1/state-templates/{st_id}/virtual-model")
        assert resp.status_code == 200
        candidates = resp.json()["candidates"]
        assert [c["key_id"] for c in candidates].count(stable_key) == 1
        assert not any(c["key_id"] == "vlm_send_control" for c in candidates)
        assert next(c for c in candidates if c["key_id"] == stable_key)["source"] == "manual"


# ---------------------------------------------------------------------------
# Tests: Invariant 7 — shared_regions via apply_to_canvas
# ---------------------------------------------------------------------------

class TestSharedRegionsInBridge:

    def test_apply_to_canvas_passes_db_session_factory(self, client):
        """apply_to_canvas with db_session_factory executes shared_regions merge."""
        from src.storage.db import Session
        from src.perception.vlm_semantic_bridge import VLMSemanticBridge
        from src.canvas.canvas_cache import get_canvas_cache
        from tests.e2e.conftest import _make_test_canvas

        bridge = VLMSemanticBridge()
        cache = get_canvas_cache()

        # Seed VLM responses with a recurring region (>= 3 times → shared)
        model = _make_vlm_model_dict(regions=[
            {"region_id": "toolbar", "role": "toolbar", "bounds": [0, 0, 800, 50], "purpose": "Top bar"},
        ])
        with Session() as session:
            pm_id, _ = _create_page_model_and_state(session, label_suffix="sr")
            for _ in range(3):
                _seed_vlm_response(session, page_model_id=pm_id, parsed_model=model)
            session.commit()

        # Create a canvas with a matching region
        canvas = _make_test_canvas()
        from src.perception.page_compiler_models import Region as CanvasRegion
        canvas.regions.append(CanvasRegion(
            region_id="test_toolbar",
            role="toolbar",
            bounds=(0, 0, 800, 50),
        ))
        cache.put(canvas)

        # Apply VLM model with db_session_factory → should trigger shared_regions
        vlm_model = _make_vlm_model_dict(regions=[
            {"region_id": "toolbar", "role": "toolbar", "bounds": [0, 0, 800, 50], "purpose": "Top bar"},
        ])
        from src.vlm.schema import PageSemanticModel
        semantic_model = PageSemanticModel.from_dict(vlm_model)

        new_canvas = bridge.apply_to_canvas(canvas, semantic_model, db_session_factory=Session)

        # The region should be marked as shared
        toolbar_regions = [r for r in new_canvas.regions if r.bounds == (0, 0, 800, 50)]
        assert len(toolbar_regions) > 0
        assert toolbar_regions[0].attributes.get("shared") is True

    def test_apply_to_canvas_without_db_factory_skips_shared(self, client):
        """apply_to_canvas without db_session_factory does NOT mark shared regions."""
        from src.perception.vlm_semantic_bridge import VLMSemanticBridge
        from src.canvas.canvas_cache import get_canvas_cache
        from tests.e2e.conftest import _make_test_canvas
        from src.vlm.schema import PageSemanticModel

        bridge = VLMSemanticBridge()
        cache = get_canvas_cache()

        canvas = _make_test_canvas()
        from src.perception.page_compiler_models import Region as CanvasRegion
        canvas.regions.append(CanvasRegion(
            region_id="test_toolbar",
            role="toolbar",
            bounds=(0, 0, 800, 50),
        ))
        cache.put(canvas)

        vlm_model = _make_vlm_model_dict()
        semantic_model = PageSemanticModel.from_dict(vlm_model)

        # No db_session_factory → shared_regions should NOT run
        new_canvas = bridge.apply_to_canvas(canvas, semantic_model)

        toolbar_regions = [r for r in new_canvas.regions if r.bounds == (0, 0, 800, 50)]
        if toolbar_regions:
            assert toolbar_regions[0].attributes.get("shared") is not True
