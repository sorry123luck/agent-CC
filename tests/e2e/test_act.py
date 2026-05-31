"""E2E tests for POST /api/v1/act."""

from __future__ import annotations


def test_act_returns_blocked(client):
    """Act currently returns blocked (executor not implemented)."""
    resp = client.post(
        "/api/v1/act",
        json={
            "candidate_id": "elem_0",
            "action": "click",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "blocked"
    assert data["error_message"] is not None
    assert "optional execution adapter" in data["error_message"]


def test_act_accepts_params(client):
    """Act accepts action parameters without error."""
    resp = client.post(
        "/api/v1/act",
        json={
            "candidate_id": "elem_0",
            "action": "type_text",
            "params": {"text": "hello"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["execution_result"] == "blocked"


def test_act_all_action_types(client):
    """Act accepts all documented action types."""
    actions = ["read", "inspect", "click", "double_click", "type_text", "send", "key_press", "hotkey", "scroll"]
    for action in actions:
        resp = client.post(
            "/api/v1/act",
            json={"candidate_id": "elem_0", "action": action},
        )
        assert resp.status_code == 200, f"action={action} failed"
        assert resp.json()["execution_result"] == "blocked"


def test_act_read_executes_without_desktop_input(client, seed_canvas):
    canvas_id, canvas = seed_canvas
    canvas.regions[0].role = "content_area"
    canvas.elements[1].text = "readable panel text"

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_1",
            "action": "read",
            "dry_run": False,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "executed"
    assert data["can_execute"] is True
    assert data["action_plan"]["action_level"] == "read_only"
    assert data["action_plan"]["execution_policy"]["executes_desktop_input"] is False
    assert data["verification_result"]["status"] == "pass"
    assert data["verification_result"]["readback_checked"] is True
    assert data["verification_result"]["readback"]["region_role"] == "content_area"
    assert "readable panel text" in data["verification_result"]["readback"]["after_read"]["text"]


def test_act_read_dry_run_has_no_execution_adapter_warning(client, seed_canvas):
    canvas_id, _canvas = seed_canvas

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_1",
            "action": "read",
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "planned"
    assert data["can_execute"] is True
    assert data["action_plan"]["action_level"] == "read_only"
    assert data["warnings"] == []


def test_act_dry_run_returns_candidate_preflight_plan(client, seed_canvas):
    canvas_id, canvas = seed_canvas
    canvas.elements[0].click_point = (150, 120)
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "click",
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "planned"
    assert data["can_execute"] is False
    assert data["action_plan"]["candidate_id"] == "elem_0"
    assert data["action_plan"]["click_point"] == [150, 120]
    assert data["action_plan"]["action_level"] == "controlled"
    assert data["action_plan"]["execution_policy"]["executes_desktop_input"] is False
    assert data["action_plan"]["execution_policy"]["confirmed_execution_would_touch_desktop"] is True
    assert data["action_plan"]["execution_policy"]["requires_execute_confirmed"] is True
    assert data["action_plan"]["execution_policy"]["enabled_for_execution"] is True
    assert data["action_plan"]["verification_plan"]["observe_after"] is True
    assert data["action_plan"]["verification_plan"]["diff_after"] is True
    assert data["action_plan"]["verification_plan"]["readback_after"] is True
    assert data["action_plan"]["verification_plan"]["readback_plan"]["request"]["region_role"] == "content_area"
    assert data["action_plan"]["verification_plan"]["readback_plan"]["after_observe"] is True
    assert "act_execution_adapter_pending" in data["warnings"]


def test_act_type_text_blocks_review_only_input(client, seed_canvas):
    canvas_id, canvas = seed_canvas
    canvas.elements[0].attributes["actionability"] = "review"
    canvas.elements[0].attributes["safe_to_type"] = False

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "type_text",
            "params": {"text": "hello"},
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "blocked"
    assert data["can_execute"] is False
    assert data["action_plan"]["action_level"] == "blocked"
    assert "type_text_requires_safe_to_type" in data["warnings"]


def test_act_type_text_safe_input_returns_readback_plan(client, seed_canvas):
    from src.perception.page_compiler_models import ElementState, SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.regions[0].role = "composer"
    canvas.elements[0].semantic_role = SemanticRole.MESSAGE_INPUT
    canvas.elements[0].text = ""
    canvas.elements[0].attributes["actionability"] = "controlled_probe"
    canvas.elements[0].attributes["safe_to_type"] = True
    canvas.elements[1].semantic_role = SemanticRole.SEND_BUTTON
    canvas.elements[1].risk_tags = ["send"]
    canvas.elements[1].state = ElementState(enabled=False, visible=True)
    canvas.regions.append(type(canvas.regions[0])(region_id="messages", role="message_stream", bounds=(0, 0, 800, 90)))

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "type_text",
            "params": {"text": "hello"},
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "planned"
    assert data["can_execute"] is False
    assert data["warnings"] == ["act_execution_adapter_pending"]
    verification = data["action_plan"]["verification_plan"]
    assert verification["readback_after"] is True
    assert verification["readback_plan"]["endpoint"] == "POST /api/v1/canvases/{canvas_id}/read-region"
    assert verification["readback_plan"]["request"]["region_role"] == "message_stream"
    assert verification["readback_plan"]["expected_text"] == "hello"
    assert verification["readback_plan"]["source"] == "params.text"
    assert verification["state_probe_plan"]["current"] == {
        "input_state": "empty",
        "send_enabled": False,
    }
    assert {"field": "input_state", "expected": "filled", "source": "params.text"} in verification["state_probe_plan"]["after_expectations"]
    assert {"field": "send_enabled", "expected": True, "condition": "send_control_present"} in verification["state_probe_plan"]["after_expectations"]
    assert verification["requires_human_or_policy_confirmation"] is False


def test_act_type_text_requires_text_for_readback(client, seed_canvas):
    from src.perception.page_compiler_models import SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.elements[0].semantic_role = SemanticRole.MESSAGE_INPUT
    canvas.elements[0].attributes["actionability"] = "controlled_probe"
    canvas.elements[0].attributes["safe_to_type"] = True

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "type_text",
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "blocked"
    assert data["action_plan"]["action_level"] == "blocked"
    assert "type_text_requires_text_for_readback" in data["warnings"]


def test_act_send_requires_expected_text_for_readback(client, seed_canvas):
    from src.perception.page_compiler_models import SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.elements[0].semantic_role = SemanticRole.SEND_BUTTON
    canvas.elements[0].risk_tags = ["send"]
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "send",
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "blocked"
    assert data["action_plan"]["action_level"] == "blocked"
    assert "send_requires_expected_text_for_readback" in data["warnings"]


def test_act_send_blocks_disabled_send_button(client, seed_canvas):
    from src.perception.page_compiler_models import ElementState, SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.elements[0].semantic_role = SemanticRole.SEND_BUTTON
    canvas.elements[0].risk_tags = ["send"]
    canvas.elements[0].state = ElementState(enabled=False, visible=True)
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "send",
            "params": {"expected_text": "DeskCanvas probe"},
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "blocked"
    assert data["action_plan"]["action_level"] == "blocked"
    assert data["action_plan"]["candidate"]["enabled"] is False
    assert "send_button_disabled" in data["warnings"]


def test_act_send_with_expected_text_returns_readback_plan(client, seed_canvas):
    from src.perception.page_compiler_models import SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.regions[0].role = "message_stream"
    canvas.elements[0].semantic_role = SemanticRole.SEND_BUTTON
    canvas.elements[0].risk_tags = ["send"]
    canvas.elements[0].text = "发送"
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "send",
            "params": {"expected_text": "DeskCanvas probe"},
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "planned"
    assert data["can_execute"] is False
    assert data["action_plan"]["action_level"] == "controlled"
    verification = data["action_plan"]["verification_plan"]
    assert verification["readback_after"] is True
    assert verification["readback_plan"]["expected_text"] == "DeskCanvas probe"
    assert verification["readback_plan"]["source"] == "params.expected_text"
    assert {"field": "send_enabled", "expected": True} in verification["state_probe_plan"]["before_requirements"]
    assert {"field": "message_stream_contains", "expected": "DeskCanvas probe", "source": "params.expected_text"} in verification["state_probe_plan"]["after_expectations"]


def test_act_send_accepts_chat_composer_geometric_send_target(client, seed_canvas):
    from src.perception.page_compiler_models import Candidate, ConfidenceLevel, ElementState, RiskLevel, SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.page.page_class = "qq/chat/main/wide"
    canvas.window.rect_client = (0, 0, 960, 640)
    canvas.regions[0].role = "message_stream"
    canvas.elements.append(
        Candidate(
            element_id="synthetic_chat_composer_input",
            semantic_role=SemanticRole.MESSAGE_INPUT,
            control_type="SyntheticInputControl",
            bounds=(315, 530, 656, 622),
            text="",
            interactable=True,
            state=ElementState(enabled=True, visible=True),
            confidence_level=ConfidenceLevel.MEDIUM,
            risk_level=RiskLevel.L0,
            attributes={"actionability": "review", "safe_to_type": False},
        )
    )
    canvas.elements.append(
        Candidate(
            element_id="vision_send_like",
            semantic_role=SemanticRole.UNKNOWN,
            control_type="icon",
            bounds=(664, 598, 762, 627),
            text="",
            interactable=True,
            state=ElementState(enabled=True, visible=True),
            confidence_level=ConfidenceLevel.MEDIUM,
            risk_level=RiskLevel.L0,
            attributes={"source": "omniparser", "actionability": "review"},
        )
    )

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "vision_send_like",
            "action": "send",
            "params": {"expected_text": "DeskCanvas probe"},
            "dry_run": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "send_requires_send_candidate" not in data["warnings"]
    assert data["action_plan"]["candidate"]["semantic_role_hint"] == "send_button"
    verification = data["action_plan"]["verification_plan"]
    assert verification["readback_plan"]["expected_text"] == "DeskCanvas probe"


def test_act_send_confirmed_execution_still_disabled(client, seed_canvas):
    from src.perception.page_compiler_models import SemanticRole

    canvas_id, canvas = seed_canvas
    canvas.window.hwnd = 98765
    canvas.elements[0].semantic_role = SemanticRole.SEND_BUTTON
    canvas.elements[0].risk_tags = ["send"]
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "send",
            "params": {"expected_text": "DeskCanvas probe"},
            "dry_run": False,
            "execute_confirmed": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "blocked"
    assert data["can_execute"] is False
    assert "act_action_not_executable" in data["warnings"]


def test_act_click_execute_confirmed_runs_click_and_diff(client, seed_canvas, monkeypatch):
    from src.execution.action_executor import ActionResult
    from tests.e2e.conftest import _make_test_canvas

    canvas_id, canvas = seed_canvas
    canvas.window.hwnd = 98765
    canvas.elements[0].bounds = (100, 100, 200, 160)
    canvas.elements[0].click_point = (150, 130)
    canvas.elements[0].region_id = "content"
    canvas.elements[0].stable_key_id = "stable_click_0"
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    after = _make_test_canvas(hwnd=98765, element_count=4)
    after.page.page_class = "test_app/panel/opened"
    after.state_template_id = "state_after_click"
    after.elements[1].text = "opened panel text"
    calls = []

    def fake_click_element(self, hwnd, element_x, element_y, element_width, element_height, button=None, double=False):
        calls.append((hwnd, element_x, element_y, element_width, element_height, double))
        return ActionResult(True, "click", "clicked")

    def fake_observe(hwnd, **kwargs):
        from src.canvas.canvas_cache import get_canvas_cache

        get_canvas_cache().put(after)
        return after, "new_state", {"used": False, "status": "skipped", "provider": ""}

    monkeypatch.setattr("src.execution.action_executor.ActionExecutor.click_element", fake_click_element)
    monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "click",
            "params": {},
            "dry_run": False,
            "execute_confirmed": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "executed"
    assert data["can_execute"] is True
    assert data["action_plan"]["action_level"] == "controlled"
    assert data["triggered_diff"]["new_canvas_id"] == after.canvas_id
    assert data["verification_result"]["status"] == "pass"
    assert data["verification_result"]["observed_after"] is True
    assert data["verification_result"]["diff_checked"] is True
    assert data["verification_result"]["readback_checked"] is True
    assert data["verification_result"]["after_canvas_id"] == after.canvas_id
    assert data["verification_result"]["readback"]["after_read"]["text"] == "Element 0\nopened panel text\nElement 2\nElement 3"
    assert data["verification_result"]["readback"]["region_role"] == "content_area"
    assert data["verification_result"]["transition"]["before_canvas_id"] == canvas_id
    assert data["verification_result"]["transition"]["after_canvas_id"] == after.canvas_id
    assert data["verification_result"]["transition"]["before_page_class"] == "test_app/main/default"
    assert data["verification_result"]["transition"]["after_page_class"] == "test_app/panel/opened"
    assert data["verification_result"]["transition"]["page_changed"] is True
    assert data["verification_result"]["transition"]["state_changed"] is True
    assert data["verification_result"]["control_transition"]["recorded"] is True
    assert data["verification_result"]["control_transition"]["candidate_key"] == "stable_click_0"
    transitions = client.get("/api/v1/control-transitions", params={"candidate_key": "stable_click_0"}).json()
    assert transitions["total"] == 1
    row = transitions["transitions"][0]
    assert row["from_page_class"] == "test_app/main/default"
    assert row["to_page_class"] == "test_app/panel/opened"
    assert row["action_type"] == "click"
    assert row["canvas_id_before"] == canvas_id
    assert row["canvas_id_after"] == after.canvas_id
    assert row["success_count"] == 1
    assert calls == [(98765, 100, 100, 100, 60, False)]


def test_act_scroll_execute_confirmed_runs_scroll_and_diff(client, seed_canvas, monkeypatch):
    from src.execution.action_executor import ActionResult
    from tests.e2e.conftest import _make_test_canvas

    canvas_id, canvas = seed_canvas
    canvas.window.hwnd = 98765
    canvas.elements[0].bounds = (100, 100, 200, 160)
    canvas.elements[0].region_id = "content"
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    after = _make_test_canvas(hwnd=98765, element_count=4)
    after.elements[1].text = "after scroll text"
    calls = []

    def fake_scroll_at(self, hwnd, client_x, client_y, delta):
        calls.append((hwnd, client_x, client_y, delta))
        return ActionResult(True, "scroll", "scrolled")

    def fake_observe(hwnd, **kwargs):
        from src.canvas.canvas_cache import get_canvas_cache

        get_canvas_cache().put(after)
        return after, "new_state", {"used": False, "status": "skipped", "provider": ""}

    monkeypatch.setattr("src.execution.action_executor.ActionExecutor.scroll_at", fake_scroll_at)
    monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "scroll",
            "params": {"direction": "down", "amount": "page"},
            "dry_run": False,
            "execute_confirmed": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "executed"
    assert data["triggered_diff"]["new_canvas_id"] == after.canvas_id
    assert data["verification_result"]["status"] == "pass"
    assert data["verification_result"]["observed_after"] is True
    assert data["verification_result"]["diff_checked"] is True
    assert data["verification_result"]["readback_checked"] is True
    assert data["verification_result"]["after_canvas_id"] == after.canvas_id
    assert data["verification_result"]["readback"]["after_read"]["text"] == "Element 0\nafter scroll text\nElement 2\nElement 3"
    assert data["verification_result"]["readback"]["region_role"] == "content_area"
    assert calls == [(98765, 150, 130, -480)]


def test_act_scroll_allows_explicit_readback_opt_out(client, seed_canvas, monkeypatch):
    from src.execution.action_executor import ActionResult
    from tests.e2e.conftest import _make_test_canvas

    canvas_id, canvas = seed_canvas
    canvas.window.hwnd = 98765
    canvas.elements[0].bounds = (100, 100, 200, 160)
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    after = _make_test_canvas(hwnd=98765, element_count=4)

    monkeypatch.setattr(
        "src.execution.action_executor.ActionExecutor.scroll_at",
        lambda self, hwnd, client_x, client_y, delta: ActionResult(True, "scroll", "scrolled"),
    )

    def fake_observe(hwnd, **kwargs):
        from src.canvas.canvas_cache import get_canvas_cache

        get_canvas_cache().put(after)
        return after, "new_state", {"used": False, "status": "skipped", "provider": ""}

    monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "scroll",
            "params": {"read_after": False},
            "dry_run": False,
            "execute_confirmed": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "executed"
    assert data["verification_result"]["readback_checked"] is False
    assert data["verification_result"]["readback"] == {}


def test_act_click_readback_missing_marks_verification_review(client, seed_canvas, monkeypatch):
    from src.execution.action_executor import ActionResult
    from tests.e2e.conftest import _make_test_canvas

    canvas_id, canvas = seed_canvas
    canvas.window.hwnd = 98765
    canvas.elements[0].bounds = (100, 100, 200, 160)
    canvas.elements[0].region_id = "content"
    canvas.elements[0].attributes["actionability"] = "controlled_probe"

    after = _make_test_canvas(hwnd=98765, element_count=4)
    after.regions = []

    monkeypatch.setattr(
        "src.execution.action_executor.ActionExecutor.click_element",
        lambda self, hwnd, element_x, element_y, element_width, element_height, button=None, double=False: ActionResult(True, "click", "clicked"),
    )

    def fake_observe(hwnd, **kwargs):
        from src.canvas.canvas_cache import get_canvas_cache

        get_canvas_cache().put(after)
        return after, "new_state", {"used": False, "status": "skipped", "provider": ""}

    monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)

    resp = client.post(
        "/api/v1/act",
        json={
            "canvas_id": canvas_id,
            "candidate_id": "elem_0",
            "action": "click",
            "dry_run": False,
            "execute_confirmed": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_result"] == "executed"
    assert data["verification_result"]["observed_after"] is True
    assert data["verification_result"]["diff_checked"] is True
    assert data["verification_result"]["readback_checked"] is True
    assert data["verification_result"]["readback"]["status"] == "missing_region"
    assert data["verification_result"]["status"] == "review"
    assert "act_scroll_readback_region_missing" in data["warnings"]
