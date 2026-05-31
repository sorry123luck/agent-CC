from src.execution.risk_policy import RiskPolicyEngine
from src.perception.page_compiler_models import (
    Candidate,
    InteractionCanvas,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def make_snapshot(surface_type: SurfaceType) -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="snap_risk",
        window=WindowInfoSnapshot(hwnd=12345, title="Risk Test"),
        surface=SurfaceInfo(surface_type=surface_type, confidence=0.9),
    )


def test_browser_send_button_blocked_by_default():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.elements = [
        Candidate(
            element_id="send_btn",
            semantic_role=SemanticRole.SEND_BUTTON,
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "send_btn", "click")

    assert decision.allowed is False
    assert decision.risk_level == "high"
    assert decision.reason == "browser_high_risk_block"


def test_browser_link_allowed_in_guarded_mode():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.elements = [
        Candidate(
            element_id="nav_link",
            semantic_role=SemanticRole.LINK,
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "nav_link", "click")

    assert decision.allowed is True
    assert decision.risk_level == "low"


def test_browser_sensitive_button_marked_critical():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.elements = [
        Candidate(
            element_id="pay_btn",
            semantic_role=SemanticRole.BUTTON,
            name="支付",
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "pay_btn", "click")

    assert decision.allowed is False
    assert decision.risk_level == "critical"


def test_browser_risk_approved_allows_high_risk_action():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.elements = [
        Candidate(
            element_id="send_btn",
            semantic_role=SemanticRole.SEND_BUTTON,
            bounds=(10, 10, 40, 40),
            attributes={"risk_approved": True},
        )
    ]

    decision = engine.assess(snapshot, "send_btn", "click")

    assert decision.allowed is True
    assert decision.reason == "risk_approved"


def test_site_policy_read_only_blocks_medium_risk_button():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.artifacts["site_policy"] = {"mode": "read_only"}
    snapshot.elements = [
        Candidate(
            element_id="btn",
            semantic_role=SemanticRole.BUTTON,
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "btn", "click")

    assert decision.allowed is False
    assert decision.mode == "read_only"
    assert decision.policy_source == "snapshot.site_policy"
    assert decision.reason == "browser_read_only_block"


def test_workflow_policy_api_first_blocks_even_low_risk_action():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.artifacts["workflow_policy"] = {"mode": "api_first"}
    snapshot.elements = [
        Candidate(
            element_id="nav_link",
            semantic_role=SemanticRole.LINK,
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "nav_link", "click")

    assert decision.allowed is False
    assert decision.mode == "api_first"
    assert decision.reason == "browser_api_first"


def test_workflow_policy_requires_confirmation():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.artifacts["workflow_policy"] = {
        "mode": "confirm_then_execute",
        "require_confirmation": True,
    }
    snapshot.elements = [
        Candidate(
            element_id="nav_link",
            semantic_role=SemanticRole.LINK,
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "nav_link", "click")

    assert decision.allowed is False
    assert decision.reason == "browser_confirmation_required"


def test_workflow_policy_max_auto_risk_level_blocks_higher_risk():
    engine = RiskPolicyEngine()
    snapshot = make_snapshot(SurfaceType.BROWSER)
    snapshot.artifacts["workflow_policy"] = {
        "mode": "guarded_automation",
        "max_auto_risk_level": "medium",
    }
    snapshot.elements = [
        Candidate(
            element_id="send_btn",
            semantic_role=SemanticRole.SEND_BUTTON,
            bounds=(10, 10, 40, 40),
        )
    ]

    decision = engine.assess(snapshot, "send_btn", "click")

    assert decision.allowed is False
    assert decision.reason == "browser_risk_above_policy_limit"
