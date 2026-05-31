from unittest.mock import MagicMock, patch

from PIL import Image

from src.execution.action_executor import ActionResult
from src.execution.action_service import ActionOutcome, ActionService
from src.execution.failure_recovery import LocatorCandidate
from src.perception.page_compiler_models import (
    Anchor,
    AnchorKind,
    CoordinateSpace,
    ElementRelation,
    Locator,
    LocatorKind,
    LocatorStatus,
    Candidate,
    InteractionCanvas,
    RelationType,
    Region,
    ScrollContext,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def make_snapshot() -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="snap_test",
        window=WindowInfoSnapshot(hwnd=12345, title="Test"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
    )


def test_execute_snapshot_action_geometry_uses_click_element_path():
    service = ActionService()
    snapshot = make_snapshot()

    locator = Locator(
        locator_id="loc_vis",
        element_ref="elem1",
        kind=LocatorKind.VISION_BBOX,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"bbox": [100, 120, 220, 180]},
    )
    resolved = {
        "locator": locator,
        "method": "geometry",
        "geometry": (100, 120, 120, 60),
        "coordinates": (160, 150),
    }

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service,
        "action_click_element",
        return_value=MagicMock(
            success=True,
            action="click",
            message="ok",
            attempts=2,
            strategy="direct",
            error=None,
        ),
    ) as mock_click_element:
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click")

    assert outcome.success is True
    mock_click_element.assert_called_once_with(
        hwnd=12345,
        element_x=100,
        element_y=120,
        element_width=120,
        element_height=60,
        verify=True,
        with_recovery=True,
    )


def test_execute_snapshot_action_coordinate_uses_recovery_after_coordinate_conversion():
    service = ActionService()
    snapshot = make_snapshot()

    locator = Locator(
        locator_id="loc_coord",
        element_ref="elem1",
        kind=LocatorKind.EPHEMERAL_COORD,
        priority=0,
        status=LocatorStatus.ACTIVE,
        coordinate_space=CoordinateSpace.WINDOW,
        selector={"x": 150, "y": 125},
    )
    resolved = {
        "locator": locator,
        "method": "coordinate",
        "geometry": None,
        "coordinates": (150, 125),
    }
    recovery_result = MagicMock()
    recovery_result.recovered = True
    recovery_result.attempts = 3
    recovery_result.strategy_used.value = "retry"
    recovery_result.final_result = ActionResult(success=True, action="click", details="ok")
    recovery_result.error = None

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._executor,
        "_client_to_screen",
        return_value=(450, 325),
    ) as mock_to_screen, patch.object(
        service._recovery,
        "click_with_recovery",
        return_value=recovery_result,
    ) as mock_recovery:
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click")

    assert outcome.success is True
    assert outcome.attempts == 3
    assert outcome.strategy == "locator_first/ephemeral_coord/retry"
    mock_to_screen.assert_called_once_with(12345, 150, 125)
    mock_recovery.assert_called_once()
    assert mock_recovery.call_args.kwargs["x"] == 450
    assert mock_recovery.call_args.kwargs["y"] == 325


def test_execute_snapshot_action_falls_back_to_adapter_for_uia():
    service = ActionService()
    snapshot = make_snapshot()

    locator = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    resolved = {
        "locator": locator,
        "method": "uia",
        "geometry": None,
        "coordinates": None,
    }
    adapter_result = ActionResult(success=True, action="click", details="uia ok")

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=adapter_result,
    ) as mock_execute:
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click", with_recovery=False)

    assert outcome.success is True
    assert outcome.strategy == "locator_first/uia/uia"
    mock_execute.assert_called_once_with(
        snapshot=snapshot,
        element_id="elem1",
        action="click",
        verify=True,
    )


def test_execute_snapshot_action_returns_element_not_found():
    service = ActionService()
    snapshot = make_snapshot()

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=None):
        outcome = service.execute_snapshot_action(snapshot, "missing", "click")

    assert outcome.success is False
    assert outcome.error == "element_not_found"


def test_execute_snapshot_action_uses_locator_chain_recovery():
    service = ActionService()
    snapshot = make_snapshot()
    locator_uia = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    locator_vision = Locator(
        locator_id="loc_vis",
        element_ref="elem1",
        kind=LocatorKind.VISION_BBOX,
        priority=1,
        status=LocatorStatus.ACTIVE,
        selector={"bbox": [100, 100, 200, 150]},
    )
    snapshot.locators = [locator_uia, locator_vision]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            locator_ids=["loc_uia", "loc_vis"],
            bounds=(100, 100, 200, 150),
        )
    ]

    resolved = {
        "locator": locator_uia,
        "method": "uia",
        "geometry": None,
        "coordinates": None,
    }
    recovery_result = MagicMock()
    recovery_result.recovered = True
    recovery_result.attempts = 2
    recovery_result.strategy_used.value = "fallback_coordinate"
    recovery_result.final_result = ActionResult(
        success=True,
        action="click",
        details="vision ok [recovery mode=replan original=elem1 resolved=elem1_new]",
    )
    recovery_result.error = None
    recovery_result.details = {"attempt_trace": [{"candidate": "uia", "stage": "action"}]}

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._recovery,
        "execute_locator_fallback_chain",
        return_value=recovery_result,
    ) as mock_chain:
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click")

    assert outcome.success is True
    assert outcome.attempts == 2
    assert outcome.strategy == "locator_first/chain/fallback_coordinate"
    assert outcome.verification_details["recovery_execution"]["resolved"] == "elem1_new"
    assert mock_chain.call_args.kwargs["candidates"][0].name == "uia"
    assert mock_chain.call_args.kwargs["candidates"][1].name == "vision_bbox"


def test_snapshot_verification_fn_defers_when_recovery_result_contains_remap_metadata():
    service = ActionService()
    snapshot = make_snapshot()

    verify_fn = service._make_snapshot_verification_fn(snapshot, "elem1")
    result = verify_fn(
        LocatorCandidate(
            name="relative",
            action_fn=lambda: ActionResult(success=True, action="click", details="ok"),
        ),
        ActionResult(
            success=True,
            action="click",
            details="vision ok [recovery mode=anchor original=elem1 resolved=elem1_new]",
        ),
    )

    assert result.verified is True
    assert result.message == "recovery_verification_deferred"
    assert result.details["candidate"] == "relative"


def test_build_locator_candidates_follows_surface_strategy_order():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.ELECTRON_WEBVIEW, confidence=0.9)
    snapshot.locators = [
        Locator(
            locator_id="loc_relative",
            element_ref="elem1",
            kind=LocatorKind.RELATIVE,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"bbox": [10, 10, 40, 40]},
        ),
        Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=10,
            status=LocatorStatus.ACTIVE,
            selector={"name": "sendButton"},
        ),
        Locator(
            locator_id="loc_uia",
            element_ref="elem1",
            kind=LocatorKind.UIA,
            priority=20,
            status=LocatorStatus.ACTIVE,
            selector={"automation_id": "send_btn"},
        ),
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            locator_ids=["loc_relative", "loc_dom", "loc_uia"],
            bounds=(100, 100, 200, 150),
        )
    ]

    candidates = service._build_locator_candidates(snapshot, "elem1", "click")

    assert [candidate.name for candidate in candidates] == ["uia", "dom", "relative"]


def test_build_locator_candidates_adds_shortcut_for_send_button():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.locators = [
        Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"name": "sendButton"},
        ),
        Locator(
            locator_id="loc_relative",
            element_ref="elem1",
            kind=LocatorKind.RELATIVE,
            priority=1,
            status=LocatorStatus.ACTIVE,
            selector={"bbox": [10, 10, 40, 40]},
        ),
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc_dom", "loc_relative"],
            bounds=(100, 100, 200, 150),
        )
    ]

    candidates = service._build_locator_candidates(snapshot, "elem1", "click")

    assert [candidate.name for candidate in candidates] == ["dom", "shortcut", "relative"]


def test_build_locator_candidates_consumes_interaction_hints_and_structure_evidence():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.artifacts["structure_evidence_score"] = 0.81
    snapshot.regions = [
        Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1", attributes={"grid_header_ids": ["header_1"]})
    ]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=800)
    ]
    snapshot.anchors = [Anchor(anchor_id="a1", kind=AnchorKind.TEXT, element_refs=["anchor_label"])]
    snapshot.locators = [
        Locator(
            locator_id="loc_relative",
            element_ref="elem1",
            kind=LocatorKind.RELATIVE,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"bbox": [10, 10, 40, 40]},
            cost_score=0.4,
            durability_score=0.4,
            confidence=0.5,
        ),
        Locator(
            locator_id="loc_coord",
            element_ref="elem1",
            kind=LocatorKind.EPHEMERAL_COORD,
            priority=1,
            status=LocatorStatus.ACTIVE,
            selector={"x": 20, "y": 20},
            cost_score=0.2,
            durability_score=0.1,
            confidence=0.4,
        ),
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            locator_ids=["loc_relative", "loc_coord"],
            bounds=(100, 100, 200, 150),
            anchor_ids=["a1", "a2"],
            content_group_id="cg_viewport",
            attributes={
                "interaction_hints": {"preferred_action": "click", "scroll_into_view": True},
                "structure_evidence_score": 0.86,
            },
        ),
        Candidate(
            element_id="anchor_label",
            region_id="r1",
            bounds=(40, 90, 90, 120),
            text="Orders",
            name="Orders",
        ),
    ]

    candidates = service._build_locator_candidates(snapshot, "elem1", "click")
    relative = next(candidate for candidate in candidates if candidate.name == "relative")
    coord = next(candidate for candidate in candidates if candidate.name == "ephemeral_coord")

    assert relative.strategy.value == "viewport_recovery"
    assert relative.structure_evidence_score == 0.86
    assert relative.interaction_hints["preferred_action"] == "click"
    assert relative.replan_hint is False
    assert coord.preferred_rank >= relative.preferred_rank


def test_build_locator_candidates_adds_viewport_recovery_callback():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.regions = [
        Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")
    ]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=640, viewport_width=900)
    ]
    snapshot.locators = [
        Locator(
            locator_id="loc_relative",
            element_ref="elem1",
            kind=LocatorKind.RELATIVE,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"bbox": [10, 10, 40, 40]},
        )
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            locator_ids=["loc_relative"],
            bounds=(100, 100, 200, 150),
            attributes={"interaction_hints": {"scroll_into_view": True}},
        )
    ]

    candidates = service._build_locator_candidates(snapshot, "elem1", "click")

    assert candidates[0].strategy.value == "viewport_recovery"
    assert candidates[0].recovery_fn is not None


def test_runtime_recovery_fn_scrolls_and_reexecutes_after_recapture():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900)
    ]
    runtime_context = {
        "has_scroll_context": True,
        "region_role": "viewport",
        "interaction_hints": {"scroll_into_view": True},
        "structure_evidence_score": 0.8,
        "viewport_index": 0,
        "rank_bias": {},
        "risk_bias": 0,
        "cost_bias": 0.0,
        "durability_bonus": 0.0,
        "confidence_bonus": 0.0,
        "replan_hint": False,
        "anchor_refs": [],
        "anchor_texts": [],
        "nearby_texts": [],
        "region_attributes": {},
        "target_name": "",
        "target_text": "",
        "target_role": "button",
        "content_group_id": "cg_viewport",
        "target_bounds": (100, 100, 200, 150),
        "target_region_id": "r1",
    }
    refreshed = make_snapshot()
    refreshed.elements = [Candidate(element_id="elem1", region_id="r1", bounds=(100, 100, 200, 150))]

    with patch.object(service._executor, "scroll", return_value=ActionResult(True, "scroll", "ok")) as mock_scroll, patch.object(
        service,
        "_capture_runtime_snapshot",
        return_value=refreshed,
    ) as mock_capture, patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "recovered"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    mock_scroll.assert_called_once_with(12345, delta=-330)
    assert mock_capture.call_count >= 1
    mock_execute.assert_called_once_with(
        snapshot=refreshed,
        element_id="elem1",
        action="click",
        verify=False,
        locator_kind=LocatorKind.RELATIVE,
    )


def test_runtime_recovery_fn_anchor_relocates_to_semantic_match():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.regions = [Region(region_id="r1", role="action_bar")]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.SUBMIT_BUTTON,
            bounds=(100, 100, 180, 140),
            anchor_ids=["a1"],
            content_group_id="cg_action_bar",
            name="Apply",
            text="Apply",
        )
    ]
    snapshot.anchors = [Anchor(anchor_id="a1", kind=AnchorKind.TEXT, element_refs=["anchor_label"])]
    runtime_context = {
        "has_scroll_context": False,
        "region_role": "action_bar",
        "interaction_hints": {"anchor_relocate": True},
        "structure_evidence_score": 0.72,
        "viewport_index": None,
        "rank_bias": {},
        "risk_bias": 0,
        "cost_bias": 0.0,
        "durability_bonus": 0.0,
        "confidence_bonus": 0.0,
        "replan_hint": False,
        "anchor_refs": ["a1"],
        "anchor_texts": ["apply"],
        "nearby_texts": ["save"],
        "region_attributes": {},
        "target_name": "Apply",
        "target_text": "Apply",
        "target_role": "submit_button",
        "content_group_id": "cg_action_bar",
        "target_bounds": (100, 100, 180, 140),
        "target_region_id": "r1",
    }
    refreshed = make_snapshot()
    refreshed.regions = [Region(region_id="r1", role="action_bar")]
    refreshed.elements = [
        Candidate(
            element_id="elem1_new",
            region_id="r1",
            semantic_role=SemanticRole.SUBMIT_BUTTON,
            bounds=(120, 100, 200, 140),
            anchor_ids=["a1"],
            content_group_id="cg_action_bar",
            name="Apply",
            text="Apply",
        )
    ]

    with patch.object(service, "_capture_runtime_snapshot", return_value=refreshed), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "relocated"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    mock_execute.assert_called_once_with(
        snapshot=refreshed,
        element_id="elem1_new",
        action="click",
        verify=False,
        locator_kind=LocatorKind.RELATIVE,
    )
    assert "resolved=elem1_new" in result.details


def test_runtime_recovery_fn_viewport_retries_until_second_recapture_finds_target():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900)
    ]
    runtime_context = {
        "has_scroll_context": True,
        "region_role": "viewport",
        "interaction_hints": {"scroll_into_view": True},
        "structure_evidence_score": 0.8,
        "viewport_index": 0,
        "rank_bias": {},
        "risk_bias": 0,
        "cost_bias": 0.0,
        "durability_bonus": 0.0,
        "confidence_bonus": 0.0,
        "replan_hint": False,
        "anchor_refs": [],
        "anchor_texts": [],
        "nearby_texts": [],
        "relation_texts": [],
        "relation_hints": [],
        "region_attributes": {},
        "target_name": "Order 42",
        "target_text": "Order 42",
        "target_role": "button",
        "content_group_id": "cg_viewport",
        "target_bounds": (100, 100, 200, 150),
        "target_region_id": "r1",
    }
    refreshed_missing = make_snapshot()
    refreshed_missing.elements = []
    refreshed_found = make_snapshot()
    refreshed_found.elements = [
        Candidate(
            element_id="candidate_order_42",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(110, 110, 210, 160),
            content_group_id="cg_viewport",
            name="Order 42",
            text="Order 42",
        )
    ]

    with patch.object(
        service._executor,
        "scroll",
        side_effect=[
            ActionResult(True, "scroll", "step1"),
            ActionResult(True, "scroll", "step2"),
        ],
    ) as mock_scroll, patch.object(
        service,
        "_capture_runtime_snapshot",
        side_effect=[refreshed_missing, refreshed_found],
    ), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "recovered"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    assert mock_scroll.call_count == 2
    mock_execute.assert_called_once_with(
        snapshot=refreshed_found,
        element_id="candidate_order_42",
        action="click",
        verify=False,
        locator_kind=LocatorKind.RELATIVE,
    )
    assert "step=2" in result.details
    assert "resolved=candidate_order_42" in result.details


def test_runtime_recovery_fn_viewport_skips_unchanged_recapture_and_uses_next_scroll_step():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=0, total_content_height=1800)
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(100, 100, 200, 150),
            name="Order 42",
            text="Order 42",
        )
    ]
    target_element = snapshot.get_element("elem1")
    runtime_context = service._build_runtime_context(snapshot, target_element)

    refreshed_same = make_snapshot()
    refreshed_same.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    refreshed_same.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=0, total_content_height=1800)
    ]
    refreshed_same.elements = []

    refreshed_changed = make_snapshot()
    refreshed_changed.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    refreshed_changed.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=360, total_content_height=1800)
    ]
    refreshed_changed.elements = [
        Candidate(
            element_id="elem1_new",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(110, 110, 210, 160),
            name="Order 42",
            text="Order 42",
        )
    ]

    with patch.object(
        service._executor,
        "scroll",
        side_effect=[
            ActionResult(True, "scroll", "step1"),
            ActionResult(True, "scroll", "step2"),
        ],
    ) as mock_scroll, patch.object(
        service,
        "_capture_runtime_snapshot",
        side_effect=[refreshed_same, refreshed_changed],
    ), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "recovered"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    assert mock_scroll.call_count == 2
    mock_execute.assert_called_once_with(
        snapshot=refreshed_changed,
        element_id="elem1_new",
        action="click",
        verify=False,
        locator_kind=LocatorKind.RELATIVE,
    )
    assert "step=2" in result.details
    assert "scroll_changed=1" in result.details


def test_runtime_recovery_fn_viewport_falls_through_from_anchor_to_replan_in_same_step():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=0, total_content_height=1800)
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(100, 100, 200, 150),
            anchor_ids=["a1"],
            name="Open Details",
            text="Open Details",
            attributes={"interaction_hints": {"anchor_relocate": True, "expected_effect": "open_detail"}},
        )
    ]
    snapshot.anchors = [Anchor(anchor_id="a1", kind=AnchorKind.TEXT, element_refs=["label1"])]
    target_element = snapshot.get_element("elem1")
    runtime_context = service._build_runtime_context(snapshot, target_element)

    refreshed = make_snapshot()
    refreshed.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    refreshed.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=320, total_content_height=1800)
    ]
    refreshed.elements = [
        Candidate(
            element_id="candidate_details",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(120, 100, 220, 150),
            name="Open Details",
            text="Open Details",
        )
    ]

    with patch.object(
        service._executor,
        "scroll",
        return_value=ActionResult(True, "scroll", "step1"),
    ), patch.object(
        service,
        "_evaluate_viewport_recovery_snapshot",
        return_value={"meaningful_change": True, "target_element": None, "diff_details": {}},
    ), patch.object(
        service,
        "_capture_runtime_snapshot",
        return_value=refreshed,
    ), patch.object(
        service,
        "_find_anchor_relocated_element",
        return_value=None,
    ) as mock_anchor, patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "recovered"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    mock_anchor.assert_called()
    mock_execute.assert_called_once_with(
        snapshot=refreshed,
        element_id="candidate_details",
        action="click",
        verify=False,
        locator_kind=LocatorKind.RELATIVE,
    )
    assert "next_step=anchor_relocate" in result.details
    assert "resolved_step=replan" in result.details


def test_runtime_recovery_fn_viewport_retries_replanned_target_when_anchor_target_execution_fails():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=0, total_content_height=1800)
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(100, 100, 200, 150),
            anchor_ids=["a1"],
            name="Open Details",
            text="Open Details",
            attributes={"interaction_hints": {"anchor_relocate": True, "expected_effect": "open_detail"}},
        )
    ]
    snapshot.anchors = [Anchor(anchor_id="a1", kind=AnchorKind.TEXT, element_refs=["label1"])]
    target_element = snapshot.get_element("elem1")
    runtime_context = service._build_runtime_context(snapshot, target_element)

    refreshed = make_snapshot()
    refreshed.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    refreshed.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=600, viewport_width=900, scroll_offset=320, total_content_height=1800)
    ]
    anchor_target = Candidate(
        element_id="anchor_target",
        region_id="r1",
        semantic_role=SemanticRole.BUTTON,
        bounds=(120, 100, 220, 150),
        name="Open Details",
        text="Open Details",
    )
    replanned_target = Candidate(
        element_id="replanned_target",
        region_id="r1",
        semantic_role=SemanticRole.BUTTON,
        bounds=(130, 105, 230, 155),
        name="Open Details",
        text="Open Details",
    )
    refreshed.elements = [anchor_target, replanned_target]

    with patch.object(
        service._executor,
        "scroll",
        return_value=ActionResult(True, "scroll", "step1"),
    ), patch.object(
        service,
        "_evaluate_viewport_recovery_snapshot",
        return_value={"meaningful_change": True, "target_element": None, "diff_details": {}},
    ), patch.object(
        service,
        "_capture_runtime_snapshot",
        return_value=refreshed,
    ), patch.object(
        service,
        "_find_anchor_relocated_element",
        return_value=anchor_target,
    ), patch.object(
        service,
        "_find_replanned_element",
        return_value=replanned_target,
    ), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        side_effect=[
            ActionResult(False, "click", "anchor miss", "anchor_miss"),
            ActionResult(True, "click", "replanned ok"),
        ],
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    assert mock_execute.call_count == 2
    assert mock_execute.call_args_list[0].kwargs["element_id"] == "anchor_target"
    assert mock_execute.call_args_list[1].kwargs["element_id"] == "replanned_target"
    assert "resolved=replanned_target" in result.details
    assert "resolved_step=replan" in result.details


def test_decide_viewport_recovery_next_step_prefers_anchor_then_replan():
    service = ActionService()

    next_step = service._decide_viewport_recovery_next_step(
        recovery_check={"meaningful_change": True, "target_element": None, "diff_details": {}},
        runtime_context={
            "anchor_refs": ["a1"],
            "interaction_hints": {},
            "replan_hint": True,
            "structure_evidence_score": 0.9,
        },
    )
    assert next_step == "anchor_relocate"

    next_step = service._decide_viewport_recovery_next_step(
        recovery_check={"meaningful_change": True, "target_element": None, "diff_details": {}},
        runtime_context={
            "anchor_refs": [],
            "interaction_hints": {"expected_effect": "navigate"},
            "replan_hint": False,
            "structure_evidence_score": 0.8,
        },
    )
    assert next_step == "replan"

    next_step = service._decide_viewport_recovery_next_step(
        recovery_check={"meaningful_change": False, "target_element": None, "diff_details": {}},
        runtime_context={
            "anchor_refs": [],
            "interaction_hints": {},
            "replan_hint": False,
            "structure_evidence_score": 0.2,
        },
    )
    assert next_step == "continue_scroll"


def test_runtime_recovery_fn_replans_to_semantic_candidate_when_original_missing():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.regions = [Region(region_id="r1", role="dialog_body", attributes={"dialog_kind": "form_dialog"})]
    runtime_context = {
        "has_scroll_context": False,
        "region_role": "dialog_body",
        "interaction_hints": {"expected_effect": "open_dialog"},
        "structure_evidence_score": 0.88,
        "viewport_index": None,
        "rank_bias": {},
        "risk_bias": 0,
        "cost_bias": 0.0,
        "durability_bonus": 0.0,
        "confidence_bonus": 0.0,
        "replan_hint": True,
        "anchor_refs": [],
        "anchor_texts": [],
        "nearby_texts": ["details"],
        "region_attributes": {"dialog_kind": "form_dialog"},
        "target_name": "Open Details",
        "target_text": "Open Details",
        "target_role": "button",
        "content_group_id": "cg_viewport",
        "target_bounds": (100, 100, 180, 140),
        "target_region_id": "r1",
    }
    refreshed = make_snapshot()
    refreshed.regions = [Region(region_id="r2", role="dialog_body", attributes={"dialog_kind": "form_dialog"})]
    refreshed.elements = [
        Candidate(
            element_id="candidate_open_details",
            region_id="r2",
            semantic_role=SemanticRole.BUTTON,
            bounds=(120, 100, 220, 150),
            content_group_id="cg_viewport",
            name="Open Details",
            text="Open Details",
        )
    ]

    with patch.object(service, "_capture_runtime_snapshot", return_value=refreshed), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "replanned"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="missing_old_id",
            action="click",
            locator_kind=LocatorKind.DOM,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    mock_execute.assert_called_once_with(
        snapshot=refreshed,
        element_id="candidate_open_details",
        action="click",
        verify=False,
        locator_kind=LocatorKind.DOM,
    )
    assert "resolved=candidate_open_details" in result.details


def test_build_runtime_context_collects_relation_hints():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.regions = [Region(region_id="r1", role="action_bar")]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(100, 100, 180, 140),
        ),
        Candidate(
            element_id="label1",
            region_id="r1",
            bounds=(40, 100, 90, 140),
            text="Billing",
            name="Billing",
        ),
    ]
    snapshot.relations = [
        ElementRelation(
            relation_id="rel1",
            from_id="elem1",
            to_id="label1",
            type=RelationType.LEFT_OF,
            offset=(-60, 0),
        )
    ]

    runtime_context = service._build_runtime_context(snapshot, snapshot.get_element("elem1"))

    assert runtime_context["relation_texts"] == ["billing"]
    assert runtime_context["relation_hints"][0]["type"] == "left_of"
    assert runtime_context["relation_hints"][0]["offset"] == (-60, 0)


def test_runtime_recovery_fn_anchor_relocates_using_relation_texts():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.regions = [Region(region_id="r1", role="action_bar")]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(100, 100, 180, 140),
            name="",
            text="",
        ),
        Candidate(
            element_id="label1",
            region_id="r1",
            bounds=(40, 100, 90, 140),
            name="Billing",
            text="Billing",
        ),
    ]
    snapshot.relations = [
        ElementRelation(
            relation_id="rel1",
            from_id="elem1",
            to_id="label1",
            type=RelationType.LEFT_OF,
        )
    ]
    target_element = snapshot.get_element("elem1")
    runtime_context = service._build_runtime_context(snapshot, target_element)
    refreshed = make_snapshot()
    refreshed.regions = [Region(region_id="r1", role="action_bar")]
    refreshed.elements = [
        Candidate(
            element_id="candidate_plain",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(260, 100, 340, 140),
            name="",
            text="",
        ),
        Candidate(
            element_id="candidate_billing",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            bounds=(130, 102, 210, 142),
            name="Billing",
            text="Billing",
        ),
    ]

    with patch.object(service, "_capture_runtime_snapshot", return_value=refreshed), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(True, "click", "relocated"),
    ) as mock_execute:
        recovery_fn = service._make_runtime_recovery_fn(
            snapshot=snapshot,
            element_id="elem1",
            action="click",
            locator_kind=LocatorKind.RELATIVE,
            runtime_context=runtime_context,
        )
        result = recovery_fn()

    assert result.success is True
    mock_execute.assert_called_once_with(
        snapshot=refreshed,
        element_id="candidate_billing",
        action="click",
        verify=False,
        locator_kind=LocatorKind.RELATIVE,
    )
    assert "resolved=candidate_billing" in result.details


def test_build_locator_candidates_uses_risk_and_cost_for_same_surface_rank():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.UNKNOWN, confidence=0.9)
    snapshot.locators = [
        Locator(
            locator_id="loc_template",
            element_ref="elem1",
            kind=LocatorKind.TEMPLATE_ICON,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"bbox": [10, 10, 40, 40]},
            cost_score=0.6,
            durability_score=0.2,
            confidence=0.5,
        ),
        Locator(
            locator_id="loc_vision",
            element_ref="elem1",
            kind=LocatorKind.VISION_BBOX,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"bbox": [10, 10, 40, 40]},
            cost_score=0.4,
            durability_score=0.6,
            confidence=0.7,
        ),
    ]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            locator_ids=["loc_template", "loc_vision"],
            bounds=(100, 100, 200, 150),
        )
    ]

    candidates = service._build_locator_candidates(snapshot, "elem1", "click")

    assert [candidate.name for candidate in candidates] == ["vision_bbox", "template_icon"]
    assert candidates[0].risk_tier <= candidates[1].risk_tier


def test_execute_snapshot_action_recaptures_and_records_diff_verification():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc_uia"],
            bounds=(10, 10, 40, 40),
        )
    ]
    locator = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    resolved = {
        "locator": locator,
        "method": "uia",
        "geometry": None,
        "coordinates": None,
    }
    after_snapshot = make_snapshot()
    after_snapshot.canvas_id = "snap_after"
    diff_result = MagicMock()
    diff_result.verified = True
    diff_result.message = "semantic_effect_detected"
    diff_result.details = {"target_changed": True, "policy": "require_page_or_structure_change", "target_element_id": "elem1_new"}

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(success=True, action="click", details="uia ok"),
    ), patch.object(
        service,
        "_capture_runtime_snapshot",
        return_value=after_snapshot,
    ) as mock_capture, patch.object(
        service._verifier,
        "verify_action_effect",
        return_value=diff_result,
    ) as mock_compare:
        outcome = service.execute_snapshot_action(
            snapshot,
            "elem1",
            "click",
            with_recovery=False,
        )

    assert outcome.success is True
    assert outcome.verification_details["status"] == "semantic_effect_detected"
    assert outcome.verification_details["after_canvas_id"] == "snap_after"
    assert outcome.verification_details["recapture_policy"] == "auto_semantic"
    assert outcome.verification_details["verification_element_id"] == "elem1"
    mock_capture.assert_called_once_with(snapshot)
    mock_compare.assert_called_once_with(
        before_snapshot=snapshot,
        after_snapshot=after_snapshot,
        target_element_id="elem1",
        action="click",
    )


def test_execute_snapshot_action_verifies_against_recovery_resolved_target():
    service = ActionService()
    snapshot = make_snapshot()
    locator_uia = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    locator_vision = Locator(
        locator_id="loc_vis",
        element_ref="elem1",
        kind=LocatorKind.VISION_BBOX,
        priority=1,
        status=LocatorStatus.ACTIVE,
        selector={"bbox": [100, 100, 200, 150]},
    )
    snapshot.locators = [locator_uia, locator_vision]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc_uia", "loc_vis"],
            bounds=(100, 100, 200, 150),
        )
    ]
    resolved = {"locator": locator_uia, "method": "uia", "geometry": None, "coordinates": None}
    recovery_result = MagicMock()
    recovery_result.recovered = True
    recovery_result.attempts = 2
    recovery_result.strategy_used.value = "replan"
    recovery_result.final_result = ActionResult(
        success=True,
        action="click",
        details="replanned [recovery mode=replan original=elem1 resolved=elem1_new]",
    )
    recovery_result.error = None
    recovery_result.details = {"attempt_trace": [{"candidate": "vision_bbox", "stage": "recovery"}]}
    after_snapshot = make_snapshot()
    diff_result = MagicMock()
    diff_result.verified = True
    diff_result.message = "semantic_effect_detected"
    diff_result.details = {"target_changed": True, "policy": "strict_snapshot_change"}

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._recovery,
        "execute_locator_fallback_chain",
        return_value=recovery_result,
    ), patch.object(
        service,
        "_capture_runtime_snapshot",
        return_value=after_snapshot,
    ) as mock_capture, patch.object(
        service._verifier,
        "verify_action_effect",
        return_value=diff_result,
        ) as mock_verify:
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click", recapture_after=True)

    assert outcome.success is True
    assert outcome.verification_details["verification_element_id"] == "elem1_new"
    assert outcome.verification_details["recovery_execution"]["resolved"] == "elem1_new"
    assert outcome.verification_details["runtime_trace"]["resolved_element_id"] == "elem1_new"
    assert outcome.verification_details["runtime_trace"]["verification_element_id"] == "elem1_new"
    assert outcome.verification_details["runtime_trace"]["verification_status"] == "semantic_effect_detected"
    assert outcome.verification_details["runtime_trace"]["verification_policy"] == "strict_snapshot_change"
    mock_capture.assert_called_once_with(snapshot)
    mock_verify.assert_called_once_with(
        before_snapshot=snapshot,
        after_snapshot=after_snapshot,
        target_element_id="elem1_new",
        action="click",
    )


def test_execute_snapshot_action_surfaces_structured_next_suggestion_on_recovery_failure():
    service = ActionService()
    snapshot = make_snapshot()
    locator_uia = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    locator_vision = Locator(
        locator_id="loc_vis",
        element_ref="elem1",
        kind=LocatorKind.VISION_BBOX,
        priority=1,
        status=LocatorStatus.ACTIVE,
        selector={"bbox": [100, 100, 200, 150]},
    )
    snapshot.locators = [locator_uia, locator_vision]
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc_uia", "loc_vis"],
            bounds=(100, 100, 200, 150),
        )
    ]
    resolved = {"locator": locator_uia, "method": "uia", "geometry": None, "coordinates": None}
    recovery_result = MagicMock()
    recovery_result.recovered = False
    recovery_result.attempts = 3
    recovery_result.strategy_used.value = "viewport_recovery"
    recovery_result.final_result = ActionResult(success=False, action="click", details="miss", error="miss")
    recovery_result.error = "locator_chain_failed_viewport_recovery_required"
    recovery_result.details = {
        "reason": "viewport_context_requires_recovery",
        "terminal_strategy": "viewport_recovery",
        "escalation_path": ["vision_bbox:viewport_recovery"],
        "attempt_trace": [{"candidate": "vision_bbox", "stage": "verify"}],
        "next_suggestion": {
            "strategy": "viewport_recovery",
            "action": "scroll_and_recapture",
            "error": "locator_chain_failed_viewport_recovery_required",
            "reason": "viewport_context_requires_recovery",
        },
    }

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._recovery,
        "execute_locator_fallback_chain",
        return_value=recovery_result,
    ):
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click", verify=False)

    assert outcome.success is False
    assert outcome.verification_details["runtime_trace"]["next_suggestion"]["action"] == "scroll_and_recapture"
    assert outcome.verification_details["runtime_trace"]["terminal_error"] == "locator_chain_failed_viewport_recovery_required"


def test_action_outcome_to_dict_keeps_runtime_trace_payload():
    outcome = ActionOutcome(
        success=False,
        action="click",
        message="failed",
        attempts=3,
        strategy="locator_first/chain/viewport_recovery",
        error="locator_chain_failed_viewport_recovery_required",
        verification_details={
            "status": "semantic_effect_missing",
            "runtime_trace": {
                "strategy": "locator_first/chain/viewport_recovery",
                "last_attempt_stage": "verify_after_recovery",
                "last_attempt_candidate": "relative",
                "final_message": "failed",
            },
        },
    )

    data = outcome.to_dict()

    assert data["attempts"] == 3
    assert data["verification_details"]["runtime_trace"]["last_attempt_stage"] == "verify_after_recovery"
    assert data["verification_details"]["runtime_trace"]["final_message"] == "failed"
    assert data["execution_record"]["runtime_trace"]["last_attempt_stage"] == "verify_after_recovery"
    assert data["execution_record"]["terminal_strategy"] is None


def test_execute_decision_action_routes_elem_candidate_to_snapshot_execution():
    service = ActionService()
    snapshot = make_snapshot()

    with patch.object(
        service,
        "execute_snapshot_action",
        return_value=ActionOutcome(
            success=True,
            action="click",
            message="ok",
            strategy="locator_first/uia/uia",
        ),
    ) as mock_execute:
        outcome = service.execute_decision_action(
            snapshot=snapshot,
            decision={
                "page_state": "search_ready",
                "selected_candidate_id": "elem::elem_search",
                "selected_role": "input_candidate",
                "next_action": {"type": "click"},
            },
        )

    mock_execute.assert_called_once_with(
        snapshot=snapshot,
        element_id="elem_search",
        action="click",
        verify=True,
        with_recovery=True,
        recapture_after=None,
    )
    assert outcome.success is True
    assert outcome.verification_details["decision_record"]["selected_candidate_id"] == "elem::elem_search"


def test_execute_decision_action_click_and_type_uses_bbox_candidate():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.regions = [Region(region_id="r1", role="filter_bar", bounds=(90, 40, 240, 80))]
    snapshot.elements = []
    snapshot.artifacts["ocr_blocks"] = [
        {"bbox": [102, 44, 137, 65], "text": "搜索", "confidence": 0.9},
    ]

    with patch.object(
        service,
        "action_click_element",
        return_value=ActionOutcome(success=True, action="click", message="clicked", strategy="direct"),
    ) as mock_click, patch.object(
        service,
        "action_send_text",
        return_value=ActionOutcome(success=True, action="send_text", message="typed", strategy="direct"),
    ) as mock_type:
        outcome = service.execute_decision_action(
            snapshot=snapshot,
            decision={
                "page_state": "search_ready",
                "selected_candidate_id": "ocr::0",
                "selected_role": "input_candidate",
                "next_action": {"type": "click_and_type", "text": "孙宇"},
            },
        )

    mock_click.assert_called_once_with(
        hwnd=12345,
        element_x=102,
        element_y=44,
        element_width=35,
        element_height=21,
        verify=False,
        with_recovery=True,
    )
    mock_type.assert_called_once_with(12345, "孙宇")
    assert outcome.success is True
    assert outcome.action == "click_and_type"
    assert outcome.verification_details["decision_candidate"]["candidate_id"] == "ocr::0"
    assert outcome.verification_details["typed_text"] == "孙宇"


def test_execute_decision_action_returns_recrop_request_for_focus_bbox():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.canvas_id = "snap_focus"

    outcome = service.execute_decision_action(
        snapshot=snapshot,
        decision={
            "page_state": "search_results_dense",
            "decision_status": "need_zoom_in",
            "reason": "候选过密",
            "focus_bbox": [60, 70, 320, 420],
            "next_action": {"type": "recrop_and_reanalyze", "text": ""},
        },
        crop_scale=1.8,
    )

    assert outcome.success is False
    assert outcome.error == "recrop_required"
    assert outcome.verification_details["recrop_request"]["source_canvas_id"] == "snap_focus"
    assert outcome.verification_details["recrop_request"]["focus_bbox"] == [60, 70, 320, 420]
    assert outcome.verification_details["recrop_request"]["crop_scale"] == 1.8


def test_build_focus_openclaw_payload_recaptures_and_builds_second_round_candidates():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.canvas_id = "snap_runtime"
    snapshot.app.process_name = "WeChat.exe"
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.ELECTRON_WEBVIEW, confidence=0.9)
    snapshot.page.page_class = "wechat/app_chat/main/wide"
    snapshot.window.rect_client = (0, 0, 400, 300)

    ocr_block = MagicMock()
    ocr_block.text = "孙宇"
    ocr_block.bbox = (15, 20, 75, 42)
    ocr_block.confidence = 0.91
    ocr_result = MagicMock()
    ocr_result.provider = "paddleocr_bridge"
    ocr_result.success = True
    ocr_result.error = None
    ocr_result.blocks = [ocr_block]

    vision_candidate = MagicMock()
    vision_candidate.element_id = "vision::0"
    vision_candidate.bounding_box = (90, 18, 125, 48)
    vision_candidate.semantic_label = "button_candidate"
    vision_candidate.text = "发送"
    vision_candidate.confidence = 0.8
    vision_result = MagicMock()
    vision_result.provider = "omniparser"
    vision_result.success = True
    vision_result.error = None
    vision_result.candidates = [vision_candidate]
    service._perception_service._vision_provider = MagicMock()
    service._perception_service._vision_provider.parse_screenshot.return_value = vision_result

    with patch("src.windows.screenshot_service.ScreenshotService") as mock_screenshot_cls, patch(
        "src.perception.ocr_service.get_ocr_service",
        return_value=MagicMock(extract_with_metadata=MagicMock(return_value=ocr_result)),
    ):
        mock_screenshot = MagicMock()
        mock_screenshot.capture.return_value = Image.new("RGB", (400, 300), color=(240, 240, 240))
        mock_screenshot_cls.return_value = mock_screenshot

        result = service.build_focus_openclaw_payload(
            snapshot=snapshot,
            decision={
                "page_state": "search_results_dense",
                "focus_bbox": [60, 70, 180, 170],
            },
            task="在微信里搜索孙宇",
            crop_scale=1.2,
        )

    assert result["page_state"] == "search_results_dense"
    assert result["ocr_provider"]["block_count"] == 1
    assert result["vision_provider"]["candidate_count"] == 1
    payload = result["focused_openclaw_payload"]
    assert payload["task"] == "在微信里搜索孙宇"
    assert payload["window_bounds"] == [0, 0, 400, 300]
    assert any(item["candidate_id"] == "focus::ocr::0" for item in payload["candidates"])
    assert any(item["candidate_id"] == "focus::vision::0" for item in payload["candidates"])
    assert any(item["bbox"] == [63, 80, 123, 102] for item in payload["candidates"])
    assert any(item["bbox"] == [138, 78, 173, 108] for item in payload["candidates"])
    assert payload["ocr_blocks"][0]["bbox"] == [63, 80, 123, 102]
    assert any(str(note).startswith("focus_source_bbox=") for note in payload["notes"])


def test_execute_openclaw_decision_loop_zooms_then_executes():
    service = ActionService()
    snapshot = make_snapshot()

    decisions = iter(
        [
            {
                "page_state": "dense_results",
                "decision_status": "need_zoom_in",
                "focus_bbox": [60, 70, 180, 170],
                "next_action": {"type": "recrop_and_reanalyze", "text": ""},
            },
            {
                "page_state": "focused_results",
                "selected_candidate_id": "focus::ocr::0",
                "selected_role": "input_candidate",
                "next_action": {"type": "click_and_type", "text": "孙宇"},
            },
        ]
    )

    with patch(
        "src.perception.debug_tools.build_openclaw_payload",
        return_value={"page_hint": "wechat/app_chat/main/wide", "candidates": [{"candidate_id": "elem::root"}]},
    ), patch.object(
        service,
        "build_focus_openclaw_payload",
        return_value={
            "crop_rect": [50, 60, 220, 200],
            "crop_size": [170, 140],
            "ocr_provider": {"block_count": 3},
            "vision_provider": {"candidate_count": 2},
            "focused_openclaw_payload": {"page_hint": "focus_region", "candidates": [{"candidate_id": "focus::ocr::0"}]},
        },
    ) as mock_focus, patch.object(
        service,
        "execute_decision_action",
        return_value=ActionOutcome(success=True, action="click_and_type", message="typed", strategy="decision_record/bbox"),
    ) as mock_execute:
        outcome = service.execute_openclaw_decision_loop(
            snapshot=snapshot,
            task="在微信里搜索孙宇",
            decision_provider=lambda payload: next(decisions),
        )

    assert outcome.success is True
    assert len(outcome.verification_details["decision_loop"]) == 2
    assert outcome.verification_details["decision_loop"][0]["decision_status"] == "need_zoom_in"
    assert outcome.verification_details["decision_loop"][1]["selected_candidate_id"] == "focus::ocr::0"
    mock_focus.assert_called_once()
    mock_execute.assert_called_once()


def test_execute_decision_action_uses_bbox_fallback_when_candidate_not_in_snapshot():
    service = ActionService()
    snapshot = make_snapshot()

    with patch.object(
        service,
        "action_click_element",
        return_value=ActionOutcome(success=True, action="click", message="clicked"),
    ) as mock_click, patch.object(
        service,
        "action_send_text",
        return_value=ActionOutcome(success=True, action="send_text", message="typed"),
    ) as mock_type:
        outcome = service.execute_decision_action(
            snapshot=snapshot,
            decision={
                "page_state": "focused_results",
                "selected_candidate_id": "focus::ocr::0",
                "selected_role": "input_candidate",
                "bbox": [120, 130, 220, 170],
                "next_action": {"type": "click_and_type", "text": "孙宇"},
                "confidence": 0.76,
            },
            verify=False,
            with_recovery=False,
        )

    assert outcome.success is True
    assert outcome.action == "click_and_type"
    assert outcome.verification_details["decision_candidate"]["bbox"] == [120, 130, 220, 170]
    assert outcome.verification_details["decision_candidate"]["attributes"]["candidate_origin"] == "decision_bbox_fallback"
    mock_click.assert_called_once()
    mock_type.assert_called_once()


def test_execute_openclaw_decision_loop_stops_after_max_rounds():
    service = ActionService()
    snapshot = make_snapshot()

    with patch(
        "src.perception.debug_tools.build_openclaw_payload",
        return_value={"page_hint": "wechat/app_chat/main/wide", "candidates": [{"candidate_id": "elem::root"}]},
    ), patch.object(
        service,
        "build_focus_openclaw_payload",
        return_value={
            "crop_rect": [50, 60, 220, 200],
            "crop_size": [170, 140],
            "ocr_provider": {"block_count": 1},
            "vision_provider": {"candidate_count": 1},
            "focused_openclaw_payload": {"page_hint": "focus_region", "candidates": [{"candidate_id": "focus::ocr::0"}]},
        },
    ):
        outcome = service.execute_openclaw_decision_loop(
            snapshot=snapshot,
            task="在微信里搜索孙宇",
            decision_provider=lambda payload: {
                "page_state": "dense_results",
                "decision_status": "need_zoom_in",
                "focus_bbox": [60, 70, 180, 170],
                "next_action": {"type": "recrop_and_reanalyze", "text": ""},
            },
            max_rounds=2,
        )

    assert outcome.success is False
    assert outcome.error == "decision_rounds_exhausted"
    assert len(outcome.verification_details["decision_loop"]) == 2


def test_execute_snapshot_action_recapture_marks_unchanged_snapshot_as_failure():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc_uia"],
            bounds=(10, 10, 40, 40),
        )
    ]
    locator = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    resolved = {
        "locator": locator,
        "method": "uia",
        "geometry": None,
        "coordinates": None,
    }
    diff_result = MagicMock()
    diff_result.verified = False
    diff_result.message = "semantic_effect_missing"
    diff_result.details = {"target_changed": False, "policy": "require_page_or_structure_change"}

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(success=True, action="click", details="uia ok"),
    ), patch.object(
        service,
        "_capture_runtime_snapshot",
        return_value=make_snapshot(),
    ), patch.object(
        service._verifier,
        "verify_action_effect",
        return_value=diff_result,
    ):
        outcome = service.execute_snapshot_action(
            snapshot,
            "elem1",
            "click",
            with_recovery=False,
        )

    assert outcome.success is False
    assert outcome.error == "snapshot_unchanged"
    assert outcome.verification_details["status"] == "semantic_effect_missing"


def test_execute_snapshot_action_does_not_auto_recapture_for_generic_button():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc_uia"],
            bounds=(10, 10, 40, 40),
        )
    ]
    locator = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "btn_ok"},
    )
    resolved = {
        "locator": locator,
        "method": "uia",
        "geometry": None,
        "coordinates": None,
    }

    with patch.object(service._snapshot_adapter, "resolve_element", return_value=resolved), patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(success=True, action="click", details="uia ok"),
    ), patch.object(
        service,
        "_capture_runtime_snapshot",
    ) as mock_capture:
        outcome = service.execute_snapshot_action(
            snapshot,
            "elem1",
            "click",
            with_recovery=False,
        )

    assert outcome.success is True
    assert outcome.verification_details is None
    mock_capture.assert_not_called()


def test_execute_snapshot_action_blocks_high_risk_browser_send_button():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc_dom"],
            bounds=(10, 10, 40, 40),
        )
    ]
    snapshot.locators = [
        Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"name": "sendButton"},
        )
    ]

    with patch.object(service._snapshot_adapter, "resolve_element") as mock_resolve:
        mock_resolve.return_value = {
            "locator": snapshot.locators[0],
            "method": "dom",
            "geometry": None,
            "coordinates": None,
        }
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click")

    assert outcome.success is False
    assert outcome.error == "risk_policy_blocked"
    assert outcome.strategy == "risk_policy/guarded_automation"
    assert outcome.verification_details["risk_level"] == "high"


def test_execute_snapshot_action_allows_browser_send_button_when_risk_approved():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc_dom"],
            bounds=(10, 10, 40, 40),
            attributes={"risk_approved": True},
        )
    ]
    snapshot.locators = [
        Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"name": "sendButton"},
        )
    ]

    with patch.object(service._snapshot_adapter, "resolve_element") as mock_resolve, patch.object(
        service._snapshot_adapter,
        "execute_element_action",
        return_value=ActionResult(success=True, action="click", details="dom ok"),
    ) as mock_execute:
        mock_resolve.return_value = {
            "locator": snapshot.locators[0],
            "method": "dom",
            "geometry": None,
            "coordinates": None,
        }
        outcome = service.execute_snapshot_action(
            snapshot,
            "elem1",
            "click",
            with_recovery=False,
            verify=False,
        )

    assert outcome.success is True
    mock_execute.assert_called_once()


def test_execute_snapshot_action_respects_workflow_policy_api_first():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.surface = SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.9)
    snapshot.artifacts["workflow_policy"] = {"mode": "api_first"}
    snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.LINK,
            locator_ids=["loc_dom"],
            bounds=(10, 10, 40, 40),
        )
    ]
    snapshot.locators = [
        Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=0,
            status=LocatorStatus.ACTIVE,
            selector={"name": "detailsLink"},
        )
    ]

    with patch.object(service._snapshot_adapter, "resolve_element") as mock_resolve:
        mock_resolve.return_value = {
            "locator": snapshot.locators[0],
            "method": "dom",
            "geometry": None,
            "coordinates": None,
        }
        outcome = service.execute_snapshot_action(snapshot, "elem1", "click")

    assert outcome.success is False
    assert outcome.error == "risk_policy_blocked"
    assert outcome.verification_details["policy_source"] == "snapshot.workflow_policy"
    assert outcome.verification_details["reason"] == "browser_api_first"


def test_execution_record_includes_decision_candidate_details():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.regions = [Region(region_id="r1", role="filter_bar", bounds=(90, 40, 240, 80))]
    snapshot.elements = []
    snapshot.artifacts["ocr_blocks"] = [
        {"bbox": [102, 44, 137, 65], "text": "搜索", "confidence": 0.9},
    ]

    with patch.object(
        service,
        "action_click_element",
        return_value=ActionOutcome(success=True, action="click", message="clicked", strategy="direct"),
    ), patch.object(
        service,
        "action_send_text",
        return_value=ActionOutcome(success=True, action="send_text", message="typed", strategy="direct"),
    ):
        outcome = service.execute_decision_action(
            snapshot=snapshot,
            decision={
                "page_state": "search_ready",
                "selected_candidate_id": "ocr::0",
                "selected_role": "input_candidate",
                "next_action": {"type": "click_and_type", "text": "孙宇"},
            },
        )

    execution_record = outcome.to_dict()["execution_record"]
    assert execution_record["decision_candidate_source"] == "ocr"
    assert execution_record["decision_candidate_kind"] == "input_candidate"
    assert execution_record["decision_candidate_bbox"] == [102, 44, 137, 65]
    assert execution_record["used_bbox_fallback"] is False


def test_execution_record_marks_decision_bbox_fallback():
    service = ActionService()
    snapshot = make_snapshot()
    snapshot.elements = []
    snapshot.artifacts["ocr_blocks"] = []

    with patch.object(
        service,
        "action_click_element",
        return_value=ActionOutcome(success=True, action="click", message="clicked", strategy="direct"),
    ):
        outcome = service.execute_decision_action(
            snapshot=snapshot,
            decision={
                "page_state": "search_ready",
                "selected_candidate_id": "missing_candidate",
                "selected_role": "input_candidate",
                "bbox": [10, 20, 110, 60],
                "next_action": {"type": "click"},
                "confidence": 0.61,
            },
        )

    execution_record = outcome.to_dict()["execution_record"]
    assert execution_record["decision_candidate_source"] == "decision_record"
    assert execution_record["decision_candidate_origin"] == "decision_bbox_fallback"
    assert execution_record["used_bbox_fallback"] is True
