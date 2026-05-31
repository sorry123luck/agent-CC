from unittest.mock import MagicMock, patch

from src.execution.action_verifier import ActionVerifier
from src.perception.page_compiler_models import (
    Anchor,
    AnchorKind,
    ContentAreaSubtype,
    ElementState,
    Locator,
    LocatorKind,
    LocatorStatus,
    Candidate,
    InteractionCanvas,
    Region,
    ScrollContext,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def make_snapshot() -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="snap_verify",
        window=WindowInfoSnapshot(hwnd=12345, title="Test"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
    )


def test_verify_snapshot_target_returns_present_for_geometry_locator():
    verifier = ActionVerifier()
    snapshot = make_snapshot()
    locator = Locator(
        locator_id="loc_vis",
        element_ref="elem1",
        kind=LocatorKind.VISION_BBOX,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"bbox": [10, 10, 40, 40]},
    )
    element = Candidate(
        element_id="elem1",
        locator_ids=["loc_vis"],
        bounds=(10, 10, 40, 40),
    )
    snapshot.locators = [locator]
    snapshot.elements = [element]

    result = verifier.verify_snapshot_target(snapshot, "elem1", LocatorKind.VISION_BBOX)

    assert result.verified is True
    assert result.method.value == "snapshot_locator"
    assert result.details["anchor_count"] == 0


def test_verify_snapshot_target_delegates_uia_locator_to_element_present():
    verifier = ActionVerifier()
    snapshot = make_snapshot()
    locator = Locator(
        locator_id="loc_uia",
        element_ref="elem1",
        kind=LocatorKind.UIA,
        priority=0,
        status=LocatorStatus.ACTIVE,
        selector={"automation_id": "send_btn"},
    )
    element = Candidate(
        element_id="elem1",
        locator_ids=["loc_uia"],
        bounds=(10, 10, 40, 40),
    )
    snapshot.locators = [locator]
    snapshot.elements = [element]

    with patch.object(verifier, "verify_element_present") as mock_verify_present:
        mock_verify_present.return_value = MagicMock(verified=True, method=None, message="ok", details={})
        result = verifier.verify_snapshot_target(snapshot, "elem1", LocatorKind.UIA)

    assert result.verified is True
    mock_verify_present.assert_called_once_with(hwnd=12345, automation_id="send_btn")


def test_verify_snapshot_target_reports_missing_locator():
    verifier = ActionVerifier()
    snapshot = make_snapshot()
    snapshot.elements = [Candidate(element_id="elem1", locator_ids=[])]

    result = verifier.verify_snapshot_target(snapshot, "elem1", LocatorKind.DOM)

    assert result.verified is False
    assert result.message == "snapshot_locator_missing"


def test_compare_snapshots_detects_target_change():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.elements = [
        Candidate(element_id="elem1", locator_ids=["loc1"], bounds=(10, 10, 40, 40))
    ]
    after_snapshot.elements = [
        Candidate(element_id="elem1", locator_ids=["loc1"], bounds=(20, 20, 60, 60))
    ]

    result = verifier.compare_snapshots(before_snapshot, after_snapshot, target_element_id="elem1")

    assert result.verified is True
    assert result.method.value == "snapshot_diff"
    assert result.details["target_changed"] is True


def test_compare_snapshots_detects_target_state_change():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.elements = [
        Candidate(
            element_id="elem1",
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            state=ElementState(selected=False),
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="elem1",
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            state=ElementState(selected=True),
        )
    ]

    result = verifier.compare_snapshots(before_snapshot, after_snapshot, target_element_id="elem1")

    assert result.verified is True
    assert result.details["target_state_changed"] is True
    assert result.details["target_changed"] is True


def test_compare_snapshots_tracks_anchor_scroll_and_content_group_changes():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    after_snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    before_snapshot.scroll_contexts = [ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=400, viewport_width=300)]
    after_snapshot.scroll_contexts = []
    before_snapshot.anchors = [Anchor(anchor_id="a1", kind=AnchorKind.SINGLE, element_refs=["elem1"])]
    after_snapshot.anchors = []
    before_snapshot.elements = [
        Candidate(element_id="elem1", locator_ids=["loc1"], bounds=(10, 10, 40, 40), region_id="r1", content_group_id="cg_viewport", anchor_ids=["a1"])
    ]
    after_snapshot.elements = [
        Candidate(element_id="elem1", locator_ids=["loc1"], bounds=(10, 10, 40, 40), region_id="r1", content_group_id="cg_action_bar", anchor_ids=[])
    ]

    result = verifier.compare_snapshots(before_snapshot, after_snapshot, target_element_id="elem1")

    assert result.verified is True
    assert result.details["before_anchor_count"] == 1
    assert result.details["after_scroll_context_count"] == 0
    assert result.details["target_content_changed"] is True


def test_compare_snapshots_detects_scroll_offset_change_without_count_change():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.scroll_contexts = [
        ScrollContext(
            scroll_context_id="scroll_r1",
            region_id="r1",
            viewport_height=400,
            viewport_width=300,
            scroll_offset=0,
            total_content_height=1600,
        )
    ]
    after_snapshot.scroll_contexts = [
        ScrollContext(
            scroll_context_id="scroll_r1",
            region_id="r1",
            viewport_height=400,
            viewport_width=300,
            scroll_offset=420,
            total_content_height=1600,
        )
    ]

    result = verifier.compare_snapshots(before_snapshot, after_snapshot)

    assert result.verified is True
    assert result.details["structure_changed"] is True
    assert result.details["before_scroll_evidence"]["scroll_offsets"] == [0]
    assert result.details["after_scroll_evidence"]["scroll_offsets"] == [420]


def test_compare_snapshots_reports_unchanged_when_same():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()

    result = verifier.compare_snapshots(before_snapshot, after_snapshot)

    assert result.verified is False
    assert result.message == "snapshot_unchanged"


def test_verify_action_effect_requires_change_for_send_button():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.SEND_BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]

    result = verifier.verify_action_effect(
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        target_element_id="elem1",
        action="click",
    )

    assert result.verified is False
    assert result.message == "semantic_effect_missing"
    assert result.details["semantic_role"] == "send_button"
    assert result.details["policy"] == "require_page_or_structure_change"


def test_verify_action_effect_uses_region_policy_for_viewport_targets():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [Region(region_id="r1", role="viewport")]
    after_snapshot.regions = [Region(region_id="r1", role="viewport")]
    before_snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(20, 10, 50, 40),
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "elem1", "click")

    assert result.verified is True
    assert result.details["region_role"] == "viewport"
    assert result.details["policy"] == "require_target_or_structure_change_in_viewport"


def test_verify_action_effect_accepts_scroll_only_change_for_viewport_targets():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    after_snapshot.regions = [Region(region_id="r1", role="viewport", scroll_context_id="scroll_r1")]
    before_snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=400, viewport_width=300, scroll_offset=0, total_content_height=1800)
    ]
    after_snapshot.scroll_contexts = [
        ScrollContext(scroll_context_id="scroll_r1", region_id="r1", viewport_height=400, viewport_width=300, scroll_offset=360, total_content_height=1800)
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="elem1",
            region_id="r1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "elem1", "click")

    assert result.verified is True
    assert result.message == "viewport_effect_detected"
    assert result.details["before_scroll_evidence"]["scroll_offsets"] == [0]
    assert result.details["after_scroll_evidence"]["scroll_offsets"] == [360]


def test_verify_action_effect_uses_interaction_hints_and_structure_evidence():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.artifacts["structure_evidence_score"] = 0.9
    after_snapshot.artifacts["structure_evidence_score"] = 0.9
    before_snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={
                "interaction_hints": {"expected_effect": "open_dialog"},
                "structure_evidence_score": 0.91,
            },
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="elem1",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={
                "interaction_hints": {"expected_effect": "open_dialog"},
                "structure_evidence_score": 0.91,
            },
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "elem1", "click")

    assert result.verified is False
    assert result.details["policy"] == "interaction_hint/open_dialog"
    assert result.details["high_structure_evidence"] is True
    assert result.details["structure_evidence_score"] == 0.91


def test_compare_snapshots_tracks_dialog_widget_and_grid_evidence_changes():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(
            region_id="dialog_r",
            role="dialog_body",
            attributes={"dialog_kind": "form_dialog", "field_count": 1, "footer_action_count": 1},
        ),
        Region(
            region_id="view_r",
            role="viewport",
            attributes={
                "widget_groups": {"left": ["chart_a"]},
                "widget_group_count": 1,
                "grid_header_ids": ["header_a"],
                "grid_row_ids": ["row_a"],
                "grid_pagination_ids": [],
                "row_group_count": 1,
            },
        ),
    ]
    after_snapshot.regions = [
        Region(
            region_id="dialog_r",
            role="dialog_body",
            attributes={"dialog_kind": "viewer_dialog", "field_count": 0, "footer_action_count": 2},
        ),
        Region(
            region_id="view_r",
            role="viewport",
            attributes={
                "widget_groups": {"left": ["chart_a"], "right": ["chart_b"]},
                "widget_group_count": 2,
                "grid_header_ids": ["header_a", "header_b"],
                "grid_row_ids": ["row_a", "row_b"],
                "grid_pagination_ids": ["page_info"],
                "row_group_count": 2,
            },
        ),
    ]

    result = verifier.compare_snapshots(before_snapshot, after_snapshot)

    assert result.verified is True
    assert result.details["before_dialog_evidence"]["dialog_count"] == 1
    assert result.details["after_widget_evidence"]["widget_group_counts"] == [2]
    assert result.details["after_grid_evidence"]["grid_pagination_counts"] == [1]


def test_verify_action_effect_uses_dialog_slot_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="dialog_r", role="dialog_body", attributes={"dialog_kind": "form_dialog", "field_count": 1})
    ]
    after_snapshot.regions = [
        Region(region_id="dialog_r", role="dialog_body", attributes={"dialog_kind": "viewer_dialog", "field_count": 0})
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="confirm_btn",
            region_id="dialog_r",
            semantic_role=SemanticRole.SUBMIT_BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"dialog_slot": "footer_actions"},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="confirm_btn",
            region_id="dialog_r",
            semantic_role=SemanticRole.SUBMIT_BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"dialog_slot": "footer_actions"},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "confirm_btn", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_dialog_structure_or_target_change"
    assert result.details["dialog_slot"] == "footer_actions"


def test_verify_action_effect_uses_widget_group_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="view_r", role="viewport", attributes={"widget_groups": {"left": ["chart_a"]}, "widget_group_count": 1})
    ]
    after_snapshot.regions = [
        Region(region_id="view_r", role="viewport", attributes={"widget_groups": {"left": ["chart_a"], "center": ["chart_b"]}, "widget_group_count": 2})
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="chart_a",
            region_id="view_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"widget_group_id": "widget_left"},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="chart_a",
            region_id="view_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"widget_group_id": "widget_left"},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "chart_a", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_widget_group_or_structure_change"
    assert result.details["widget_group_id"] == "widget_left"


def test_verify_action_effect_uses_grid_pagination_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(
            region_id="view_r",
            role="viewport",
            attributes={"grid_header_ids": ["header"], "grid_row_ids": ["row1"], "grid_pagination_ids": [], "row_group_count": 1},
        )
    ]
    after_snapshot.regions = [
        Region(
            region_id="view_r",
            role="viewport",
            attributes={"grid_header_ids": ["header"], "grid_row_ids": ["row2"], "grid_pagination_ids": ["page_info"], "row_group_count": 1},
        )
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="page_info",
            region_id="view_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"grid_role": "pagination"},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="page_info",
            region_id="view_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"grid_role": "pagination"},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "page_info", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_grid_pagination_change"
    assert result.details["grid_role"] == "pagination"


def test_verify_action_effect_uses_grid_header_sort_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(
            region_id="view_r",
            role="viewport",
            attributes={"grid_header_ids": ["header"], "grid_row_ids": ["row1"], "grid_pagination_ids": [], "row_group_count": 1},
            element_ids=["header"],
        )
    ]
    after_snapshot.regions = [
        Region(
            region_id="view_r",
            role="viewport",
            attributes={"grid_header_ids": ["header"], "grid_row_ids": ["row1"], "grid_pagination_ids": [], "row_group_count": 1},
            element_ids=["header"],
        )
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="header",
            region_id="view_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Name",
            attributes={"grid_role": "header"},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="header",
            region_id="view_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Name ↓",
            attributes={"grid_role": "header"},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "header", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_grid_header_or_sort_change"
    assert result.details["grid_role"] == "header"


def test_verify_action_effect_uses_list_detail_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.LIST_DETAIL, child_region_ids=["list_r", "detail_r"]),
        Region(region_id="list_r", role="list_panel", parent_region_id="content_r"),
        Region(region_id="detail_r", role="detail_panel", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.LIST_DETAIL, child_region_ids=["list_r", "detail_r"]),
        Region(region_id="list_r", role="list_panel", parent_region_id="content_r"),
        Region(region_id="detail_r", role="detail_panel", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="thread_item",
            region_id="list_r",
            semantic_role=SemanticRole.LIST_ITEM,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="thread_item",
            region_id="detail_r",
            semantic_role=SemanticRole.LIST_ITEM,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "thread_item", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_list_detail_target_or_structure_change"
    assert result.details["content_subtype"] == "list_detail"
    assert result.details["target_region_before"] == "list_r"
    assert result.details["target_region_after"] == "detail_r"


def test_verify_action_effect_uses_list_detail_navigation_policy_from_interaction_hint():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.LIST_DETAIL, child_region_ids=["list_r", "detail_r"]),
        Region(region_id="list_r", role="list_panel", parent_region_id="content_r"),
        Region(region_id="detail_r", role="detail_panel", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.LIST_DETAIL, child_region_ids=["list_r", "detail_r"]),
        Region(region_id="list_r", role="list_panel", parent_region_id="content_r"),
        Region(region_id="detail_r", role="detail_panel", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="thread_item",
            region_id="list_r",
            semantic_role=SemanticRole.LIST_ITEM,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"interaction_hints": {"expected_effect": "open_detail"}},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="thread_item",
            region_id="detail_r",
            semantic_role=SemanticRole.LIST_ITEM,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            attributes={"interaction_hints": {"expected_effect": "open_detail"}},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "thread_item", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_list_detail_navigation_change"


def test_verify_action_effect_uses_editor_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.EDITOR, child_region_ids=["toolbar_r", "view_r"]),
        Region(region_id="toolbar_r", role="toolbar", parent_region_id="content_r"),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.EDITOR, child_region_ids=["toolbar_r", "view_r"]),
        Region(region_id="toolbar_r", role="toolbar", parent_region_id="content_r"),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="run_btn",
            region_id="toolbar_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Run",
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="run_btn",
            region_id="toolbar_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(20, 10, 50, 40),
            text="Run",
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "run_btn", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_editor_target_or_structure_change"
    assert result.details["content_subtype"] == "editor"


def test_verify_action_effect_uses_editor_run_code_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.EDITOR, child_region_ids=["toolbar_r", "view_r"]),
        Region(region_id="toolbar_r", role="toolbar", parent_region_id="content_r"),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.EDITOR, child_region_ids=["toolbar_r", "view_r"]),
        Region(region_id="toolbar_r", role="toolbar", parent_region_id="content_r"),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="run_btn",
            region_id="toolbar_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Run",
            attributes={"interaction_hints": {"expected_effect": "run_code"}},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="run_btn",
            region_id="toolbar_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(20, 10, 50, 40),
            text="Run",
            attributes={"interaction_hints": {"expected_effect": "run_code"}},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "run_btn", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_editor_run_code_change"


def test_verify_action_effect_uses_editor_open_panel_policy_from_editor_evidence():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.EDITOR, child_region_ids=["toolbar_r", "side_r", "view_r"]),
        Region(region_id="toolbar_r", role="toolbar", parent_region_id="content_r", element_ids=["toggle_panel"]),
        Region(region_id="side_r", role="side_panel", parent_region_id="content_r", element_ids=[]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.EDITOR, child_region_ids=["toolbar_r", "side_r", "view_r"]),
        Region(region_id="toolbar_r", role="toolbar", parent_region_id="content_r", element_ids=["toggle_panel"]),
        Region(region_id="side_r", role="side_panel", parent_region_id="content_r", element_ids=["outline_item"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="toggle_panel",
            region_id="toolbar_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Outline",
            attributes={"interaction_hints": {"expected_effect": "open_panel"}},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="toggle_panel",
            region_id="toolbar_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Outline",
            attributes={"interaction_hints": {"expected_effect": "open_panel"}},
        ),
        Candidate(
            element_id="outline_item",
            region_id="side_r",
            semantic_role=SemanticRole.TEXT,
            locator_ids=[],
            bounds=(60, 80, 120, 110),
            text="Chapter 1",
        ),
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "toggle_panel", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_editor_open_panel_change"


def test_verify_action_effect_uses_canvas_doc_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.CANVAS_DOC_VIEWER, child_region_ids=["view_r", "actions_r"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
        Region(region_id="actions_r", role="action_bar", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.CANVAS_DOC_VIEWER, child_region_ids=["view_r", "actions_r"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
        Region(region_id="actions_r", role="action_bar", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="next_page",
            region_id="actions_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Next",
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="next_page",
            region_id="actions_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Page 2",
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "next_page", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_canvas_doc_navigation_or_structure_change"
    assert result.details["content_subtype"] == "canvas_doc_viewer"


def test_verify_action_effect_uses_canvas_doc_next_page_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.CANVAS_DOC_VIEWER, child_region_ids=["view_r", "actions_r"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
        Region(region_id="actions_r", role="action_bar", parent_region_id="content_r"),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.CANVAS_DOC_VIEWER, child_region_ids=["view_r", "actions_r"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r"),
        Region(region_id="actions_r", role="action_bar", parent_region_id="content_r"),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="next_page",
            region_id="actions_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Next",
            attributes={"interaction_hints": {"expected_effect": "next_page"}},
        )
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="next_page",
            region_id="actions_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Page 2",
            attributes={"interaction_hints": {"expected_effect": "next_page"}},
        )
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "next_page", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_canvas_doc_next_page_change"


def test_verify_action_effect_uses_canvas_doc_zoom_policy():
    verifier = ActionVerifier()
    before_snapshot = make_snapshot()
    after_snapshot = make_snapshot()
    before_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.CANVAS_DOC_VIEWER, child_region_ids=["view_r", "actions_r"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r", element_ids=["page_canvas"]),
        Region(region_id="actions_r", role="action_bar", parent_region_id="content_r", element_ids=["zoom_in"]),
    ]
    after_snapshot.regions = [
        Region(region_id="content_r", role="content_area", subtype=ContentAreaSubtype.CANVAS_DOC_VIEWER, child_region_ids=["view_r", "actions_r"]),
        Region(region_id="view_r", role="viewport", parent_region_id="content_r", element_ids=["page_canvas"]),
        Region(region_id="actions_r", role="action_bar", parent_region_id="content_r", element_ids=["zoom_in"]),
    ]
    before_snapshot.elements = [
        Candidate(
            element_id="zoom_in",
            region_id="actions_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Zoom +",
            attributes={"interaction_hints": {"expected_effect": "zoom_in"}},
        ),
        Candidate(
            element_id="page_canvas",
            region_id="view_r",
            semantic_role=SemanticRole.IMAGE,
            locator_ids=[],
            bounds=(100, 100, 700, 900),
            text="125%",
        ),
    ]
    after_snapshot.elements = [
        Candidate(
            element_id="zoom_in",
            region_id="actions_r",
            semantic_role=SemanticRole.BUTTON,
            locator_ids=["loc1"],
            bounds=(10, 10, 40, 40),
            text="Zoom +",
            attributes={"interaction_hints": {"expected_effect": "zoom_in"}},
        ),
        Candidate(
            element_id="page_canvas",
            region_id="view_r",
            semantic_role=SemanticRole.IMAGE,
            locator_ids=[],
            bounds=(100, 100, 700, 900),
            text="150%",
        ),
    ]

    result = verifier.verify_action_effect(before_snapshot, after_snapshot, "zoom_in", "click")

    assert result.verified is True
    assert result.details["policy"] == "require_canvas_doc_zoom_in_change"
