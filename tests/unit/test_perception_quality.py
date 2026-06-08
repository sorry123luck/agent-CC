"""Tests for perception quality, visual pattern, and ROI planning."""

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    PageInfo,
    ProviderTrace,
    Region,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
)
from src.perception.perception_quality import PerceptionQualityEvaluator
from src.perception.roi_selection import RoiSelectionPlanner
from src.perception.visual_pattern import VisualPatternClassifier


def _canvas(
    *,
    page_class: str = "unknown/canvas/main/wide",
    surface_type: SurfaceType = SurfaceType.CANVAS_SELF_DRAWN,
    elements: list[Candidate] | None = None,
    regions: list[Region] | None = None,
    geometric_regions: list[GeometricRegion] | None = None,
    provider_details: dict | None = None,
    width: int = 1000,
    height: int = 700,
) -> InteractionCanvas:
    canvas = InteractionCanvas(
        app=AppInfo(process_name=page_class.split("/")[0]),
        surface=SurfaceInfo(surface_type=surface_type),
        page=PageInfo(page_class=page_class),
        elements=elements or [],
        regions=regions or [],
        provider_trace=ProviderTrace(provider_details=provider_details or {}),
    )
    canvas.artifacts["geometric_regions"] = [
        region.to_dict() for region in (geometric_regions or [])
    ]
    canvas.artifacts["screenshot_size"] = [width, height]
    return canvas


def _candidate(
    element_id: str,
    role: SemanticRole = SemanticRole.UNKNOWN,
    bounds: tuple[int, int, int, int] = (0, 0, 10, 10),
    text: str = "",
    control_type: str = "",
) -> Candidate:
    return Candidate(
        element_id=element_id,
        semantic_role=role,
        bounds=bounds,
        text=text,
        control_type=control_type,
    )


def _geo(region_id: str, bounds: tuple[int, int, int, int]) -> GeometricRegion:
    return GeometricRegion(region_id=region_id, bounds=bounds, geometry_confidence=0.7)


def test_quality_flags_sparse_fast_path_as_unreliable():
    canvas = _canvas(
        elements=[_candidate("root", SemanticRole.CONTAINER, (0, 0, 1000, 700))],
        regions=[Region("r0", bounds=(0, 0, 1000, 700)), Region("r1", bounds=(0, 0, 1000, 700))],
        geometric_regions=[_geo("R0", (0, 0, 100, 700)) for _ in range(14)],
    )

    quality = PerceptionQualityEvaluator().evaluate(canvas)

    assert quality.usable_state == "unreliable"
    assert "sparse_elements" in quality.warnings
    assert quality.element_count == 1
    assert quality.geometric_region_count == 14


def test_quality_flags_layout_audit_when_local_partition_is_insufficient():
    canvas = _canvas(
        elements=[_candidate("root", SemanticRole.CONTAINER, (0, 0, 1000, 700))],
        regions=[Region("r0", bounds=(0, 0, 1000, 700))],
        geometric_regions=[],
    )

    quality = PerceptionQualityEvaluator().evaluate(canvas)

    assert quality.usable_state == "unreliable"
    assert "layout_audit_needed" in quality.warnings
    assert quality.local_coordinate_trust == "low"


def test_quality_flags_layout_audit_blocked_when_screenshot_is_missing():
    canvas = _canvas(
        elements=[],
        regions=[Region("r0", bounds=(0, 0, 1000, 700))],
        geometric_regions=[],
    )
    canvas.artifacts.pop("screenshot_size", None)

    quality = PerceptionQualityEvaluator().evaluate(canvas)

    assert "screenshot_missing" in quality.warnings
    assert "layout_audit_needed" in quality.warnings
    assert "layout_audit_blocked_no_screenshot" in quality.warnings


def test_quality_warns_unknown_heavy_and_overfragmented():
    elements = [
        _candidate(f"e{i}", SemanticRole.UNKNOWN, (i, i, i + 5, i + 5))
        for i in range(30)
    ]
    elements.append(_candidate("button", SemanticRole.BUTTON, (10, 10, 20, 20)))
    canvas = _canvas(
        elements=elements,
        regions=[Region("content", bounds=(0, 0, 1000, 700))],
        geometric_regions=[_geo(f"R{i}", (i * 5, 0, i * 5 + 3, 100)) for i in range(39)],
    )

    quality = PerceptionQualityEvaluator().evaluate(canvas)

    assert quality.usable_state == "usable_with_warnings"
    assert quality.unknown_role_ratio > 0.9
    assert "unknown_role_heavy" in quality.warnings
    assert "geometric_overfragmented" in quality.warnings


def test_quality_treats_roi_vlm_refined_labels_as_known_semantics():
    elements = [
        _candidate(f"e{i}", SemanticRole.UNKNOWN, (i, i, i + 5, i + 5))
        for i in range(10)
    ]
    for idx, element in enumerate(elements[:8]):
        element.role_label = f"roi label {idx}"
        element.role_source = "roi_vlm"
        element.semantic_tags = ["roi_vlm.role.action"]
        element.refine_status = "refined"
    canvas = _canvas(
        elements=elements,
        regions=[
            Region("left", bounds=(0, 0, 250, 700)),
            Region("content", bounds=(250, 0, 1000, 700)),
            Region("footer", bounds=(250, 620, 1000, 700)),
        ],
    )

    quality = PerceptionQualityEvaluator().evaluate(canvas)

    assert quality.unknown_role_ratio == 0.2
    assert "unknown_role_heavy" not in quality.warnings


def test_visual_pattern_detects_loading_and_list_management():
    loading = _canvas(elements=[_candidate("txt", text="启动中")])
    list_page = _canvas(
        page_class="adspower global/viewer/main/wide",
        elements=[
            _candidate("search", SemanticRole.SEARCH_INPUT, control_type="Edit"),
            _candidate("row1", SemanticRole.LIST_ITEM, control_type="ListItem"),
            _candidate("row2", SemanticRole.LIST_ITEM, control_type="ListItem"),
            _candidate("row3", SemanticRole.LIST_ITEM, control_type="ListItem"),
        ],
    )

    classifier = VisualPatternClassifier()

    assert classifier.classify(loading).mode == "loading_state"
    assert classifier.classify(list_page).mode == "list_management"


def test_visual_pattern_prefers_specific_app_hints_over_generic_list_shape():
    winrar = _canvas(
        page_class="winrar/form/main/wide",
        elements=[
            _candidate("search", SemanticRole.SEARCH_INPUT, control_type="Edit"),
            _candidate("row1", SemanticRole.LIST_ITEM, control_type="ListItem"),
            _candidate("row2", SemanticRole.LIST_ITEM, control_type="ListItem"),
        ],
    )
    baidu_welcome = _canvas(
        page_class="unknown/canvas/main/wide",
        elements=[_candidate("root", SemanticRole.CONTAINER, text="百度网盘")],
    )
    baidu_welcome.app.process_name = "baidunetdisk"

    classifier = VisualPatternClassifier()

    assert classifier.classify(winrar).mode == "archive_file_manager"
    assert classifier.classify(baidu_welcome).mode == "loading_state"


def test_visual_pattern_detects_chat_and_media_from_evidence():
    chat = _canvas(
        page_class="wechat/app/main/wide",
        elements=[_candidate("composer", SemanticRole.UNKNOWN, text="发送")],
    )
    media = _canvas(
        page_class="cloudmusic/canvas/main/wide",
        elements=[_candidate("play", SemanticRole.BUTTON, text="播放全部")],
    )

    classifier = VisualPatternClassifier()

    assert classifier.classify(chat).mode == "chat_workspace"
    assert classifier.classify(media).mode == "media_home"


def test_visual_pattern_splits_wechat_browse_page_from_chat_workspace():
    classifier = VisualPatternClassifier()
    chat = _canvas(
        page_class="wechat/app/main/wide",
        width=1000,
        height=730,
        elements=[
            _candidate("emoji", bounds=(310, 680, 345, 714)),
            _candidate("file", bounds=(390, 681, 420, 711)),
            _candidate("send", bounds=(920, 683, 980, 709)),
        ],
    )
    contacts = _canvas(
        page_class="wechat/app/main/wide",
        width=1000,
        height=730,
        elements=[
            _candidate("rail", bounds=(8, 676, 55, 716)),
            _candidate("list_bottom", bounds=(86, 695, 301, 731)),
            _candidate("contact_row", bounds=(90, 350, 302, 406)),
        ],
    )

    assert classifier.classify(chat).mode == "chat_workspace"
    contact_result = classifier.classify(contacts)
    assert contact_result.mode == "chat_app_browse_page"
    assert "wechat_without_composer_geometry" in contact_result.evidence


def test_visual_pattern_splits_qq_chat_from_account_switcher():
    classifier = VisualPatternClassifier()
    chat = _canvas(
        page_class="qq/viewer/main/wide",
        width=960,
        height=640,
        elements=[
            _candidate("toolbar_emoji", bounds=(330, 441, 354, 465), text="表情"),
            _candidate("toolbar_file", bounds=(413, 441, 437, 465), text="文件"),
            _candidate("editor", SemanticRole.TEXT_INPUT, bounds=(310, 465, 780, 583), control_type="EditControl"),
            _candidate("send", SemanticRole.BUTTON, bounds=(665, 599, 729, 625), text="发送"),
        ],
    )
    account = _canvas(
        page_class="qq/viewer/main/wide",
        width=960,
        height=640,
        elements=[
            _candidate("account", bounds=(315, 91, 706, 215)),
            _candidate("login_action", SemanticRole.BUTTON, bounds=(520, 237, 570, 260), text="登录"),
        ],
    )

    chat_result = classifier.classify(chat)
    assert chat_result.mode == "chat_workspace"
    assert "qq_chat_evidence" in chat_result.evidence
    assert classifier.classify(account).mode == "account_switcher"


def test_visual_pattern_marks_qq_private_and_group_variants():
    classifier = VisualPatternClassifier()
    private_chat = _canvas(
        page_class="qq/viewer/main/wide",
        width=960,
        height=640,
        elements=[
            _candidate("title", SemanticRole.BUTTON, bounds=(330, 32, 360, 54), text="jz"),
            _candidate("toolbar_group", SemanticRole.BUTTON, bounds=(876, 31, 900, 55), text="发起群聊"),
            _candidate("toolbar_emoji", bounds=(330, 441, 354, 465), text="表情"),
            _candidate("editor", SemanticRole.TEXT_INPUT, bounds=(310, 465, 780, 583), control_type="EditControl"),
            _candidate("send", SemanticRole.BUTTON, bounds=(665, 599, 729, 625), text="发送"),
        ],
    )
    group_chat = _canvas(
        page_class="qq/viewer/main/wide",
        width=960,
        height=640,
        elements=[
            _candidate("title", SemanticRole.BUTTON, bounds=(330, 32, 420, 54), text="项目交流群"),
            _candidate("member", SemanticRole.BUTTON, bounds=(760, 80, 910, 110), text="群成员"),
            _candidate("toolbar_emoji", bounds=(330, 441, 354, 465), text="表情"),
            _candidate("editor", SemanticRole.TEXT_INPUT, bounds=(310, 465, 780, 583), control_type="EditControl"),
            _candidate("send", SemanticRole.BUTTON, bounds=(665, 599, 729, 625), text="发送"),
        ],
    )

    private_result = classifier.classify(private_chat)
    group_result = classifier.classify(group_chat)

    assert "qq_variant:private_chat" in private_result.evidence
    assert "qq_variant:group_chat" in group_result.evidence


def test_visual_pattern_splits_feishu_browse_page_from_inbox():
    classifier = VisualPatternClassifier()
    inbox = _canvas(
        page_class="lark/viewer/main/wide",
        width=1018,
        height=768,
        elements=[
            _candidate("tool_1", bounds=(731, 702, 759, 730)),
            _candidate("tool_2", bounds=(785, 704, 812, 732)),
            _candidate("send", bounds=(958, 704, 982, 732)),
        ],
    )
    browse = _canvas(
        page_class="lark/viewer/main/wide",
        width=1018,
        height=768,
        elements=[
            _candidate("rail", bounds=(20, 100, 60, 140)),
            _candidate("list_item", bounds=(200, 140, 480, 190)),
            _candidate("content", bounds=(520, 90, 960, 680)),
        ],
    )

    assert classifier.classify(inbox).mode == "collaboration_inbox"
    browse_result = classifier.classify(browse)
    assert browse_result.mode == "collaboration_browse_page"
    assert "collaboration_without_composer_geometry" in browse_result.evidence


def test_visual_pattern_detects_chat_search_results_for_qq_and_wechat():
    classifier = VisualPatternClassifier()
    qq_search = _canvas(
        page_class="qq/viewer/main/wide",
        width=960,
        height=640,
        elements=[
            _candidate("search", SemanticRole.SEARCH_INPUT, bounds=(76, 28, 260, 56), text="Qjz", control_type="ComboBox"),
            _candidate("contact_tab", bounds=(76, 70, 120, 90), text="联系人"),
            _candidate("global_search", SemanticRole.SEARCH_INPUT, bounds=(124, 331, 294, 369), text="进入全网搜索jz"),
            _candidate("editor", SemanticRole.TEXT_INPUT, bounds=(310, 465, 780, 583), control_type="EditControl"),
            _candidate("send", SemanticRole.BUTTON, bounds=(665, 599, 729, 625), text="发送"),
        ],
    )
    wechat_search = _canvas(
        page_class="wechat/app/main/wide",
        width=1002,
        height=731,
        elements=[
            _candidate("search_icon", bounds=(78, 44, 99, 68)),
            _candidate("query_text", bounds=(98, 43, 122, 68)),
            _candidate("clear", bounds=(231, 45, 252, 69)),
            _candidate("plus", bounds=(259, 36, 291, 74)),
            _candidate("result_avatar", bounds=(80, 115, 126, 155)),
            _candidate("result_info", bounds=(359, 124, 379, 150)),
        ],
    )

    qq_result = classifier.classify(qq_search)
    wechat_result = classifier.classify(wechat_search)

    assert qq_result.mode == "chat_search_results"
    assert "search_state:qq_results" in qq_result.evidence
    assert wechat_result.mode == "chat_search_results"
    assert "search_state:wechat_dropdown" in wechat_result.evidence


def test_visual_pattern_does_not_treat_wechat_sidebar_as_search_dropdown():
    classifier = VisualPatternClassifier()
    wechat_workspace = _canvas(
        page_class="wechat/app/main/wide",
        width=1002,
        height=731,
        elements=[
            _candidate("search_icon", bounds=(78, 44, 99, 68)),
            _candidate("normal_plus_button", bounds=(259, 37, 291, 75)),
            _candidate("plus", bounds=(310, 42, 337, 69)),
            _candidate("normal_chat_item", bounds=(60, 146, 295, 212)),
            _candidate("composer_icon_1", bounds=(512, 669, 536, 694)),
            _candidate("composer_icon_2", bounds=(552, 669, 576, 694)),
            _candidate("composer_icon_3", bounds=(592, 669, 616, 694)),
        ],
    )

    result = classifier.classify(wechat_workspace)

    assert result.mode == "chat_workspace"
    assert "search_state:wechat_dropdown" not in result.evidence


def test_visual_pattern_detects_wechat_screen_region_search_popup():
    classifier = VisualPatternClassifier()
    wechat_search = _canvas(
        page_class="wechat/app/main/wide",
        width=1002,
        height=731,
        elements=[
            _candidate("search_icon", bounds=(76, 43, 98, 68)),
            _candidate("query_text", bounds=(98, 43, 122, 68)),
            _candidate("clear", bounds=(231, 45, 252, 69)),
            _candidate("plus", bounds=(259, 36, 291, 74)),
            _candidate("result_avatar", bounds=(80, 115, 126, 155)),
            _candidate("result_info", bounds=(359, 124, 379, 150)),
            _candidate("composer_icon_1", bounds=(512, 669, 536, 694)),
            _candidate("composer_icon_2", bounds=(552, 669, 576, 694)),
            _candidate("composer_icon_3", bounds=(592, 669, 616, 694)),
        ],
    )

    result = classifier.classify(wechat_search)

    assert result.mode == "chat_search_results"
    assert "search_state:wechat_dropdown" in result.evidence


def test_visual_pattern_detects_feishu_search_overlay():
    classifier = VisualPatternClassifier()
    feishu_search = _canvas(
        page_class="lark/viewer/main/wide",
        width=1018,
        height=768,
        elements=[
            _candidate("modal", bounds=(101, 50, 917, 717), text="search-command-bar"),
            _candidate("query", SemanticRole.SEARCH_INPUT, bounds=(163, 81, 353, 113), text="大笨蛋"),
            _candidate("result", bounds=(122, 189, 890, 277), text="大笨蛋 Openclaw"),
            _candidate("tool_1", bounds=(731, 702, 759, 730)),
            _candidate("send", bounds=(958, 704, 982, 732)),
        ],
    )

    result = classifier.classify(feishu_search)

    assert result.mode == "collaboration_search_overlay"
    assert "search_state:feishu_command_bar" in result.evidence


def test_roi_plan_uses_search_specific_regions():
    planner = RoiSelectionPlanner()
    qq_search = _canvas(
        page_class="qq/viewer/main/wide",
        width=960,
        height=640,
        elements=[
            _candidate("search", SemanticRole.SEARCH_INPUT, bounds=(76, 28, 260, 56), text="Qjz", control_type="ComboBox"),
            _candidate("global_search", SemanticRole.SEARCH_INPUT, bounds=(124, 331, 294, 369), text="进入全网搜索jz"),
            _candidate("editor", SemanticRole.TEXT_INPUT, bounds=(310, 465, 780, 583), control_type="EditControl"),
            _candidate("send", SemanticRole.BUTTON, bounds=(665, 599, 729, 625), text="发送"),
        ],
    )
    feishu_search = _canvas(
        page_class="lark/viewer/main/wide",
        width=1018,
        height=768,
        elements=[_candidate("modal", bounds=(101, 50, 917, 717), text="search-command-bar")],
    )

    qq_plan = planner.plan(qq_search)
    feishu_plan = planner.plan(feishu_search)

    assert qq_plan.mode == "chat_search_results"
    assert [roi.purpose for roi in qq_plan.rois] == ["app_rail", "search_results_list", "conversation_context"]
    assert feishu_plan.mode == "collaboration_search_overlay"
    assert [roi.purpose for roi in feishu_plan.rois] == ["search_overlay", "background_context"]


def test_roi_plan_returns_zero_for_loading_and_bounded_rois_for_normal_page():
    loading = _canvas(elements=[_candidate("txt", text="启动中")])
    normal = _canvas(
        page_class="bilibili/canvas/main/wide",
        elements=[_candidate(f"e{i}", bounds=(i * 20, 20, i * 20 + 10, 60)) for i in range(10)],
        geometric_regions=[_geo(f"R{i}", (i * 20, 0, i * 20 + 30, 100)) for i in range(20)],
        width=1200,
        height=730,
    )

    planner = RoiSelectionPlanner()

    assert planner.plan(loading).rois == []

    plan = planner.plan(normal)
    assert plan.mode == "video_feed_home"
    assert 3 <= len(plan.rois) <= 8
    assert all(roi.allowed_outputs == ["region_semantics", "candidate_annotations", "review_only_hints"] for roi in plan.rois)
    assert all(roi.vlm_task in {"region_annotation", "missing_audit", "none"} for roi in plan.rois)


def test_roi_plan_uses_review_only_layout_audit_regions_for_unknown_sparse_page():
    canvas = _canvas(page_class="unknown/canvas/main/wide", width=1000, height=700)
    canvas.artifacts["vlm_layout_audit"] = {
        "review_only": True,
        "layout_regions": [
            {
                "role": "left_navigation",
                "relative_bounds": [0.0, 0.0, 0.24, 1.0],
                "confidence": 0.72,
            },
            {
                "role": "main_content",
                "relative_bounds": [0.24, 0.0, 1.0, 1.0],
                "confidence": 0.70,
            },
        ],
    }

    plan = RoiSelectionPlanner().plan(canvas)

    assert [roi.source for roi in plan.rois[:2]] == ["vlm_layout_audit", "vlm_layout_audit"]
    assert [roi.purpose for roi in plan.rois[:2]] == ["left_navigation", "main_content"]
    assert plan.rois[0].bounds == (0, 0, 240, 700)
    assert plan.rois[0].vlm_task == "region_annotation"


def test_roi_plan_uses_mode_specific_regions_for_non_list_apps():
    planner = RoiSelectionPlanner()

    cases = [
        ("flclash/canvas/main/wide", ["left_navigation", "mode_tabs", "dashboard_or_cards", "right_status_actions"]),
        ("chatgpt/canvas/main/wide", ["conversation_sidebar", "document_thread", "composer"]),
        ("wechat/contact/main/wide", ["app_rail", "item_list", "detail_pane"]),
        ("lark/browse/main/wide", ["app_rail", "workspace_navigation", "main_workspace"]),
        ("hipsmain/canvas/main/wide", ["security_status", "feature_grid", "bottom_actions"]),
        ("winrar/form/main/wide", ["menu_toolbar", "file_list", "status_bar"]),
    ]

    for page_class, expected_purposes in cases:
        canvas = _canvas(page_class=page_class, width=1200, height=760)
        plan = planner.plan(canvas)
        assert [roi.purpose for roi in plan.rois[: len(expected_purposes)]] == expected_purposes


def test_roi_plan_splits_collaboration_inbox_thread_and_composer():
    canvas = _canvas(
        page_class="feishu/canvas/main/wide",
        width=1000,
        height=700,
        elements=[
            _candidate("emoji", bounds=(560, 640, 590, 670)),
            _candidate("attach", bounds=(610, 640, 640, 670)),
            _candidate("message", bounds=(620, 220, 940, 300), text="hello"),
        ],
    )

    plan = RoiSelectionPlanner().plan(canvas)

    assert [roi.purpose for roi in plan.rois[:4]] == [
        "app_rail",
        "inbox_list",
        "message_thread",
        "composer",
    ]
    assert plan.rois[2].bounds[3] <= plan.rois[3].bounds[1]
    assert "emoji" in plan.rois[3].candidate_ids
    assert "message" in plan.rois[2].candidate_ids


def test_roi_plan_does_not_append_overlapping_candidate_regions_for_complete_skeleton_modes():
    planner = RoiSelectionPlanner()
    canvas = _canvas(
        page_class="hipsmain/canvas/main/wide",
        width=916,
        height=616,
        geometric_regions=[
            _geo("main", (88, 0, 916, 616)),
            _geo("rail", (0, 0, 88, 616)),
        ],
    )

    plan = planner.plan(canvas)

    assert [roi.purpose for roi in plan.rois] == [
        "security_status",
        "feature_grid",
        "bottom_actions",
    ]


def test_visual_pattern_detects_new_live_sample_modes():
    classifier = VisualPatternClassifier()

    cases = [
        ("everything/canvas/main/wide", "file_search"),
        ("localsend/canvas/main/wide", "local_transfer_dashboard"),
        ("qq/account/main/wide", "account_switcher"),
        ("hubstudio/canvas/main/wide", "browser_profile_manager"),
        ("teamviewer/canvas/main/wide", "remote_access_dashboard"),
        ("qyclient/canvas/main/wide", "media_video_home"),
    ]

    for page_class, expected_mode in cases:
        assert classifier.classify(_canvas(page_class=page_class)).mode == expected_mode


def test_roi_plan_uses_mode_specific_regions_for_new_live_samples():
    planner = RoiSelectionPlanner()

    cases = [
        ("everything/canvas/main/wide", ["search_bar", "result_list", "status_or_filters"]),
        ("localsend/canvas/main/wide", ["device_status", "send_receive_actions", "transfer_history"]),
        ("qq/account/main/wide", ["account_list", "login_actions", "window_controls"]),
        ("hubstudio/canvas/main/wide", ["left_navigation", "profile_table_or_cards", "toolbar_search_filters"]),
        ("teamviewer/canvas/main/wide", ["connection_panel", "remote_control_actions", "status_or_recent"]),
        ("qyclient/canvas/main/wide", ["top_navigation_search", "media_feed", "player_or_bottom_actions"]),
    ]

    for page_class, expected_purposes in cases:
        canvas = _canvas(page_class=page_class, width=1200, height=760)
        plan = planner.plan(canvas)
        assert [roi.purpose for roi in plan.rois[: len(expected_purposes)]] == expected_purposes
