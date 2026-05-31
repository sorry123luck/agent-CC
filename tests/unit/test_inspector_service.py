import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from src.inspector.service import InspectorService
from src.perception.debug_tools import save_debug_bundle
from src.perception.openclaw_protocol import build_openclaw_payload
from src.perception.page_compiler_models import InteractionCanvas, WindowInfoSnapshot
from src.perception.uia_client import UIAElementInfo
from src.windows.window_enum import WindowInfoExt


def make_window(hwnd: int = 1001) -> WindowInfoExt:
    return WindowInfoExt(
        hwnd=hwnd,
        title="Test Window",
        class_name="TestClass",
        process_name="test.exe",
        process_id=1234,
        rect=(0, 0, 1280, 720),
        is_visible=True,
        is_enabled=True,
    )


def test_inspect_window_exports_minimal_bundle(tmp_path: Path):
    window_enum_service = MagicMock()
    window_enum_service.enumerate_all.return_value = [make_window()]

    screenshot_service = MagicMock()
    screenshot = Image.new("RGB", (1280, 720), color=(32, 32, 32))
    preview = Image.new("RGB", (640, 360), color=(48, 48, 48))
    screenshot_service.capture.return_value = screenshot
    screenshot_service.create_preview.return_value = preview

    perception_service = MagicMock()
    zone_page = MagicMock()
    zone_page.screenshot = screenshot
    zone_page.window_info = make_window()
    perception_service.analyze.return_value = zone_page
    perception_service.create_page_snapshot.return_value = InteractionCanvas(canvas_id="snap_test")

    ocr_service = MagicMock()
    ocr_block = MagicMock()
    ocr_block.text = "hello"
    ocr_block.bbox = (10, 10, 40, 20)
    ocr_block.confidence = 0.9
    ocr_service.extract.return_value = [ocr_block]

    inspector = InspectorService(
        window_enum_service=window_enum_service,
        screenshot_service=screenshot_service,
        perception_service=perception_service,
        ocr_service=ocr_service,
    )

    tree_element = UIAElementInfo(
        name="Send",
        automation_id="btnSend",
        control_type="ButtonControl",
        bounding_rect=(1, 2, 3, 4),
        is_enabled=True,
        element_id="uia_1",
    )

    with patch("src.inspector.service.UIAClient") as mock_uia_client_cls, patch(
        "src.inspector.service.save_debug_bundle"
    ) as mock_save_debug_bundle:
        mock_uia_client = MagicMock()
        mock_uia_client.get_element_tree.return_value = [(0, tree_element)]
        mock_uia_client_cls.return_value = mock_uia_client
        mock_save_debug_bundle.return_value = {
            "bundle_name": "inspect_1001_20260406",
            "files": {"snapshot": str(tmp_path / "snapshot.json")},
            "summary": {
                "surface_type": "native_uia",
                "confidence": 0.8,
                "page_class": "unknown/unknown/unknown",
                "region_count": 1,
                "element_count": 1,
                "locator_count": 1,
                "anchor_count": 0,
                "relation_count": 0,
            },
        }

        result = inspector.inspect_window(hwnd=1001, output_dir=tmp_path)

    assert result["files"]["preview"].endswith("_preview.png")
    assert result["files"]["uia_tree"].endswith("_uia_tree.json")
    assert result["files"]["ocr"].endswith("_ocr.json")
    assert result["files"]["window"].endswith("_window.json")
    assert result["summary"]["uia_tree_node_count"] == 1
    assert result["summary"]["ocr_block_count"] == 1

    uia_tree = json.loads((tmp_path / "inspect_1001_uia_tree.json").read_text(encoding="utf-8"))
    assert uia_tree[0]["automation_id"] == "btnSend"


def test_inspect_foreground_window_uses_foreground_hwnd(tmp_path: Path):
    foreground = make_window(hwnd=2002)
    window_enum_service = MagicMock()
    window_enum_service.get_foreground_window.return_value = foreground

    inspector = InspectorService(
        window_enum_service=window_enum_service,
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )

    with patch.object(inspector, "inspect_window", return_value={"hwnd": 2002}) as mock_inspect:
        result = inspector.inspect_foreground_window(output_dir=tmp_path, include_ocr=False)

    assert result["hwnd"] == 2002
    mock_inspect.assert_called_once()
    assert mock_inspect.call_args.kwargs["hwnd"] == 2002


def test_compare_snapshot_files_reports_structure_and_provider_diffs(tmp_path: Path):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(
        json.dumps(
            {
                "surface": {"surface_type": "browser"},
                "page": {"page_class": "app/list/default"},
                "regions": [{"role": "content_area", "child_region_ids": ["r_toolbar"]}],
                "elements": [{"element_id": "elem1"}],
                "anchors": [{"anchor_id": "a1"}],
                "scroll_contexts": [{"scroll_context_id": "scroll_r1"}],
                "provider_trace": {"provider_details": {"vision": {"layout_region_count": 1, "control_group_count": 0, "interaction_hint_count": 0}}},
                "artifacts": {"ocr_blocks": [{}], "vision_candidates": [{}], "structure_evidence_score": 0.45},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "surface": {"surface_type": "browser"},
                "page": {"page_class": "app/detail/default"},
                "regions": [{"role": "content_area", "child_region_ids": ["r_toolbar", "r_viewport"]}, {"role": "viewport", "child_region_ids": []}],
                "elements": [{"element_id": "elem1"}, {"element_id": "elem2"}],
                "anchors": [],
                "scroll_contexts": [],
                "provider_trace": {"provider_details": {"vision": {"layout_region_count": 2, "control_group_count": 1, "interaction_hint_count": 1}}},
                "artifacts": {"ocr_blocks": [{}, {}], "vision_candidates": [{}, {}], "structure_evidence_score": 0.79},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.compare_snapshot_files(before_path, after_path)

    assert result["region_count_delta"] == 1
    assert result["element_count_delta"] == 1
    assert result["added_region_roles"] == ["viewport"]
    assert result["before_anchor_count"] == 1
    assert result["after_scroll_context_count"] == 0
    assert result["provider_diff"]["after_control_group_count"] == 1
    assert result["after_structure_evidence_score"] == 0.79
    assert result["before_trace"]["vision_layout_region_count"] == 0
    assert result["after_trace"]["vision_layout_region_count"] == 0


def test_compare_snapshot_files_includes_target_debug(tmp_path: Path):
    before_path = tmp_path / "before_target.json"
    after_path = tmp_path / "after_target.json"
    before_path.write_text(
        json.dumps(
            {
                "surface": {"surface_type": "browser"},
                "page": {"page_class": "app/chat/default"},
                "regions": [
                    {
                        "region_id": "r_viewport",
                        "role": "viewport",
                        "bounds": [0, 0, 800, 600],
                        "child_region_ids": [],
                        "scroll_context_id": "scroll_1",
                    }
                ],
                "elements": [
                    {
                        "element_id": "elem_send",
                        "region_id": "r_viewport",
                        "semantic_role": "send_button",
                        "bounds": [700, 520, 780, 560],
                        "content_group_id": "cg_composer",
                        "provider_sources": ["ocr", "vision"],
                        "locator_ids": ["loc_send"],
                        "anchor_ids": ["anchor_send"],
                        "attributes": {
                            "interaction_hints": {"expected_effect": "send_message"},
                            "structure_evidence_score": 0.83,
                            "ocr_text": "Send",
                        },
                    }
                ],
                "locators": [
                    {
                        "locator_id": "loc_send",
                        "kind": "ocr",
                        "priority": 3,
                        "confidence": 0.74,
                        "durability_score": 0.66,
                        "cost_score": 0.22,
                        "anchor_refs": ["anchor_send"],
                        "verification_hints": {"text_equals": "Send"},
                    }
                ],
                "anchors": [
                    {
                        "anchor_id": "anchor_send",
                        "kind": "text",
                        "stability_score": 0.71,
                        "signature": {"text": "Send"},
                    }
                ],
                "scroll_contexts": [
                    {
                        "scroll_context_id": "scroll_1",
                        "scroll_type": "vertical",
                        "viewport_height": 600,
                        "viewport_width": 800,
                        "total_content_height": 1600,
                        "is_virtual": False,
                    }
                ],
                "artifacts": {"structure_evidence_score": 0.8},
                "provider_trace": {"provider_details": {}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    after_path.write_text(before_path.read_text(encoding="utf-8"), encoding="utf-8")

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.compare_snapshot_files(
        before_path,
        after_path,
        target_element_id="elem_send",
    )

    assert result["target_element_id"] == "elem_send"
    assert result["target_debug"]["before"]["semantic_role"] == "send_button"


def test_trace_action_outcome_returns_runtime_and_recovery_trace(tmp_path: Path):
    outcome_path = tmp_path / "outcome.json"
    outcome_path.write_text(
        json.dumps(
            {
                "success": False,
                "action": "click",
                "message": "miss",
                "attempts": 3,
                "strategy": "locator_first/chain/viewport_recovery",
                "error": "locator_chain_failed_viewport_recovery_required",
                "verification_details": {
                    "status": "semantic_effect_missing",
                    "policy": "strict_snapshot_change",
                    "verification_element_id": "elem1_new",
                    "risk_level": "high",
                    "policy_mode": "guarded_automation",
                    "policy_source": "snapshot.workflow_policy",
                    "reason": "browser_high_risk_block",
                    "recovery": {
                        "reason": "viewport_context_requires_recovery",
                        "terminal_strategy": "viewport_recovery",
                        "next_suggestion": {
                            "strategy": "viewport_recovery",
                            "action": "scroll_and_recapture",
                        },
                        "attempt_trace": [
                            {"candidate": "relative", "stage": "action", "error": "miss"},
                            {
                                "candidate": "relative",
                                "stage": "verify_after_recovery",
                                "error": "verification_failed",
                                "verification_message": "semantic_effect_missing",
                            },
                        ],
                    },
                    "recovery_execution": {
                        "mode": "replan",
                        "original": "elem1",
                        "resolved": "elem1_new",
                    },
                    "runtime_trace": {
                        "last_attempt_stage": "verify_after_recovery",
                        "last_attempt_candidate": "relative",
                        "final_message": "miss",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.trace_action_outcome(outcome_path)

    assert result["action_trace"]["strategy"] == "locator_first/chain/viewport_recovery"
    assert result["action_trace"]["verification_element_id"] == "elem1_new"
    assert result["action_trace"]["resolved_element_id"] == "elem1_new"
    assert result["action_trace"]["next_suggestion"]["action"] == "scroll_and_recapture"
    assert result["action_trace"]["runtime_trace"]["last_attempt_stage"] == "verify_after_recovery"
    assert result["action_trace"]["risk_level"] == "high"
    assert result["action_trace"]["policy_mode"] == "guarded_automation"


def test_trace_action_outcome_keeps_decision_and_recrop_fields(tmp_path: Path):
    outcome_path = tmp_path / "outcome_decision.json"
    outcome_path.write_text(
        json.dumps(
            {
                "success": False,
                "action": "recrop",
                "message": "openclaw_requested_recrop",
                "attempts": 1,
                "strategy": "decision_record/recrop",
                "verification_details": {
                    "decision_record": {
                        "page_state": "search_results_dense",
                        "selected_candidate_id": "",
                        "decision_status": "need_zoom_in",
                        "focus_bbox": [60, 70, 180, 170],
                    },
                    "recrop_request": {
                        "source_canvas_id": "snap_focus_1",
                        "focus_bbox": [60, 70, 180, 170],
                        "crop_scale": 1.5,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.trace_action_outcome(outcome_path)

    assert result["action_trace"]["decision_record"]["decision_status"] == "need_zoom_in"
    assert result["action_trace"]["recrop_request"]["focus_bbox"] == [60, 70, 180, 170]


def test_trace_action_target_reads_snapshot_file(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_1",
                "surface": {"surface_type": "canvas_self_drawn"},
                "page": {"page_class": "app/form/default"},
                "regions": [
                    {
                        "region_id": "r_form",
                        "role": "form_fields",
                        "bounds": [0, 0, 900, 500],
                        "child_region_ids": [],
                    }
                ],
                "elements": [
                    {
                        "element_id": "input_email",
                        "region_id": "r_form",
                        "semantic_role": "text_input",
                        "bounds": [120, 100, 520, 140],
                        "provider_sources": ["ocr"],
                        "locator_ids": [],
                        "anchor_ids": [],
                        "attributes": {
                            "interaction_hints": {"expected_effect": "focus_input"},
                            "structure_evidence_score": 0.64,
                            "ocr_text": "Email",
                        },
                    }
                ],
                "locators": [],
                "anchors": [],
                "scroll_contexts": [],
                "artifacts": {
                    "ocr_blocks": [{}],
                    "vision_candidates": [{}],
                    "vision_layout_regions": [{}],
                    "vision_control_groups": [],
                    "structure_evidence_score": 0.61,
                },
                "provider_trace": {"provider_details": {"vision": {"layout_region_count": 1}}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.trace_action_target(snapshot_path, "input_email")

    assert result["surface_type"] == "canvas_self_drawn"
    assert result["evidence_trace"]["vision_layout_region_count"] == 1
    assert result["target_debug"]["semantic_role"] == "text_input"
    assert result["target_debug"]["interaction_hints"]["expected_effect"] == "focus_input"


def test_build_openclaw_payload_exposes_weak_candidates(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot_openclaw.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_1",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 896, 648]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [
                    {
                        "region_id": "r_filter",
                        "role": "filter_bar",
                        "subtype": "unknown",
                        "bounds": [100, 40, 240, 72],
                    },
                    {
                        "region_id": "r_action",
                        "role": "action_bar",
                        "subtype": "unknown",
                        "bounds": [820, 598, 860, 624],
                    },
                ],
                "elements": [
                    {
                        "element_id": "elem_search",
                        "semantic_role": "search_input",
                        "control_type": "EditControl",
                        "region_id": "r_filter",
                        "bounds": [102, 44, 137, 65],
                        "provider_sources": ["ocr"],
                        "locator_ids": ["loc1"],
                        "anchor_ids": [],
                        "attributes": {"ocr_text": "鎼滅储"},
                    }
                ],
                "artifacts": {
                    "ocr_blocks": [{"bbox": [102, 44, 137, 65], "text": "鎼滅储", "confidence": 0.9}],
                    "vision_candidates": [
                        {
                            "candidate_id": "vision_send",
                            "bbox": [824, 598, 857, 621],
                            "kind": "send_button",
                            "source": "omniparser",
                            "confidence": 0.8,
                            "text": "发送",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.build_openclaw_payload(
        snapshot_path=snapshot_path,
        task="鍦ㄥ井淇′腑鎼滅储瀛欏畤",
    )

    payload = result["openclaw_payload"]
    assert payload["task"] == "鍦ㄥ井淇′腑鎼滅储瀛欏畤"
    assert payload["app"] == "WeChat"
    assert payload["surface_type"] == "electron_webview"
    assert payload["page_hint"] == "wechat/app_chat/main/wide"
    assert any(item["candidate_id"] == "elem::elem_search" for item in payload["candidates"])
    assert any(item["candidate_id"] == "vision_send" for item in payload["candidates"])
    assert payload["regions"][0]["role_hint"] == "filter_bar"


def test_export_candidate_regression_writes_candidates_payload_and_overlay(tmp_path: Path):
    snapshot_path = tmp_path / "wechat_snapshot.json"
    image_path = tmp_path / "wechat_raw.png"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_reg_1",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 400, 300]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [
                    {
                        "region_id": "r_filter",
                        "role": "filter_bar",
                        "subtype": "unknown",
                        "bounds": [80, 30, 220, 60],
                    }
                ],
                "elements": [],
                "artifacts": {
                    "ocr_blocks": [{"bbox": [90, 35, 130, 55], "text": "鎼滅储", "confidence": 0.9}],
                    "vision_candidates": [],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(245, 245, 245)).save(image_path)

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.export_candidate_regression(
        snapshot_path=snapshot_path,
        task="鍦ㄥ井淇′腑鎼滅储瀛欏畤",
        image_path=image_path,
        output_dir=tmp_path,
        draw_labels=True,
    )

    assert result["candidate_count"] >= 1
    assert Path(result["candidates_path"]).exists()
    assert Path(result["payload_path"]).exists()
    assert Path(result["overlay_path"]).exists()
    payload = json.loads(Path(result["payload_path"]).read_text(encoding="utf-8"))
    assert payload["task"] == "鍦ㄥ井淇′腑鎼滅储瀛欏畤"
    assert any(item["candidate_id"] == "ocr::0" for item in payload["candidates"])


def test_normalize_decision_record_returns_canonical_shape(tmp_path: Path):
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "page_state": "search_ready",
                "task_understanding": "先搜索孙宇",
                "selected_candidate_id": "cand_1",
                "selected_role": "input_candidate",
                "bbox": [78, 48, 173, 31],
                "reason": "顶部输入框",
                "alternatives": [
                    {
                        "candidate_id": "cand_2",
                        "selected_role": "button_candidate",
                        "reject_reason": "涓嶆槸褰撳墠姝ラ",
                    }
                ],
                "next_action": {"type": "click_and_type", "text": "孙宇"},
                "confidence": 0.91,
                "decision_status": "ready",
                "focus_bbox": [70, 40, 200, 40],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.normalize_decision_record(decision_path)

    record = result["decision_record"]
    assert record["page_state"] == "search_ready"
    assert record["selected_candidate_id"] == "cand_1"
    assert record["bbox"] == [78, 48, 173, 31]
    assert record["next_action"]["type"] == "click_and_type"
    assert record["focus_bbox"] == [70, 40, 200, 40]


def test_build_recrop_request_uses_focus_bbox(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot_focus.json"
    decision_path = tmp_path / "decision_focus.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_focus_1",
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    decision_path.write_text(
        json.dumps(
            {
                "page_state": "search_results_dense",
                "reason": "鍊欓€夎繃瀵嗭紝闇€瑕佸眬閮ㄩ噸閲囨牱",
                "focus_bbox": [60, 70, 320, 420],
                "next_action": {"type": "recrop_and_reanalyze", "text": ""},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.build_recrop_request(
        snapshot_path=snapshot_path,
        decision_path=decision_path,
        crop_scale=1.8,
    )

    recrop = result["recrop_request"]
    assert recrop["source_canvas_id"] == "snap_focus_1"
    assert recrop["focus_bbox"] == [60, 70, 320, 420]
    assert recrop["crop_scale"] == 1.8
    assert recrop["next_action"]["type"] == "recrop_and_reanalyze"


def test_reanalyze_focus_region_builds_second_round_payload(tmp_path: Path):
    snapshot_path = tmp_path / "inspect_1001_snapshot.json"
    decision_path = tmp_path / "decision_focus.json"
    image_path = tmp_path / "inspect_1001_raw.png"

    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_focus_1",
                "app": {"process_name": "WeChat.exe"},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    decision_path.write_text(
        json.dumps(
            {
                "page_state": "search_results_dense",
                "reason": "鍊欓€夎繃瀵嗭紝闇€瑕佸眬閮ㄩ噸閲囨牱",
                "focus_bbox": [60, 70, 180, 170],
                "next_action": {"type": "recrop_and_reanalyze", "text": ""},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(240, 240, 240)).save(image_path)

    ocr_service = MagicMock()
    ocr_block = MagicMock()
    ocr_block.text = "瀛欏畤"
    ocr_block.bbox = (20, 30, 70, 55)
    ocr_block.confidence = 0.95
    ocr_result = MagicMock()
    ocr_result.provider = "paddleocr_bridge"
    ocr_result.success = True
    ocr_result.error = None
    ocr_result.blocks = [ocr_block]
    ocr_service.extract_with_metadata.return_value = ocr_result

    vision_candidate = MagicMock()
    vision_candidate.element_id = "vision::0"
    vision_candidate.bounding_box = (90, 40, 130, 70)
    vision_candidate.semantic_label = "button_candidate"
    vision_candidate.text = "发送"
    vision_candidate.confidence = 0.81
    vision_result = MagicMock()
    vision_result.provider = "omniparser"
    vision_result.success = True
    vision_result.error = None
    vision_result.candidates = [vision_candidate]

    perception_service = MagicMock()
    perception_service._vision_provider = MagicMock()
    perception_service._vision_provider.parse_screenshot.return_value = vision_result

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=perception_service,
        ocr_service=ocr_service,
    )
    result = inspector.reanalyze_focus_region(
        snapshot_path=snapshot_path,
        decision_path=decision_path,
        task="鍦ㄥ井淇￠噷鎼滅储瀛欏畤",
        image_path=image_path,
        crop_scale=1.2,
    )

    assert result["focus_bbox"] == [60, 70, 180, 170]
    assert result["ocr_provider"]["block_count"] == 1
    assert result["vision_provider"]["candidate_count"] == 1
    assert Path(result["crop_path"]).exists()
    payload = result["focused_openclaw_payload"]
    assert payload["task"] == "鍦ㄥ井淇￠噷鎼滅储瀛欏畤"
    assert any(item["candidate_id"] == "ocr::0" for item in payload["candidates"])
    assert any(item["candidate_id"] == "vision::0" for item in payload["candidates"])


def test_save_debug_bundle_writes_candidate_overlay(tmp_path: Path):
    snapshot = InteractionCanvas(canvas_id="snap_candidates")
    snapshot.window = WindowInfoSnapshot(hwnd=1001, title="Test", rect_client=(0, 0, 320, 240))
    snapshot.artifacts["boundary_candidates"] = [
        {
            "candidate_id": "ocr::0",
            "bbox": [20, 30, 120, 60],
            "source": "ocr",
            "confidence": 0.92,
            "candidate_kind": "input_candidate",
            "text": "Search",
            "control_hint": "edit_like",
            "region_hint": "filter_bar",
            "region_id": None,
            "attributes": {},
        },
        {
            "candidate_id": "vision_group::cg_footer",
            "bbox": [210, 180, 300, 220],
            "source": "omniparser",
            "confidence": 0.74,
            "candidate_kind": "panel_candidate",
            "text": "Footer",
            "control_hint": "group_like",
            "region_hint": "action_bar",
            "region_id": None,
            "attributes": {"candidate_origin": "vision_control_group"},
        },
    ]
    snapshot.artifacts["boundary_candidate_stats"] = {
        "count": 2,
        "by_source": {"ocr": 1, "omniparser": 1},
        "by_kind": {"input_candidate": 1, "panel_candidate": 1},
    }

    bundle = save_debug_bundle(
        snapshot,
        output_dir=tmp_path,
        bundle_name="candidate_debug",
        screenshot=Image.new("RGB", (320, 240), color=(30, 30, 30)),
    )

    assert "candidate_overlay" in bundle["files"]
    assert Path(bundle["files"]["candidate_overlay"]).exists()
    assert Path(bundle["files"]["candidates"]).exists()


def test_openclaw_payload_includes_provider_health_and_candidate_summary(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot_openclaw_health.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_health_1",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 896, 648]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [
                    {
                        "region_id": "r_filter",
                        "role": "filter_bar",
                        "subtype": "unknown",
                        "bounds": [100, 40, 240, 72],
                    }
                ],
                "elements": [],
                "artifacts": {
                    "ocr_blocks": [{"bbox": [102, 44, 160, 65], "text": "鎼滅储", "confidence": 0.9}],
                    "vision_candidates": [
                        {
                            "candidate_id": "vision_send",
                            "bbox": [824, 598, 857, 621],
                            "kind": "send_button",
                            "source": "omniparser",
                            "confidence": 0.8,
                            "text": "发送",
                        }
                    ],
                },
                "provider_trace": {
                    "uia_used": False,
                    "ocr_used": True,
                    "vision_used": True,
                    "provider_details": {
                        "vision": {
                            "provider": "omniparser",
                            "success": True,
                            "candidate_count": 1,
                            "layout_region_count": 0,
                            "control_group_count": 0,
                        },
                        "ocr": {
                            "provider": "paddleocr_bridge",
                            "success": True,
                            "block_count": 1,
                        },
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    payload = inspector.build_openclaw_payload(
        snapshot_path=snapshot_path,
        task="鍦ㄥ井淇′腑鎼滅储瀛欏畤",
    )["openclaw_payload"]

    assert payload["candidate_summary"]["count"] == len(payload["candidates"])
    assert payload["candidate_summary"]["by_source"]["omniparser"] == 1
    assert payload["provider_health"]["vision_provider"]["success"] is True
    assert payload["provider_health"]["ocr_provider"]["provider"] == "paddleocr_bridge"
    assert payload["provider_health"]["boundary_candidates"]["count"] == len(payload["candidates"])


def test_export_real_app_regressions_writes_summary(tmp_path: Path):
    snapshot_path = tmp_path / "wechat_snapshot.json"
    provider_path = tmp_path / "wechat_provider.json"
    image_path = tmp_path / "wechat_preview.png"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_reg_batch",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 400, 300]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [{"region_id": "r1", "role": "filter_bar", "bounds": [50, 20, 220, 60]}],
                "elements": [],
                "artifacts": {
                    "ocr_blocks": [{"bbox": [60, 25, 130, 45], "text": "鎼滅储", "confidence": 0.95}],
                    "vision_candidates": [
                        {
                            "candidate_id": "vision_send",
                            "bbox": [300, 240, 360, 270],
                            "kind": "send_button",
                            "source": "omniparser",
                            "confidence": 0.8,
                            "text": "发送",
                        }
                    ],
                },
                "provider_trace": {
                    "uia_used": False,
                    "ocr_used": True,
                    "vision_used": True,
                    "provider_details": {
                        "vision": {"provider": "omniparser", "success": True, "candidate_count": 1},
                        "ocr": {"provider": "paddleocr_bridge", "success": True, "block_count": 1},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider_path.write_text(
        json.dumps(
            {
                "details": {
                    "vision_provider": {"provider": "omniparser", "success": True, "candidate_count": 1}
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(245, 245, 245)).save(image_path)

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.export_real_app_regressions(
        baseline_dir=tmp_path,
        task="分析当前页面中的可交互候选",
    )

    assert result["snapshot_count"] == 1
    assert Path(result["summary_path"]).exists()
    entry = result["summary"][0]
    assert entry["app"] == "WeChat"
    assert entry["provider_health"]["vision_provider"]["success"] is True
    assert entry["candidate_summary"]["by_source"]["omniparser"] == 1
    assert "no_omniparser_candidates_retained" not in entry["issues"]


def test_build_openclaw_payload_hydrates_sidecar_vision_and_provider_files(tmp_path: Path):
    snapshot_path = tmp_path / "wechat_20260408_135208_snapshot.json"
    vision_path = tmp_path / "wechat_20260408_135208_vision.json"
    provider_path = tmp_path / "wechat_20260408_135208_provider.json"
    preview_path = tmp_path / "wechat_preview.png"

    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_sidecar_1",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 400, 300]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [{"region_id": "r1", "role": "action_bar", "bounds": [280, 220, 390, 290]}],
                "elements": [],
                "artifacts": {"ocr_blocks": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vision_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "vision_send",
                    "bbox": [300, 240, 360, 270],
                    "kind": "send_button",
                    "source": "omniparser",
                    "confidence": 0.8,
                    "text": "发送",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider_path.write_text(
        json.dumps(
            {
                "vision_used": True,
                "details": {
                    "vision_provider": {
                        "provider": "omniparser",
                        "success": True,
                        "candidate_count": 1,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(245, 245, 245)).save(preview_path)

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    payload_result = inspector.build_openclaw_payload(
        snapshot_path=snapshot_path,
        task="鍦ㄥ井淇′腑鎼滅储瀛欏畤",
    )
    regression_result = inspector.export_candidate_regression(
        snapshot_path=snapshot_path,
        task="鍦ㄥ井淇′腑鎼滅储瀛欏畤",
        output_dir=tmp_path,
    )

    payload = payload_result["openclaw_payload"]
    assert any(item["candidate_id"] == "vision_send" for item in payload["candidates"])
    assert payload["provider_health"]["vision_provider"]["success"] is True
    assert regression_result["overlay_path"] is not None


def test_build_openclaw_payload_retains_omniparser_candidates_when_truncated():
    snapshot = {
        "canvas_id": "snap_rank_1",
        "app": {"process_name": "WeChat.exe"},
        "window": {"rect_client": [0, 0, 896, 648]},
        "surface": {"surface_type": "electron_webview"},
        "page": {"page_class": "wechat/app_chat/main/wide"},
        "regions": [
            {"region_id": "r_filter", "role": "filter_bar", "bounds": [40, 30, 260, 72]},
            {"region_id": "r_main", "role": "content_area", "bounds": [260, 30, 896, 648]},
        ],
        "elements": [
            {
                "element_id": f"uia_{index}",
                "bounds": [30 + index * 10, 100, 110 + index * 10, 132],
                "provider_sources": ["uia"],
                "semantic_role": "button",
                "control_type": "ButtonControl",
                "region_id": "r_main",
                "attributes": {},
            }
            for index in range(8)
        ],
        "artifacts": {
            "vision_candidates": [
                {
                    "candidate_id": "vision_send",
                    "bbox": [820, 590, 860, 620],
                    "kind": "send_button",
                    "source": "omniparser",
                    "confidence": 0.85,
                    "text": "发送",
                }
            ]
        },
        "provider_trace": {
            "uia_used": True,
            "ocr_used": False,
            "vision_used": True,
            "provider_details": {
                "vision_provider": {
                    "provider": "omniparser",
                    "success": True,
                    "candidate_count": 1,
                }
            },
        },
    }

    payload = build_openclaw_payload(snapshot, task="发送一条消息", max_candidates=2)

    assert len(payload["candidates"]) == 2
    assert any(item["candidate_id"] == "vision_send" for item in payload["candidates"])
    assert payload["candidate_summary"]["omniparser_retained_count"] == 1
    assert payload["provider_health"]["omniparser_retained_count"] == 1


def test_build_openclaw_payload_hydrates_dict_vision_sidecar(tmp_path: Path):
    snapshot_path = tmp_path / "clash_vision_snapshot.json"
    vision_path = tmp_path / "clash_vision_vision.json"
    provider_path = tmp_path / "clash_vision_provider.json"
    preview_path = tmp_path / "clash_preview.png"

    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_sidecar_dict",
                "app": {"process_name": "Clash.exe"},
                "window": {"rect_client": [0, 0, 400, 300]},
                "surface": {"surface_type": "native_uia"},
                "page": {"page_class": "clash/main/wide"},
                "regions": [{"region_id": "r1", "role": "viewport", "bounds": [0, 0, 400, 300]}],
                "elements": [],
                "artifacts": {"ocr_blocks": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vision_path.write_text(
        json.dumps(
            {
                "vision_candidate_count": 1,
                "vision_layout_region_count": 1,
                "vision_control_group_count": 0,
                "vision_candidates": [
                    {
                        "candidate_id": "vision_node",
                        "bbox": [40, 60, 180, 110],
                        "kind": "list_item",
                        "source": "omniparser",
                        "confidence": 0.82,
                        "text": "鑺傜偣 A",
                    }
                ],
                "vision_layout_regions": [
                    {
                        "region_id": "vision_region::node_panel",
                        "bbox": [24, 48, 220, 220],
                        "role": "list_panel",
                        "confidence": 0.7,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider_path.write_text(
        json.dumps(
            {
                "vision_used": True,
                "details": {
                    "vision_provider": {
                        "provider": "omniparser",
                        "success": True,
                        "candidate_count": 1,
                        "layout_region_count": 1,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(240, 240, 240)).save(preview_path)

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    payload = inspector.build_openclaw_payload(
        snapshot_path=snapshot_path,
        task="查看节点候选",
    )["openclaw_payload"]

    assert any(item["candidate_id"] == "vision_node" for item in payload["candidates"])
    assert payload["provider_health"]["vision_provider"]["layout_region_count"] == 1
    assert payload["provider_health"]["omniparser_total_count"] >= 1


def test_export_real_app_regressions_marks_sidecar_only_when_omniparser_is_unavailable(tmp_path: Path):
    snapshot_path = tmp_path / "cc_switch_snapshot.json"
    provider_path = tmp_path / "cc_switch_provider.json"
    vision_path = tmp_path / "cc_switch_vision.json"
    image_path = tmp_path / "cc_switch_preview.png"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_sidecar_only",
                "app": {"process_name": "cc-switch.exe"},
                "window": {"rect_client": [0, 0, 360, 240]},
                "surface": {"surface_type": "native_uia"},
                "page": {"page_class": "ccswitch/main/wide"},
                "regions": [{"region_id": "r1", "role": "content_area", "bounds": [0, 0, 360, 240]}],
                "elements": [{"element_id": "uia_root", "bounds": [0, 0, 360, 240], "provider_sources": ["uia"], "semantic_role": "pane", "control_type": "PaneControl", "region_id": "r1", "attributes": {}}],
                "artifacts": {"ocr_blocks": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider_path.write_text(
        json.dumps(
            {
                "vision_used": True,
                "details": {
                    "vision_provider": {
                        "provider": "omniparser",
                        "success": False,
                        "error": "connection refused",
                        "candidate_count": 0,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vision_path.write_text(
        json.dumps(
            {
                "vision_candidate_count": 2,
                "vision_candidates": [
                    {
                        "candidate_id": "vision_ocr_1",
                        "bbox": [20, 20, 120, 40],
                        "kind": "ocr_text_candidate",
                        "source": "paddleocr_bridge",
                        "confidence": 0.8,
                        "text": "CC Switch",
                    },
                    {
                        "candidate_id": "vision_ocr_2",
                        "bbox": [20, 60, 120, 80],
                        "kind": "ocr_text_candidate",
                        "source": "paddleocr_bridge",
                        "confidence": 0.8,
                        "text": "MiniMax",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (360, 240), color=(245, 245, 245)).save(image_path)

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=MagicMock(),
        ocr_service=MagicMock(),
    )
    result = inspector.export_real_app_regressions(
        baseline_dir=tmp_path,
        task="分析当前页面中的可交互候选",
    )

    issues = result["summary"][0]["issues"]
    assert "omniparser_provider_unavailable" in issues
    assert "vision_sidecar_only" in issues
    assert "no_omniparser_candidates_retained" in issues


def test_export_candidate_regression_refreshes_live_ocr_and_vision_from_image(tmp_path: Path):
    snapshot_path = tmp_path / "wechat_live_snapshot.json"
    image_path = tmp_path / "wechat_preview.png"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_live_refresh",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 400, 300]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [{"region_id": "r1", "role": "filter_bar", "bounds": [40, 20, 220, 60]}],
                "elements": [],
                "artifacts": {"ocr_blocks": []},
                "provider_trace": {
                    "uia_used": False,
                    "ocr_used": False,
                    "vision_used": False,
                    "provider_details": {
                        "vision": {"used": False, "candidate_count": 0},
                        "ocr": {"used": False, "block_count": 0},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(245, 245, 245)).save(image_path)

    perception_service = MagicMock()
    perception_service._vision_provider = MagicMock()
    perception_service._vision_provider.parse_screenshot.return_value = type(
        "VisionResult",
        (),
        {
            "provider": "omniparser",
            "success": True,
            "error": None,
            "visual_score": 0.74,
            "layout_regions": [{"region_id": "vr1", "role": "list_panel", "bbox": [20, 50, 180, 260]}],
            "control_groups": [{"group_id": "cg1", "member_ids": ["vision_item_1"]}],
            "interaction_hints": [{"candidate_id": "vision_item_1", "preferred_action": "click"}],
            "structure_evidence_score": 0.68,
            "candidates": [
                type(
                    "Candidate",
                    (),
                    {
                        "element_id": "vision_item_1",
                        "bounding_box": (32, 82, 188, 126),
                        "semantic_label": "list_item",
                        "confidence": 0.83,
                        "text": "孙宇",
                        "icon_type": None,
                        "region_role": "list_panel",
                        "group_id": "cg1",
                        "interaction_hints": {"preferred_action": "click"},
                        "structure_evidence_score": 0.68,
                        "attributes": {"source": "omniparser"},
                    },
                )()
            ],
        },
    )()

    ocr_service = MagicMock()
    ocr_service.extract_with_metadata.return_value = type(
        "OCRResult",
        (),
        {
            "provider": "paddleocr_bridge",
            "success": True,
            "error": None,
            "elapsed_seconds": 0.21,
            "used_region": None,
            "worker_command": ["python", "worker.py"],
            "blocks": [
                type("Block", (), {"bbox": (44, 24, 116, 48), "text": "搜索", "confidence": 0.93})(),
            ],
        },
    )()

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=perception_service,
        ocr_service=ocr_service,
    )
    result = inspector.export_candidate_regression(
        snapshot_path=snapshot_path,
        task="在微信里搜索孙宇",
        output_dir=tmp_path,
        image_path=image_path,
    )
    payload = json.loads(Path(result["payload_path"]).read_text(encoding="utf-8"))

    assert payload["provider_health"]["vision_used"] is True
    assert payload["provider_health"]["ocr_used"] is True
    assert payload["provider_health"]["vision_provider"]["candidate_count"] == 1
    assert payload["provider_health"]["ocr_provider"]["block_count"] == 1
    assert payload["provider_health"]["omniparser_total_count"] >= 1
    assert any(item["candidate_id"] == "vision_item_1" for item in payload["candidates"])


def test_export_real_app_regressions_ignores_stale_element_sparse_when_live_candidates_are_rich(tmp_path: Path):
    snapshot_path = tmp_path / "wechat_snapshot.json"
    image_path = tmp_path / "wechat_preview.png"
    snapshot_path.write_text(
        json.dumps(
            {
                "canvas_id": "snap_live_rich",
                "app": {"process_name": "WeChat.exe"},
                "window": {"rect_client": [0, 0, 400, 300]},
                "surface": {"surface_type": "electron_webview"},
                "page": {"page_class": "wechat/app_chat/main/wide"},
                "regions": [{"region_id": "r1", "role": "content_area", "bounds": [0, 0, 400, 300]}],
                "elements": [{"element_id": "uia_root", "bounds": [0, 0, 400, 300], "provider_sources": ["uia"], "semantic_role": "pane", "control_type": "PaneControl", "region_id": "r1", "attributes": {}}],
                "artifacts": {"ocr_blocks": []},
                "provider_trace": {"uia_used": True, "ocr_used": False, "vision_used": False, "provider_details": {}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (400, 300), color=(240, 240, 240)).save(image_path)

    candidates = []
    for index in range(16):
        candidates.append(
            type(
                "Candidate",
                (),
                {
                    "element_id": f"vision_{index}",
                    "bounding_box": (20 + index * 5, 40 + index * 5, 80 + index * 5, 72 + index * 5),
                    "semantic_label": "icon_button" if index % 2 == 0 else "list_item",
                    "confidence": 0.72,
                    "text": f"item_{index}",
                    "icon_type": None,
                    "region_role": "content_area",
                    "group_id": None,
                    "interaction_hints": {},
                    "structure_evidence_score": 0.6,
                    "attributes": {"source": "omniparser"},
                },
            )()
        )

    perception_service = MagicMock()
    perception_service._vision_provider = MagicMock()
    perception_service._vision_provider.parse_screenshot.return_value = type(
        "VisionResult",
        (),
        {
            "provider": "omniparser",
            "success": True,
            "error": None,
            "visual_score": 0.8,
            "layout_regions": [],
            "control_groups": [],
            "interaction_hints": [],
            "structure_evidence_score": 0.7,
            "candidates": candidates,
        },
    )()

    ocr_service = MagicMock()
    ocr_service.extract_with_metadata.return_value = type(
        "OCRResult",
        (),
        {
            "provider": "paddleocr_bridge",
            "success": True,
            "error": None,
            "elapsed_seconds": 0.2,
            "used_region": None,
            "worker_command": [],
            "blocks": [],
        },
    )()

    inspector = InspectorService(
        window_enum_service=MagicMock(),
        screenshot_service=MagicMock(),
        perception_service=perception_service,
        ocr_service=ocr_service,
    )
    result = inspector.export_real_app_regressions(
        baseline_dir=tmp_path,
        task="分析当前页面中的可交互候选",
    )

    entry = result["summary"][0]
    assert entry["provider_health"]["omniparser_total_count"] >= 16
    assert "element_count_sparse" not in entry["issues"]


