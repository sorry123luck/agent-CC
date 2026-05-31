"""Tests for the live sample matrix collection script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "collect_live_sample_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_live_sample_matrix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_window_filter_keeps_visible_third_party_apps():
    module = _load_module()
    windows = [
        {"hwnd": 1, "title": "微信", "process_name": "weixin.exe", "is_minimized": False},
        {"hwnd": 2, "title": "设置", "process_name": "systemsettings.exe", "is_minimized": False},
        {"hwnd": 3, "title": "Chrome", "process_name": "chrome.exe", "is_minimized": True},
        {"hwnd": 4, "title": "Codex", "process_name": "codex.exe", "is_minimized": False},
    ]

    selected = module.select_sample_windows(windows, include_processes=[], max_windows=10)

    assert [item["hwnd"] for item in selected] == [1, 4]


def test_collect_matrix_excludes_tiny_no_screenshot_windows(tmp_path, monkeypatch):
    module = _load_module()

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class Session:
        def get(self, url, timeout=None):
            if url.endswith("/api/v1/windows"):
                return Response([
                    {"hwnd": 1, "title": "QQ", "process_name": "qq.exe", "is_minimized": False},
                    {"hwnd": 2, "title": "微信", "process_name": "weixin.exe", "is_minimized": False},
                ])
            if url.endswith("/snap_tiny"):
                return Response({
                    "canvas_id": "snap_tiny",
                    "elements": [{"element_id": "merged_1", "bounds": [0, 0, 64, 64]}],
                    "provider_trace": {
                        "provider_details": {
                            "screenshot_provider": {"success": False, "error": "capture_invalid"}
                        }
                    },
                    "visual_pattern": {"mode": "account_switcher"},
                })
            return Response({
                "canvas_id": "snap_chat",
                "elements": [{"element_id": "merged_1", "bounds": [0, 0, 960, 640]}],
                "provider_trace": {
                    "provider_details": {
                        "screenshot_provider": {"success": True}
                    }
                },
                "roi_selection_plan": {"mode": "chat_workspace", "rois": []},
            })

        def post(self, url, json=None, timeout=None):
            if url.endswith("/api/v1/observe"):
                canvas_id = "snap_tiny" if json["hwnd"] == 1 else "snap_chat"
                return Response({"canvas_id": canvas_id, "elapsed_ms": 1})
            if url.endswith("/semantic-completion"):
                return Response({"status": "dry_run", "stages": [], "roi_vlm": {"jobs": []}})
            raise AssertionError(url)

    monkeypatch.setattr(module.requests, "Session", lambda: Session())

    summary = module.collect_matrix(
        base_url="http://api",
        output_dir=tmp_path,
        include_processes=["qq.exe", "weixin.exe"],
        max_windows=8,
        run_vlm=False,
        max_rois_per_window=1,
        wait_late_seconds=0,
        wait_enhance_seconds=0,
        deadline_ms=2000,
        clean_output=False,
        async_enhance=False,
    )

    assert [row["sample"] for row in summary["rows"]] == ["weixin.exe_2"]
    assert summary["excluded_count"] == 1
    assert summary["excluded_rows"][0]["sample"] == "qq.exe_1_QQ"
    assert summary["excluded_rows"][0]["excluded_reason"] == "invalid_tiny_window_no_screenshot"


def test_row_from_results_records_semantic_completion_metrics():
    module = _load_module()
    row = module.build_matrix_row(
        window={"hwnd": 7, "title": "微信", "process_name": "weixin.exe"},
        observe={
            "canvas_id": "snap_1",
            "elapsed_ms": 430,
            "processing_state": "local_ready",
            "element_count": 12,
            "region_count": 2,
            "geometric_region_count": 18,
            "perception_quality": {"warnings": ["unknown_role_heavy"], "unknown_role_ratio": 0.5},
            "roi_selection_plan": {"mode": "chat_workspace", "rois": [{"roi_id": "roi_0"}]},
        },
        semantic={
            "status": "timeout",
            "elapsed_ms": 1900,
            "next_action": "accept_late_roi_results",
            "stages": [
                {"name": "layout_audit", "status": "skipped"},
                {"name": "roi_vlm", "status": "timeout", "job_count": 1},
            ],
            "roi_vlm": {
                "result_status_counts": {"timeout": 1},
                "jobs": [
                    {
                        "roi_id": "roi_0",
                        "purpose": "composer",
                        "candidate_ids": ["icon_1", "icon_2"],
                        "candidate_refs": [
                            {"candidate_id": "icon_1", "local_hint": "emoji_button", "bounds": [320, 600, 350, 630]},
                            {"candidate_id": "icon_2", "local_hint": "send_button", "bounds": [900, 600, 960, 630]},
                        ],
                        "local_text_candidate_ids": ["text_1"],
                    }
                ],
            },
        },
        detail={
            "processing_state": "semantic_late_merged",
            "provider_trace": {
                "provider_details": {
                    "screenshot_provider": {
                        "success": True,
                        "size": [1002, 731],
                    }
                }
            },
            "roi_selection_plan": {
                "mode": "chat_workspace",
                "rois": [
                    {"purpose": "composer", "bounds": [300, 580, 960, 640]},
                ],
            },
            "roi_vlm_semantic_supplements": [{"status": "timeout_late_success"}],
            "elements": [
                {
                    "element_id": "synthetic_input",
                    "role_label": "message_input",
                    "bounds": [320, 590, 900, 630],
                    "click_point": [340, 610],
                    "actionability": "review",
                    "attributes": {"needs_manual_label": True, "safe_to_type": False},
                }
            ],
        },
        error="",
    )

    assert row["canvas_id"] == "snap_1"
    assert row["processing_state"] == "semantic_late_merged"
    assert row["mode"] == "chat_workspace"
    assert row["chat_variant"] == ""
    assert row["chat_variant_source"] == ""
    assert row["semantic_status"] == "timeout"
    assert row["semantic_stage_statuses"] == ["layout_audit:skipped", "roi_vlm:timeout"]
    assert row["roi_vlm_result_status_counts"] == {"timeout": 1}
    assert row["late_supplement_count"] == 1
    assert row["roi_vlm_job_count"] == 1
    assert row["roi_vlm_candidate_count"] == 2
    assert row["composer_vlm_candidate_count"] == 2
    assert row["composer_vlm_hint_labels"] == ["emoji_button", "send_button"]
    assert row["composer_vlm_send_hint_count"] == 1
    assert row["composer_send_target_candidate_id"] == "icon_2"
    assert row["composer_send_target_bounds"] == [900, 600, 960, 630]
    assert row["composer_send_target_width"] == 60
    assert row["composer_send_target_height"] == 30
    assert row["roi_vlm_local_text_candidate_count"] == 1
    assert row["roi_vlm_max_candidates_per_job"] == 2
    assert row["roi_vlm_avg_candidates_per_job"] == 2.0
    assert row["composer_input_candidate_count"] == 1
    assert row["composer_input_safe_count"] == 0
    assert row["composer_input_review_count"] == 1
    assert row["composer_input_review_bounds"] == [[320, 590, 900, 630]]
    assert row["composer_input_review_click_points"] == [[340, 610]]
    assert row["composer_input_primary_click_point"] == [340, 610]
    assert row["search_candidate_count"] == 0
    assert row["composer_input_review_in_composer_count"] == 1
    assert row["composer_input_review_min_width"] == 580
    assert row["composer_input_review_min_height"] == 40
    assert row["screenshot_success"] is True
    assert row["screenshot_error"] == ""


def test_row_marks_root_only_detail_as_quality_warning():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 9, "title": "飞书", "process_name": "feishu.exe"},
        observe={"canvas_id": "snap_root", "elapsed_ms": 500},
        semantic={},
        detail={
            "canvas_id": "snap_root",
            "provider_trace": {
                "provider_details": {
                    "screenshot_provider": {"success": True}
                }
            },
            "roi_selection_plan": {"mode": "collaboration_browse_page", "rois": []},
            "elements": [
                {
                    "element_id": "root",
                    "control_type": "PaneControl",
                    "semantic_role": "layout",
                    "bounds": [0, 0, 1000, 760],
                }
            ],
        },
        error="",
    )

    assert "root_only_detail" in row["quality_warnings"]


def test_row_records_narrow_composer_send_target():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 8, "title": "飞书", "process_name": "feishu.exe"},
        observe={"canvas_id": "snap_2", "roi_selection_plan": {"mode": "collaboration_inbox", "rois": []}},
        semantic={
            "roi_vlm": {
                "jobs": [
                    {
                        "purpose": "composer",
                        "candidate_ids": ["send_dropdown"],
                        "candidate_refs": [
                            {
                                "candidate_id": "send_dropdown",
                                "local_hint": "send_button",
                                "bounds": [958, 704, 982, 732],
                            }
                        ],
                    }
                ]
            }
        },
        detail={},
        error="",
    )

    assert row["composer_send_target_candidate_id"] == "send_dropdown"
    assert row["composer_send_target_bounds"] == [958, 704, 982, 732]
    assert row["composer_send_target_width"] == 24
    assert row["composer_send_target_height"] == 28


def test_row_prefers_local_text_send_button_parent_over_narrow_icon_hint():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 9, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={
            "roi_vlm": {
                "jobs": [
                    {
                        "purpose": "composer",
                        "candidate_ids": ["send_icon"],
                        "candidate_refs": [
                            {
                                "candidate_id": "send_icon",
                                "local_hint": "send_button",
                                "bounds": [918, 606, 932, 621],
                            }
                        ],
                        "local_text_candidate_ids": ["send_parent"],
                    }
                ]
            }
        },
        detail={
            "roi_selection_plan": {
                "mode": "chat_workspace",
                "rois": [{"purpose": "composer", "bounds": [172, 433, 960, 640]}],
            },
            "elements": [
                {
                    "element_id": "send_parent",
                    "semantic_role": "send_button",
                    "text": "发送",
                    "bounds": [845, 599, 909, 625],
                }
            ],
        },
        error="",
    )

    assert row["composer_send_target_candidate_id"] == "send_parent"
    assert row["composer_send_target_bounds"] == [845, 599, 909, 625]
    assert row["composer_send_target_width"] == 64
    assert row["composer_send_target_height"] == 26


def test_row_derives_review_chat_input_click_point_from_bounds():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 8, "title": "微信", "process_name": "weixin.exe"},
        observe={"canvas_id": "snap_input", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={},
        detail={
            "roi_selection_plan": {
                "mode": "chat_workspace",
                "rois": [{"purpose": "composer", "bounds": [300, 560, 980, 720]}],
            },
            "elements": [
                {
                    "element_id": "synthetic_chat_composer_input",
                    "role_label": "message_input",
                    "bounds": [320, 590, 900, 630],
                    "actionability": "review",
                    "attributes": {"needs_manual_label": True, "safe_to_type": False},
                }
            ],
        },
        error="",
    )

    assert row["composer_input_review_click_points"] == [[610, 610]]
    assert row["composer_input_primary_click_point"] == [610, 610]


def test_row_records_feishu_search_target_from_ocr():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 12, "title": "飞书", "process_name": "feishu.exe"},
        observe={"canvas_id": "snap_feishu", "roi_selection_plan": {"mode": "collaboration_inbox", "rois": []}},
        semantic={},
        detail={"ocr_blocks": [{"text": "搜索（Ctrl+K）", "bbox": [37, 64, 118, 81]}]},
        error="",
    )

    assert row["search_candidate_count"] == 1
    assert row["search_source"] == "ocr"
    assert row["search_primary_click_point"] == [94, 72]
    assert row["search_bounds"] == [15, 54, 174, 91]


def test_row_records_wechat_search_target_from_sidebar_geometry():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 13, "title": "微信", "process_name": "weixin.exe"},
        observe={"canvas_id": "snap_wechat", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={},
        detail={
            "elements": [
                {"element_id": "search_icon", "bounds": [77, 43, 99, 69]},
                {"element_id": "plus", "bounds": [258, 38, 292, 76]},
            ]
        },
        error="",
    )

    assert row["search_candidate_count"] == 1
    assert row["search_source"] == "sidebar_geometry"
    assert row["search_primary_click_point"] == [158, 55]


def test_row_records_qq_search_target_from_sidebar_geometry():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 14, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={},
        detail={
            "elements": [
                {"element_id": "search_icon", "bounds": [77, 28, 101, 57]},
                {"element_id": "plus", "bounds": [267, 28, 295, 57]},
            ]
        },
        error="",
    )

    assert row["search_candidate_count"] == 1
    assert row["search_source"] == "sidebar_geometry"
    assert row["search_primary_click_point"] == [163, 42]


def test_row_records_qq_search_target_from_large_search_box_geometry():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 15, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={},
        detail={
            "elements": [
                {"element_id": "search_box", "bounds": [73, 27, 262, 58]},
                {"element_id": "plus", "bounds": [267, 28, 296, 57]},
            ]
        },
        error="",
    )

    assert row["search_candidate_count"] == 1
    assert row["search_source"] == "sidebar_geometry"
    assert row["search_primary_click_point"] == [167, 42]


def test_row_records_feishu_search_target_from_top_bar_geometry_when_ocr_missing():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 16, "title": "飞书", "process_name": "feishu.exe"},
        observe={"canvas_id": "snap_feishu", "roi_selection_plan": {"mode": "collaboration_browse_page", "rois": []}},
        semantic={},
        detail={"screenshot_width": 1018, "screenshot_height": 768, "elements": []},
        error="",
    )

    assert row["search_candidate_count"] == 1
    assert row["search_source"] == "app_default_geometry"
    assert row["search_primary_click_point"] == [92, 73]


def test_row_records_qq_private_chat_variant_from_visual_pattern_evidence():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 9, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq"},
        semantic={},
        detail={
            "visual_pattern": {
                "mode": "chat_workspace",
                "evidence": ["app_hint:qq", "qq_variant:private_chat"],
            },
            "roi_selection_plan": {"mode": "chat_workspace", "rois": []},
        },
        error="",
    )

    assert row["mode"] == "chat_workspace"
    assert row["chat_variant"] == "private_chat"
    assert row["chat_variant_source"] == "visual_pattern"


def test_row_infers_qq_group_chat_variant_from_detail_elements():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 10, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq_group"},
        semantic={},
        detail={
            "screenshot_width": 960,
            "visual_pattern": {"mode": "chat_workspace", "evidence": ["app_hint:qq"]},
            "roi_selection_plan": {"mode": "chat_workspace", "rois": []},
            "elements": [
                {"text": "项目交流群", "bounds": [330, 30, 430, 54]},
                {"text": "群成员", "bounds": [760, 82, 900, 112]},
            ],
        },
        error="",
    )

    assert row["chat_variant"] == "group_chat"
    assert row["chat_variant_source"] == "local_text"


def test_row_infers_qq_group_chat_variant_from_right_member_panel_geometry():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 10, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq_group"},
        semantic={},
        detail={
            "screenshot_width": 960,
            "visual_pattern": {
                "mode": "chat_workspace",
                "evidence": ["app_hint:qq", "qq_variant:unknown"],
            },
            "roi_selection_plan": {"mode": "chat_workspace", "rois": []},
            "elements": [
                {"element_id": "member_1", "bounds": [785, 116, 960, 149]},
                {"element_id": "member_2", "bounds": [784, 150, 960, 181]},
                {"element_id": "member_3", "bounds": [787, 184, 960, 209]},
            ],
            "roi_vlm_semantic_supplements": [
                {
                    "roi_id": "roi_1",
                    "region_semantics": {
                        "role": "message_stream",
                        "summary": "QQ chat message area showing private chats",
                    },
                }
            ],
        },
        error="",
    )

    assert row["chat_variant"] == "group_chat"
    assert row["chat_variant_source"] == "local_geometry"
    assert row["chat_variant_evidence"] == "right_member_list_like"


def test_row_infers_qq_chat_variant_from_roi_vlm_region_semantics():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 11, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq_vlm"},
        semantic={},
        detail={
            "visual_pattern": {
                "mode": "chat_workspace",
                "evidence": ["app_hint:qq", "qq_variant:unknown"],
            },
            "roi_selection_plan": {"mode": "chat_workspace", "rois": []},
            "roi_vlm_semantic_supplements": [
                {
                    "roi_id": "roi_1",
                    "region_semantics": {"role": "message_stream", "summary": "QQ chat with private messages"},
                    "review_only_hints": [],
                }
            ],
        },
        error="",
    )

    assert row["chat_variant"] == "private_chat"
    assert row["chat_variant_source"] == "roi_vlm"
    assert "private messages" in row["chat_variant_evidence"]


def test_row_records_screenshot_capture_failure_reason():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 7, "title": "QQ", "process_name": "qq.exe"},
        observe={"canvas_id": "snap_qq"},
        semantic={},
        detail={
            "provider_trace": {
                "provider_details": {
                    "screenshot_provider": {
                        "success": False,
                        "error": "窗口截图失败: capture_invalid",
                    }
                }
            }
        },
        error="",
    )

    assert row["screenshot_success"] is False
    assert row["screenshot_error"] == "窗口截图失败: capture_invalid"


def test_collaboration_input_quality_accepts_detail_pane_bottom_candidate():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 8, "title": "飞书", "process_name": "feishu.exe"},
        observe={"canvas_id": "snap_2", "roi_selection_plan": {"mode": "collaboration_inbox", "rois": []}},
        semantic={"status": "dry_run"},
        detail={
            "roi_selection_plan": {
                "mode": "collaboration_inbox",
                "rois": [
                    {"purpose": "app_rail", "bounds": [0, 0, 80, 720]},
                    {"purpose": "inbox_list", "bounds": [80, 0, 360, 720]},
                    {"purpose": "detail_pane", "bounds": [360, 0, 1024, 720]},
                ],
            },
            "elements": [
                {
                    "element_id": "synthetic_feishu_input",
                    "role_label": "message_input",
                    "bounds": [420, 620, 980, 700],
                    "actionability": "review",
                    "attributes": {"needs_manual_label": True, "safe_to_type": False},
                }
            ],
        },
        error="",
    )

    assert row["composer_input_review_bounds"] == [[420, 620, 980, 700]]
    assert row["composer_input_review_in_composer_count"] == 1


def test_input_quality_accepts_candidate_with_major_composer_overlap():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 9, "title": "微信", "process_name": "weixin.exe"},
        observe={"canvas_id": "snap_3", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={"status": "dry_run"},
        detail={
            "roi_selection_plan": {
                "mode": "chat_workspace",
                "rois": [
                    {"purpose": "composer", "bounds": [180, 629, 1002, 731]},
                ],
            },
            "elements": [
                {
                    "element_id": "synthetic_chat_input",
                    "role_label": "message_input",
                    "bounds": [300, 615, 857, 713],
                    "actionability": "review",
                    "attributes": {"needs_manual_label": True, "safe_to_type": False},
                }
            ],
        },
        error="",
    )

    assert row["composer_input_review_in_composer_count"] == 1


def test_row_from_results_records_rejected_roi_vlm_diagnostics():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 7, "title": "微信", "process_name": "weixin.exe"},
        observe={"canvas_id": "snap_1", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={"status": "timeout"},
        detail={
            "roi_vlm_rejected_responses": [
                {"status": "timeout_late_success", "errors": ["unknown_roi_id"]}
            ]
        },
        error="",
    )

    assert row["roi_vlm_rejected_count"] == 1
    assert row["roi_vlm_rejected_errors"] == ["unknown_roi_id"]


def test_row_records_late_vlm_pending_count():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 7, "title": "微信", "process_name": "weixin.exe"},
        observe={"canvas_id": "snap_1", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={"status": "timeout"},
        detail={
            "roi_vlm_timeouts": [{"roi_id": "roi_0"}, {"roi_id": "roi_1"}, {"roi_id": "roi_2"}],
            "roi_vlm_semantic_supplements": [
                {"roi_id": "roi_0", "status": "timeout_late_success"}
            ],
            "roi_vlm_late_failures": [{"roi_id": "roi_1", "status": "timeout_late_failed"}],
            "roi_vlm_rejected_responses": [
                {"roi_id": "roi_2", "status": "timeout_late_rejected", "errors": ["bad_id"]}
            ],
        },
        error="",
    )

    assert row["roi_vlm_late_pending_count"] == 0

    pending = module.build_matrix_row(
        window={"hwnd": 7, "title": "微信", "process_name": "weixin.exe"},
        observe={"canvas_id": "snap_1", "roi_selection_plan": {"mode": "chat_workspace", "rois": []}},
        semantic={"status": "timeout"},
        detail={
            "roi_vlm_timeouts": [{"roi_id": "roi_0"}, {"roi_id": "roi_1"}, {"roi_id": "roi_2"}],
            "roi_vlm_semantic_supplements": [
                {"roi_id": "roi_0", "status": "timeout_late_success"}
            ],
        },
        error="",
    )

    assert pending["roi_vlm_late_pending_count"] == 2


def test_row_prefers_detail_quality_after_late_semantic_merge():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 7, "title": "火绒", "process_name": "hipsmain.exe"},
        observe={
            "canvas_id": "snap_1",
            "perception_quality": {
                "warnings": ["unknown_role_heavy"],
                "unknown_role_ratio": 0.95,
            },
            "roi_selection_plan": {"mode": "security_dashboard", "rois": []},
        },
        semantic={"status": "timeout"},
        detail={
            "perception_quality": {
                "warnings": ["coarse_regions"],
                "unknown_role_ratio": 0.2,
            },
            "roi_vlm_semantic_supplements": [{"status": "timeout_late_success"}],
        },
        error="",
    )

    assert row["unknown_role_ratio"] == 0.2
    assert row["quality_warnings"] == ["coarse_regions"]


def test_row_prefers_detail_counts_after_async_enhancement():
    module = _load_module()

    row = module.build_matrix_row(
        window={"hwnd": 7, "title": "飞书", "process_name": "feishu.exe"},
        observe={
            "canvas_id": "snap_1",
            "element_count": 1,
            "region_count": 2,
            "geometric_region_count": 3,
            "roi_selection_plan": {"mode": "collaboration_inbox", "rois": []},
        },
        semantic={"status": "dry_run"},
        detail={
            "elements": [{"id": idx} for idx in range(72)],
            "regions": [{"id": idx} for idx in range(4)],
            "geometric_regions": [{"id": idx} for idx in range(25)],
            "roi_selection_plan": {"mode": "collaboration_inbox", "rois": []},
        },
        error="",
    )

    assert row["element_count"] == 72
    assert row["region_count"] == 4
    assert row["geometric_region_count"] == 25


def test_write_matrix_outputs_persists_summary_files(tmp_path):
    module = _load_module()
    rows = [
        {"sample": "wechat", "semantic_status": "timeout", "mode": "chat_workspace", "error": ""},
        {"sample": "huorong", "semantic_status": "dry_run", "mode": "security_dashboard", "error": ""},
    ]
    summary = module.build_summary(rows)

    module.write_matrix_outputs(summary, tmp_path)

    saved = json.loads((tmp_path / "sample_matrix_summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "sample_matrix_summary.md").read_text(encoding="utf-8")
    assert saved["sample_matrix_schema_version"] == module.SAMPLE_MATRIX_SCHEMA_VERSION
    assert "Schema:" in report
    assert saved["sample_count"] == 2
    assert saved["excluded_count"] == 0
    assert saved["semantic_status_counts"] == {"timeout": 1, "dry_run": 1}
    assert saved["chat_variant_counts"] == {"none": 1}
    assert "Live Sample Matrix" in report
    assert "vlm_candidates" in report
    assert "input_safe/review" in report
    assert "input_click" in report
    assert "search_click" in report
    assert "| sample | process | mode | variant | variant_source | state |" in report


def test_write_matrix_outputs_lists_excluded_rows(tmp_path):
    module = _load_module()
    summary = module.build_summary([])
    summary["excluded_count"] = 1
    summary["excluded_rows"] = [
        {
            "sample": "qq.exe_1_QQ",
            "process_name": "qq.exe",
            "excluded_reason": "invalid_tiny_window_no_screenshot",
        }
    ]

    module.write_matrix_outputs(summary, tmp_path)

    report = (tmp_path / "sample_matrix_summary.md").read_text(encoding="utf-8")
    assert "Excluded Samples" in report
    assert "invalid_tiny_window_no_screenshot" in report


def test_write_detail_output_creates_output_directory(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "missing" / "matrix"

    module.write_detail_output(
        {"sample": "wechat"},
        {"canvas_id": "snap_1"},
        output_dir,
    )

    saved = json.loads((output_dir / "wechat.detail.json").read_text(encoding="utf-8"))
    assert saved["canvas_id"] == "snap_1"


def test_write_response_output_persists_stage_response(tmp_path):
    module = _load_module()

    module.write_response_output(
        {"sample": "wechat"},
        {"status": "timeout"},
        tmp_path,
        suffix="semantic",
    )

    saved = json.loads((tmp_path / "wechat.semantic.json").read_text(encoding="utf-8"))
    assert saved["status"] == "timeout"


def test_prepare_output_dir_clean_removes_stale_json_only(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    (output_dir / "old.detail.json").write_text("{}", encoding="utf-8")
    (output_dir / "keep.png").write_text("image", encoding="utf-8")

    module.prepare_output_dir(output_dir, clean=True)

    assert not (output_dir / "old.detail.json").exists()
    assert (output_dir / "keep.png").exists()


def test_collect_one_window_disables_async_enhance_by_default(tmp_path):
    module = _load_module()

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.observe_payload = None

        def post(self, url, json=None, timeout=None):
            if url.endswith("/api/v1/observe"):
                self.observe_payload = json
                return Response({"canvas_id": "snap_1", "elapsed_ms": 1})
            if url.endswith("/semantic-completion"):
                return Response({"status": "dry_run", "stages": [], "roi_vlm": {"jobs": []}})
            raise AssertionError(url)

        def get(self, url, timeout=None):
            return Response({"canvas_id": "snap_1"})

    session = Session()

    module._collect_one_window(
        session=session,
        base_url="http://api",
        output_dir=tmp_path,
        window={"hwnd": 1, "title": "微信", "process_name": "weixin.exe"},
        run_vlm=False,
        max_rois_per_window=1,
        wait_late_seconds=0,
        deadline_ms=2000,
        async_enhance=False,
        wait_enhance_seconds=0,
    )

    assert session.observe_payload["async_enhance"] is False


def test_collect_one_window_can_wait_for_async_enhancement(tmp_path):
    module = _load_module()

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.processing_calls = 0
            self.detail_calls = 0

        def post(self, url, json=None, timeout=None):
            if url.endswith("/api/v1/observe"):
                return Response({"canvas_id": "snap_1", "elapsed_ms": 1})
            if url.endswith("/semantic-completion"):
                return Response({"status": "dry_run", "stages": [], "roi_vlm": {"jobs": []}})
            raise AssertionError(url)

        def get(self, url, timeout=None):
            if url.endswith("/processing"):
                self.processing_calls += 1
                state = "fusion_running" if self.processing_calls == 1 else "enhanced_ready"
                return Response({"processing_state": state})
            self.detail_calls += 1
            return Response({"canvas_id": "snap_1", "detail_call": self.detail_calls})

    session = Session()

    row, _observe, _semantic, detail = module._collect_one_window(
        session=session,
        base_url="http://api",
        output_dir=tmp_path,
        window={"hwnd": 1, "title": "飞书", "process_name": "feishu.exe"},
        run_vlm=False,
        max_rois_per_window=1,
        wait_late_seconds=0,
        deadline_ms=2000,
        async_enhance=True,
        wait_enhance_seconds=1,
    )

    assert row["error"] == ""
    assert session.processing_calls == 2
    assert detail["detail_call"] == 2


def test_collect_one_window_polls_for_late_vlm_merge(tmp_path):
    module = _load_module()

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.processing_calls = 0
            self.detail_calls = 0

        def post(self, url, json=None, timeout=None):
            if url.endswith("/api/v1/observe"):
                return Response({"canvas_id": "snap_1", "elapsed_ms": 1})
            if url.endswith("/semantic-completion") and json.get("dry_run"):
                return Response(
                    {
                        "status": "dry_run",
                        "stages": [],
                        "roi_vlm": {"jobs": [{"roi_id": "roi_1", "candidate_ids": ["a"]}]},
                    }
                )
            if url.endswith("/semantic-completion"):
                return Response({"status": "timeout", "stages": [], "roi_vlm": {"jobs": []}})
            raise AssertionError(url)

        def get(self, url, timeout=None):
            if url.endswith("/processing"):
                self.processing_calls += 1
                state = "semantic_timeout" if self.processing_calls == 1 else "semantic_late_merged"
                return Response({"processing_state": state})
            self.detail_calls += 1
            return Response(
                {
                    "canvas_id": "snap_1",
                    "detail_call": self.detail_calls,
                    "roi_vlm_semantic_supplements": [{"status": "timeout_late_success"}],
                }
            )

    session = Session()

    row, _observe, _semantic, detail = module._collect_one_window(
        session=session,
        base_url="http://api",
        output_dir=tmp_path,
        window={"hwnd": 1, "title": "微信", "process_name": "weixin.exe"},
        run_vlm=True,
        max_rois_per_window=1,
        wait_late_seconds=1,
        deadline_ms=2000,
        async_enhance=False,
        wait_enhance_seconds=0,
    )

    assert row["late_supplement_count"] == 1
    assert session.processing_calls == 2
    assert detail["detail_call"] == 2


def test_collect_one_window_can_filter_vlm_rois_by_purpose(tmp_path):
    module = _load_module()

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.run_payload = None

        def post(self, url, json=None, timeout=None):
            if url.endswith("/api/v1/observe"):
                return Response({"canvas_id": "snap_1", "elapsed_ms": 1})
            if url.endswith("/semantic-completion") and json.get("dry_run"):
                return Response(
                    {
                        "status": "dry_run",
                        "stages": [],
                        "roi_vlm": {
                            "jobs": [
                                {"roi_id": "roi_nav", "purpose": "navigation_and_list"},
                                {"roi_id": "roi_msg", "purpose": "message_stream"},
                                {"roi_id": "roi_comp", "purpose": "composer"},
                            ]
                        },
                    }
                )
            if url.endswith("/semantic-completion"):
                self.run_payload = json
                return Response({"status": "success", "stages": [], "roi_vlm": {"jobs": []}})
            raise AssertionError(url)

        def get(self, url, timeout=None):
            return Response({"canvas_id": "snap_1"})

    session = Session()

    module._collect_one_window(
        session=session,
        base_url="http://api",
        output_dir=tmp_path,
        window={"hwnd": 1, "title": "微信", "process_name": "weixin.exe"},
        run_vlm=True,
        max_rois_per_window=3,
        wait_late_seconds=0,
        deadline_ms=2000,
        async_enhance=False,
        wait_enhance_seconds=0,
        roi_purposes=["composer"],
    )

    assert session.run_payload["roi_ids"] == ["roi_comp"]
