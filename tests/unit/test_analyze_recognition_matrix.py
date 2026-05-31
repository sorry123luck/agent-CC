"""Tests for recognition-chain sample matrix acceptance analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_recognition_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_recognition_matrix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_chat_workspace_requires_review_input_and_composer_roi():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "chat_variant": "private_chat",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_primary_click_point": [610, 610],
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["sample_class"] == "chat_workspace"
    assert verdict["chat_variant"] == "private_chat"
    assert verdict["status"] == "pass"
    assert verdict["failures"] == []


def test_chat_workspace_fails_when_input_is_safe_before_validation():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 1,
        "composer_input_review_count": 0,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "chat_input_must_remain_review_only" in verdict["failures"]


def test_chat_workspace_fails_without_composer_roi():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "missing_composer_roi" in verdict["failures"]


def test_chat_workspace_fails_when_review_input_is_outside_composer_roi():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 0,
        "composer_input_review_min_width": 240,
        "composer_input_review_min_height": 48,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "review_chat_input_outside_composer_roi" in verdict["failures"]


def test_chat_workspace_fails_when_review_input_is_too_narrow():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 64,
        "composer_input_review_min_height": 48,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "review_chat_input_too_narrow" in verdict["failures"]


def test_chat_workspace_fails_when_review_input_has_no_click_point():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 240,
        "composer_input_review_min_height": 48,
        "composer_input_primary_click_point": [],
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "missing_review_chat_input_click_point" in verdict["failures"]


def test_account_switcher_does_not_require_chat_input():
    module = _load_module()
    row = {
        "sample": "qq",
        "process_name": "qq.exe",
        "mode": "account_switcher",
        "roi_purposes": ["account_list", "login_actions", "window_controls"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 0,
        "observe_elapsed_ms": 301,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["sample_class"] == "account_switcher"
    assert verdict["status"] == "pass"


def test_qq_chat_workspace_warns_when_private_or_group_variant_is_unknown():
    module = _load_module()
    row = {
        "sample": "qq",
        "process_name": "qq.exe",
        "mode": "chat_workspace",
        "chat_variant": "unknown",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "observe_elapsed_ms": 301,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "warn"
    assert "qq_chat_variant_unknown" in verdict["warnings"]


def test_row_warns_when_screenshot_is_missing():
    module = _load_module()
    row = {
        "sample": "qq",
        "process_name": "qq.exe",
        "mode": "account_switcher",
        "roi_purposes": [],
        "quality_warnings": ["screenshot_missing", "sparse_elements"],
        "observe_elapsed_ms": 301,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "warn"
    assert "screenshot_missing" in verdict["warnings"]


def test_collaboration_inbox_requires_thread_and_composer_rois():
    module = _load_module()
    row = {
        "sample": "feishu",
        "process_name": "feishu.exe",
        "mode": "collaboration_inbox",
        "roi_purposes": ["app_rail", "inbox_list", "message_thread", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 420,
        "composer_input_review_min_height": 64,
        "observe_elapsed_ms": 430,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["sample_class"] == "collaboration_inbox"
    assert verdict["status"] == "pass"


def test_dense_control_matrix_warns_when_roi_candidate_budget_is_high():
    module = _load_module()
    row = {
        "sample": "voicemeeter",
        "process_name": "voicemeeter.exe",
        "mode": "control_matrix",
        "roi_purposes": ["left_control_columns", "middle_control_columns", "right_master_section"],
        "roi_vlm_max_candidates_per_job": 14,
        "observe_elapsed_ms": 980,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "warn"
    assert "dense_roi_candidate_budget_high" in verdict["warnings"]


def test_row_warns_when_late_vlm_results_are_pending():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 240,
        "composer_input_review_min_height": 48,
        "roi_vlm_late_pending_count": 2,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "warn"
    assert "roi_vlm_late_results_pending" in verdict["warnings"]


def test_chat_workspace_warns_when_composer_vlm_candidate_budget_is_high():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 240,
        "composer_input_review_min_height": 48,
        "composer_vlm_candidate_count": 7,
        "composer_vlm_send_hint_count": 1,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "warn"
    assert "composer_vlm_candidate_budget_high" in verdict["warnings"]


def test_chat_workspace_fails_when_composer_has_no_send_hint():
    module = _load_module()
    row = {
        "sample": "wechat",
        "process_name": "weixin.exe",
        "mode": "chat_workspace",
        "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 240,
        "composer_input_review_min_height": 48,
        "composer_vlm_candidate_count": 4,
        "composer_vlm_send_hint_count": 0,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "missing_composer_send_hint" in verdict["failures"]


def test_chat_workspace_fails_when_send_target_is_too_narrow():
    module = _load_module()
    row = {
        "sample": "feishu",
        "process_name": "feishu.exe",
        "mode": "collaboration_inbox",
        "roi_purposes": ["app_rail", "inbox_list", "message_thread", "composer"],
        "composer_input_candidate_count": 1,
        "composer_input_safe_count": 0,
        "composer_input_review_count": 1,
        "composer_input_review_in_composer_count": 1,
        "composer_input_review_min_width": 420,
        "composer_input_review_min_height": 64,
        "composer_vlm_candidate_count": 5,
        "composer_vlm_send_hint_count": 1,
        "composer_send_target_width": 24,
        "composer_send_target_height": 28,
        "observe_elapsed_ms": 724,
        "error": "",
    }

    verdict = module.evaluate_row(row)

    assert verdict["status"] == "fail"
    assert "composer_send_target_too_narrow" in verdict["failures"]


def test_root_only_detail_is_warning_not_pass():
    module = _load_module()

    verdict = module.evaluate_row(
        {
            "sample": "feishu_root",
            "process_name": "feishu.exe",
            "mode": "collaboration_browse_page",
            "quality_warnings": ["root_only_detail"],
        }
    )

    assert verdict["status"] == "warn"
    assert "root_only_detail" in verdict["warnings"]


def test_build_acceptance_report_counts_pass_warn_fail():
    module = _load_module()
    summary = {
        "sample_matrix_schema_version": module.ACCEPTED_SAMPLE_MATRIX_SCHEMA_VERSION,
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "mode": "chat_workspace",
                "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
                "composer_input_candidate_count": 1,
                "composer_input_safe_count": 0,
                "composer_input_review_count": 1,
                "observe_elapsed_ms": 724,
                "error": "",
            },
            {
                "sample": "broken",
                "process_name": "broken.exe",
                "mode": "unknown_pattern",
                "roi_purposes": [],
                "observe_elapsed_ms": 0,
                "error": "observe failed",
            },
        ]
    }

    report = module.build_acceptance_report(summary)

    assert report["counts"] == {"pass": 1, "warn": 0, "fail": 1}
    assert report["overall_status"] == "fail"


def test_build_acceptance_report_warns_for_legacy_sample_matrix():
    module = _load_module()
    summary = {
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "mode": "chat_workspace",
                "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
                "composer_input_candidate_count": 1,
                "composer_input_safe_count": 0,
                "composer_input_review_count": 1,
                "observe_elapsed_ms": 724,
                "error": "",
            }
        ]
    }

    report = module.build_acceptance_report(summary)

    assert report["overall_status"] == "warn"
    assert report["matrix_warnings"] == ["stale_or_legacy_sample_matrix"]
    assert report["sample_matrix_schema_version"] == ""


def test_write_acceptance_outputs(tmp_path):
    module = _load_module()
    report = {
        "overall_status": "pass",
        "counts": {"pass": 1, "warn": 0, "fail": 0},
        "matrix_warnings": [],
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "sample_class": "chat_workspace",
                "chat_variant": "private_chat",
                "status": "pass",
                "failures": [],
                "warnings": [],
            }
        ],
    }

    module.write_acceptance_outputs(report, tmp_path)

    saved = json.loads((tmp_path / "recognition_acceptance.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "recognition_acceptance.md").read_text(encoding="utf-8")
    assert saved["overall_status"] == "pass"
    assert "Recognition Acceptance" in markdown
    assert "| sample | process | class | variant | status |" in markdown
    assert "| wechat | weixin.exe | chat_workspace | private_chat | pass |" in markdown
