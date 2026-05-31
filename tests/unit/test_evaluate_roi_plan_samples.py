"""Tests for offline ROI plan sample evaluation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_roi_plan_samples.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_roi_plan_samples", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canvas_from_detail_json_preserves_shape_for_planners():
    module = _load_module()
    sample = {
        "canvas_id": "sample_1",
        "page_class": "adspower global/viewer/main/wide",
        "window_title": "AdsPower Browser",
        "screenshot_width": 1200,
        "screenshot_height": 760,
        "elements": [
            {
                "element_id": "search",
                "semantic_role": "search_input",
                "bounds": [240, 120, 620, 150],
                "text": "keyword",
                "name": "",
                "control_type": "EditControl",
            },
            {
                "element_id": "row_1",
                "semantic_role": "list_item",
                "bounds": [260, 200, 1100, 250],
                "text": "row",
                "name": "",
                "control_type": "ListItemControl",
            },
        ],
        "regions": [{"region_id": "content", "bounds": [0, 0, 1200, 760]}],
        "geometric_regions": [
            {"region_id": "R1", "bounds": [220, 160, 1180, 660], "geometry_confidence": 0.8}
        ],
    }

    canvas = module.canvas_from_sample(sample, source_name="AdsPower.detail.json")

    assert canvas.canvas_id == "sample_1"
    assert canvas.page_class == "adspower global/viewer/main/wide"
    assert canvas.app.process_name == "adspower"
    assert canvas.artifacts["screenshot_size"] == [1200, 760]
    assert canvas.artifacts["geometric_regions"][0]["region_id"] == "R1"
    assert canvas.elements[0].semantic_role == "search_input"


def test_expected_mode_prefers_filename_and_page_class():
    module = _load_module()

    assert module.expected_mode_for_sample("WeChatCurrent.detail.json", "wechat/main") == "chat_workspace"
    assert module.expected_mode_for_sample("CloudMusic_snap.json", "unknown") == "media_home"
    assert module.expected_mode_for_sample("BitBrowserMain_snap.json", "bitbrowser/main") == "list_management"
    assert module.expected_mode_for_sample("Everything.detail.json", "everything/canvas") == "file_search"
    assert module.expected_mode_for_sample("LocalSend.detail.json", "localsend/canvas") == "local_transfer_dashboard"
    assert module.expected_mode_for_sample("QQ.detail.json", "qq/account") == "account_switcher"
    assert module.expected_mode_for_sample("Hubstudio.detail.json", "hubstudio/canvas") == "browser_profile_manager"
    assert module.expected_mode_for_sample("TeamViewer.detail.json", "teamviewer/canvas") == "remote_access_dashboard"
    assert module.expected_mode_for_sample("iQIYI.detail.json", "qyclient/canvas") == "media_video_home"
    assert module.expected_mode_for_sample("Unknown.detail.json", "unknown") == ""


def test_collect_sample_paths_excludes_generated_summaries(tmp_path):
    module = _load_module()
    (tmp_path / "App.detail.json").write_text("{}", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "roi_selection_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "App.observe.json").write_text("{}", encoding="utf-8")

    paths = module.collect_sample_paths(tmp_path)

    assert [path.name for path in paths] == ["App.detail.json"]


def test_collect_sample_paths_excludes_vlm_result_aggregates(tmp_path):
    module = _load_module()
    (tmp_path / "App.detail.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results.json").write_text("[]", encoding="utf-8")
    (tmp_path / "model_smoke_results.json").write_text("[]", encoding="utf-8")
    (tmp_path / "worker_profile_results.json").write_text("[]", encoding="utf-8")
    (tmp_path / "endpoint_text_probe.json").write_text("[]", encoding="utf-8")
    (tmp_path / "model_compare.json").write_text("[]", encoding="utf-8")
    (tmp_path / "roi_vlm.compare.json").write_text("{}", encoding="utf-8")

    paths = module.collect_sample_paths(tmp_path)

    assert [path.name for path in paths] == ["App.detail.json"]


def test_collect_sample_paths_excludes_live_matrix_stage_responses(tmp_path):
    module = _load_module()
    (tmp_path / "App.detail.json").write_text("{}", encoding="utf-8")
    (tmp_path / "App.observe.json").write_text("{}", encoding="utf-8")
    (tmp_path / "App.semantic.json").write_text("{}", encoding="utf-8")

    paths = module.collect_sample_paths(tmp_path)

    assert [path.name for path in paths] == ["App.detail.json"]


def test_collect_sample_paths_excludes_regression_outputs(tmp_path):
    module = _load_module()
    (tmp_path / "App.detail.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "roi_vlm_regression"
    output_dir.mkdir()
    (output_dir / "quality_summary.json").write_text("{}", encoding="utf-8")
    (output_dir / "regression_summary.json").write_text("{}", encoding="utf-8")
    (output_dir / "summary.json").write_text("{}", encoding="utf-8")

    paths = module.collect_sample_paths(tmp_path)

    assert [path.name for path in paths] == ["App.detail.json"]


def test_evaluate_sample_reports_roi_quality_and_vlm_effect_metrics(tmp_path):
    module = _load_module()
    sample_path = tmp_path / "WeChat.detail.json"
    sample_path.write_text(
        """
        {
          "canvas_id": "sample_chat",
          "page_class": "wechat/app/main/wide",
          "window_title": "微信",
          "screenshot_width": 1000,
          "screenshot_height": 800,
          "elements": [
            {"element_id": "nav", "semantic_role": "button", "bounds": [20, 20, 80, 80]},
            {"element_id": "msg", "semantic_role": "text", "bounds": [360, 180, 920, 260]},
            {"element_id": "send", "semantic_role": "button", "bounds": [900, 720, 980, 770]}
          ],
          "regions": [{"region_id": "content", "bounds": [0, 0, 1000, 800]}],
          "artifacts": {
            "roi_vlm_semantic_supplements": [
              {
                "roi_id": "roi_1",
                "status": "success",
                "candidate_annotations": [
                  {"candidate_id": "msg", "label": "latest message"},
                  {"candidate_id": "send", "label": "send button"}
                ]
              }
            ],
            "roi_vlm_timeouts": [{"roi_id": "roi_2"}],
            "roi_vlm_late_failures": [{"roi_id": "roi_3"}]
          }
        }
        """,
        encoding="utf-8",
    )

    row = module.evaluate_sample(sample_path)

    assert row["expected_required_purposes"] == [
        "navigation_and_list",
        "message_stream",
        "composer",
    ]
    assert row["missing_required_purposes"] == []
    assert row["roi_coverage_ratio"] > 0.7
    assert row["roi_candidate_coverage_ratio"] == 1.0
    assert row["roi_vlm_status"] == "supplemented_with_timeout"
    assert row["roi_vlm_supplement_count"] == 1
    assert row["roi_vlm_annotation_count"] == 2
    assert row["roi_vlm_timeout_count"] == 1
    assert row["roi_vlm_late_failure_count"] == 1
