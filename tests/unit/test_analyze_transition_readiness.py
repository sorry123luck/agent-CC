"""Tests for multi-page transition readiness reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_transition_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_transition_readiness", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_search_probe_roundtrip_becomes_observed_transition_memory():
    module = _load_module()

    report = module.build_transition_readiness_report(
        search_probe={
            "execute": True,
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "captured",
                    "query": "jz",
                    "before_canvas_id": "snap_before",
                    "after_canvas_id": "snap_search",
                    "selected_canvas_id": "snap_selected",
                    "state_transition": {
                        "path": "chat_workspace -> chat_search_results -> chat_workspace",
                        "selected_returned_to_before_state": True,
                    },
                    "notes": ["search_state_captured", "selected_first_result"],
                }
            ],
        }
    )

    assert report["overall_status"] == "observed_transition_memory_ready"
    row = report["rows"][0]
    assert row["status"] == "observed_roundtrip"
    assert row["source"] == "search_probe"
    assert row["transition_path"] == "chat_workspace -> chat_search_results -> chat_workspace"
    assert row["execution_boundary"] == "observed_only"
    assert row["autonomous_execution"] is False
    assert row["agent_contract"]["allowed_default"] == "read_transition_memory"
    assert row["agent_contract"]["blocked_default"] == ["click", "type_text", "send"]


def test_control_transition_requires_confirmation_even_when_successful():
    module = _load_module()

    report = module.build_transition_readiness_report(
        control_transition_graph={
            "transitions": [
                {
                    "candidate_key": "stable:pm:settings",
                    "from_page_class": "chat_workspace",
                    "to_page_class": "settings_panel",
                    "action_type": "click",
                    "observe_count": 2,
                    "success_count": 2,
                    "failure_count": 0,
                    "success_rate": 1.0,
                    "canvas_id_before": "snap_a",
                    "canvas_id_after": "snap_b",
                }
            ]
        }
    )

    assert report["overall_status"] == "controlled_transition_memory_ready"
    row = report["rows"][0]
    assert row["status"] == "controlled_transition_observed"
    assert row["execution_boundary"] == "requires_confirmed_execution"
    assert row["required_verification"] == ["observe_after", "diff_after", "readback_after"]
    assert row["autonomous_execution"] is False


def test_missing_before_after_identity_is_incomplete():
    module = _load_module()

    report = module.build_transition_readiness_report(
        control_transition_graph={
            "transitions": [
                {
                    "candidate_key": "stable:pm:bad",
                    "from_page_class": "chat_workspace",
                    "to_page_class": "",
                    "action_type": "click",
                    "observe_count": 1,
                    "success_count": 1,
                    "success_rate": 1.0,
                }
            ]
        }
    )

    assert report["overall_status"] == "incomplete"
    assert report["rows"][0]["status"] == "incomplete_transition"
    assert "missing_to_page_class" in report["rows"][0]["blockers"]


def test_analyze_transition_readiness_dirs_writes_outputs(tmp_path):
    module = _load_module()
    input_dir = tmp_path / "probe"
    input_dir.mkdir()
    (input_dir / "search_probe_report.json").write_text(
        json.dumps(
            {
                "execute": True,
                "rows": [
                    {
                        "sample": "wechat",
                        "process_name": "weixin.exe",
                        "status": "captured",
                        "state_transition": {
                            "path": "chat_workspace -> chat_search_results -> chat_workspace",
                            "selected_returned_to_before_state": True,
                        },
                        "notes": ["search_state_captured"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    report = module.analyze_transition_readiness_dirs(search_probe_dir=input_dir, output_dir=output_dir)

    assert report["overall_status"] == "observed_transition_memory_ready"
    assert (output_dir / "transition_readiness_report.json").exists()
    markdown = (output_dir / "transition_readiness_report.md").read_text(encoding="utf-8")
    assert "Transition Readiness Report" in markdown
    assert "weixin.exe" in markdown
