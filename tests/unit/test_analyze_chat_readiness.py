"""Tests for chat readiness aggregation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_chat_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_chat_readiness", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_chat_readiness_keeps_inputs_review_only_even_when_evidence_passes():
    module = _load_module()
    report = module.build_chat_readiness(
        action_plan={
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "ready",
                    "safe_to_type": False,
                    "input_click": [485, 576],
                    "send_click": [713, 612],
                }
            ]
        },
        send_report={"rows": [{"app": "qq", "status": "pass", "text_observed_after": True}]},
        snapshot_acceptance={"rows": [{"sample": "qq", "process_name": "qq.exe", "status": "pass", "message_text_count": 2}]},
    )

    assert report["overall_status"] == "controlled_ready"
    row = report["rows"][0]
    assert row["status"] == "controlled_ready"
    assert row["safe_to_type"] is False
    assert row["can_run_controlled_send_probe"] is True
    assert row["can_default_agent_type"] is False
    assert "review_only_input" in row["warnings"]


def test_build_chat_readiness_marks_readonly_when_no_send_evidence():
    module = _load_module()
    report = module.build_chat_readiness(
        action_plan={
            "rows": [
                {
                    "sample": "wechat-vincci",
                    "process_name": "weixin.exe",
                    "status": "ready",
                    "safe_to_type": False,
                    "input_click": [578, 668],
                    "send_click": [950, 701],
                }
            ]
        },
        send_report={"rows": []},
        snapshot_acceptance={
            "rows": [{"sample": "wechat-vincci", "process_name": "weixin.exe", "status": "pass", "message_text_count": 8}]
        },
    )

    assert report["overall_status"] == "readonly_ready"
    row = report["rows"][0]
    assert row["status"] == "readonly_ready"
    assert row["can_run_controlled_send_probe"] is False
    assert row["can_run_readonly_snapshot"] is True
    assert "missing_send_probe_pass" in row["warnings"]


def test_build_chat_readiness_preserves_target_mismatch_status():
    module = _load_module()
    report = module.build_chat_readiness(
        action_plan={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "status": "blocked_target_mismatch",
                    "safe_to_type": False,
                    "input_click": [749, 698],
                    "send_click": [935, 715],
                    "reasons": ["target_identity_check_stop"],
                }
            ]
        },
        send_report={"rows": []},
        snapshot_acceptance={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "status": "pass",
                    "message_text_count": 14,
                }
            ]
        },
    )

    assert report["overall_status"] == "blocked"
    row = report["rows"][0]
    assert row["status"] == "blocked_target_mismatch"
    assert row["can_run_controlled_send_probe"] is False
    assert "target_identity_check_stop" in row["warnings"]
    assert "missing_send_probe_pass" not in row["warnings"]


def test_build_chat_readiness_preserves_evidence_mismatch_status():
    module = _load_module()
    report = module.build_chat_readiness(
        action_plan={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "status": "blocked_evidence_mismatch",
                    "safe_to_type": False,
                    "reasons": ["evidence_gate_stop"],
                }
            ]
        },
        send_report={"rows": []},
        snapshot_acceptance={"rows": [{"sample": "feishu.exe_68378", "process_name": "feishu.exe", "status": "pass"}]},
    )

    row = report["rows"][0]
    assert row["status"] == "blocked_evidence_mismatch"
    assert "evidence_gate_stop" in row["warnings"]


def test_build_chat_readiness_applies_page_evidence_gate_stop():
    module = _load_module()
    report = module.build_chat_readiness(
        action_plan={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "status": "ready",
                    "safe_to_type": False,
                    "input_click": [749, 698],
                    "send_click": [935, 715],
                }
            ]
        },
        send_report={"rows": [{"app": "feishu", "status": "pass"}]},
        snapshot_acceptance={"rows": [{"sample": "feishu.exe_68378", "process_name": "feishu.exe", "status": "pass"}]},
        evidence_gate={
            "gate_type": "page_evidence",
            "checks": [
                {
                    "sample": "feishu.exe_68378",
                    "decision": "stop",
                    "reason": "required evidence text not observed",
                }
            ],
        },
    )

    row = report["rows"][0]
    assert row["status"] == "blocked_evidence_mismatch"
    assert row["can_run_controlled_send_probe"] is False
    assert row["action_plan_status"] == "ready"
    assert row["evidence_gate_decision"] == "stop"
    assert "evidence_gate_stop" in row["warnings"]
    assert report["counts"]["blocked"] == 1
    assert report["counts"]["blocked_evidence_mismatch"] == 1


def test_analyze_chat_readiness_dirs_loads_page_evidence_gate(tmp_path: Path):
    module = _load_module()
    action_dir = tmp_path / "matrix"
    send_dir = tmp_path / "send"
    snapshot_dir = tmp_path / "snapshot"
    gate_dir = tmp_path / "gate"
    action_dir.mkdir()
    send_dir.mkdir()
    snapshot_dir.mkdir()
    gate_dir.mkdir()
    (action_dir / "chat_action_plan.json").write_text(
        json.dumps({"rows": [{"sample": "feishu.exe_68378", "process_name": "feishu.exe", "status": "ready"}]}),
        encoding="utf-8",
    )
    (send_dir / "chat_send_probe_report.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (snapshot_dir / "chat_readback_snapshot_acceptance.json").write_text(
        json.dumps({"rows": [{"sample": "feishu.exe_68378", "process_name": "feishu.exe", "status": "pass"}]}),
        encoding="utf-8",
    )
    (gate_dir / "page_evidence_gate.json").write_text(
        json.dumps(
            {
                "gate_type": "page_evidence",
                "checks": [{"sample": "feishu.exe_68378", "decision": "stop", "reason": "wrong page"}],
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_chat_readiness_dirs(
        action_plan_dir=action_dir,
        send_report_dir=send_dir,
        snapshot_acceptance_dir=snapshot_dir,
        evidence_gate_dir=gate_dir,
        output_dir=tmp_path / "out",
    )

    assert report["rows"][0]["status"] == "blocked_evidence_mismatch"


def test_analyze_chat_readiness_dirs_writes_report(tmp_path: Path):
    module = _load_module()
    action_dir = tmp_path / "matrix"
    send_dir = tmp_path / "send"
    snapshot_dir = tmp_path / "snapshot"
    action_dir.mkdir()
    send_dir.mkdir()
    snapshot_dir.mkdir()
    (action_dir / "chat_action_plan.json").write_text(
        json.dumps({"rows": [{"sample": "qq", "process_name": "qq.exe", "status": "ready", "safe_to_type": False}]}),
        encoding="utf-8",
    )
    (send_dir / "chat_send_probe_report.json").write_text(
        json.dumps({"rows": [{"app": "qq", "status": "pass", "text_observed_after": True}]}),
        encoding="utf-8",
    )
    (snapshot_dir / "chat_readback_snapshot_acceptance.json").write_text(
        json.dumps({"rows": [{"sample": "qq", "process_name": "qq.exe", "status": "pass", "message_text_count": 1}]}),
        encoding="utf-8",
    )

    report = module.analyze_chat_readiness_dirs(
        action_plan_dir=action_dir,
        send_report_dir=send_dir,
        snapshot_acceptance_dir=snapshot_dir,
        output_dir=tmp_path / "out",
    )

    assert report["overall_status"] == "controlled_ready"
    assert (tmp_path / "out" / "chat_readiness_report.json").exists()
