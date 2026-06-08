"""Tests for input safety sample report generation."""

from __future__ import annotations

import json
import subprocess
import sys

from scripts.input_safety_report import (
    build_input_safety_report,
    write_archived_input_safety_report,
    write_input_safety_report,
)


def _snapshot(
    *,
    app_id: str,
    page_class: str,
    region_id: str,
    input_text: str,
    send_enabled: bool,
    input_bounds: list[int] | None = None,
) -> dict:
    return {
        "canvas_id": f"{app_id}_{'has_text' if input_text else 'empty'}",
        "app": {"app_id": app_id, "process_name": f"{app_id}.exe"},
        "page": {"page_class": page_class},
        "elements": [
            {
                "element_id": f"{app_id}_input",
                "region_id": region_id,
                "semantic_role": "message_input",
                "control_type": "Edit",
                "bounds": input_bounds,
                "text": input_text,
                "interactable": True,
                "state": {"enabled": True, "visible": True},
            },
            {
                "element_id": f"{app_id}_send",
                "region_id": region_id,
                "semantic_role": "send_button",
                "control_type": "Button",
                "bounds": [620, 520, 700, 580],
                "text": "send",
                "interactable": send_enabled,
                "state": {"enabled": send_enabled, "visible": True},
            },
        ],
    }


def test_build_input_safety_report_summarizes_wechat_qq_feishu_matrix():
    snapshots = [
        _snapshot(app_id="wechat", page_class="wechat/chat/main", region_id="bottom_input", input_text="", send_enabled=False, input_bounds=[80, 520, 600, 580]),
        _snapshot(app_id="qq", page_class="qq/chat/main", region_id="bottom_input", input_text="hello", send_enabled=True, input_bounds=[80, 520, 600, 580]),
        _snapshot(app_id="feishu", page_class="feishu/chat/main", region_id="composer", input_text="hello", send_enabled=False, input_bounds=[80, 520, 600, 580]),
    ]

    report = build_input_safety_report(snapshots)

    assert report["schema_version"] == "input_safety_report.v1"
    assert report["summary"] == {
        "total_samples": 3,
        "passed": 2,
        "failed": 1,
        "apps": {
            "wechat": {"total": 1, "passed": 1, "failed": 0},
            "qq": {"total": 1, "passed": 1, "failed": 0},
            "feishu": {"total": 1, "passed": 0, "failed": 1},
        },
    }
    feishu = next(sample for sample in report["samples"] if sample["profile"]["app_id"] == "feishu")
    assert feishu["status"] == "failed"
    assert feishu["input_state"] == "has_text"
    assert feishu["send_button_state"]["enabled"] is False
    assert "send_button_disabled_while_input_has_text" in feishu["issues"]
    assert feishu["input_gate_evaluation"]["eligible_for_safe_to_send"] is False
    assert "send_button_state" in feishu["input_gate_evaluation"]["failed_checks"]

    wechat = next(sample for sample in report["samples"] if sample["profile"]["app_id"] == "wechat")
    assert wechat["input_gate_evaluation"]["eligible_for_safe_to_type"] is True
    assert wechat["input_gate_evaluation"]["gate_status"] == "blocked_pending_target2_acceptance"


def test_write_input_safety_report_reads_snapshot_directory(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    (sample_dir / "wechat_empty_snapshot.json").write_text(
        json.dumps(
            _snapshot(app_id="wechat", page_class="wechat/chat/main", region_id="bottom_input", input_text="", send_enabled=False, input_bounds=[80, 520, 600, 580]),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    report = write_input_safety_report(sample_dir, output_path)

    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == report
    assert persisted["summary"]["passed"] == 1
    assert persisted["samples"][0]["safe_to_type"] is False
    assert persisted["samples"][0]["safe_to_send"] is False


def test_write_archived_input_safety_report_scans_real_app_matrix_dirs(tmp_path):
    sample_root = tmp_path / "data" / "baselines" / "new"
    archive_root = tmp_path / "data" / "reports"
    for app_id, page_class, region_id in [
        ("wechat", "wechat/chat/main", "bottom_input"),
        ("qq", "qq/chat/main", "bottom_input"),
    ]:
        app_dir = sample_root / app_id
        app_dir.mkdir(parents=True)
        (app_dir / f"{app_id}_20260602_120000_snapshot.json").write_text(
            json.dumps(
                _snapshot(
                    app_id=app_id,
                    page_class=page_class,
                    region_id=region_id,
                    input_text="",
                    send_enabled=False,
                    input_bounds=[80, 520, 600, 580],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    report, report_path = write_archived_input_safety_report(
        sample_root,
        archive_root,
        matrix_apps=["wechat", "qq", "feishu"],
        run_id="20260602_120000",
    )

    assert report_path == archive_root / "input_safety" / "20260602_120000" / "input_safety_report.json"
    assert report_path.exists()
    assert report["matrix"] == {
        "expected_apps": ["wechat", "qq", "feishu"],
        "covered_apps": ["qq", "wechat"],
        "missing_apps": ["feishu"],
        "coverage_status": "incomplete",
    }
    assert report["summary"]["total_samples"] == 2
    assert {sample["source_path"] for sample in report["samples"]} == {
        str(sample_root / "wechat" / "wechat_20260602_120000_snapshot.json"),
        str(sample_root / "qq" / "qq_20260602_120000_snapshot.json"),
    }


def test_input_safety_report_cli_writes_archived_report(tmp_path):
    sample_root = tmp_path / "samples"
    archive_root = tmp_path / "reports"
    app_dir = sample_root / "wechat"
    app_dir.mkdir(parents=True)
    (app_dir / "wechat_20260602_120000_snapshot.json").write_text(
        json.dumps(
            _snapshot(
                app_id="wechat",
                page_class="wechat/chat/main",
                region_id="bottom_input",
                input_text="",
                send_enabled=False,
                input_bounds=[80, 520, 600, 580],
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/input_safety_report.py",
            "--input-dir",
            str(sample_root),
            "--archive-root",
            str(archive_root),
            "--run-id",
            "20260602_120000",
            "--matrix-apps",
            "wechat,qq,feishu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    report_path = archive_root / "input_safety" / "20260602_120000" / "input_safety_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matrix"]["missing_apps"] == ["qq", "feishu"]
    assert "Input safety report: 1/1 passed" in result.stdout


def test_build_input_safety_report_uses_fallback_for_unlabeled_chat_composer():
    report = build_input_safety_report(
        [
            {
                "canvas_id": "qq_fallback",
                "app": {"app_id": "qq", "process_name": "qq.exe"},
                "page": {"page_class": "qq/chat/main"},
                "elements": [
                    {
                        "element_id": "qq_edit",
                        "region_id": "bottom_input",
                        "semantic_role": "unknown",
                        "control_type": "Edit",
                        "bounds": [80, 520, 600, 580],
                        "text": "",
                        "interactable": True,
                        "state": {"enabled": True, "visible": True},
                    },
                    {
                        "element_id": "qq_send",
                        "region_id": "bottom_input",
                        "semantic_role": "button",
                        "control_type": "Button",
                        "bounds": [620, 520, 700, 580],
                        "text": "send",
                        "name": "发送",
                        "interactable": False,
                        "state": {"enabled": False, "visible": True},
                    },
                ],
            }
        ],
        matrix_apps=["qq"],
    )

    sample = report["samples"][0]
    assert sample["status"] == "passed"
    assert sample["input_state"] == "empty"
    assert sample["target_input"]["candidate_id"] == "qq_edit"
    assert sample["associated_send_button"]["candidate_id"] == "qq_send"
    assert sample["matrix_detection"] == {
        "input_source": "fallback",
        "send_button_source": "fallback",
    }
    assert sample["input_gate_evaluation"]["eligible_for_safe_to_type"] is True


def test_build_input_safety_report_marks_target2_acceptance_ready_only_after_full_two_state_matrix():
    snapshots = []
    for app_id, page_class, region_id in [
        ("wechat", "wechat/chat/main", "bottom_input"),
        ("qq", "qq/chat/main", "bottom_input"),
        ("feishu", "feishu/chat/main", "composer"),
    ]:
        snapshots.extend(
            [
                _snapshot(app_id=app_id, page_class=page_class, region_id=region_id, input_text="", send_enabled=False, input_bounds=[80, 520, 600, 580]),
                _snapshot(app_id=app_id, page_class=page_class, region_id=region_id, input_text="hello", send_enabled=True, input_bounds=[80, 520, 600, 580]),
            ]
        )

    report = build_input_safety_report(snapshots, matrix_apps=["wechat", "qq", "feishu"])

    assert report["target2_acceptance"] == {
        "status": "ready_for_manual_review",
        "safe_to_type_can_be_considered": True,
        "safe_to_send_can_be_considered": True,
        "required_states": ["empty", "has_text"],
        "missing_state_coverage": {},
        "failed_apps": [],
        "blocking_reasons": [],
    }
    assert all(sample["safe_to_type"] is False for sample in report["samples"])
    assert all(sample["safe_to_send"] is False for sample in report["samples"])


def test_build_input_safety_report_marks_target2_acceptance_not_ready_when_state_coverage_missing():
    report = build_input_safety_report(
        [
            _snapshot(app_id="wechat", page_class="wechat/chat/main", region_id="bottom_input", input_text="", send_enabled=False, input_bounds=[80, 520, 600, 580]),
            _snapshot(app_id="qq", page_class="qq/chat/main", region_id="bottom_input", input_text="", send_enabled=False, input_bounds=[80, 520, 600, 580]),
        ],
        matrix_apps=["wechat", "qq", "feishu"],
    )

    acceptance = report["target2_acceptance"]
    assert acceptance["status"] == "not_ready"
    assert acceptance["safe_to_type_can_be_considered"] is False
    assert acceptance["safe_to_send_can_be_considered"] is False
    assert acceptance["missing_state_coverage"] == {
        "wechat": ["has_text"],
        "qq": ["has_text"],
        "feishu": ["empty", "has_text"],
    }
    assert "matrix_coverage_incomplete" in acceptance["blocking_reasons"]
    assert "state_coverage_incomplete" in acceptance["blocking_reasons"]
