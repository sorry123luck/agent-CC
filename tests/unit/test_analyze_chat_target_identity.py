"""Tests for chat target identity gate reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_chat_target_identity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_chat_target_identity", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_target_identity_check_marks_expected_texts_proceed():
    module = _load_module()

    report = module.build_target_identity_check(
        snapshot_report={
            "rows": [
                {
                    "sample": "qq.exe_4590900_QQ",
                    "process_name": "qq.exe",
                    "readback_result": {
                        "events": [
                            {"text": "消息测试123", "sender": "me"},
                            {"text": "消息测试456", "sender": "peer"},
                        ]
                    },
                }
            ]
        },
        targets={"qq.exe_4590900_QQ": ["消息测试123", "消息测试456"]},
    )

    assert report["overall_decision"] == "proceed"
    check = report["checks"][0]
    assert check["decision"] == "proceed"
    assert check["matched_texts"] == ["消息测试123", "消息测试456"]
    assert check["missing_texts"] == []


def test_build_target_identity_check_marks_missing_texts_stop():
    module = _load_module()

    report = module.build_target_identity_check(
        snapshot_report={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "readback_result": {
                        "events": [
                            {"text": "目文希社区安全漏洞测试报告", "sender": "peer"},
                            {"text": "We detected unusual activity fromyourdevice or", "sender": "unknown"},
                        ]
                    },
                }
            ]
        },
        targets={"feishu.exe_68378": ["示例联系人"]},
    )

    assert report["overall_decision"] == "stop"
    check = report["checks"][0]
    assert check["decision"] == "stop"
    assert check["missing_texts"] == ["示例联系人"]
    assert check["evidence_texts"] == [
        "目文希社区安全漏洞测试报告",
        "We detected unusual activity fromyourdevice or",
    ]


def test_analyze_target_identity_dir_loads_nested_readback_and_writes_reports(tmp_path: Path):
    module = _load_module()
    snapshot_dir = tmp_path / "snapshot"
    readback_dir = snapshot_dir / "readback"
    readback_dir.mkdir(parents=True)
    readback_path = readback_dir / "qq_readback.json"
    readback_path.write_text(
        json.dumps({"events": [{"text": "消息测试123"}, {"text": "消息测试456"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "chat_readback_snapshot_report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample": "qq.exe_4590900_QQ",
                        "process_name": "qq.exe",
                        "readback_result_path": str(readback_path),
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps({"targets": [{"sample": "qq.exe_4590900_QQ", "expected_texts": ["消息测试123"]}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = module.analyze_target_identity_dir(
        snapshot_dir=snapshot_dir,
        targets_file=target_file,
        output_dir=tmp_path / "out",
    )

    assert report["overall_decision"] == "proceed"
    assert (tmp_path / "out" / "target_identity_check.json").exists()
    assert (tmp_path / "out" / "target_identity_check.md").exists()

