"""Tests for generic page evidence gate reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_page_evidence_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_page_evidence_gate", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_page_evidence_gate_uses_generic_required_texts():
    module = _load_module()

    report = module.build_page_evidence_gate(
        snapshot_report={
            "rows": [
                {
                    "sample": "qq.exe_4590900_QQ",
                    "process_name": "qq.exe",
                    "readback_result": {"events": [{"text": "消息测试123"}, {"text": "消息测试456"}]},
                }
            ]
        },
        requirements={"qq.exe_4590900_QQ": ["消息测试123"]},
    )

    assert report["overall_decision"] == "proceed"
    check = report["checks"][0]
    assert check["decision"] == "proceed"
    assert check["matched_texts"] == ["消息测试123"]
    assert check["gate_type"] == "page_evidence"


def test_build_page_evidence_gate_stops_when_required_text_missing():
    module = _load_module()

    report = module.build_page_evidence_gate(
        snapshot_report={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "readback_result": {"events": [{"text": "目文希社区安全漏洞测试报告"}]},
                }
            ]
        },
        requirements={"feishu.exe_68378": ["示例联系人"]},
    )

    assert report["overall_decision"] == "stop"
    check = report["checks"][0]
    assert check["decision"] == "stop"
    assert check["missing_texts"] == ["示例联系人"]
    assert check["reason"] == "required evidence text not observed"


def test_analyze_page_evidence_dir_writes_page_evidence_files(tmp_path: Path):
    module = _load_module()
    snapshot_dir = tmp_path / "snapshot"
    readback_dir = snapshot_dir / "readback"
    readback_dir.mkdir(parents=True)
    readback_path = readback_dir / "qq_readback.json"
    readback_path.write_text(json.dumps({"events": [{"text": "消息测试123"}]}, ensure_ascii=False), encoding="utf-8")
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
    requirements_file = tmp_path / "requirements.json"
    requirements_file.write_text(
        json.dumps(
            {"requirements": [{"sample": "qq.exe_4590900_QQ", "required_texts": ["消息测试123"]}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.analyze_page_evidence_dir(
        snapshot_dir=snapshot_dir,
        requirements_file=requirements_file,
        output_dir=tmp_path / "out",
    )

    assert report["overall_decision"] == "proceed"
    assert (tmp_path / "out" / "page_evidence_gate.json").exists()
    assert (tmp_path / "out" / "page_evidence_gate.md").exists()

