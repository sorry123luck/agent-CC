from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_chat_style_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_chat_style_baseline", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_readback_results_reads_nested_json(tmp_path):
    module = _load_module()
    path = tmp_path / "probe" / "readback" / "wechat_sent_readback.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"app_process": "weixin.exe", "events": []}),
        encoding="utf-8",
    )
    (tmp_path / "not_readback.json").write_text("{}", encoding="utf-8")

    results = module.find_readback_results([tmp_path])

    assert [item["app_process"] for item in results] == ["weixin.exe"]


def test_write_baseline_report_persists_json_and_markdown(tmp_path):
    module = _load_module()
    baseline = {
        "apps": {
            "weixin.exe": {
                "sender_style_hints": {"me": [146, 224, 148]},
                "sample_counts": {"me": 1},
                "usable": True,
            }
        },
        "sample_count": 1,
    }

    module.write_baseline_report(tmp_path, baseline)

    data = json.loads((tmp_path / "chat_sender_style_baseline.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "chat_sender_style_baseline.md").read_text(encoding="utf-8")
    assert data["apps"]["weixin.exe"]["sender_style_hints"]["me"] == [146, 224, 148]
    assert "| weixin.exe | true | me=1 | me=[146, 224, 148] |" in markdown
