from __future__ import annotations

from scripts import p0_entry_report
from scripts.p0_entry_report import build_p0_entry_report, classify_entry_stage


def test_build_p0_entry_report_counts_product_stages():
    discovery = [
        {"app_id": "chrome", "software_name": "Chrome", "exe_path": "chrome.exe", "launchable": True},
        {"app_id": "wechat", "software_name": "WeChat", "exe_path": None, "launchable": False},
    ]
    app_runs = [
        {
            "app_id": "chrome",
            "software_name": "Chrome",
            "discovery_status": "found",
            "launch_status": "bound",
            "bind_status": "bound",
            "capture_status": "captured",
            "observe_status": "canvas_created",
            "query_status": "matched",
            "query_results": [{"query": "search", "matched": True}],
        },
        {
            "app_id": "wechat",
            "software_name": "WeChat",
            "discovery_status": "not_found",
            "failure_stage": "discovery",
            "next_fix_task": "Add or scan launch target.",
        },
    ]

    report = build_p0_entry_report(discovery, app_runs)

    assert report["schema_version"] == "deskcanvas_p0_entry_report.v1"
    assert report["summary"] == {
        "total_apps": 2,
        "discovered": 1,
        "launchable": 1,
        "bound": 1,
        "captured": 1,
        "canvas_created": 1,
        "query_matched": 1,
        "full_chain_passed": 1,
    }
    assert report["apps"][0]["product_value"] == "Agent can open or attach Chrome, see its window, build a canvas, and find target controls."


def test_classify_entry_stage_returns_next_fix_task():
    app_run = {
        "discovery_status": "found",
        "launch_status": "failed",
        "failure_stage": "bind",
        "failure_reason": "multiple_candidate_windows",
    }

    stage, task = classify_entry_stage(app_run)

    assert stage == "bind"
    assert task == "Tighten process-to-hwnd binding and candidate ranking for this app."


def test_discover_software_matrix_initializes_database(monkeypatch):
    calls = []

    monkeypatch.setattr(p0_entry_report, "_ensure_db_schema", lambda: calls.append("init"))
    monkeypatch.setattr(
        "src.indexer.catalog_service.CatalogService.search",
        lambda _self, _query: [],
    )
    monkeypatch.setattr(
        "src.windows.window_resolver.WindowResolver.resolve",
        lambda _self, app_id, prefer_session=True: type(
            "Resolve",
            (),
            {"launch_targets": [], "candidates": []},
        )(),
    )

    p0_entry_report.discover_software_matrix(["chrome"])

    assert calls == ["init"]


def test_suspected_wrong_target_is_not_selected(monkeypatch):
    calls = []

    monkeypatch.setattr(p0_entry_report, "_ensure_db_schema", lambda: calls.append("init"))
    monkeypatch.setattr(
        "src.indexer.catalog_service.CatalogService.search",
        lambda _self, _query: [
            {
                "launch_path": "E:\\WXWork\\WXWork.exe",
                "source": "startmenu",
                "confidence": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        "src.windows.window_resolver.WindowResolver.resolve",
        lambda _self, app_id, prefer_session=True: type(
            "Resolve",
            (),
            {"launch_targets": [], "candidates": []},
        )(),
    )

    matrix = p0_entry_report.discover_software_matrix(["wechat"])

    assert matrix[0]["discovery_status"] == "not_found"
    assert matrix[0]["exe_path"] is None
    assert matrix[0]["suspected_error_paths"] == ["E:\\WXWork\\WXWork.exe"]


def test_discover_software_matrix_uses_existing_window_process_path(monkeypatch):
    monkeypatch.setattr(p0_entry_report, "_ensure_db_schema", lambda: None)
    monkeypatch.setattr(
        "src.indexer.catalog_service.CatalogService.search",
        lambda _self, _query: [],
    )
    monkeypatch.setattr(
        "src.windows.window_resolver.WindowResolver.resolve",
        lambda _self, app_id, prefer_session=True: type(
            "Resolve",
            (),
            {
                "launch_targets": [],
                "candidates": [
                    type(
                        "WindowCandidate",
                        (),
                        {
                            "process_id": 1234,
                            "process_name": "qq.exe",
                            "confidence": 0.6,
                            "title": "QQ",
                        },
                    )()
                ],
            },
        )(),
    )
    monkeypatch.setattr(p0_entry_report, "_process_exe_path", lambda pid: "C:\\Program Files\\Tencent\\QQNT\\QQ.exe")

    matrix = p0_entry_report.discover_software_matrix(["qq"])

    assert matrix[0]["discovery_status"] == "found"
    assert matrix[0]["exe_path"] == "C:\\Program Files\\Tencent\\QQNT\\QQ.exe"
    assert matrix[0]["source"] == "existing_window"
    assert matrix[0]["launchable"] is True
