"""Generate DeskCanvas P0 product-entry reports.

Product chain:
software discovery -> launch -> bind hwnd -> capture diagnostics -> observe
InteractionCanvas -> query key controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "deskcanvas_p0_entry_report.v1"
DEFAULT_TARGETS = ["wechat", "qq", "feishu", "chrome", "vscode", "netease_cloud_music", "flclash", "voicemeeter"]
QUERY_MATRIX: dict[str, list[dict[str, str]]] = {
    "wechat": [{"text": "聊天"}, {"text": "联系人"}, {"text": "搜索"}, {"semantic_role": "message_input"}],
    "qq": [{"text": "聊天"}, {"text": "联系人"}, {"text": "搜索"}, {"semantic_role": "message_input"}],
    "feishu": [{"text": "消息"}, {"text": "搜索"}, {"semantic_role": "message_input"}],
    "chrome": [{"semantic_role": "search_input"}, {"text": "地址"}, {"text": "标签"}, {"semantic_role": "tab"}],
    "vscode": [{"text": "文件"}, {"text": "终端"}, {"text": "资源管理器"}, {"semantic_role": "tree_item"}],
    "netease_cloud_music": [{"text": "搜索"}, {"text": "发现音乐"}, {"semantic_role": "button"}],
    "flclash": [{"text": "代理"}, {"text": "配置"}, {"semantic_role": "button"}],
    "voicemeeter": [{"text": "A1"}, {"text": "Menu"}, {"semantic_role": "button"}],
}
DISPLAY_NAMES = {
    "wechat": "WeChat",
    "qq": "QQ",
    "feishu": "Feishu",
    "chrome": "Chrome",
    "vscode": "VSCode",
    "netease_cloud_music": "NetEase Cloud Music",
    "flclash": "FlClash",
    "voicemeeter": "VoiceMeeter",
}


TARGET_ALIASES = {
    "wechat": ["wechat.exe", "weixin", "微信"],
    "qq": ["qq", "tencent qq", "腾讯qq"],
    "feishu": ["feishu", "lark", "飞书"],
    "chrome": ["chrome", "google chrome", "谷歌浏览器"],
    "vscode": ["vscode", "visual studio code", "code.exe"],
    "netease_cloud_music": ["cloudmusic", "cloud music", "netease cloud music", "网易云音乐"],
    "flclash": ["flclash", "clash"],
    "voicemeeter": ["voicemeeter", "voice meeter"],
}
NEGATIVE_ALIASES = {
    "wechat": ["wxwork", "wecom", "企业微信", "work"],
    "netease_cloud_music": ["uu_", "uu.exe", "uu加速", "uu accelerator"],
}


def discover_software_matrix(targets: list[str], *, scan: bool = False) -> list[dict[str, Any]]:
    from src.indexer.catalog_service import CatalogService
    from src.windows.window_resolver import get_window_resolver

    _ensure_db_schema()
    service = CatalogService()
    if scan:
        service.scan_all()
    resolver = get_window_resolver()
    rows: list[dict[str, Any]] = []
    for app_id in targets:
        catalog_hits = _scan_catalog_hits(app_id) if scan else service.search(app_id)
        resolve_result = resolver.resolve(app_id, prefer_session=True)
        launch_targets = resolve_result.launch_targets
        candidates: list[dict[str, Any]] = []
        for hit in catalog_hits:
            if hit.get("launch_path"):
                candidates.append(
                    {
                        "exe_path": hit.get("launch_path"),
                        "source": hit.get("source") or "catalog",
                        "confidence": hit.get("confidence", 0.0),
                        "launchable": Path(str(hit.get("launch_path"))).is_file(),
                    }
                )
        for target in launch_targets:
            candidates.append(
                {
                    "exe_path": target.executable,
                    "source": "window_resolver",
                    "confidence": 0.5,
                    "launchable": Path(target.executable).is_file() or _is_path_like_command(target.executable),
                }
            )
        for window in getattr(resolve_result, "candidates", []) or []:
            exe_path = _process_exe_path(getattr(window, "process_id", None))
            if exe_path:
                candidates.append(
                    {
                        "exe_path": exe_path,
                        "source": "existing_window",
                        "confidence": getattr(window, "confidence", 0.6),
                        "launchable": Path(exe_path).is_file(),
                        "window_title": getattr(window, "title", ""),
                        "process_name": getattr(window, "process_name", ""),
                    }
                )
        seen_paths = set()
        unique = []
        duplicate_paths = []
        for candidate in candidates:
            path_key = str(candidate.get("exe_path") or "").lower()
            if path_key in seen_paths:
                duplicate_paths.append(candidate.get("exe_path"))
                continue
            seen_paths.add(path_key)
            unique.append(candidate)
        suspected_error_paths = [
            item.get("exe_path")
            for item in unique
            if _is_suspected_wrong_target(app_id, str(item.get("exe_path") or ""))
        ]
        viable = [item for item in unique if item.get("exe_path") not in suspected_error_paths]
        best = viable[0] if viable else {}
        suspicious = [
            item.get("exe_path")
            for item in unique
            if any(token in str(item.get("exe_path") or "").lower() for token in ["uninstall", "setup", "update", "helper"])
        ]
        rows.append(
            {
                "app_id": app_id,
                "software_name": DISPLAY_NAMES.get(app_id, app_id),
                "exe_path": best.get("exe_path"),
                "source": best.get("source"),
                "confidence": best.get("confidence", 0.0),
                "launchable": bool(best.get("launchable")),
                "duplicate_count": len(duplicate_paths),
                "duplicate_paths": duplicate_paths,
                "suspicious_paths": suspicious,
                "suspected_error_paths": suspected_error_paths,
                "candidates": unique,
                "discovery_status": "found" if viable else "not_found",
            }
        )
    return rows


def _scan_catalog_hits(app_id: str) -> list[dict[str, Any]]:
    from src.storage.db import Session
    from src.storage.repositories import AppRepository, LaunchTargetRepository

    aliases = [alias.lower() for alias in TARGET_ALIASES.get(app_id, [app_id])]
    hits: list[dict[str, Any]] = []
    with Session() as session:
        app_repo = AppRepository(session)
        launch_repo = LaunchTargetRepository(session)
        for app in app_repo.list_all():
            app_blob = " ".join(
                [
                    str(app.canonical_name or ""),
                    str(app.display_name or ""),
                    str(app.install_location or ""),
                ]
            ).lower()
            if not any(alias in app_blob for alias in aliases):
                continue
            if any(alias in app_blob for alias in NEGATIVE_ALIASES.get(app_id, [])):
                continue
            for target in launch_repo.get_by_app_id(app.id):
                if _is_suspected_wrong_target(app_id, target.path):
                    hits.append(
                        {
                            "canonical_name": app.canonical_name,
                            "display_name": app.display_name,
                            "launch_path": target.path,
                            "source": target.source,
                            "confidence": float(app.confidence or 0.0),
                        }
                    )
                    continue
                hits.append(
                    {
                        "canonical_name": app.canonical_name,
                        "display_name": app.display_name,
                        "launch_path": target.path,
                        "source": target.source,
                        "confidence": float(app.confidence or 0.0),
                    }
                )
    return hits


def _is_suspected_wrong_target(app_id: str, path: str) -> bool:
    blob = path.lower()
    return any(alias in blob for alias in NEGATIVE_ALIASES.get(app_id, []))


def run_entry_flow(client, app: dict[str, Any], *, allow_launch: bool, allow_existing: bool) -> dict[str, Any]:
    app_id = app["app_id"]
    run: dict[str, Any] = {
        "app_id": app_id,
        "software_name": app["software_name"],
        "product_value": "",
        "discovery_status": app.get("discovery_status", "not_found"),
        "launch_status": "not_run",
        "bind_status": "not_run",
        "capture_status": "not_run",
        "observe_status": "not_run",
        "query_status": "not_run",
        "query_results": [],
    }
    if run["discovery_status"] != "found":
        run["failure_stage"] = "discovery"
        run["failure_reason"] = "path_not_found"
        run["next_fix_task"] = "Add or scan launch target for this app."
        return run
    if not allow_launch and not allow_existing:
        run["failure_stage"] = "launch"
        run["failure_reason"] = "launch_not_allowed"
        run["next_fix_task"] = "Run the report with --allow-launch or --allow-existing for real chain validation."
        return run

    launch_resp = client.post(
        "/api/v1/apps/launch-bind",
        json={
            "app_id": app_id,
            "exe_path": app.get("exe_path") if allow_launch else None,
            "timeout_seconds": 12,
            "allow_existing": allow_existing,
        },
    )
    launch_data = launch_resp.json()
    run["launch_bind"] = launch_data
    run["launch_status"] = launch_data.get("status")
    run["bind_status"] = "bound" if launch_data.get("bound_window") else "failed"
    if launch_data.get("status") != "bound":
        run["failure_stage"] = launch_data.get("stage") or "bind"
        run["failure_reason"] = launch_data.get("failure_reason") or "launch_bind_failed"
        run["next_fix_task"] = classify_entry_stage(run)[1]
        return run

    diagnostics = launch_data.get("capture_diagnostics") or {}
    run["capture_status"] = "captured" if diagnostics.get("valid") else "failed"
    hwnd = (launch_data.get("bound_window") or {}).get("hwnd")
    observe_resp = client.post("/api/v1/observe", json={"hwnd": hwnd, "async_enhance": False})
    if observe_resp.status_code != 200:
        run["observe_status"] = "failed"
        run["failure_stage"] = "observe"
        run["failure_reason"] = f"http_{observe_resp.status_code}"
        run["next_fix_task"] = classify_entry_stage(run)[1]
        return run
    observe_data = observe_resp.json()
    run["observe"] = observe_data
    run["observe_status"] = "canvas_created" if observe_data.get("canvas_id") else "failed"
    canvas_id = observe_data.get("canvas_id")

    matched = 0
    for query in QUERY_MATRIX.get(app_id, [{"semantic_role": "button"}])[:5]:
        query_resp = client.post(
            "/api/v1/query",
            json={"canvas_id": canvas_id, "target": query, "max_results": 3, "min_confidence": 0.0},
        )
        result = {"query": query, "matched": False, "candidates": []}
        if query_resp.status_code == 200:
            qdata = query_resp.json()
            candidates = qdata.get("candidates", [])
            result["matched"] = bool(candidates)
            result["candidates"] = [
                {
                    "element_id": c.get("element_id"),
                    "text": c.get("text"),
                    "semantic_role": c.get("semantic_role"),
                    "confidence": c.get("confidence"),
                    "risk_level": c.get("risk_level"),
                    "provider_sources": c.get("provider_sources"),
                }
                for c in candidates
            ]
            matched += 1 if candidates else 0
        run["query_results"].append(result)
    run["query_status"] = "matched" if matched else "no_match"
    if matched:
        run["product_value"] = (
            f"Agent can open or attach {run['software_name']}, see its window, build a canvas, and find target controls."
        )
    else:
        run["failure_stage"] = "query"
        run["failure_reason"] = "key_controls_not_found"
        run["next_fix_task"] = classify_entry_stage(run)[1]
    return run


def classify_entry_stage(app_run: dict[str, Any]) -> tuple[str, str]:
    stage = app_run.get("failure_stage")
    reason = app_run.get("failure_reason")
    if stage == "discovery":
        return stage, "Add software path discovery or target alias coverage for this app."
    if stage == "launch":
        return stage, "Fix launch target command, args, or launch permissions for this app."
    if stage == "bind" and reason == "multiple_candidate_windows":
        return stage, "Tighten process-to-hwnd binding and candidate ranking for this app."
    if stage == "bind":
        return stage, "Wait for the launched process window and distinguish old windows from new ones."
    if stage == "capture":
        return stage, "Fix window bounds, DPI, minimization, or screenshot capture diagnostics."
    if stage == "observe":
        return stage, "Fix perception startup for this hwnd and ensure InteractionCanvas is generated."
    if stage == "query":
        return stage, "Improve key-control detection or query scoring for this app."
    return "passed", "No fix needed for this stage."


def build_p0_entry_report(discovery: list[dict[str, Any]], app_runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total_apps": len(discovery),
        "discovered": sum(1 for item in discovery if item.get("discovery_status") == "found" or item.get("exe_path")),
        "launchable": sum(1 for item in discovery if item.get("launchable")),
        "bound": sum(1 for item in app_runs if item.get("bind_status") == "bound"),
        "captured": sum(1 for item in app_runs if item.get("capture_status") == "captured"),
        "canvas_created": sum(1 for item in app_runs if item.get("observe_status") == "canvas_created"),
        "query_matched": sum(1 for item in app_runs if item.get("query_status") == "matched"),
        "full_chain_passed": sum(
            1
            for item in app_runs
            if item.get("bind_status") == "bound"
            and item.get("capture_status") == "captured"
            and item.get("observe_status") == "canvas_created"
            and item.get("query_status") == "matched"
        ),
    }
    apps = []
    for app in app_runs:
        if not app.get("next_fix_task"):
            _stage, task = classify_entry_stage(app)
            app["next_fix_task"] = task
        if not app.get("product_value") and app.get("query_status") == "matched":
            app["product_value"] = (
                f"Agent can open or attach {app['software_name']}, see its window, build a canvas, and find target controls."
            )
        apps.append(app)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "product_chain": [
            "software_discovery",
            "launch",
            "bind_hwnd",
            "capture_diagnostics",
            "observe_interaction_canvas",
            "query_key_controls",
        ],
        "summary": summary,
        "discovery_matrix": discovery,
        "apps": apps,
        "acceptance": {
            "required_full_chain_passed": 3,
            "status": "passed" if summary["full_chain_passed"] >= 3 else "not_ready",
        },
    }


def write_report(report: dict[str, Any], output_root: Path = Path("reports/p0_entry")) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "p0_entry_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _ensure_db_schema() -> None:
    from src.storage.db import init_db

    init_db().create_all()


def _process_exe_path(process_id: int | None) -> str | None:
    if not process_id:
        return None
    try:
        import psutil

        path = psutil.Process(int(process_id)).exe()
        return path if path else None
    except Exception:
        return None


def _is_path_like_command(value: str) -> bool:
    return bool(value and not any(sep in value for sep in ["/", "\\"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeskCanvas P0 entry chain report")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--output-root", default="reports/p0_entry")
    parser.add_argument("--scan", action="store_true", help="Run software catalog scan before discovery matrix")
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from src.integration.api_server import app

    discovery = discover_software_matrix(args.targets, scan=args.scan)
    with TestClient(app) as client:
        app_runs = [
            run_entry_flow(client, item, allow_launch=args.allow_launch, allow_existing=args.allow_existing)
            for item in discovery
        ]
    report = build_p0_entry_report(discovery, app_runs)
    out_path = write_report(report, Path(args.output_root))
    print(json.dumps({"report_path": str(out_path), "summary": report["summary"], "acceptance": report["acceptance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
