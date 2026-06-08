"""Generate DeskCanvas P1 real-interface understanding quality reports.

P1 validates whether InteractionCanvas is useful for an Agent to understand
real software screens. It does not open input/send execution gates.
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

from scripts.p0_entry_report import (  # noqa: E402
    DEFAULT_TARGETS,
    DISPLAY_NAMES,
    classify_entry_stage,
    discover_software_matrix,
)


SCHEMA_VERSION = "deskcanvas_p1_quality_report.v1"
P05_SCHEMA_VERSION = "deskcanvas_p05_stage_report.v1"
DEFAULT_MIN_CORRECT_RATE = 0.7

P1_QUERY_MATRIX: dict[str, list[dict[str, Any]]] = {
    "wechat": [
        {"label": "搜索框", "target": {"natural_language": "找搜索框"}, "expected": {"semantic_roles": ["search_input"], "text_any": ["搜索"]}},
        {"label": "会话列表项", "target": {"natural_language": "找会话列表"}, "expected": {"semantic_roles": ["chat_item", "list_item"]}},
        {"label": "聊天输入区", "target": {"natural_language": "找聊天输入区"}, "expected": {"semantic_roles": ["message_input", "text_input"]}},
        {"label": "发送按钮", "target": {"natural_language": "找发送按钮"}, "expected": {"semantic_roles": ["send_button"], "text_any": ["发送"], "risk_required": True}},
    ],
    "qq": [
        {"label": "搜索框", "target": {"natural_language": "找搜索框"}, "expected": {"semantic_roles": ["search_input"], "text_any": ["搜索"]}},
        {"label": "聊天列表", "target": {"natural_language": "找聊天列表"}, "expected": {"semantic_roles": ["chat_item", "list_item"]}},
        {"label": "聊天输入框", "target": {"natural_language": "找输入框"}, "expected": {"semantic_roles": ["message_input", "text_input"]}},
        {"label": "发送按钮", "target": {"natural_language": "找发送按钮"}, "expected": {"semantic_roles": ["send_button"], "text_any": ["发送"]}},
        {"label": "联系人入口", "target": {"natural_language": "找联系人"}, "expected": {"semantic_roles": ["nav_item", "tab", "button"], "forbidden_roles": ["chat_item"]}},
    ],
    "feishu": [
        {"label": "搜索框", "target": {"natural_language": "找搜索框"}, "expected": {"semantic_roles": ["search_input"], "text_any": ["搜索"]}},
        {"label": "消息入口", "target": {"natural_language": "找会话列表"}, "expected": {"semantic_roles": ["nav_item", "tab", "list_item"], "text_any": ["消息"]}},
        {"label": "聊天输入区", "target": {"natural_language": "找聊天输入区"}, "expected": {"semantic_roles": ["message_input", "text_input"]}},
        {"label": "发送按钮", "target": {"natural_language": "找发送按钮"}, "expected": {"semantic_roles": ["send_button"], "text_any": ["发送"], "risk_required": True}},
    ],
    "chrome": [
        {"label": "地址/搜索框", "target": {"natural_language": "找地址栏"}, "expected": {"semantic_roles": ["search_input", "text_input"]}},
        {"label": "标签页", "target": {"natural_language": "找标签页"}, "expected": {"semantic_roles": ["tab", "menu_item"]}},
        {"label": "后退按钮", "target": {"natural_language": "找后退按钮"}, "expected": {"semantic_roles": ["button", "icon_button"], "negative_text": ["http", "www", ".com", ".cn", "google", "chrome", "chatgpt"], "forbidden_roles": ["search_input", "text_input"]}},
        {"label": "刷新按钮", "target": {"natural_language": "找刷新按钮"}, "expected": {"semantic_roles": ["button", "icon_button"], "negative_text": ["http", "www", ".com", ".cn", "google", "chrome", "chatgpt"], "forbidden_roles": ["search_input", "text_input"]}},
        {"label": "主内容区域", "target": {"natural_language": "找主内容区域"}, "expected": {"semantic_roles": ["text", "list_item", "document", "pane"]}},
    ],
    "vscode": [
        {"label": "搜索框", "target": {"natural_language": "找搜索框"}, "expected": {"semantic_roles": ["search_input", "text_input"]}},
        {"label": "文件资源管理器", "target": {"natural_language": "找文件资源管理器"}, "expected": {"semantic_roles": ["sidebar", "tree_item", "list_item"], "expected_region_roles": ["sidebar", "side_panel", "navigation"]}},
        {"label": "终端", "target": {"natural_language": "找终端"}, "expected": {"semantic_roles": ["text", "pane"], "expected_region_roles": ["panel", "bottom_panel", "terminal"], "negative_text": ["资源管理器", "文件", "设置"]}},
        {"label": "编辑区域", "target": {"natural_language": "找编辑区域"}, "expected": {"semantic_roles": ["text", "text_input", "document"], "expected_region_roles": ["editor", "main_content", "document"]}},
        {"label": "设置按钮", "target": {"natural_language": "找设置菜单"}, "expected": {"semantic_roles": ["menu_item", "button", "icon_button"]}},
    ],
    "netease_cloud_music": [
        {"label": "搜索框", "target": {"natural_language": "找搜索框"}, "expected": {"semantic_roles": ["search_input", "text_input"], "text_any": ["搜索"]}},
        {"label": "发现音乐", "target": {"natural_language": "找会话列表"}, "expected": {"semantic_roles": ["nav_item", "tab", "button"], "text_any": ["发现音乐"]}},
        {"label": "播放按钮", "target": {"natural_language": "找主内容区域"}, "expected": {"semantic_roles": ["button", "icon_button", "text", "list_item"]}},
    ],
    "flclash": [
        {"label": "代理入口", "target": {"natural_language": "代理"}, "expected": {"semantic_roles": ["nav_item", "button", "tab", "list_item"], "text_any": ["代理"]}},
        {"label": "主内容区域", "target": {"natural_language": "找主内容区域"}, "expected": {"semantic_roles": ["nav_item", "button", "tab", "text", "list_item"]}},
        {"label": "设置按钮", "target": {"natural_language": "找设置菜单"}, "expected": {"semantic_roles": ["button", "icon_button", "menu_item"], "text_any": ["设置"]}},
    ],
    "voicemeeter": [
        {"label": "A1 输出", "target": {"natural_language": "找 A1 输出按钮"}, "expected": {"semantic_roles": ["button", "toggle_button"], "text_any": ["A1"]}},
        {"label": "菜单按钮", "target": {"natural_language": "找设置菜单"}, "expected": {"semantic_roles": ["button", "menu_item"], "text_any": ["Menu"]}},
        {"label": "硬件输入", "target": {"natural_language": "找主内容区域"}, "expected": {"semantic_roles": ["text", "button"], "text_any": ["Hardware", "Input"]}},
    ],
}


def evaluate_canvas_quality(canvas, capture_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute P1 quality metrics from an InteractionCanvas."""
    capture_diagnostics = capture_diagnostics or {}
    elements = list(getattr(canvas, "elements", []) or [])
    artifacts = dict(getattr(canvas, "artifacts", {}) or {})
    provider_trace = getattr(canvas, "provider_trace", None)
    provider_details = dict(getattr(provider_trace, "provider_details", {}) or {})

    ocr_blocks = list(artifacts.get("ocr_blocks") or [])
    vision_candidates = list(artifacts.get("vision_candidates") or [])
    fallback_vision = [
        item for item in vision_candidates
        if str(item.get("source") or "").lower() not in {"vision", "omniparser"}
    ]

    uia_detail = provider_details.get("uia") or {}
    uia_provider = artifacts.get("uia_provider") or provider_details.get("uia_provider") or {}
    if not isinstance(uia_provider, dict):
        uia_provider = {}
    ocr_provider = artifacts.get("ocr_provider") or provider_details.get("ocr_bridge") or provider_details.get("ocr_provider") or {}
    if not isinstance(ocr_provider, dict):
        ocr_provider = {}
    element_sources = provider_details.get("element_sources") or {}
    uia_count = int(uia_detail.get("element_count") or element_sources.get("uia") or 0)
    if uia_count <= 0:
        uia_count = sum(1 for element in elements if _has_source(element, {"uia", "zone_partition"}))

    risky_count = sum(1 for element in elements if _is_risky(element))
    dangerous_unmarked_candidates = [
        _candidate_problem_summary(element)
        for element in elements
        if _is_dangerous(element) and not _is_risky(element)
    ]
    outside_candidates = [
        _candidate_problem_summary(element)
        for element in elements
        if _outside_capture_bounds(_element_bounds(element), capture_diagnostics)
    ]
    unknown_count = sum(1 for element in elements if _element_role(element) == "unknown")
    evidence_count = sum(1 for element in elements if _has_evidence(element))
    app_layout_candidates = [
        _candidate_problem_summary(element)
        for element in elements
        if _depends_on_app_layout(element)
    ]

    total = len(elements)
    return {
        "canvas_created": bool(getattr(canvas, "canvas_id", "")),
        "element_count": total,
        "ocr_block_count": len(ocr_blocks),
        "uia_candidate_count": uia_count,
        "fallback_vision_candidate_count": len(fallback_vision),
        "query_count": 0,
        "query_matched_count": 0,
        "query_correct_count": 0,
        "false_positive_count": 0,
        "risky_candidate_count": risky_count,
        "dangerous_unmarked_count": len(dangerous_unmarked_candidates),
        "dangerous_unmarked_candidates": dangerous_unmarked_candidates[:20],
        "candidates_outside_capture_bounds_count": len(outside_candidates),
        "outside_capture_bounds_candidates": outside_candidates[:20],
        "unknown_candidate_ratio": (unknown_count / total) if total else 0.0,
        "evidence_coverage": (evidence_count / total) if total else 0.0,
        "app_layout_dependency_count": len(app_layout_candidates),
        "app_layout_dependency_ratio": (len(app_layout_candidates) / total) if total else 0.0,
        "app_layout_dependent_candidates": app_layout_candidates[:20],
        "uia_provider_mode": uia_provider.get("mode"),
        "uia_provider_truncated": bool(uia_provider.get("truncated")),
        "uia_provider_elapsed_seconds": uia_provider.get("elapsed_seconds"),
        "uia_provider_element_count": uia_provider.get("element_count"),
        "ocr_provider_name": ocr_provider.get("provider"),
        "ocr_provider_success": ocr_provider.get("success"),
        "ocr_provider_error": ocr_provider.get("error"),
        "ocr_provider_elapsed_seconds": ocr_provider.get("elapsed_seconds"),
        "ocr_provider_worker_reused": ocr_provider.get("worker_reused"),
        "ocr_provider_startup_seconds": ocr_provider.get("startup_seconds"),
        "ocr_provider_fallback_reason": ocr_provider.get("fallback_reason"),
        "ocr_provider_block_count": ocr_provider.get("block_count"),
    }


def diagnose_query_match(
    *,
    app_id: str,
    query_spec: dict[str, Any],
    candidate: Any | None,
    screenshot_path: str | None = None,
) -> dict[str, Any]:
    """Build a human-readable diagnostic for a single query's top candidate.

    Verdict types:
    - TP (true positive): role/text match, no negative/forbidden hit
    - FP (false positive): matched but negative/forbidden hit, or role/text mismatch
    - AMB (ambiguous): role match but low confidence, or partial match
    - MISS (candidate missing): no candidate returned
    - MIN (window minimized): observe returned 422
    """
    target = query_spec.get("target") or {}
    expected = query_spec.get("expected") or {}

    if candidate is None:
        return {
            "app_id": app_id,
            "query_label": query_spec.get("label") or _format_target(target),
            "target": target,
            "matched": False,
            "verdict": "MISS",
            "reasons": ["no_candidate_matched"],
            "candidate_id": None,
            "candidate_text": "",
            "role": "",
            "semantic_role": "",
            "center": None,
            "confidence": 0.0,
            "evidence": [],
            "screenshot_evidence_path": screenshot_path,
        }

    role = _element_role(candidate)
    text = _element_text(candidate)
    bounds = _element_bounds(candidate)
    confidence = _element_confidence(candidate)
    evidence = _element_evidence(candidate)

    expected_roles = set(expected.get("semantic_roles") or [])
    expected_text = [str(item).lower() for item in expected.get("text_any") or []]
    text_blob = _candidate_text_blob(candidate).lower()
    target_text = str(target.get("text") or "").lower()

    role_match = role in expected_roles
    text_match = bool(target_text and target_text in text_blob) or any(token and token in text_blob for token in expected_text)

    # Strict evaluation: negative text, forbidden roles
    negative_text = [str(t).lower() for t in expected.get("negative_text") or []]
    negative_hit = any(token in text_blob for token in negative_text) if negative_text else False

    forbidden_roles = set(expected.get("forbidden_roles") or [])
    forbidden_hit = role in forbidden_roles if forbidden_roles else False

    # Build reasons
    reasons: list[str] = []
    if role_match:
        reasons.append(f"role_match:{role}")
    if text_match:
        reasons.append(f"text_match:{text or expected.get('text_any')}")
    if negative_hit:
        reasons.append(f"negative_hit:{[t for t in negative_text if t in text_blob]}")
    if forbidden_hit:
        reasons.append(f"forbidden_role:{role}")
    if confidence < 0.45:
        reasons.append(f"low_confidence:{confidence:.2f}")

    # Verdict logic
    if forbidden_hit:
        verdict = "FP"
        reasons.append("verdict:forbidden_role_hit")
    elif negative_hit and not text_match:
        verdict = "FP"
        reasons.append("verdict:negative_text_excludes")
    elif not role_match and not text_match:
        verdict = "FP"
        reasons.append("verdict:no_role_or_text_match")
    elif role_match and not negative_hit and confidence >= 0.45:
        verdict = "TP"
    elif text_match and not negative_hit and confidence >= 0.45:
        verdict = "TP"
    elif (role_match or text_match) and confidence < 0.45:
        verdict = "AMB"
        reasons.append("verdict:low_confidence_ambiguous")
    else:
        verdict = "FP"
        reasons.append("verdict:default_fp")

    return {
        "app_id": app_id,
        "query_label": query_spec.get("label") or _format_target(target),
        "target": target,
        "matched": True,
        "verdict": verdict,
        "reasons": reasons,
        "candidate_id": _element_id(candidate),
        "candidate_text": text,
        "role": role,
        "semantic_role": role,
        "center": _center(bounds),
        "confidence": confidence,
        "evidence": evidence,
        "risk_tags": _element_risk_tags(candidate),
        "risk_level": _element_risk_level(candidate),
        "screenshot_evidence_path": screenshot_path,
    }


def run_quality_flow(
    client,
    app: dict[str, Any],
    *,
    allow_launch: bool,
    allow_existing: bool,
    screenshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Run discovery->bind->observe->query and add P1 quality diagnostics."""
    from src.canvas.canvas_cache import get_canvas_cache

    app_id = app["app_id"]
    run: dict[str, Any] = {
        "app_id": app_id,
        "software_name": app.get("software_name") or DISPLAY_NAMES.get(app_id, app_id),
        "discovery_status": app.get("discovery_status", "not_found"),
        "launch_status": "not_run",
        "bind_status": "not_run",
        "capture_status": "not_run",
        "observe_status": "not_run",
        "quality_metrics": {},
        "query_diagnostics": [],
    }
    if run["discovery_status"] != "found":
        run["failure_stage"] = "discovery"
        run["failure_reason"] = "path_not_found"
        run["next_fix_task"] = "Add software path discovery or target alias coverage for this app."
        return run
    if not allow_launch and not allow_existing:
        run["failure_stage"] = "launch"
        run["failure_reason"] = "launch_not_allowed"
        run["next_fix_task"] = "Run with --allow-launch or --allow-existing for P1 real quality validation."
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

    capture_diagnostics = launch_data.get("capture_diagnostics") or {}
    run["capture_status"] = "captured" if capture_diagnostics.get("valid") else "failed"
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
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    screenshot_path = _write_screenshot_evidence(cache, canvas_id, app_id, screenshot_dir)
    run["screenshot_evidence_path"] = screenshot_path

    metrics = evaluate_canvas_quality(canvas, observe_data.get("capture_diagnostics") or capture_diagnostics) if canvas else {}
    matched = 0
    correct = 0
    false_positive = 0
    query_specs = P1_QUERY_MATRIX.get(app_id, [{"label": "按钮", "target": {"semantic_role": "button"}, "expected": {"semantic_roles": ["button", "icon_button"]}}])
    for query_spec in query_specs[:5]:
        query_resp = client.post(
            "/api/v1/query",
            json={"canvas_id": canvas_id, "target": query_spec["target"], "max_results": 3, "min_confidence": 0.0},
        )
        top_candidate: dict[str, Any] | None = None
        candidates: list[dict[str, Any]] = []
        if query_resp.status_code == 200:
            qdata = query_resp.json()
            candidates = list(qdata.get("candidates") or [])
            top_candidate = candidates[0] if candidates else None
        diag = diagnose_query_match(
            app_id=app_id,
            query_spec=query_spec,
            candidate=top_candidate,
            screenshot_path=screenshot_path,
        )
        diag["candidate_count"] = len(candidates)
        diag["candidate_ids"] = [item.get("element_id") for item in candidates]
        matched += 1 if diag["matched"] else 0
        correct += 1 if diag["heuristic_correct"] else 0
        false_positive += 1 if diag["matched"] and not diag["heuristic_correct"] else 0
        run["query_diagnostics"].append(diag)

    metrics.update(
        {
            "query_count": len(query_specs[:5]),
            "query_matched_count": matched,
            "query_correct_count": correct,
            "false_positive_count": false_positive,
        }
    )
    run["quality_metrics"] = metrics
    run["next_fix_task"] = _next_quality_fix_task(run)
    return run


def build_p1_quality_report(
    discovery: list[dict[str, Any]],
    app_runs: list[dict[str, Any]],
    *,
    min_apps: int = 5,
    min_correct_rate: float = DEFAULT_MIN_CORRECT_RATE,
) -> dict[str, Any]:
    coverage_gaps = [_coverage_gap(run) for run in app_runs if _coverage_gap(run)]
    app_layout_dependency_by_app = {
        str(run.get("app_id") or ""): int((run.get("quality_metrics") or {}).get("app_layout_dependency_count") or 0)
        for run in app_runs
        if int((run.get("quality_metrics") or {}).get("app_layout_dependency_count") or 0) > 0
    }
    summary = {
        "total_apps": len(app_runs),
        "canvas_created": sum(1 for run in app_runs if (run.get("quality_metrics") or {}).get("canvas_created")),
        "total_query_count": sum(int((run.get("quality_metrics") or {}).get("query_count") or 0) for run in app_runs),
        "query_matched_count": sum(int((run.get("quality_metrics") or {}).get("query_matched_count") or 0) for run in app_runs),
        "query_correct_count": sum(int((run.get("quality_metrics") or {}).get("query_correct_count") or 0) for run in app_runs),
        "false_positive_count": sum(int((run.get("quality_metrics") or {}).get("false_positive_count") or 0) for run in app_runs),
        "risky_candidate_count": sum(int((run.get("quality_metrics") or {}).get("risky_candidate_count") or 0) for run in app_runs),
        "dangerous_unmarked_count": sum(int((run.get("quality_metrics") or {}).get("dangerous_unmarked_count") or 0) for run in app_runs),
        "app_layout_dependency_count": sum(int((run.get("quality_metrics") or {}).get("app_layout_dependency_count") or 0) for run in app_runs),
        "candidates_outside_capture_bounds_count": sum(
            int((run.get("quality_metrics") or {}).get("candidates_outside_capture_bounds_count") or 0)
            for run in app_runs
        ),
        "coverage_gap_count": len(coverage_gaps),
        "coverage_gaps": coverage_gaps,
        "app_layout_dependency_by_app": app_layout_dependency_by_app,
    }
    total_query_count = summary["total_query_count"]
    summary["query_correct_rate"] = (
        summary["query_correct_count"] / total_query_count if total_query_count else 0.0
    )
    passed = (
        summary["canvas_created"] >= min_apps
        and total_query_count >= min_apps * 3
        and summary["query_correct_rate"] >= min_correct_rate
        and summary["dangerous_unmarked_count"] == 0
        and summary["candidates_outside_capture_bounds_count"] == 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "quality_chain": [
            "observe",
            "interaction_canvas",
            "query_key_controls",
            "query_diagnostics",
            "risk_and_bounds_audit",
        ],
        "summary": summary,
        "discovery_matrix": discovery,
        "apps": app_runs,
        "acceptance": {
            "required_canvas_created_apps": min_apps,
            "required_min_queries_per_app": 3,
            "required_min_correct_rate": min_correct_rate,
            "requires_zero_dangerous_unmarked": True,
            "requires_zero_outside_capture_bounds": True,
            "status": "passed" if passed else "not_ready",
        },
    }


def write_report(report: dict[str, Any], output_root: Path = Path("reports/p1_quality")) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "p1_quality_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, out_dir / "p1_quality_report.md")
    return out_path


def write_markdown_report(report: dict[str, Any], out_path: Path) -> None:
    """Write a compact P1.1 query-quality report for handoff review."""
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), dict) else {}
    lines = [
        "# P1.1 Natural Query Quality Report",
        "",
        f"- schema_version: `{report.get('schema_version', '')}`",
        f"- acceptance: `{acceptance.get('status', '')}`",
        f"- total_apps: {summary.get('total_apps', 0)}",
        f"- canvas_created: {summary.get('canvas_created', 0)}",
        f"- total_query_count: {summary.get('total_query_count', 0)}",
        f"- query_correct_count: {summary.get('query_correct_count', 0)}",
        f"- query_correct_rate: {_format_rate(summary.get('query_correct_rate', 0.0))}",
        f"- false_positive_count: {summary.get('false_positive_count', 0)}",
        f"- dangerous_unmarked_count: {summary.get('dangerous_unmarked_count', 0)}",
        f"- outside_bounds: {summary.get('candidates_outside_capture_bounds_count', 0)}",
        f"- app_layout_dependency_count: {summary.get('app_layout_dependency_count', 0)}",
        f"- coverage_gap_count: {summary.get('coverage_gap_count', 0)}",
        f"- app_layout_dependency_by_app: `{json.dumps(summary.get('app_layout_dependency_by_app') or {}, ensure_ascii=False)}`",
        "",
        "| app | queries | correct | false_positive | risk_missing | outside_bounds | app_layout | uia | ocr | next_fix |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for run in list(report.get("apps") or []):
        metrics = run.get("quality_metrics") if isinstance(run.get("quality_metrics"), dict) else {}
        lines.append(
            "| {app} | {queries} | {correct} | {fp} | {risk} | {outside} | {layout} | {uia} | {ocr} | {fix} |".format(
                app=_md(run.get("software_name") or run.get("app_id") or ""),
                queries=int(metrics.get("query_count") or 0),
                correct=int(metrics.get("query_correct_count") or 0),
                fp=int(metrics.get("false_positive_count") or 0),
                risk=int(metrics.get("dangerous_unmarked_count") or 0),
                outside=int(metrics.get("candidates_outside_capture_bounds_count") or 0),
                layout=int(metrics.get("app_layout_dependency_count") or 0),
                uia=_md(_format_uia_provider(metrics)),
                ocr=_md(_format_ocr_provider(metrics)),
                fix=_md(run.get("next_fix_task") or ""),
            )
        )
    coverage_gaps = [gap for gap in list(summary.get("coverage_gaps") or []) if isinstance(gap, dict)]
    if coverage_gaps:
        lines.extend(["", "## Coverage Gaps", "", "| app | stage | reason | next_fix |", "|---|---|---|---|"])
        for gap in coverage_gaps:
            lines.append(
                "| {app} | {stage} | {reason} | {fix} |".format(
                    app=_md(gap.get("software_name") or gap.get("app_id") or ""),
                    stage=_md(gap.get("stage") or ""),
                    reason=_md(gap.get("reason") or ""),
                    fix=_md(gap.get("next_fix_task") or ""),
                )
            )
    lines.extend(["", "## Query Diagnostics", ""])
    for run in list(report.get("apps") or []):
        app_name = str(run.get("software_name") or run.get("app_id") or "")
        for diag in list(run.get("query_diagnostics") or []):
            if not isinstance(diag, dict):
                continue
            status = "pass" if diag.get("heuristic_correct") else ("miss" if not diag.get("matched") else "false_positive")
            lines.append(
                "- {app} / {query}: `{status}` candidate=`{candidate}` reason=`{reason}`".format(
                    app=app_name,
                    query=str(diag.get("query_label") or ""),
                    status=status,
                    candidate=str(diag.get("candidate_id") or ""),
                    reason=str(diag.get("false_positive_reason") or diag.get("why_correct") or ""),
                )
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "0.00%"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def _format_uia_provider(metrics: dict[str, Any]) -> str:
    mode = str(metrics.get("uia_provider_mode") or "")
    if not mode:
        return ""
    elapsed_text = ""
    try:
        elapsed_text = f" {float(metrics.get('uia_provider_elapsed_seconds')):.1f}s"
    except (TypeError, ValueError):
        pass
    count = metrics.get("uia_provider_element_count")
    count_text = f" {count}el" if count is not None else ""
    truncated = " truncated" if metrics.get("uia_provider_truncated") else ""
    return f"{mode}{elapsed_text}{count_text}{truncated}".strip()


def _format_ocr_provider(metrics: dict[str, Any]) -> str:
    provider = str(metrics.get("ocr_provider_name") or "")
    if not provider:
        return ""
    elapsed_text = ""
    try:
        elapsed_text = f" {float(metrics.get('ocr_provider_elapsed_seconds')):.1f}s"
    except (TypeError, ValueError):
        pass
    startup_text = ""
    try:
        startup_text = f" startup:{float(metrics.get('ocr_provider_startup_seconds')):.1f}s"
    except (TypeError, ValueError):
        pass
    reused = metrics.get("ocr_provider_worker_reused")
    reused_text = ""
    if reused is True:
        reused_text = " reused"
    elif reused is False:
        reused_text = " cold"
    count = metrics.get("ocr_provider_block_count")
    count_text = f" {count}blk" if count is not None else ""
    fallback = str(metrics.get("ocr_provider_fallback_reason") or "")
    fallback_text = f" {fallback}" if fallback else ""
    error = str(metrics.get("ocr_provider_error") or "")
    error_text = f" error:{error}" if error else ""
    return f"{provider}{elapsed_text}{startup_text}{reused_text}{count_text}{fallback_text}{error_text}".strip()


def _coverage_gap(run: dict[str, Any]) -> dict[str, str] | None:
    metrics = run.get("quality_metrics") or {}
    if metrics.get("canvas_created") and int(metrics.get("query_count") or 0) > 0:
        return None
    stage = str(run.get("failure_stage") or "")
    if not stage:
        if run.get("discovery_status") != "found":
            stage = "discovery"
        elif run.get("observe_status") != "canvas_created":
            stage = "observe"
        elif int(metrics.get("query_count") or 0) == 0:
            stage = "query_matrix"
    if not stage:
        return None
    return {
        "app_id": str(run.get("app_id") or ""),
        "software_name": str(run.get("software_name") or DISPLAY_NAMES.get(str(run.get("app_id") or ""), run.get("app_id") or "")),
        "stage": stage,
        "reason": str(run.get("failure_reason") or ""),
        "next_fix_task": str(run.get("next_fix_task") or _next_quality_fix_task(run)),
    }


def build_p05_stage_report(
    discovery: list[dict[str, Any]],
    app_runs: list[dict[str, Any]],
    *,
    min_apps: int = 5,
    min_correct_rate: float = DEFAULT_MIN_CORRECT_RATE,
) -> dict[str, Any]:
    """Build the P0.5 real-app entry/canvas usability stage report."""
    discovery_by_app = {item.get("app_id"): item for item in discovery}
    rows = [_p05_stage_row(run, discovery_by_app.get(run.get("app_id"), {})) for run in app_runs]
    total_query_count = sum(int(row["query_count"]) for row in rows)
    query_correct_count = sum(int(row["query_correct_count"]) for row in rows)
    query_correct_rate = query_correct_count / total_query_count if total_query_count else 0.0
    summary = {
        "total_apps": len(rows),
        "discovered": sum(1 for row in rows if row["discovered"]),
        "launchable": sum(1 for row in rows if row["launchable"]),
        "bound": sum(1 for row in rows if row["bound"]),
        "captured": sum(1 for row in rows if row["captured"]),
        "observe_ok": sum(1 for row in rows if row["observe_ok"]),
        "canvas_created": sum(1 for row in rows if row["canvas_created"]),
        "total_query_count": total_query_count,
        "query_matched_count": sum(int(row["query_matched_count"]) for row in rows),
        "query_correct_count": query_correct_count,
        "false_positive_count": sum(int(row["false_positive_count"]) for row in rows),
        "risky_candidate_count": sum(int(row["risky_candidate_count"]) for row in rows),
        "outside_bounds_count": sum(int(row["outside_bounds_count"]) for row in rows),
        "query_correct_rate": query_correct_rate,
        "failed_apps": sum(1 for row in rows if row["failed_stage"]),
    }
    passed = (
        summary["canvas_created"] >= min_apps
        and total_query_count >= min_apps * 3
        and all(row["query_count"] >= 3 for row in rows)
        and query_correct_rate >= min_correct_rate
        and summary["outside_bounds_count"] == 0
        and all(not row["failed_stage"] for row in rows)
    )
    return {
        "schema_version": P05_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "stage_chain": [
            "discovered",
            "launchable",
            "bound",
            "captured",
            "observe_ok",
            "canvas_created",
            "query_key_controls",
            "risk_and_bounds_audit",
        ],
        "summary": summary,
        "apps": rows,
        "acceptance": {
            "required_canvas_created_apps": min_apps,
            "required_min_queries_per_app": 3,
            "required_min_correct_rate": min_correct_rate,
            "requires_zero_outside_bounds": True,
            "requires_no_failed_stage": True,
            "status": "passed" if passed else "not_ready",
        },
    }


def _p05_stage_row(run: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("quality_metrics") or {}
    failed_stage, next_fix_task = _p05_failure(run, metrics)
    return {
        "app_id": run.get("app_id"),
        "software_name": run.get("software_name") or DISPLAY_NAMES.get(str(run.get("app_id") or ""), run.get("app_id")),
        "discovered": run.get("discovery_status") == "found",
        "launchable": bool(discovery.get("launchable") or run.get("launchable")),
        "bound": run.get("bind_status") == "bound",
        "captured": run.get("capture_status") == "captured",
        "observe_ok": run.get("observe_status") == "canvas_created",
        "canvas_created": bool(metrics.get("canvas_created")),
        "candidate_count": int(metrics.get("element_count") or 0),
        "ocr_block_count": int(metrics.get("ocr_block_count") or 0),
        "uia_candidate_count": int(metrics.get("uia_candidate_count") or 0),
        "fallback_candidate_count": int(metrics.get("fallback_vision_candidate_count") or 0),
        "query_count": int(metrics.get("query_count") or 0),
        "query_matched_count": int(metrics.get("query_matched_count") or 0),
        "query_correct_count": int(metrics.get("query_correct_count") or 0),
        "false_positive_count": int(metrics.get("false_positive_count") or 0),
        "risky_candidate_count": int(metrics.get("risky_candidate_count") or 0),
        "outside_bounds_count": int(metrics.get("candidates_outside_capture_bounds_count") or 0),
        "failed_stage": failed_stage,
        "next_fix_task": next_fix_task,
    }


def _p05_failure(run: dict[str, Any], metrics: dict[str, Any]) -> tuple[str | None, str]:
    if run.get("discovery_status") != "found":
        return "discovery", "Add software discovery alias or launch path coverage for this app."
    if run.get("bind_status") != "bound":
        return "bind", "Fix launch/bind candidate selection for this app."
    if run.get("capture_status") != "captured":
        diagnostics = ((run.get("launch_bind") or {}).get("capture_diagnostics") or {})
        if diagnostics.get("is_minimized"):
            return "capture", "Restore the minimized window or bind a non-minimized top-level window before capture."
        return "capture", "Fix window bounds, minimization, DPI, or screenshot capture diagnostics."
    if run.get("observe_status") != "canvas_created":
        return "observe", "Fix capture/observe so this app produces InteractionCanvas."
    if not metrics.get("canvas_created"):
        return "canvas", "Fix InteractionCanvas creation for this app."
    if int(metrics.get("element_count") or 0) < 3:
        return "canvas_sparse", "Improve OCR/fallback candidate generation so this real app has enough queryable controls."
    if int(metrics.get("candidates_outside_capture_bounds_count") or 0) > 0:
        return "bounds", "Fix UIA/OCR/fallback coordinate normalization so candidates stay inside capture bounds."
    if int(metrics.get("dangerous_unmarked_count") or 0) > 0:
        return "risk", "Tag dangerous send/delete/submit/payment candidates with risk metadata."
    if int(metrics.get("query_count") or 0) < 3:
        return "query_matrix", "Define at least 3 key query targets for this real app."
    if int(metrics.get("query_correct_count") or 0) < int(metrics.get("query_count") or 0):
        return "query_quality", "Tighten query scoring and semantic role inference to reduce false positives."
    return None, "No P0.5 fix needed for this app."


def _write_screenshot_evidence(cache, canvas_id: str | None, app_id: str, screenshot_dir: Path | None) -> str | None:
    if not canvas_id or screenshot_dir is None:
        return None
    screenshot = cache.get_screenshot(canvas_id)
    if screenshot is None:
        return None
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    path = screenshot_dir / f"{app_id}_{canvas_id}.png"
    screenshot.save(str(path))
    return str(path)


def _next_quality_fix_task(run: dict[str, Any]) -> str:
    metrics = run.get("quality_metrics") or {}
    if run.get("observe_status") != "canvas_created":
        return "Fix observe so this app produces InteractionCanvas."
    if metrics.get("candidates_outside_capture_bounds_count", 0) > 0:
        return "Fix UIA/OCR coordinate normalization so candidates stay inside capture bounds."
    if metrics.get("dangerous_unmarked_count", 0) > 0:
        return "Tag dangerous send/delete/submit/payment candidates with risk metadata."
    if metrics.get("query_matched_count", 0) < 3:
        return "Improve search/input/menu/list query detection for this app."
    if metrics.get("false_positive_count", 0) > 0:
        return "Tighten query scoring and semantic role inference to reduce false positives."
    if metrics.get("evidence_coverage", 0.0) < 0.8:
        return "Improve candidate evidence coverage with provider sources, locators, or role evidence."
    return "No P1 quality fix needed for this report."


def _element_id(candidate: Any) -> str:
    return str(_get(candidate, "element_id") or "")


def _element_role(candidate: Any) -> str:
    role = _get(candidate, "semantic_role")
    return str(role.value if hasattr(role, "value") else role or "unknown")


def _element_text(candidate: Any) -> str:
    return str(_get(candidate, "text") or _get(candidate, "name") or _get(candidate, "role_label") or "")


def _candidate_text_blob(candidate: Any) -> str:
    tags = _get(candidate, "semantic_tags") or []
    return " ".join(
        [
            str(_get(candidate, "text") or ""),
            str(_get(candidate, "name") or ""),
            str(_get(candidate, "placeholder") or ""),
            str(_get(candidate, "role_label") or ""),
            " ".join(str(tag) for tag in tags),
            _element_role(candidate),
            str(_get(candidate, "control_type") or ""),
        ]
    )


def _element_bounds(candidate: Any) -> tuple[int, int, int, int] | None:
    bounds = _get(candidate, "bounds")
    if not bounds or len(bounds) < 4:
        return None
    return tuple(int(value) for value in bounds[:4])


def _element_confidence(candidate: Any) -> float:
    try:
        return float(_get(candidate, "confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _element_risk_tags(candidate: Any) -> list[str]:
    return list(_get(candidate, "risk_tags") or [])


def _element_risk_level(candidate: Any) -> str:
    level = _get(candidate, "risk_level")
    return str(level.value if hasattr(level, "value") else level or "L0")


def _element_sources(candidate: Any) -> list[str]:
    return list(_get(candidate, "provider_sources") or [])


def _element_evidence(candidate: Any) -> list[str]:
    evidence: list[str] = []
    for source in _element_sources(candidate):
        evidence.append(f"provider:{source}")
    for locator_id in list(_get(candidate, "locator_ids") or []):
        evidence.append(f"locator:{locator_id}")
    for item in list(_get(candidate, "role_evidence") or []):
        evidence.append(f"role:{item}")
    attrs = _get(candidate, "attributes") or {}
    if isinstance(attrs, dict):
        for key in ["ocr_text", "ocr_bbox", "vision_candidate", "automation_id"]:
            if attrs.get(key):
                evidence.append(f"attr:{key}")
    return evidence


def _candidate_problem_summary(candidate: Any) -> dict[str, Any]:
    return {
        "element_id": _element_id(candidate),
        "text": _element_text(candidate),
        "semantic_role": _element_role(candidate),
        "bounds": list(_element_bounds(candidate) or []),
        "center": _center(_element_bounds(candidate)),
        "provider_sources": _element_sources(candidate),
        "risk_tags": _element_risk_tags(candidate),
        "risk_level": _element_risk_level(candidate),
    }


def _has_evidence(candidate: Any) -> bool:
    return bool(_element_evidence(candidate))


def _has_source(candidate: Any, sources: set[str]) -> bool:
    return bool(set(_element_sources(candidate)) & sources)


def _depends_on_app_layout(candidate: Any) -> bool:
    if "app_layout" in set(_element_sources(candidate)):
        return True
    attrs = _get(candidate, "attributes") or {}
    if isinstance(attrs, dict):
        return str(attrs.get("synthetic_source") or attrs.get("candidate_origin") or "") == "app_layout"
    return False


def _is_risky(candidate: Any) -> bool:
    return bool(_element_risk_tags(candidate)) or _element_risk_level(candidate) not in {"", "L0", "low"}


def _is_dangerous(candidate: Any) -> bool:
    role = _element_role(candidate)
    text = _candidate_text_blob(candidate).lower()
    if role in {"send_button", "submit_button", "delete_button"}:
        return _is_short_action_text(_element_text(candidate))
    control_type = str(_get(candidate, "control_type") or "").lower()
    actionable_roles = {
        "button",
        "icon_button",
        "menu_item",
        "toggle_button",
        "checkbox",
        "radio_button",
    }
    actionable_control = any(token in control_type for token in ["button", "menu", "item", "split"])
    if role not in actionable_roles and not actionable_control:
        return False
    return any(
        token in text
        for token in ["发送", "send", "删除", "delete", "提交", "submit", "支付", "付款", "pay", "transfer", "转账"]
    )


def _is_short_action_text(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return not compact or (len(compact) <= 32 and len(str(text or "").split()) <= 5)


def _outside_capture_bounds(bounds: tuple[int, int, int, int] | None, capture_diagnostics: dict[str, Any]) -> bool:
    if not bounds:
        return False
    size = capture_diagnostics.get("screenshot_size") or []
    width = int(size[0]) if len(size) >= 2 else 0
    height = int(size[1]) if len(size) >= 2 else 0
    if width <= 0 or height <= 0:
        capture_bounds = capture_diagnostics.get("capture_bounds") or []
        if len(capture_bounds) >= 4:
            width = int(capture_bounds[2]) - int(capture_bounds[0])
            height = int(capture_bounds[3]) - int(capture_bounds[1])
    if width <= 0 or height <= 0:
        return False
    left, top, right, bottom = bounds
    return left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top


def _center(bounds: tuple[int, int, int, int] | None) -> list[int] | None:
    if not bounds:
        return None
    left, top, right, bottom = bounds
    return [left + (right - left) // 2, top + (bottom - top) // 2]


def _format_target(target: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in target.items() if value is not None)


def _get(candidate: Any, key: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key)
    return getattr(candidate, key, None)


def configure_p05_local_only_providers() -> None:
    """Disable slow external OCR/vision providers for deterministic P0.5 local runs."""
    try:
        from src.perception.ocr_service import get_ocr_service

        ocr = get_ocr_service()
        ocr._config.bridge_enabled = False
        ocr._config.worker_timeout_seconds = min(int(ocr._config.worker_timeout_seconds), 5)
    except Exception:
        pass

    try:
        from src.perception.providers.remote_vision_provider import (
            OmniParserRemoteVisionProvider,
            VisionParseResult,
        )

        def _local_only_parse(self, image):
            return VisionParseResult(
                candidates=[],
                provider=getattr(self._config, "provider", "omniparser"),
                success=False,
                error="p05_local_only_provider_disabled",
            )

        OmniParserRemoteVisionProvider.parse_screenshot = _local_only_parse
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeskCanvas P1 interface understanding quality report")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--output-root", default="reports/p1_quality")
    parser.add_argument("--scan", action="store_true", help="Run software catalog scan before discovery matrix")
    parser.add_argument("--min-apps", type=int, default=5)
    parser.add_argument("--min-correct-rate", type=float, default=DEFAULT_MIN_CORRECT_RATE)
    parser.add_argument(
        "--p05-local-only",
        action="store_true",
        help="Disable slow external OCR/vision providers for P0.5 local UIA/app-layout validation.",
    )
    args = parser.parse_args()

    if args.p05_local_only:
        configure_p05_local_only_providers()

    from fastapi.testclient import TestClient
    from src.integration.api_server import app

    output_root = Path(args.output_root)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    screenshot_dir = run_dir / "screenshots"

    discovery = discover_software_matrix(args.targets, scan=args.scan)
    with TestClient(app) as client:
        app_runs = [
            run_quality_flow(
                client,
                item,
                allow_launch=args.allow_launch,
                allow_existing=args.allow_existing,
                screenshot_dir=screenshot_dir,
            )
            for item in discovery
        ]
    report = build_p1_quality_report(
        discovery,
        app_runs,
        min_apps=args.min_apps,
        min_correct_rate=args.min_correct_rate,
    )
    stage_report = build_p05_stage_report(
        discovery,
        app_runs,
        min_apps=args.min_apps,
        min_correct_rate=args.min_correct_rate,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "p1_quality_report.json"
    stage_out_path = run_dir / "p05_stage_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, run_dir / "p1_quality_report.md")
    stage_out_path.write_text(json.dumps(stage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report_path": str(out_path),
        "stage_report_path": str(stage_out_path),
        "summary": report["summary"],
        "stage_summary": stage_report["summary"],
        "acceptance": report["acceptance"],
        "stage_acceptance": stage_report["acceptance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
