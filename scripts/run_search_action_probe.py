"""Run controlled search probes from a search action plan.

The script clicks the recorded search entry, pastes a query, waits for the UI
to settle, captures screenshots, observes the target window, and optionally
clicks the first extracted search result for a second observe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.windows import window_action_context as action_ctx


DEFAULT_QUERIES = {
    "qq.exe": "jz",
    "weixin.exe": "念",
    "wechat.exe": "念",
    "feishu.exe": "大笨蛋",
    "lark.exe": "大笨蛋",
}

def build_probe_targets(
    plan: dict[str, Any],
    *,
    queries: dict[str, str],
    current_windows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return executable search targets with per-app queries attached."""
    normalized_queries = {key.lower(): value for key, value in queries.items() if value}
    targets: list[dict[str, Any]] = []
    for row in list(plan.get("rows") or []):
        if str(row.get("status") or "") != "ready":
            continue
        process = str(row.get("process_name") or "").lower()
        click = _point(row.get("search_click"))
        if not process or not click:
            continue
        query = normalized_queries.get(process) or DEFAULT_QUERIES.get(process)
        if not query:
            continue
        target = dict(row)
        target["process_name"] = process
        target["search_click"] = click
        target["query"] = query
        target["probe_strategy"] = _probe_strategy_for_process(process)
        original_hwnd = int(target.get("hwnd") or 0)
        refreshed_hwnd = 0 if _window_is_actionable(original_hwnd) else _fresh_hwnd_for_process(process, current_windows or [])
        if refreshed_hwnd and refreshed_hwnd != original_hwnd:
            target["original_hwnd"] = int(target.get("hwnd") or 0)
            target["hwnd"] = refreshed_hwnd
            target["hwnd_refreshed"] = True
        targets.append(target)
    return targets


def _missing_ready_processes(plan: dict[str, Any], *, current_windows: list[dict[str, Any]]) -> list[str]:
    """Return ready plan processes that currently have no visible window."""
    missing: list[str] = []
    for row in list(plan.get("rows") or []):
        if str(row.get("status") or "") != "ready":
            continue
        process = str(row.get("process_name") or "").lower()
        if not process or process in missing:
            continue
        if _fresh_hwnd_for_process(process, current_windows):
            continue
        missing.append(process)
    return missing


def _fresh_hwnd_for_process(process: str, current_windows: list[dict[str, Any]]) -> int:
    """Return the best visible hwnd for a process from a live window listing."""
    process = process.lower()
    for item in current_windows:
        if str(item.get("process_name") or "").lower() != process:
            continue
        hwnd = int(item.get("hwnd") or 0)
        if hwnd <= 0:
            continue
        if item.get("is_minimized") is True:
            continue
        if not _window_is_actionable(hwnd):
            continue
        width = item.get("width")
        height = item.get("height")
        if width is not None and height is not None and int(width or 0) <= 0 and int(height or 0) <= 0:
            continue
        return hwnd
    return 0


def _tray_control_matches(process: str, control_name: str) -> bool:
    """Return whether a system tray control name belongs to the process."""
    return action_ctx.tray_control_matches(process, control_name)


def extract_result_options(
    detail: dict[str, Any],
    *,
    process_name: str,
    query: str,
) -> list[dict[str, Any]]:
    """Extract selectable search result options from an observed search state."""
    mode = str((detail.get("roi_selection_plan") or {}).get("mode") or "")
    elements = [item for item in list(detail.get("elements") or []) if isinstance(item, dict)]
    process = process_name.lower()
    if process in {"qq.exe"} and mode == "chat_search_results":
        return _extract_qq_result_options(elements, query=query)
    if process in {"weixin.exe", "wechat.exe"} and mode == "chat_search_results":
        text_options = _extract_text_result_options(elements, query=query, x_range=(55, 340), y_min=80)
        return text_options or _extract_wechat_contact_fragment_options(elements) or _extract_wechat_visual_result_options(elements)
    if process in {"feishu.exe", "lark.exe"} and mode == "collaboration_search_overlay":
        return _extract_text_result_options(elements, query=query, x_range=(100, 930), y_min=115)
    return []


def _extract_qq_result_options(elements: list[dict[str, Any]], *, query: str) -> list[dict[str, Any]]:
    options = []
    needle = query.lower()
    for item in elements:
        text = _element_text(item)
        bounds = _rect(item.get("bounds"))
        if not bounds:
            continue
        left, top, right, bottom = bounds
        if left < 55 or right > 330 or top < 80:
            continue
        if needle not in text.lower():
            continue
        if "进入全网搜索" in text or "查找用户" in text:
            continue
        options.append(_result_option(item, bounds, source="text_match"))
    return _dedupe_options(options)


def _extract_text_result_options(
    elements: list[dict[str, Any]],
    *,
    query: str,
    x_range: tuple[int, int],
    y_min: int,
) -> list[dict[str, Any]]:
    options = []
    needle = query.lower()
    for item in elements:
        text = _element_text(item)
        bounds = _rect(item.get("bounds"))
        if not bounds or not text:
            continue
        left, top, right, bottom = bounds
        if left < x_range[0] or right > x_range[1] or top < y_min:
            continue
        if needle not in text.lower():
            continue
        if "搜索" in text and top < 160:
            continue
        options.append(_result_option(item, bounds, source="text_match"))
    return _dedupe_options(options)


def _extract_wechat_visual_result_options(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    web_section_top = _wechat_web_search_section_top(elements)
    options = []
    for item in elements:
        bounds = _rect(item.get("bounds"))
        if not bounds:
            continue
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        if web_section_top is not None and top >= web_section_top:
            continue
        if 55 <= left <= 70 and 180 <= right <= 340 and 80 <= top <= 520 and 42 <= height <= 90 and width >= 180:
            options.append(_result_option(item, bounds, source="visual_row_fallback"))
    return _dedupe_options(options)


def _wechat_web_search_section_top(elements: list[dict[str, Any]]) -> int | None:
    for item in elements:
        text = _element_text(item)
        if not text:
            continue
        normalized = text.replace(" ", "")
        if "搜索网络结果" not in normalized and "搜一搜" not in normalized:
            continue
        bounds = _rect(item.get("bounds"))
        if not bounds:
            continue
        _left, top, _right, _bottom = bounds
        return top
    return None


def _extract_wechat_contact_fragment_options(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    options = []
    for item in elements:
        bounds = _rect(item.get("bounds"))
        if not bounds:
            continue
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        if not (62 <= left <= 90 and 80 <= top <= 165 and 35 <= width <= 65 and 35 <= height <= 70):
            continue
        if not _has_adjacent_wechat_result_text_fragment(elements, bounds):
            continue
        row_bounds = [60, max(80, top - 6), 396, min(175, bottom + 10)]
        option = _result_option(item, row_bounds, source="contact_row_fragment")
        option["click"] = [178, int((top + bottom) / 2)]
        options.append(option)
    return _dedupe_options(options)


def _has_adjacent_wechat_result_text_fragment(elements: list[dict[str, Any]], avatar_bounds: list[int]) -> bool:
    _left, top, right, bottom = avatar_bounds
    for item in elements:
        bounds = _rect(item.get("bounds"))
        if not bounds:
            continue
        left, text_top, text_right, text_bottom = bounds
        if left < right - 4 or text_right > 260:
            continue
        if text_bottom < top - 8 or text_top > bottom + 8:
            continue
        if text_right - left >= 18 and text_bottom - text_top >= 10:
            return True
    return False


def _result_option(item: dict[str, Any], bounds: list[int], *, source: str) -> dict[str, Any]:
    return {
        "element_id": str(item.get("element_id") or ""),
        "text": _element_text(item),
        "bounds": bounds,
        "click": _center(bounds),
        "source": source,
        "selection_policy": "first_matching_result",
    }


def _dedupe_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int]] = set()
    deduped = []
    for option in sorted(options, key=lambda item: (item["bounds"][1], item["bounds"][0], -(item["bounds"][3] - item["bounds"][1]))):
        bounds_key = tuple(option["bounds"])
        if bounds_key in seen:
            continue
        seen.add(bounds_key)
        deduped.append(option)
    return deduped


def run_probe_plan(
    *,
    plan: dict[str, Any],
    output_dir: Path,
    base_url: str,
    queries: dict[str, str],
    execute: bool,
    wait_seconds: float,
) -> dict[str, Any]:
    """Execute or dry-run search probes and return a report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    before_windows = _fetch_windows(base_url)
    restored = []
    if execute:
        for process in _missing_ready_processes(plan, current_windows=before_windows):
            if _restore_process_from_tray(process):
                restored.append(process)
        if restored:
            time.sleep(1.0)
            before_windows = _fetch_windows(base_url)
    targets = build_probe_targets(plan, queries=queries, current_windows=before_windows)
    for target in targets:
        row = _run_one_target(
            target=target,
            output_dir=output_dir,
            base_url=base_url,
            before_windows=before_windows,
            execute=execute,
            wait_seconds=wait_seconds,
        )
        rows.append(row)
    overall = _probe_overall_status(rows, execute=execute)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "execute": execute,
        "restored_from_tray": restored,
        "rows": rows,
    }
    write_probe_report(report, output_dir)
    return report


def _probe_overall_status(rows: list[dict[str, Any]], *, execute: bool) -> str:
    if not rows:
        return "blocked"
    if not execute:
        return "dry_run"
    for row in rows:
        notes = set(row.get("notes") or [])
        transition = row.get("state_transition") if isinstance(row.get("state_transition"), dict) else {}
        path = str(transition.get("path") or "")
        if row.get("status") != "captured":
            return "review"
        if "search_state_captured" not in notes or "selected_first_result" not in notes:
            return "review"
        if "->" not in path:
            return "review"
    return "pass"


def load_plan(path: Path) -> dict[str, Any]:
    """Load a JSON probe plan, accepting UTF-8 files with or without BOM."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def extract_canvas_state(detail: dict[str, Any]) -> dict[str, Any]:
    """Summarize a canvas detail into the model/state fields needed for transition auditing."""
    roi_plan = detail.get("roi_selection_plan") if isinstance(detail.get("roi_selection_plan"), dict) else {}
    visual_pattern = detail.get("visual_pattern") if isinstance(detail.get("visual_pattern"), dict) else {}
    mode = str(roi_plan.get("mode") or visual_pattern.get("mode") or detail.get("page_class") or "")
    return {
        "canvas_id": str(detail.get("canvas_id") or ""),
        "page_model_id": str(detail.get("page_model_id") or ""),
        "state_template_id": str(detail.get("state_template_id") or ""),
        "page_class": str(detail.get("page_class") or ""),
        "state_label": str(detail.get("state_label") or ""),
        "mode": mode,
        "window_title": str(detail.get("window_title") or ""),
    }


def build_state_transition_audit(
    *,
    before: dict[str, Any],
    search: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact transition report for before -> search -> selected stages."""
    return {
        "before_to_search": _compare_stage_states(before, search),
        "search_to_selected": _compare_stage_states(search, selected),
        "selected_returned_to_before_state": bool(
            before.get("state_template_id")
            and selected.get("state_template_id")
            and before.get("state_template_id") == selected.get("state_template_id")
        ),
        "selected_same_page_model_as_before": bool(
            before.get("page_model_id")
            and selected.get("page_model_id")
            and before.get("page_model_id") == selected.get("page_model_id")
        ),
        "path": " -> ".join(
            [str(item.get("mode") or "") for item in (before, search, selected) if item.get("mode")]
        ),
    }


def _compare_stage_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    return {
        "page_model_changed": bool(left.get("page_model_id") and right.get("page_model_id") and left.get("page_model_id") != right.get("page_model_id")),
        "state_template_changed": bool(
            left.get("state_template_id") and right.get("state_template_id") and left.get("state_template_id") != right.get("state_template_id")
        ),
        "mode_changed": bool(left.get("mode") and right.get("mode") and left.get("mode") != right.get("mode")),
    }


def write_probe_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist probe report JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Search Probe Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Execute: {report.get('execute', '')}",
        "",
        "| sample | process | query | status | after_canvas | selected_canvas | result_count | selected_click | state_path | after_screenshot | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        selected_result = row.get("selected_result") if isinstance(row.get("selected_result"), dict) else {}
        state_transition = row.get("state_transition") if isinstance(row.get("state_transition"), dict) else {}
        lines.append(
            "| {sample} | {process} | {query} | {status} | {canvas} | {selected_canvas} | {result_count} | {selected_click} | {state_path} | {shot} | {notes} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                query=str(row.get("query") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                canvas=str(row.get("after_canvas_id") or "").replace("|", "/"),
                selected_canvas=str(row.get("selected_canvas_id") or "").replace("|", "/"),
                result_count=str(row.get("result_candidate_count") or 0),
                selected_click=str(selected_result.get("click") or "").replace("|", "/"),
                state_path=str(state_transition.get("path") or "").replace("|", "/"),
                shot=str(row.get("after_screenshot") or "").replace("|", "/"),
                notes=",".join(row.get("notes") or []),
            )
        )
    (output_dir / "search_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_one_target(
    *,
    target: dict[str, Any],
    output_dir: Path,
    base_url: str,
    before_windows: list[dict[str, Any]],
    execute: bool,
    wait_seconds: float,
) -> dict[str, Any]:
    hwnd = int(target.get("hwnd") or 0)
    process = str(target.get("process_name") or "")
    sample = str(target.get("sample") or f"{process}_{hwnd}")
    query = str(target.get("query") or "")
    notes: list[str] = []
    before_canvas_id = ""
    after_canvas_id = ""
    selected_canvas_id = ""
    before_screenshot = ""
    after_screenshot = ""
    selected_screenshot = ""
    result_options: list[dict[str, Any]] = []
    selected_result: dict[str, Any] = {}
    stage_states: dict[str, dict[str, Any]] = {}
    state_transition: dict[str, Any] = {}
    status = "dry_run"
    error = ""
    try:
        if execute:
            _ensure_window_actionable(hwnd, process)
            _reset_probe_surface(hwnd=hwnd, process=process)
            before_canvas_id = _observe(base_url, hwnd)
            before_detail = _fetch_canvas_detail(base_url, before_canvas_id)
            _save_detail(output_dir, sample, "before", before_canvas_id, before_detail)
            stage_states["before"] = extract_canvas_state(before_detail)
            before_screenshot = _save_screen(output_dir, sample, "before")
            _focus_and_type(
                hwnd=hwnd,
                click=_point(target.get("search_click")),
                text=query,
                strategy=str(target.get("probe_strategy") or "click_search"),
            )
            time.sleep(wait_seconds)
            after_screenshot = _save_screen(output_dir, sample, "after")
            after_canvas_id = _observe(base_url, hwnd, capture_mode=_search_observe_capture_mode(process))
            after_detail = _fetch_canvas_detail(base_url, after_canvas_id)
            _save_detail(output_dir, sample, "search", after_canvas_id, after_detail)
            stage_states["search"] = extract_canvas_state(after_detail)
            search_mode_captured = _search_mode_captured(process, stage_states["search"])
            if search_mode_captured:
                notes.append("search_state_captured")
                result_options = extract_result_options(after_detail, process_name=process, query=query)
            else:
                notes.append("search_state_not_captured")
            if result_options:
                selected_result = result_options[0]
                _click_relative(hwnd=hwnd, click=selected_result["click"])
                _post_selection_cleanup(hwnd=hwnd, process=process)
                time.sleep(wait_seconds)
                selected_screenshot = _save_screen(output_dir, sample, "selected")
                selected_canvas_id = _observe(base_url, hwnd)
                selected_detail = _fetch_canvas_detail(base_url, selected_canvas_id)
                _save_detail(output_dir, sample, "selected", selected_canvas_id, selected_detail)
                stage_states["selected"] = extract_canvas_state(selected_detail)
                notes.append("selected_first_result")
            state_transition = build_state_transition_audit(
                before=stage_states.get("before", {}),
                search=stage_states.get("search", {}),
                selected=stage_states.get("selected", {}),
            )
            after_windows = _fetch_windows(base_url)
            if _new_related_window_seen(before_windows, after_windows, process):
                notes.append("new_related_window_seen")
            status = "captured"
    except Exception as exc:  # pragma: no cover - depends on live desktop.
        status = "error"
        error = str(exc)
    return {
        "sample": sample,
        "hwnd": hwnd,
        "title": str(target.get("title") or ""),
        "process_name": process,
        "query": query,
        "search_click": _point(target.get("search_click")),
        "status": status,
        "before_canvas_id": before_canvas_id,
        "after_canvas_id": after_canvas_id,
        "selected_canvas_id": selected_canvas_id,
        "before_screenshot": before_screenshot,
        "after_screenshot": after_screenshot,
        "selected_screenshot": selected_screenshot,
        "result_candidate_count": len(result_options),
        "result_options": result_options[:8],
        "selected_result": selected_result,
        "stage_states": stage_states,
        "state_transition": state_transition,
        "notes": notes,
        "error": error,
    }


def _search_mode_captured(process: str, state: dict[str, Any]) -> bool:
    mode = str(state.get("mode") or "")
    process = process.lower()
    if process in {"weixin.exe", "wechat.exe", "qq.exe"}:
        return mode == "chat_search_results"
    if process in {"feishu.exe", "lark.exe"}:
        return mode == "collaboration_search_overlay"
    return bool(mode and mode != "unknown_pattern")


def _search_observe_capture_mode(process: str) -> str:
    return action_ctx.chat_search_capture_mode(process)


def _focus_and_type(*, hwnd: int, click: list[int], text: str, strategy: str) -> None:
    import pyautogui
    import win32gui

    if strategy == "click_search" and not click:
        raise ValueError("missing search click")
    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        raise ValueError(f"invalid hwnd: {hwnd}")
    _bring_window_to_front(hwnd)
    left, top, _right, _bottom = win32gui.GetWindowRect(hwnd)
    if strategy == "global_search_hotkey":
        pyautogui.hotkey("ctrl", "k")
        time.sleep(0.45)
        overlay_click = _global_search_input_click([left, top, _right, _bottom])
        pyautogui.click(left + overlay_click[0], top + overlay_click[1])
        time.sleep(0.15)
    else:
        pyautogui.click(left + click[0], top + click[1])
        time.sleep(0.25)
    pyautogui.hotkey("ctrl", "a")
    _paste_text(text)


def _paste_text(text: str) -> None:
    import pyautogui
    import win32clipboard
    import win32con

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()
    pyautogui.hotkey("ctrl", "v")


def _reset_probe_surface(*, hwnd: int, process: str) -> None:
    import pyautogui
    import win32gui

    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        return
    _ensure_window_actionable(hwnd, process)
    _bring_window_to_front(hwnd)
    for _ in range(_pre_probe_escape_count(process)):
        pyautogui.press("esc")
        time.sleep(0.15)


def _pre_probe_escape_count(process: str) -> int:
    return action_ctx.pre_probe_escape_count(process)


def _post_selection_cleanup(*, hwnd: int, process: str) -> None:
    import pyautogui

    _bring_window_to_front(hwnd)
    for _ in range(_post_selection_escape_count(process)):
        pyautogui.press("esc")
        time.sleep(0.15)


def _post_selection_escape_count(process: str) -> int:
    return action_ctx.post_selection_escape_count(process)


def _bring_window_to_front(hwnd: int) -> None:
    action_ctx.bring_window_to_front(hwnd)


def _ensure_window_actionable(hwnd: int, process: str) -> None:
    action_ctx.ensure_window_actionable(hwnd, process)


def _restore_hidden_window(hwnd: int) -> bool:
    return action_ctx.restore_hidden_window(hwnd)


def _wait_until_window_actionable(hwnd: int, *, timeout_seconds: float = 8.0, interval_seconds: float = 0.25) -> bool:
    return action_ctx.wait_until_window_actionable(hwnd, timeout_seconds=timeout_seconds, interval_seconds=interval_seconds)


def _window_is_actionable(hwnd: int) -> bool:
    return action_ctx.window_is_actionable(hwnd)


def _foreground_matches_target(hwnd: int) -> bool:
    return action_ctx.foreground_matches_target(hwnd)


def _global_search_input_click(window_rect: list[int]) -> list[int]:
    left, top, right, bottom = window_rect
    return [int((right - left) * 0.5), min(90, max(60, int((bottom - top) * 0.12)))]


def _save_screen(output_dir: Path, sample: str, stage: str) -> str:
    import pyautogui

    path = output_dir / "screenshots" / f"{_safe_filename(sample)}_{stage}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pyautogui.screenshot(str(path))
    return str(path)


def _save_detail(output_dir: Path, sample: str, stage: str, canvas_id: str, detail: dict[str, Any]) -> str:
    if not canvas_id or not detail:
        return ""
    path = output_dir / "details" / f"{_safe_filename(sample)}_{stage}_{_safe_filename(canvas_id)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _click_relative(*, hwnd: int, click: list[int]) -> None:
    import pyautogui
    import win32gui

    if not click:
        raise ValueError("missing result click")
    _bring_window_to_front(hwnd)
    left, top, _right, _bottom = win32gui.GetWindowRect(hwnd)
    pyautogui.click(left + click[0], top + click[1])


def _observe(base_url: str, hwnd: int, *, capture_mode: str = "window") -> str:
    if hwnd <= 0:
        return ""
    response = requests.post(
        f"{base_url.rstrip('/')}/api/v1/observe",
        json={
            "hwnd": hwnd,
            "async_enhance": False,
            "allow_vlm": False,
            "force_vlm": False,
            "capture_mode": capture_mode,
        },
        timeout=45,
    )
    response.raise_for_status()
    return str((response.json() or {}).get("canvas_id") or "")


def _fetch_canvas_detail(base_url: str, canvas_id: str) -> dict[str, Any]:
    if not canvas_id:
        return {}
    response = requests.get(f"{base_url.rstrip('/')}/api/v1/canvases/{canvas_id}", timeout=20)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _fetch_windows(base_url: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/v1/windows", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in list(data.get("windows") or []) if isinstance(item, dict)]
    return []


def _restore_process_from_tray(process: str) -> bool:
    """Best-effort restore for apps hidden in the Windows notification area."""
    return action_ctx.restore_process_from_tray(process)


def _find_tray_control(auto: Any, process: str) -> Any | None:
    return action_ctx._find_tray_control(auto, process)


def _iter_controls(control: Any, *, max_depth: int) -> Any:
    return iter(action_ctx._iter_controls(control, max_depth=max_depth))


def _new_related_window_seen(
    before_windows: list[dict[str, Any]],
    after_windows: list[dict[str, Any]],
    process: str,
) -> bool:
    before = {int(item.get("hwnd") or 0) for item in before_windows}
    for item in after_windows:
        if int(item.get("hwnd") or 0) in before:
            continue
        if str(item.get("process_name") or "").lower() == process:
            return True
    return False


def _point(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return []
    try:
        x, y = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if x < 0 or y < 0:
        return []
    return [x, y]


def _rect(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if right <= left or bottom <= top:
        return []
    return [left, top, right, bottom]


def _center(bounds: list[int]) -> list[int]:
    if not bounds:
        return []
    return [int((bounds[0] + bounds[2]) / 2), int((bounds[1] + bounds[3]) / 2)]


def _element_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("name") or item.get("role_label") or "").strip()


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("_") or "sample"


def _parse_query_override(items: list[str]) -> dict[str, str]:
    queries: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip() and value:
            queries[key.strip().lower()] = value
    return queries


def _probe_strategy_for_process(process: str) -> str:
    if process.lower() in {"feishu.exe", "lark.exe"}:
        return "global_search_hotkey"
    return "click_search"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", action="append", default=[], help="Override query, e.g. qq.exe=jz")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=2.5)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    report = run_probe_plan(
        plan=plan,
        output_dir=args.output_dir,
        base_url=args.base_url,
        queries=_parse_query_override(args.query),
        execute=args.execute,
        wait_seconds=args.wait_seconds,
    )
    print(f"overall={report['overall_status']} rows={len(report['rows'])}")
    return 0 if report["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
