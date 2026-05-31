"""Capture read-only chat message-stream snapshots from a chat action plan.

This probe intentionally never types into the composer and never clicks send.
It brings each ready chat window forward, saves a screenshot, crops the visible
message stream, and runs the OCR-first readback worker over that crop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.message_stream_crop import message_stream_crop_box
from src.chat.readback_worker import ChatReadbackWorker
from src.windows import window_action_context as action_ctx


def load_plan(path: Path) -> dict[str, Any]:
    """Load a JSON chat action plan, accepting UTF-8 files with or without BOM."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def build_snapshot_targets(
    plan: dict[str, Any],
    *,
    current_windows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return ready chat rows suitable for read-only message stream capture."""
    targets: list[dict[str, Any]] = []
    for row in list(plan.get("rows") or []):
        if str(row.get("status") or "") != "ready":
            continue
        process = str(row.get("process_name") or "").lower()
        input_click = _point(row.get("input_click"))
        if not process or not input_click:
            continue
        target = dict(row)
        target.pop("text", None)
        target["process_name"] = process
        target["input_click"] = input_click
        target["input_bounds"] = _rects(row.get("input_bounds"))
        original_hwnd = int(target.get("hwnd") or 0) or _hwnd_from_sample(str(target.get("sample") or ""))
        if original_hwnd:
            target["hwnd"] = original_hwnd
        refreshed_hwnd = 0 if _window_is_actionable(original_hwnd) else _fresh_hwnd_for_process(
            process,
            current_windows or [],
            preferred_hwnd=original_hwnd,
        )
        if refreshed_hwnd and refreshed_hwnd != original_hwnd:
            target["original_hwnd"] = original_hwnd
            target["hwnd"] = refreshed_hwnd
            target["hwnd_refreshed"] = True
        targets.append(target)
    return targets


def run_snapshot_plan(
    *,
    plan: dict[str, Any],
    output_dir: Path,
    base_url: str,
    execute: bool,
    wait_seconds: float = 0.5,
    loading_retry_attempts: int = 0,
    loading_retry_seconds: float = 1.0,
) -> dict[str, Any]:
    """Capture read-only snapshots and return a report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    current_windows = _fetch_windows(base_url)
    targets = build_snapshot_targets(plan, current_windows=current_windows)
    rows = [
        _run_one_target(
            target=target,
            output_dir=output_dir,
            execute=execute,
            wait_seconds=wait_seconds,
            loading_retry_attempts=loading_retry_attempts,
            loading_retry_seconds=loading_retry_seconds,
        )
        for target in targets
    ]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": _snapshot_overall_status(rows, execute=execute),
        "execute": execute,
        "rows": rows,
    }
    write_snapshot_report(report, output_dir)
    return report


def write_snapshot_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist readback snapshot JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat_readback_snapshot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Readback Snapshot",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Execute: {report.get('execute', '')}",
        "",
        "| sample | process | status | crop | events | has_events | warnings |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {status} | {crop} | {events} | {has_events} | {warnings} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                crop=str(row.get("message_stream_crop") or "").replace("|", "/"),
                events=int(row.get("readback_event_count") or 0),
                has_events="true" if row.get("readback_has_events") else "false",
                warnings=",".join(row.get("readback_warnings") or []).replace("|", "/"),
            )
        )
    (output_dir / "chat_readback_snapshot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _snapshot_overall_status(rows: list[dict[str, Any]], *, execute: bool) -> str:
    if not rows:
        return "blocked"
    if not execute:
        return "dry_run"
    if any(row.get("status") == "error" for row in rows):
        return "review"
    return "review"


def _run_one_target(
    *,
    target: dict[str, Any],
    output_dir: Path,
    execute: bool,
    wait_seconds: float,
    loading_retry_attempts: int,
    loading_retry_seconds: float,
) -> dict[str, Any]:
    hwnd = int(target.get("hwnd") or 0)
    process = str(target.get("process_name") or "")
    sample = str(target.get("sample") or f"{process}_{hwnd}")
    screenshot = ""
    message_stream_crop = ""
    readback_result: dict[str, Any] = {}
    status = "dry_run"
    error = ""
    try:
        if execute:
            _ensure_window_actionable(hwnd, process)
            _bring_window_to_front(hwnd)
            time.sleep(max(0.0, wait_seconds))
            screenshot, message_stream_crop, readback_result, retry_count = _capture_until_not_loading(
                output_dir=output_dir,
                sample=sample,
                process=process,
                hwnd=hwnd,
                input_bounds=_rects(target.get("input_bounds")),
                send_bounds=_rect(target.get("send_bounds")),
                loading_retry_attempts=loading_retry_attempts,
                loading_retry_seconds=loading_retry_seconds,
            )
            status = "captured"
    except Exception as exc:  # pragma: no cover - depends on live desktop.
        status = "error"
        error = str(exc)
    return {
        "app": _app_name_for_process(process),
        "sample": sample,
        "hwnd": hwnd,
        "process_name": process,
        "status": status,
        "ok": status == "captured" if execute else None,
        "would_send": False,
        "screenshot": screenshot,
        "message_stream_crop": message_stream_crop,
        "retry_count": retry_count if execute else 0,
        **readback_result,
        "error": error,
    }


def _capture_until_not_loading(
    *,
    output_dir: Path,
    sample: str,
    process: str,
    hwnd: int,
    input_bounds: list[list[int]],
    send_bounds: list[int],
    loading_retry_attempts: int,
    loading_retry_seconds: float,
) -> tuple[str, str, dict[str, Any], int]:
    attempts = max(0, loading_retry_attempts) + 1
    last_screenshot = ""
    last_crop = ""
    last_readback: dict[str, Any] = {}
    retry_count = 0
    for attempt in range(attempts):
        if attempt > 0:
            time.sleep(max(0.0, loading_retry_seconds))
            retry_count += 1
        stage = "snapshot" if attempt == 0 else f"snapshot_retry{attempt}"
        last_screenshot = _save_screen(output_dir, sample, stage)
        last_crop = _save_message_stream_crop(
            output_dir=output_dir,
            sample=sample if attempt == 0 else f"{sample}_retry{attempt}",
            screenshot_path=last_screenshot,
            input_bounds=input_bounds,
            send_bounds=send_bounds,
            window_rect=_window_rect(hwnd),
        )
        last_readback = _run_readback_for_stream_crop(
            output_dir=output_dir,
            sample=sample if attempt == 0 else f"{sample}_retry{attempt}",
            process=process,
            hwnd=hwnd,
            message_stream_crop=last_crop,
        )
        if "message_stream_loading" not in list(last_readback.get("readback_warnings") or []):
            break
    return last_screenshot, last_crop, last_readback, retry_count


def _save_screen(output_dir: Path, sample: str, stage: str) -> str:
    import pyautogui

    path = output_dir / "screenshots" / f"{_safe_filename(sample)}_{stage}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pyautogui.screenshot(str(path))
    return str(path)


def _save_message_stream_crop(
    *,
    output_dir: Path,
    sample: str,
    screenshot_path: str,
    input_bounds: list[list[int]],
    send_bounds: list[int],
    window_rect: list[int] | None = None,
) -> str:
    from PIL import Image

    if not screenshot_path:
        return ""
    image_path = Path(screenshot_path)
    if not image_path.exists():
        return ""
    image = Image.open(image_path)
    crop_box = message_stream_crop_box(
        window_rect=window_rect,
        input_bounds=input_bounds,
        send_bounds=send_bounds,
        screenshot_size=image.size,
    )
    if not crop_box:
        return ""
    path = output_dir / "message_stream_crops" / f"{_safe_filename(sample)}_message_stream.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(path)
    return str(path)


def _run_readback_for_stream_crop(
    *,
    output_dir: Path,
    sample: str,
    process: str,
    hwnd: int,
    message_stream_crop: str,
) -> dict[str, Any]:
    if not message_stream_crop:
        return {
            "readback_result_path": "",
            "readback_event_count": 0,
            "readback_has_events": False,
            "readback_warnings": ["missing_message_stream_crop"],
        }
    crop_path = Path(message_stream_crop)
    if not crop_path.exists():
        return {
            "readback_result_path": "",
            "readback_event_count": 0,
            "readback_has_events": False,
            "readback_warnings": [f"missing crop: {message_stream_crop}"],
        }
    from PIL import Image

    with Image.open(crop_path) as image:
        source_image = image.copy()
    result = ChatReadbackWorker().read_crop(
        image=source_image,
        image_path=crop_path,
        app_process=process,
        window_id=str(hwnd),
        stream_bounds=(0, 0, source_image.width, source_image.height),
        crop_origin=(0, 0),
    )
    path = output_dir / "readback" / f"{_safe_filename(sample)}_snapshot_readback.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "readback_result_path": str(path),
        "readback_event_count": len(result.events),
        "readback_has_events": bool(result.events),
        "readback_warnings": result.warnings,
    }


def _fetch_windows(base_url: str) -> list[dict[str, Any]]:
    try:
        import requests

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


def _fresh_hwnd_for_process(process: str, current_windows: list[dict[str, Any]], *, preferred_hwnd: int = 0) -> int:
    process = process.lower()
    if preferred_hwnd:
        for item in current_windows:
            if int(item.get("hwnd") or 0) == preferred_hwnd and str(item.get("process_name") or "").lower() == process:
                return preferred_hwnd
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
        if width is not None and height is not None and (int(width or 0) < 240 or int(height or 0) < 160):
            continue
        return hwnd
    return 0


def _window_rect(hwnd: int) -> list[int]:
    try:
        import win32gui

        if hwnd <= 0 or not win32gui.IsWindow(hwnd):
            return []
        return [int(value) for value in win32gui.GetWindowRect(hwnd)]
    except Exception:
        return []


def _ensure_window_actionable(hwnd: int, process: str) -> None:
    action_ctx.ensure_window_actionable(hwnd, process)


def _window_is_actionable(hwnd: int) -> bool:
    return action_ctx.window_is_actionable(hwnd)


def _bring_window_to_front(hwnd: int) -> None:
    action_ctx.bring_window_to_front(hwnd)


def _hwnd_from_sample(sample: str) -> int:
    for part in sample.replace("-", "_").split("_"):
        if not part.isdigit():
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            return value
    return 0


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


def _rects(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    return [rect for rect in (_rect(item) for item in value) if rect]


def _app_name_for_process(process: str) -> str:
    process = process.lower()
    if process in {"weixin.exe", "wechat.exe"}:
        return "wechat"
    if process in {"feishu.exe", "lark.exe"}:
        return "feishu"
    if process == "qq.exe":
        return "qq"
    return process.removesuffix(".exe")


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("_") or "sample"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=0.5)
    parser.add_argument("--loading-retry-attempts", type=int, default=0)
    parser.add_argument("--loading-retry-seconds", type=float, default=1.0)
    args = parser.parse_args()
    report = run_snapshot_plan(
        plan=load_plan(args.plan),
        output_dir=args.output_dir,
        base_url=args.base_url,
        execute=args.execute,
        wait_seconds=args.wait_seconds,
        loading_retry_attempts=args.loading_retry_attempts,
        loading_retry_seconds=args.loading_retry_seconds,
    )
    print(f"overall={report['overall_status']} rows={len(report['rows'])}")
    return 0 if report["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
