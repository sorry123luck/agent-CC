"""Run controlled chat input/send probes from a chat action plan.

The probe is intentionally conservative: dry-run is the default, executing
only types into the detected input area, and sending requires an explicit
``--send`` flag.
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

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.message_stream_crop import input_crop_box, sent_message_crop_box
from src.chat.readback_worker import ChatReadbackWorker
from src.windows import window_action_context as action_ctx


DEFAULT_TEXTS = {
    "qq.exe": "DeskCanvas send probe",
    "weixin.exe": "DeskCanvas send probe",
    "wechat.exe": "DeskCanvas send probe",
    "feishu.exe": "DeskCanvas send probe",
    "lark.exe": "DeskCanvas send probe",
}


def load_plan(path: Path) -> dict[str, Any]:
    """Load a JSON probe plan, accepting UTF-8 files with or without BOM."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def build_probe_targets(
    plan: dict[str, Any],
    *,
    texts: dict[str, str],
    current_windows: list[dict[str, Any]] | None = None,
    identity_gate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return executable chat input targets with per-app probe text attached."""
    normalized_texts = {key.lower(): value for key, value in texts.items() if value}
    blocked = _evidence_gate_blocked_rows(identity_gate or {})
    targets: list[dict[str, Any]] = []
    for row in list(plan.get("rows") or []):
        sample = str(row.get("sample") or "")
        if sample and sample.lower() in blocked:
            continue
        if str(row.get("status") or "") != "ready":
            continue
        process = str(row.get("process_name") or "").lower()
        input_click = _point(row.get("input_click"))
        send_click = _point(row.get("send_click"))
        if not process or not input_click or not send_click:
            continue
        text = normalized_texts.get(process) or DEFAULT_TEXTS.get(process)
        if not text:
            continue
        target = dict(row)
        target["process_name"] = process
        target["input_click"] = input_click
        target["input_bounds"] = _rects(row.get("input_bounds"))
        target["send_click"] = send_click
        target["send_bounds"] = _rect(row.get("send_bounds"))
        target["text"] = text
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


def run_probe_plan(
    *,
    plan: dict[str, Any],
    output_dir: Path,
    base_url: str,
    texts: dict[str, str],
    execute: bool,
    send: bool,
    wait_seconds: float,
    cleanup_after_type: bool = True,
    readback_after_send: bool = True,
    identity_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or dry-run chat send probes and return a report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    current_windows = _fetch_windows(base_url)
    targets = build_probe_targets(
        plan,
        texts=texts,
        current_windows=current_windows,
        identity_gate=identity_gate,
    )
    skipped_rows = _evidence_gate_skipped_action_rows(plan, identity_gate or {})
    rows = [
        _run_one_target(
            target=target,
            output_dir=output_dir,
            base_url=base_url,
            execute=execute,
            send=send,
            wait_seconds=wait_seconds,
            cleanup_after_type=cleanup_after_type,
            readback_after_send=readback_after_send,
        )
        for target in targets
    ] + skipped_rows
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": _probe_overall_status(rows, execute=execute),
        "execute": execute,
        "send": send,
        "readback_after_send": readback_after_send,
        "rows": rows,
    }
    write_probe_report(report, output_dir)
    return report


def _probe_overall_status(rows: list[dict[str, Any]], *, execute: bool) -> str:
    if not rows:
        return "blocked"
    if all(row.get("status") in {"blocked_target_mismatch", "blocked_evidence_mismatch"} for row in rows):
        return "blocked"
    if not execute:
        return "dry_run"
    if any(row.get("status") == "error" for row in rows):
        return "review"
    return "review"


def write_probe_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist action JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "send_probe_actions.json").write_text(
        json.dumps(report.get("rows") or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "chat_send_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Send Probe",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Execute: {report.get('execute', '')}",
        f"- Send: {report.get('send', '')}",
        "",
        "| sample | process | status | text | input_click | send_click | would_send | before | typed | sent | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {status} | {text} | {input_click} | {send_click} | {would_send} | {before} | {typed} | {sent} | {notes} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                text=str(row.get("text") or "").replace("|", "/"),
                input_click=str(row.get("input_click") or "").replace("|", "/"),
                send_click=str(row.get("send_click") or "").replace("|", "/"),
                would_send="true" if row.get("would_send") else "false",
                before=str(row.get("before_canvas_id") or "").replace("|", "/"),
                typed=str(row.get("typed_canvas_id") or "").replace("|", "/"),
                sent=str(row.get("sent_canvas_id") or "").replace("|", "/"),
                notes=",".join(row.get("notes") or []),
            )
        )
    (output_dir / "chat_send_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_one_target(
    *,
    target: dict[str, Any],
    output_dir: Path,
    base_url: str,
    execute: bool,
    send: bool,
    wait_seconds: float,
    cleanup_after_type: bool,
    readback_after_send: bool,
) -> dict[str, Any]:
    hwnd = int(target.get("hwnd") or 0)
    process = str(target.get("process_name") or "")
    sample = str(target.get("sample") or f"{process}_{hwnd}")
    text = str(target.get("text") or "")
    input_click = _point(target.get("input_click"))
    send_click = _point(target.get("send_click"))
    notes: list[str] = []
    before_canvas_id = ""
    typed_canvas_id = ""
    sent_canvas_id = ""
    before_screenshot = ""
    typed_screenshot = ""
    sent_screenshot = ""
    typed_input_crop = ""
    sent_message_crop = ""
    readback_result: dict[str, Any] = {}
    status = "dry_run"
    error = ""
    try:
        if execute:
            _ensure_window_actionable(hwnd, process)
            _reset_probe_surface(hwnd=hwnd, process=process)
            before_canvas_id = _observe(base_url, hwnd)
            before_detail = _fetch_canvas_detail(base_url, before_canvas_id)
            _save_detail(output_dir, sample, "before", before_canvas_id, before_detail)
            before_screenshot = _save_screen(output_dir, sample, "before")
            _focus_and_type(hwnd=hwnd, click=input_click, text=text)
            time.sleep(wait_seconds)
            typed_screenshot = _save_screen(output_dir, sample, "typed")
            typed_input_crop = _save_input_crop(
                output_dir=output_dir,
                sample=sample,
                stage="typed",
                screenshot_path=typed_screenshot,
                input_bounds=_rects(target.get("input_bounds")),
                input_click=input_click,
                window_rect=_window_rect(hwnd),
            )
            typed_canvas_id = _observe(base_url, hwnd)
            typed_detail = _fetch_canvas_detail(base_url, typed_canvas_id)
            _save_detail(output_dir, sample, "typed", typed_canvas_id, typed_detail)
            notes.append("typed_probe_text")
            if send:
                _click_relative(hwnd=hwnd, click=send_click)
                time.sleep(wait_seconds)
                sent_screenshot = _save_screen(output_dir, sample, "sent")
                sent_message_crop = _save_sent_message_crop(
                    output_dir=output_dir,
                    sample=sample,
                    screenshot_path=sent_screenshot,
                    input_bounds=_rects(target.get("input_bounds")),
                    window_rect=_window_rect(hwnd),
                )
                sent_canvas_id = _observe(base_url, hwnd)
                sent_detail = _fetch_canvas_detail(base_url, sent_canvas_id)
                _save_detail(output_dir, sample, "sent", sent_canvas_id, sent_detail)
                if readback_after_send:
                    readback_result = _run_readback_for_sent_crop(
                        output_dir=output_dir,
                        sample=sample,
                        process=process,
                        hwnd=hwnd,
                        text=text,
                        sent_message_crop=sent_message_crop,
                    )
                notes.append("clicked_send")
            elif cleanup_after_type:
                _clear_input(hwnd=hwnd, click=input_click)
                notes.append("cleared_probe_text")
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
        "text": text,
        "input_click": input_click,
        "send_click": send_click,
        "send_bounds": _rect(target.get("send_bounds")),
        "would_send": bool(send),
        "before_canvas_id": before_canvas_id,
        "typed_canvas_id": typed_canvas_id,
        "sent_canvas_id": sent_canvas_id,
        "before_screenshot": before_screenshot,
        "typed_screenshot": typed_screenshot,
        "sent_screenshot": sent_screenshot,
        "typed_input_crop": typed_input_crop,
        "sent_message_crop": sent_message_crop,
        **readback_result,
        "notes": notes,
        "error": error,
    }


def _focus_and_type(*, hwnd: int, click: list[int], text: str) -> None:
    import pyautogui
    import win32gui

    if not click:
        raise ValueError("missing input click")
    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        raise ValueError(f"invalid hwnd: {hwnd}")
    _bring_window_to_front(hwnd)
    left, top, _right, _bottom = win32gui.GetWindowRect(hwnd)
    pyautogui.click(left + click[0], top + click[1])
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "a")
    _paste_text(text)


def _clear_input(*, hwnd: int, click: list[int]) -> None:
    import pyautogui
    import win32gui

    if not click:
        return
    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        return
    _bring_window_to_front(hwnd)
    left, top, _right, _bottom = win32gui.GetWindowRect(hwnd)
    pyautogui.click(left + click[0], top + click[1])
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "a")
    pyautogui.press("backspace")


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

    _ensure_window_actionable(hwnd, process)
    _bring_window_to_front(hwnd)
    for _ in range(action_ctx.pre_probe_escape_count(process)):
        pyautogui.press("esc")
        time.sleep(0.15)


def _click_relative(*, hwnd: int, click: list[int]) -> None:
    import pyautogui
    import win32gui

    if not click:
        raise ValueError("missing send click")
    _bring_window_to_front(hwnd)
    left, top, _right, _bottom = win32gui.GetWindowRect(hwnd)
    pyautogui.click(left + click[0], top + click[1])


def _observe(base_url: str, hwnd: int) -> str:
    if hwnd <= 0:
        return ""
    response = requests.post(
        f"{base_url.rstrip('/')}/api/v1/observe",
        json={
            "hwnd": hwnd,
            "async_enhance": False,
            "allow_vlm": False,
            "force_vlm": False,
            "capture_mode": "window",
        },
        timeout=45,
    )
    response.raise_for_status()
    return str((response.json() or {}).get("canvas_id") or "")


def _fetch_canvas_detail(base_url: str, canvas_id: str) -> dict[str, Any]:
    if not canvas_id:
        return {}
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/v1/canvases/{canvas_id}", timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}
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


def _fresh_hwnd_for_process(process: str, current_windows: list[dict[str, Any]], *, preferred_hwnd: int = 0) -> int:
    """Return the best visible hwnd for a process from a live window listing."""
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


def _save_screen(output_dir: Path, sample: str, stage: str) -> str:
    import pyautogui

    path = output_dir / "screenshots" / f"{_safe_filename(sample)}_{stage}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pyautogui.screenshot(str(path))
    return str(path)


def _save_input_crop(
    *,
    output_dir: Path,
    sample: str,
    stage: str,
    screenshot_path: str,
    input_bounds: list[list[int]],
    input_click: list[int],
    window_rect: list[int] | None = None,
) -> str:
    from PIL import Image

    if not screenshot_path:
        return ""
    image_path = Path(screenshot_path)
    if not image_path.exists():
        return ""
    image = Image.open(image_path)
    crop_box = input_crop_box(
        window_rect=window_rect,
        input_bounds=input_bounds,
        input_click=input_click,
        screenshot_size=image.size,
    )
    if not crop_box:
        return ""
    path = output_dir / "input_crops" / f"{_safe_filename(sample)}_{stage}_input.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(path)
    return str(path)


def _save_sent_message_crop(
    *,
    output_dir: Path,
    sample: str,
    screenshot_path: str,
    input_bounds: list[list[int]],
    window_rect: list[int] | None = None,
) -> str:
    from PIL import Image

    if not screenshot_path:
        return ""
    image_path = Path(screenshot_path)
    if not image_path.exists():
        return ""
    image = Image.open(image_path)
    crop_box = sent_message_crop_box(
        window_rect=window_rect,
        input_bounds=input_bounds,
        screenshot_size=image.size,
    )
    if not crop_box:
        return ""
    path = output_dir / "message_crops" / f"{_safe_filename(sample)}_sent_message.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(path)
    return str(path)


def _run_readback_for_sent_crop(
    *,
    output_dir: Path,
    sample: str,
    process: str,
    hwnd: int,
    text: str,
    sent_message_crop: str,
) -> dict[str, Any]:
    if not sent_message_crop:
        return {}
    crop_path = Path(sent_message_crop)
    if not crop_path.exists():
        return {
            "readback_result_path": "",
            "readback_observed_after": False,
            "readback_event_count": 0,
            "readback_error": f"missing crop: {sent_message_crop}",
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
        expected_self_texts=[text],
    )
    path = output_dir / "readback" / f"{_safe_filename(sample)}_sent_readback.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "readback_result_path": str(path),
        "readback_observed_after": result.has_text(text),
        "readback_event_count": len(result.events),
    }


def _window_rect(hwnd: int) -> list[int]:
    try:
        import win32gui

        if hwnd <= 0 or not win32gui.IsWindow(hwnd):
            return []
        return [int(value) for value in win32gui.GetWindowRect(hwnd)]
    except Exception:
        return []


def _save_detail(output_dir: Path, sample: str, stage: str, canvas_id: str, detail: dict[str, Any]) -> str:
    if not canvas_id or not detail:
        return ""
    path = output_dir / "details" / f"{_safe_filename(sample)}_{stage}_{_safe_filename(canvas_id)}.detail.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _ensure_window_actionable(hwnd: int, process: str) -> None:
    action_ctx.ensure_window_actionable(hwnd, process)


def _window_is_actionable(hwnd: int) -> bool:
    return action_ctx.window_is_actionable(hwnd)


def _bring_window_to_front(hwnd: int) -> None:
    action_ctx.bring_window_to_front(hwnd)


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


def _parse_text_override(items: list[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip() and value:
            texts[key.strip().lower()] = value
    return texts


def load_identity_gate(path: Path | None) -> dict[str, Any]:
    """Load an optional target identity gate report."""
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"identity gate not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _evidence_gate_blocked_rows(identity_gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocked: dict[str, dict[str, Any]] = {}
    for check in list(identity_gate.get("checks") or []):
        if not isinstance(check, dict):
            continue
        sample = str(check.get("sample") or "").strip().lower()
        decision = str(check.get("decision") or "").strip().lower()
        if sample and decision in {"stop", "blocked", "blocked_target_mismatch", "blocked_evidence_mismatch"}:
            blocked[sample] = check
    return blocked


def _evidence_gate_skipped_action_rows(plan: dict[str, Any], identity_gate: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = _evidence_gate_blocked_rows(identity_gate)
    generic_gate = str(identity_gate.get("gate_type") or "").strip().lower() == "page_evidence"
    skipped: list[dict[str, Any]] = []
    for row in list(plan.get("rows") or []):
        if not isinstance(row, dict):
            continue
        sample = str(row.get("sample") or "")
        check = blocked.get(sample.lower())
        if not check:
            continue
        status = "blocked_evidence_mismatch" if generic_gate else "blocked_target_mismatch"
        note = "evidence_gate_stop" if generic_gate else "identity_gate_stop"
        skipped.append(
            {
                "app": _app_name_for_process(str(row.get("process_name") or "")),
                "sample": sample,
                "hwnd": int(row.get("hwnd") or 0) or _hwnd_from_sample(sample),
                "process_name": str(row.get("process_name") or ""),
                "status": status,
                "ok": False,
                "text": "",
                "input_click": _point(row.get("input_click")),
                "send_click": _point(row.get("send_click")),
                "send_bounds": _rect(row.get("send_bounds")),
                "would_send": False,
                "before_canvas_id": "",
                "typed_canvas_id": "",
                "sent_canvas_id": "",
                "before_screenshot": "",
                "typed_screenshot": "",
                "sent_screenshot": "",
                "typed_input_crop": "",
                "sent_message_crop": "",
                "readback_observed_after": False,
                "readback_event_count": 0,
                "notes": [note],
                "error": str(check.get("reason") or "required page evidence was not observed"),
            }
        )
    return skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--text", action="append", default=[], help="Override text, e.g. qq.exe=DeskCanvas probe")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--keep-text", action="store_true", help="Keep typed probe text when --execute is used without --send")
    parser.add_argument("--no-readback", action="store_true", help="Skip OCR readback after --send")
    parser.add_argument("--identity-gate", type=Path, help="Optional target_identity_check.json; rows with decision=stop are skipped")
    parser.add_argument("--evidence-gate", type=Path, help="Optional page_evidence_gate.json; rows with decision=stop are skipped")
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    report = run_probe_plan(
        plan=plan,
        output_dir=args.output_dir,
        base_url=args.base_url,
        texts=_parse_text_override(args.text),
        execute=args.execute,
        send=args.send,
        wait_seconds=args.wait_seconds,
        cleanup_after_type=not args.keep_text,
        readback_after_send=not args.no_readback,
        identity_gate=load_identity_gate(args.evidence_gate or args.identity_gate),
    )
    print(f"overall={report['overall_status']} rows={len(report['rows'])}")
    return 0 if report["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
