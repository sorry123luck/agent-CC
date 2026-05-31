"""Shared desktop window action helpers.

These helpers are intentionally small and conservative. They keep probe/action
code from reimplementing process-specific window activation and capture policy.
"""

from __future__ import annotations

import time
from typing import Any


TRAY_RESTORE_KEYWORDS = {
    "weixin.exe": ["微信", "wechat", "weixin"],
    "wechat.exe": ["微信", "wechat", "weixin"],
    "qq.exe": ["qq"],
    "feishu.exe": ["飞书", "feishu", "lark"],
    "lark.exe": ["飞书", "feishu", "lark"],
}


def chat_search_capture_mode(process: str) -> str:
    """Return observe capture mode for a chat search result surface."""
    process = process.lower()
    if process in {"weixin.exe", "wechat.exe"}:
        return "screen_region"
    return "window"


def pre_probe_escape_count(process: str) -> int:
    """Return how many Esc presses are safe before probing a search surface."""
    process = process.lower()
    if process in {"feishu.exe", "lark.exe"}:
        return 2
    if process == "qq.exe":
        return 1
    return 0


def post_selection_escape_count(process: str) -> int:
    """Return how many Esc presses are safe after selecting a search result."""
    return 0


def tray_control_matches(process: str, control_name: str) -> bool:
    """Return whether a system tray control name belongs to the process."""
    name = control_name.strip().lower()
    if not name:
        return False
    keywords = TRAY_RESTORE_KEYWORDS.get(process.lower(), [])
    return any(keyword.lower() in name for keyword in keywords)


def window_is_actionable(hwnd: int) -> bool:
    """Return whether a hwnd is visible and has a usable rectangle."""
    import win32gui

    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        return False
    if not win32gui.IsWindowVisible(hwnd):
        return False
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return right > left and bottom > top


def restore_hidden_window(hwnd: int) -> bool:
    """Best-effort restore for an existing hidden top-level window."""
    import win32con
    import win32gui

    if hwnd <= 0 or not win32gui.IsWindow(hwnd):
        return False
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )
        return True
    except Exception:
        return False


def wait_until_window_actionable(hwnd: int, *, timeout_seconds: float = 8.0, interval_seconds: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if window_is_actionable(hwnd):
            return True
        time.sleep(interval_seconds)
    return window_is_actionable(hwnd)


def ensure_window_actionable(hwnd: int, process: str) -> None:
    """Restore a target window if possible; raise if still unsafe to click."""
    if window_is_actionable(hwnd):
        return
    if restore_hidden_window(hwnd) and wait_until_window_actionable(hwnd):
        return
    restored = restore_process_from_tray(process)
    if restored and wait_until_window_actionable(hwnd):
        return
    if not restored or not window_is_actionable(hwnd):
        raise ValueError(f"window not actionable: {hwnd}")


def bring_window_to_front(hwnd: int) -> None:
    """Activate a target window and verify it is the foreground root."""
    import pyautogui
    import win32con
    import win32gui

    if not window_is_actionable(hwnd):
        raise ValueError(f"window not actionable: {hwnd}")
    try:
        from src.windows.window_activator import WindowActivator

        WindowActivator().activate(hwnd)
        time.sleep(0.2)
    except Exception:
        pass
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.25)
    try:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
        )
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
        )
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.25)
    except Exception:
        left, top, _right, _bottom = win32gui.GetWindowRect(hwnd)
        pyautogui.click(left + 20, top + 20)
        time.sleep(0.35)
    if not foreground_matches_target(hwnd):
        raise ValueError(f"window activation failed: {hwnd}")


def foreground_matches_target(hwnd: int) -> bool:
    import win32con
    import win32gui

    foreground = win32gui.GetForegroundWindow()
    if foreground == hwnd:
        return True
    return win32gui.GetAncestor(foreground, win32con.GA_ROOT) == hwnd


def restore_process_from_tray(process: str) -> bool:
    """Best-effort restore for apps hidden in the Windows notification area."""
    if process.lower() not in TRAY_RESTORE_KEYWORDS:
        return False
    try:
        import uiautomation as auto
    except Exception:
        return False
    control = _find_tray_control(auto, process)
    if control is None:
        return False
    try:
        control.DoubleClick(simulateMove=False)
        return True
    except Exception:
        try:
            control.Click(simulateMove=False)
            time.sleep(0.15)
            control.Click(simulateMove=False)
            return True
        except Exception:
            return False


def _find_tray_control(auto: Any, process: str) -> Any | None:
    root = auto.GetRootControl()
    for control in _iter_controls(root, max_depth=5):
        name = str(getattr(control, "Name", "") or "")
        class_name = str(getattr(control, "ClassName", "") or "")
        if class_name != "SystemTray.NormalButton" and "SystemTray" not in class_name:
            continue
        if tray_control_matches(process, name):
            return control
    return None


def _iter_controls(control: Any, *, max_depth: int) -> list[Any]:
    if max_depth < 0:
        return []
    result = [control]
    try:
        children = control.GetChildren()
    except Exception:
        children = []
    for child in children or []:
        result.extend(_iter_controls(child, max_depth=max_depth - 1))
    return result
