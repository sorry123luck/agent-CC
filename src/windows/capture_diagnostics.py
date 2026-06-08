"""Window capture coordinate diagnostics for DeskCanvas P0 entry checks."""

from __future__ import annotations

from typing import Any


def _as_list(rect) -> list[int] | None:
    if rect is None:
        return None
    try:
        return [int(value) for value in rect[:4]]
    except Exception:
        return None


def _bounds_size(bounds: list[int] | None) -> list[int] | None:
    if not bounds or len(bounds) < 4:
        return None
    return [max(0, bounds[2] - bounds[0]), max(0, bounds[3] - bounds[1])]


def window_capture_diagnostics(hwnd: int, *, screenshot_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Return best-effort window/screenshot coordinate evidence.

    The function is intentionally non-fatal: diagnostics should explain capture
    risk without breaking observe/launch-bind when one Windows API call fails.
    """
    window_bounds: list[int] | None = None
    dpi_scale = 1.0
    is_minimized = False
    is_visible = None
    monitor: dict[str, Any] = {"index": 0}
    error = None

    try:
        import win32api
        import win32con
        import win32gui
        from ctypes import windll

        window_bounds = _as_list(win32gui.GetWindowRect(hwnd))
        is_minimized = bool(win32gui.IsIconic(hwnd))
        is_visible = bool(win32gui.IsWindowVisible(hwnd))

        try:
            dpi = windll.user32.GetDpiForWindow(hwnd)
            if dpi:
                dpi_scale = round(float(dpi) / 96.0, 4)
        except Exception:
            dpi_scale = 1.0

        try:
            monitor_handle = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
            info = win32api.GetMonitorInfo(monitor_handle)
            monitor = {
                "index": 0,
                "handle": int(monitor_handle),
                "monitor_bounds": _as_list(info.get("Monitor")),
                "work_area": _as_list(info.get("Work")),
                "primary": bool(info.get("Flags", 0) & win32con.MONITORINFOF_PRIMARY),
            }
        except Exception:
            pass
    except Exception as exc:
        error = str(exc)

    size = list(screenshot_size) if screenshot_size else _bounds_size(window_bounds)
    capture_bounds = [0, 0, size[0], size[1]] if size else None
    valid = bool(hwnd and window_bounds and size and size[0] > 0 and size[1] > 0 and not is_minimized)

    return {
        "hwnd": int(hwnd) if hwnd else 0,
        "window_bounds": window_bounds,
        "capture_bounds": capture_bounds,
        "screenshot_size": size,
        "dpi_scale": dpi_scale,
        "monitor": monitor,
        "is_minimized": is_minimized,
        "is_visible": is_visible,
        "is_occluded": False,
        "valid": valid,
        "coordinate_system": "window_local_original_pixels",
        "error": error,
    }


def canvas_capture_diagnostics(canvas, screenshot=None) -> dict[str, Any]:
    """Build diagnostics from an InteractionCanvas and optional cached screenshot."""
    window = getattr(canvas, "window", None)
    hwnd = int(getattr(window, "hwnd", 0) or 0)
    screenshot_size = getattr(screenshot, "size", None)
    diagnostics = window_capture_diagnostics(hwnd, screenshot_size=screenshot_size)

    rect_screen = _as_list(getattr(window, "rect_screen", None))
    if rect_screen:
        diagnostics["window_bounds"] = rect_screen
    rect_client = _as_list(getattr(window, "rect_client", None))
    if rect_client:
        diagnostics["client_bounds"] = rect_client
    if window is not None:
        diagnostics["dpi_scale"] = float(getattr(window, "dpi_scale", diagnostics["dpi_scale"]) or 1.0)
        diagnostics["is_minimized"] = bool(getattr(window, "is_minimized", diagnostics["is_minimized"]))
        diagnostics["is_occluded"] = bool(getattr(window, "occluded", False))
    diagnostics["monitor"]["index"] = int(getattr(canvas, "monitor_id", 0) or 0)
    diagnostics["valid"] = bool(
        hwnd
        and diagnostics.get("window_bounds")
        and diagnostics.get("screenshot_size")
        and not diagnostics.get("is_minimized")
    )
    return diagnostics
