"""Window bounds helpers shared by capture and coordinate normalization."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import win32gui


DWMWA_EXTENDED_FRAME_BOUNDS = 9


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = win32gui.GetWindowRect(hwnd)
    return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))


def get_extended_frame_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
    """Return DWM visible frame bounds, excluding invisible resize borders."""
    try:
        rect = _RECT()
        hr = ctypes.WinDLL("dwmapi").DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if hr != 0:
            return None
        bounds = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return None
        return bounds
    except Exception:
        return None


def get_capture_frame_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """Canonical screen-space bounds used by screenshots and UIA normalization."""
    window_rect = get_window_rect(hwnd)
    extended = get_extended_frame_bounds(hwnd)
    if extended is None:
        return window_rect

    wl, wt, wr, wb = window_rect
    el, et, er, eb = extended
    # DWM bounds should sit inside or very near GetWindowRect. If not, keep the
    # Win32 rect to avoid producing an invalid crop for unusual windows.
    if el < wl - 2 or et < wt - 2 or er > wr + 2 or eb > wb + 2:
        return window_rect
    return extended


def crop_offsets_from_window_rect(
    window_rect: tuple[int, int, int, int],
    frame_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return crop box in a GetWindowRect-sized image for the target frame."""
    wl, wt, wr, wb = window_rect
    fl, ft, fr, fb = frame_rect
    return (
        max(0, fl - wl),
        max(0, ft - wt),
        max(0, fr - wl),
        max(0, fb - wt),
    )
