"""Geometry helpers for chat input and message-stream readback crops."""

from __future__ import annotations

from typing import Iterable, Sequence


Rect = tuple[int, int, int, int]
Point = tuple[int, int]


def input_crop_box(
    *,
    window_rect: Sequence[int] | None,
    input_bounds: Sequence[int] | Sequence[Sequence[int]] | None = None,
    input_click: Sequence[int] | None = None,
    screenshot_size: tuple[int, int] | None = None,
) -> Rect | None:
    """Return a screen-coordinate crop for the typed input area."""
    rect = _first_rect(input_bounds)
    if rect is None and input_click:
        click = _point(input_click)
        if click:
            x, y = click
            rect = (x - 80, y - 40, x + 160, y + 40)
    if rect is None:
        return None
    screen_rect = _to_screen_rect(rect, window_rect)
    left, top, right, bottom = screen_rect
    top -= min(40, max(0, bottom - top))
    return _clamp_rect((left, top, right, bottom), screenshot_size)


def sent_message_crop_box(
    *,
    window_rect: Sequence[int] | None,
    input_bounds: Sequence[int] | Sequence[Sequence[int]] | None = None,
    screenshot_size: tuple[int, int] | None = None,
) -> Rect | None:
    """Return a screen-coordinate crop for the latest sent message area."""
    win = _rect(window_rect)
    if win is None:
        return None
    win_left, win_top, win_right, win_bottom = win
    input_rect = _first_rect(input_bounds)
    input_top = win_top + input_rect[1] if input_rect else win_bottom - 140
    window_width = win_right - win_left
    left = int(win_left + window_width * 0.30)
    if input_rect:
        left = max(left, win_left + input_rect[0])
    top = max(win_top, input_top - 220)
    right = win_right - 10
    bottom = input_top - 10
    return _clamp_rect((left, top, right, bottom), screenshot_size)


def message_stream_crop_box(
    *,
    window_rect: Sequence[int] | None,
    input_bounds: Sequence[int] | Sequence[Sequence[int]] | None = None,
    send_bounds: Sequence[int] | None = None,
    screenshot_size: tuple[int, int] | None = None,
) -> Rect | None:
    """Return a screen-coordinate crop for the visible message stream."""
    win = _rect(window_rect)
    if win is None:
        return None
    win_left, win_top, win_right, win_bottom = win
    input_rect = _first_rect(input_bounds)
    input_top = win_top + input_rect[1] if input_rect else win_bottom - 140
    window_width = win_right - win_left
    left = int(win_left + window_width * 0.30)
    if input_rect and _looks_like_wide_left_navigation(input_rect, window_width):
        left = win_left + max(0, input_rect[0] - 20)
    top = win_top + 80
    right = win_right
    if input_rect and _looks_like_right_side_panel(input_rect, win):
        content_right = input_rect[2]
        send_rect = _rect(send_bounds)
        if send_rect:
            content_right = max(content_right, send_rect[2] + 20)
        right = win_left + content_right
    bottom = input_top
    return _clamp_rect((left, top, right, bottom), screenshot_size)


def _looks_like_right_side_panel(input_rect: Rect, window_rect: Rect) -> bool:
    win_left, _win_top, win_right, _win_bottom = window_rect
    window_width = win_right - win_left
    right_gap = window_width - input_rect[2]
    return right_gap >= 240 and right_gap >= window_width * 0.22


def _looks_like_wide_left_navigation(input_rect: Rect, window_width: int) -> bool:
    default_left = int(window_width * 0.30)
    return input_rect[0] - default_left >= 120


def classify_message_side(*, bounds: Sequence[int], stream_bounds: Sequence[int]) -> str:
    """Classify a message-like bounding box as self, peer, or unknown by horizontal position."""
    box = _rect(bounds)
    stream = _rect(stream_bounds)
    if box is None or stream is None:
        return "unknown"
    left, _top, right, _bottom = box
    stream_left, _stream_top, stream_right, _stream_bottom = stream
    width = stream_right - stream_left
    if width <= 0:
        return "unknown"
    center = (left + right) / 2
    relative = (center - stream_left) / width
    if relative >= 0.68:
        return "me"
    if relative <= 0.42:
        return "peer"
    return "unknown"


def _first_rect(value: Sequence[int] | Sequence[Sequence[int]] | None) -> Rect | None:
    if value is None:
        return None
    direct = _rect(value)  # type: ignore[arg-type]
    if direct is not None:
        return direct
    if isinstance(value, Iterable):
        for item in value:
            rect = _rect(item)  # type: ignore[arg-type]
            if rect is not None:
                return rect
    return None


def _to_screen_rect(rect: Rect, window_rect: Sequence[int] | None) -> Rect:
    win = _rect(window_rect)
    if win is None:
        return rect
    win_left, win_top, _win_right, _win_bottom = win
    left, top, right, bottom = rect
    return (left + win_left, top + win_top, right + win_left, bottom + win_top)


def _clamp_rect(rect: Rect, screenshot_size: tuple[int, int] | None) -> Rect | None:
    left, top, right, bottom = rect
    if screenshot_size:
        width, height = screenshot_size
        if width <= 0 or height <= 0:
            return None
        left = max(0, min(width, left))
        top = max(0, min(height, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return (int(left), int(top), int(right), int(bottom))


def _rect(value: Sequence[int] | None) -> Rect | None:
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _point(value: Sequence[int] | None) -> Point | None:
    if not isinstance(value, Sequence) or len(value) != 2:
        return None
    try:
        x, y = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x < 0 or y < 0:
        return None
    return (x, y)
