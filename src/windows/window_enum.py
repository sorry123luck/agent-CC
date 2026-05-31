"""
窗口枚举与句柄获取服务

提供窗口枚举、标题匹配、进程过滤等功能
"""
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

import psutil
from win32gui import (
    EnumWindows, GetWindowText, IsWindowVisible, GetClassName,
    IsWindowEnabled, IsIconic, GetWindowPlacement,
    GetForegroundWindow as win32_GetForegroundWindow,
)
from win32process import GetWindowThreadProcessId

from src.common.errors import WindowNotFoundError
from src.windows.window_bounds import get_capture_frame_bounds


class WindowState(Enum):
    """窗口状态"""
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    OTHER = "other"


class WindowInfoExt(BaseModel):
    """扩展窗口信息"""
    hwnd: int
    title: str
    class_name: str | None = None
    process_name: str | None = None
    process_id: int | None = None
    rect: tuple[int, int, int, int] | None = None
    is_visible: bool = True
    is_enabled: bool = True
    state: WindowState = WindowState.NORMAL

    model_config = {"from_attributes": True}


# 模块级回调列表，用于 EnumWindows
_window_list: list[WindowInfoExt] = []


def _enum_callback(hwnd: int, _) -> bool:
    """EnumWindows 的回调函数（模块级）"""
    try:
        if not IsWindowVisible(hwnd):
            return True

        title = GetWindowText(hwnd)
        if not title or not title.strip():
            return True

        class_name = GetClassName(hwnd)
        rect = get_capture_frame_bounds(hwnd)
        is_enabled = IsWindowEnabled(hwnd)

        process_id = None
        process_name = None
        try:
            # GetWindowThreadProcessId returns (thread_id, process_id)
            # We need process_id which is index [1]
            pid = GetWindowThreadProcessId(hwnd)[1]
            process_id = pid
            process = psutil.Process(pid)
            process_name = process.name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        rect_tuple = (rect[0], rect[1], rect[2], rect[3])

        # 判断窗口状态
        if IsIconic(hwnd):
            state = WindowState.MINIMIZED
        else:
            placement = GetWindowPlacement(hwnd)
            if placement[1] == 3:  # SW_SHOWMAXIMIZED == 3
                state = WindowState.MAXIMIZED
            else:
                state = WindowState.NORMAL

        win_info = WindowInfoExt(
            hwnd=hwnd,
            title=title,
            class_name=class_name,
            process_name=process_name,
            process_id=process_id,
            rect=rect_tuple,
            is_visible=True,
            is_enabled=is_enabled,
            state=state,
        )
        _window_list.append(win_info)
    except Exception:
        pass
    return True


class WindowEnumService:
    """窗口枚举服务"""

    def __init__(self) -> None:
        self._windows: list[WindowInfoExt] = []
        self._refresh()

    def _refresh(self) -> None:
        """刷新窗口列表"""
        global _window_list
        _window_list = []
        EnumWindows(_enum_callback, None)
        self._windows = list(_window_list)

    def enumerate_all(self, refresh: bool = False) -> list[WindowInfoExt]:
        """
        枚举所有顶层窗口

        Args:
            refresh: 是否强制刷新，默认 False（使用缓存）

        Returns:
            窗口信息列表
        """
        if refresh:
            self._refresh()
        return list(self._windows)

    def find_by_title(
        self, pattern: str, refresh: bool = False
    ) -> list[WindowInfoExt]:
        """
        按窗口标题模糊匹配查找

        支持通配符 *，如 "*Notepad*"

        Args:
            pattern: 标题模式（支持 * 通配符）
            refresh: 是否强制刷新

        Returns:
            匹配的窗口列表
        """
        if refresh:
            self._refresh()

        if "*" not in pattern:
            # 精确匹配
            return [w for w in self._windows if pattern.lower() in w.title.lower()]

        # 通配符匹配
        import fnmatch
        return [
            w for w in self._windows
            if fnmatch.fnmatch(w.title.lower(), pattern.lower())
        ]

    def find_by_process(
        self, process_name: str, refresh: bool = False
    ) -> list[WindowInfoExt]:
        """
        按进程名过滤窗口

        Args:
            process_name: 进程名（如 "explorer.exe"），不区分大小写
            refresh: 是否强制刷新

        Returns:
            匹配的窗口列表
        """
        if refresh:
            self._refresh()

        process_name_lower = process_name.lower()
        return [
            w for w in self._windows
            if w.process_name and w.process_name.lower() == process_name_lower
        ]

    def find_by_class(
        self, class_name: str, refresh: bool = False
    ) -> list[WindowInfoExt]:
        """
        按窗口类名过滤

        Args:
            class_name: 类名（不区分大小写）
            refresh: 是否强制刷新

        Returns:
            匹配的窗口列表
        """
        if refresh:
            self._refresh()

        class_name_lower = class_name.lower()
        return [
            w for w in self._windows
            if w.class_name and w.class_name.lower() == class_name_lower
        ]

    def find_one(self, pattern: str, refresh: bool = False) -> WindowInfoExt:
        """
        查找单个窗口，匹配多个时返回第一个

        Args:
            pattern: 标题模式
            refresh: 是否强制刷新

        Returns:
            第一个匹配的窗口

        Raises:
            WindowNotFoundError: 未找到窗口
        """
        results = self.find_by_title(pattern, refresh=refresh)
        if not results:
            raise WindowNotFoundError(f"未找到标题匹配 '{pattern}' 的窗口")
        return results[0]

    def filter(
        self,
        predicate: Callable[[WindowInfoExt], bool],
        refresh: bool = False,
    ) -> list[WindowInfoExt]:
        """
        使用自定义过滤器筛选窗口

        Args:
            predicate: 过滤函数，接收 WindowInfoExt，返回 bool
            refresh: 是否强制刷新

        Returns:
            满足条件的窗口列表
        """
        if refresh:
            self._refresh()
        return [w for w in self._windows if predicate(w)]

    def get_foreground_window(self) -> WindowInfoExt | None:
        """
        获取当前前台窗口

        Returns:
            前台窗口信息，未找到返回 None
        """
        hwnd = win32_GetForegroundWindow()
        if not hwnd:
            return None

        try:
            title = GetWindowText(hwnd)
            class_name = GetClassName(hwnd)
            rect = get_capture_frame_bounds(hwnd)
            rect_tuple = (rect[0], rect[1], rect[2], rect[3])
            is_enabled = IsWindowEnabled(hwnd)
            is_visible = IsWindowVisible(hwnd)

            try:
                # GetWindowThreadProcessId returns (thread_id, process_id)
                pid = GetWindowThreadProcessId(hwnd)[1]
                process_id = pid
                process = psutil.Process(pid)
                process_name = process.name().lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                process_id = None
                process_name = None

            if IsIconic(hwnd):
                state = WindowState.MINIMIZED
            else:
                placement = GetWindowPlacement(hwnd)
                if placement[1] == 3:
                    state = WindowState.MAXIMIZED
                else:
                    state = WindowState.NORMAL

            return WindowInfoExt(
                hwnd=hwnd,
                title=title,
                class_name=class_name,
                process_name=process_name,
                process_id=process_id,
                rect=rect_tuple,
                is_visible=is_visible,
                is_enabled=is_enabled,
                state=state,
            )
        except Exception:
            return None
