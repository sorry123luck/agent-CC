"""
窗口前景切换服务

提供窗口激活、前景切换等功能
"""
from typing import Union

import win32gui
import win32con
import win32api

from src.windows.window_enum import WindowInfoExt
from src.common.errors import WindowNotFoundError


# ShowWindow 命令
SW_RESTORE = win32con.SW_RESTORE
SW_SHOW = win32con.SW_SHOW
SW_SHOWMAXIMIZED = win32con.SW_SHOWMAXIMIZED
SW_SHOWMINIMIZED = win32con.SW_SHOWMINIMIZED


class WindowActivator:
    """窗口激活服务"""

    def activate(
        self, window: Union[WindowInfoExt, int]
    ) -> bool:
        """
        将窗口置前并激活

        Args:
            window: WindowInfoExt 对象或窗口句柄(int)

        Returns:
            是否成功激活

        Raises:
            WindowNotFoundError: 窗口句柄无效
        """
        hwnd = self._to_hwnd(window)
        if not hwnd:
            raise WindowNotFoundError(f"无效的窗口句柄: {hwnd}")

        try:
            # 检查窗口是否最小化，如果是，还原窗口
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, SW_RESTORE)

            # 将窗口置前
            # 使用多种方法确保成功
            success = False

            # 方法1: SetForegroundWindow
            try:
                win32gui.SetForegroundWindow(hwnd)
                success = True
            except Exception:
                pass

            # 方法2: 如果方法1失败，尝试先 BringWindowToTop 再 SetForegroundWindow
            if not success:
                try:
                    win32gui.BringWindowToTop(hwnd)
                    win32gui.SetForegroundWindow(hwnd)
                    success = True
                except Exception:
                    pass

            # 方法3: 尝试 SetWindowPos
            if not success:
                try:
                    win32gui.SetWindowPos(
                        hwnd,
                        win32con.HWND_TOP,
                        0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
                    )
                    success = True
                except Exception:
                    pass

            # 确保窗口可见
            if success:
                try:
                    win32gui.ShowWindow(hwnd, SW_SHOW)
                except Exception:
                    pass

            return success

        except Exception as e:
            raise WindowNotFoundError(f"激活窗口失败: {e}")

    def _to_hwnd(self, window: Union[WindowInfoExt, int]) -> int | None:
        """从 WindowInfoExt 或 int 获取窗口句柄"""
        if isinstance(window, int):
            return window
        if isinstance(window, WindowInfoExt):
            return window.hwnd
        return None

    def minimize(self, window: Union[WindowInfoExt, int]) -> bool:
        """
        最小化窗口

        Args:
            window: WindowInfoExt 对象或窗口句柄

        Returns:
            是否成功
        """
        hwnd = self._to_hwnd(window)
        if not hwnd:
            raise WindowNotFoundError(f"无效的窗口句柄: {hwnd}")

        try:
            win32gui.ShowWindow(hwnd, SW_SHOWMINIMIZED)
            return True
        except Exception as e:
            raise WindowNotFoundError(f"最小化窗口失败: {e}")

    def maximize(self, window: Union[WindowInfoExt, int]) -> bool:
        """
        最大化窗口

        Args:
            window: WindowInfoExt 对象或窗口句柄

        Returns:
            是否成功
        """
        hwnd = self._to_hwnd(window)
        if not hwnd:
            raise WindowNotFoundError(f"无效的窗口句柄: {hwnd}")

        try:
            win32gui.ShowWindow(hwnd, SW_SHOWMAXIMIZED)
            return True
        except Exception as e:
            raise WindowNotFoundError(f"最大化窗口失败: {e}")

    def restore(self, window: Union[WindowInfoExt, int]) -> bool:
        """
        还原窗口（从最小化或最大化状态）

        Args:
            window: WindowInfoExt 对象或窗口句柄

        Returns:
            是否成功
        """
        hwnd = self._to_hwnd(window)
        if not hwnd:
            raise WindowNotFoundError(f"无效的窗口句柄: {hwnd}")

        try:
            win32gui.ShowWindow(hwnd, SW_RESTORE)
            return True
        except Exception as e:
            raise WindowNotFoundError(f"还原窗口失败: {e}")
