"""
WindowEnumService 单元测试

测试窗口枚举服务的核心逻辑
"""
import pytest
from unittest.mock import MagicMock, patch

from src.windows.window_enum import (
    WindowEnumService,
    WindowInfoExt,
    WindowState,
)


class TestWindowState:
    """WindowState 枚举测试"""

    def test_window_state_values(self):
        assert WindowState.NORMAL.value == "normal"
        assert WindowState.MINIMIZED.value == "minimized"
        assert WindowState.MAXIMIZED.value == "maximized"
        assert WindowState.OTHER.value == "other"


class TestWindowInfoExt:
    """WindowInfoExt 模型测试"""

    def test_default_state(self):
        """默认状态为 NORMAL"""
        info = WindowInfoExt(
            hwnd=123,
            title="Test",
        )
        assert info.state == WindowState.NORMAL

    def test_all_fields(self):
        """完整字段"""
        info = WindowInfoExt(
            hwnd=456,
            title="My Window",
            class_name="Notepad",
            process_name="notepad.exe",
            process_id=1234,
            rect=(0, 0, 800, 600),
            is_visible=True,
            is_enabled=True,
            state=WindowState.MAXIMIZED,
        )
        assert info.hwnd == 456
        assert info.title == "My Window"
        assert info.class_name == "Notepad"
        assert info.process_name == "notepad.exe"
        assert info.process_id == 1234
        assert info.rect == (0, 0, 800, 600)
        assert info.is_visible is True
        assert info.is_enabled is True
        assert info.state == WindowState.MAXIMIZED

    def test_rect_properties(self):
        """rect 属性计算"""
        info = WindowInfoExt(
            hwnd=1,
            title="Test",
            rect=(100, 200, 300, 400),
        )
        # rect = (left, top, right, bottom)
        assert info.rect == (100, 200, 300, 400)


class TestWindowInfoExtRect:
    """WindowInfoExt rect 相关属性测试"""

    def test_rect_parsing(self):
        """从 rect 元组正确解析宽高"""
        info = WindowInfoExt(
            hwnd=1,
            title="Test",
            rect=(0, 0, 1920, 1080),
        )
        # rect[2]-rect[0] = width, rect[3]-rect[1] = height
        assert info.rect[2] - info.rect[0] == 1920  # width
        assert info.rect[3] - info.rect[1] == 1080  # height
