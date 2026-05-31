"""
UIAElementInfo 单元测试

测试 UIA 元素信息模型的属性计算
"""
import pytest

from src.perception.uia_client import UIAElementInfo


class TestUIAElementInfo:
    """UIAElementInfo 数据模型测试"""

    def test_basic_creation(self):
        elem = UIAElementInfo(
            name="Test Button",
            automation_id="btn1",
            control_type="ButtonControl",
            bounding_rect=(10, 20, 100, 50),
            is_enabled=True,
        )
        assert elem.name == "Test Button"
        assert elem.automation_id == "btn1"
        assert elem.control_type == "ButtonControl"
        assert elem.bounding_rect == (10, 20, 100, 50)
        assert elem.is_enabled is True

    def test_rect_properties(self):
        """rect 属性计算"""
        elem = UIAElementInfo(
            name="Panel",
            automation_id="panel1",
            control_type="PaneControl",
            bounding_rect=(0, 0, 1920, 1032),
            is_enabled=True,
        )
        assert elem.left == 0
        assert elem.top == 0
        assert elem.right == 1920
        assert elem.bottom == 1032
        assert elem.width == 1920
        assert elem.height == 1032

    def test_rect_with_frame_offset(self):
        """带窗口边框偏移的 rect"""
        elem = UIAElementInfo(
            name="Frame",
            automation_id="frame1",
            control_type="WindowControl",
            bounding_rect=(-8, -8, 1928, 1040),
            is_enabled=True,
        )
        assert elem.left == -8
        assert elem.top == -8
        assert elem.width == 1936
        assert elem.height == 1048

    def test_optional_fields(self):
        """可选字段"""
        elem = UIAElementInfo(
            name=None,
            automation_id=None,
            control_type="PaneControl",
            bounding_rect=(0, 0, 100, 100),
            is_enabled=False,
        )
        assert elem.name is None
        assert elem.automation_id is None
        assert elem.is_enabled is False

    def test_handle_field(self):
        """handle 字段"""
        elem = UIAElementInfo(
            name="Element",
            automation_id="el1",
            control_type="Control",
            bounding_rect=(0, 0, 50, 50),
            is_enabled=True,
            handle=12345,
        )
        assert elem.handle == 12345
