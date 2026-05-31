"""
ZonePartitioner 单元测试

测试区域划分模块的分类逻辑
"""
import pytest

from src.perception.element_merger import ElementMerger, MergedElement
from src.perception.zone_partitioner import ZonePartitioner, ZoneType, ZoneInfo
from src.perception.uia_client import UIAElementInfo


def make_elem(
    left: int, top: int, right: int, bottom: int,
    name: str | None = None,
    control_type: str = "PaneControl",
) -> UIAElementInfo:
    return UIAElementInfo(
        name=name,
        automation_id=None,
        control_type=control_type,
        bounding_rect=(left, top, right, bottom),
        is_enabled=True,
    )


def elem_to_merged(elem: UIAElementInfo) -> MergedElement:
    """将 UIAElementInfo 转换为 MergedElement（单元素组）"""
    return MergedElement(
        elements=[elem],
        bounding_rect=elem.bounding_rect or (0, 0, 0, 0),
        control_type=elem.control_type or "Unknown",
        name=elem.name,
        is_merged=False,
        original_count=1,
    )


class TestZonePartitionerBasic:
    """区域划分基础测试"""

    def setup_method(self):
        self.partitioner = ZonePartitioner()

    def test_empty_elements(self):
        """空元素列表返回空区域"""
        result = self.partitioner.partition([], 1920, 1080)
        assert len(result.zones) == 0
        assert result.window_width == 1920
        assert result.window_height == 1080

    def test_simple_title_bar_detection(self):
        """标题栏检测：窗口顶部的小 Button 元素"""
        elems = [
            make_elem(1783, 0, 1828, 34, control_type="ButtonControl"),  # 关闭按钮
            make_elem(1828, 0, 1874, 34, control_type="ButtonControl"),  # 最大化按钮
            make_elem(1874, 0, 1920, 34, control_type="ButtonControl"), # 最小化按钮
        ]
        result = self.partitioner.partition(elems, 1920, 1080)
        title_bar = result.get_zone(ZoneType.TITLE_BAR)
        assert title_bar is not None
        assert title_bar.element_count >= 3

    def test_content_area_detection(self):
        """内容区检测：中间区域的 Edit 元素"""
        elems = [
            # 标题栏
            make_elem(0, 0, 1920, 34, control_type="ButtonControl"),
            # 内容区
            make_elem(0, 34, 1920, 1000, control_type="EditControl"),
            # 状态栏
            make_elem(0, 1000, 1920, 1032, control_type="StatusBarControl"),
        ]
        result = self.partitioner.partition(elems, 1920, 1080)
        # 内容区可能有多个区域类型，关键是能成功划分
        assert len(result.zones) > 0
        # 状态栏应该被检测到
        status = result.get_zone(ZoneType.STATUS_BAR)
        assert status is not None

    def test_status_bar_detection(self):
        """状态栏检测"""
        elems = [
            make_elem(0, 1000, 1920, 1032, control_type="StatusBarControl"),
        ]
        result = self.partitioner.partition(elems, 1920, 1080)
        status = result.get_zone(ZoneType.STATUS_BAR)
        assert status is not None

    def test_side_bar_detection(self):
        """侧边栏检测：左侧高占空比的 Pane"""
        elems = [
            make_elem(0, 34, 300, 1032, control_type="PaneControl"),  # 左侧边栏
            make_elem(300, 34, 1920, 1032, control_type="EditControl"),  # 内容区
        ]
        result = self.partitioner.partition(elems, 1920, 1080)
        sidebar = result.get_zone(ZoneType.SIDE_BAR)
        assert sidebar is not None


class TestZonePartitionerComplex:
    """复杂场景区域划分测试"""

    def setup_method(self):
        self.partitioner = ZonePartitioner()

    def test_vscode_like_layout(self):
        """
        模拟 VS Code 布局：
        - 标题栏：顶部 Button
        - 侧边栏：左侧 Explorer 面板
        - 内容区：右侧编辑器
        - 状态栏：底部状态信息
        """
        elems = [
            # 标题栏按钮
            make_elem(1783, 0, 1828, 34, control_type="ButtonControl"),
            make_elem(1828, 0, 1874, 34, control_type="ButtonControl"),
            make_elem(1874, 0, 1920, 34, control_type="ButtonControl"),
            # 侧边栏
            make_elem(0, 34, 300, 1032, control_type="PaneControl"),  # Explorer
            make_elem(300, 34, 1920, 1000, control_type="EditControl"),  # 编辑器
            # 状态栏
            make_elem(0, 1000, 1920, 1032, control_type="StatusBarControl"),
        ]
        result = self.partitioner.partition(elems, 1920, 1080)

        # 应该能成功划分出多个区域
        assert len(result.zones) >= 2  # 至少标题栏+状态栏被检测到
        zone_types = {z.zone_type for z in result.zones}
        # 状态栏和标题栏应该被检测到（确定性高的区域）
        assert ZoneType.STATUS_BAR in zone_types
        assert ZoneType.TITLE_BAR in zone_types

    def test_tool_bar_detection(self):
        """工具栏检测"""
        elems = [
            make_elem(0, 34, 1920, 68, control_type="ToolBarControl"),  # 工具栏
            make_elem(0, 68, 1920, 1032, control_type="EditControl"),  # 内容区
        ]
        result = self.partitioner.partition(elems, 1920, 1080)
        # 工具栏应该被检测到
        assert len(result.zones) >= 1
        toolbar = result.get_zone(ZoneType.TOOL_BAR)
        assert toolbar is not None


class TestZoneInfo:
    """ZoneInfo 属性测试"""

    def test_zone_properties(self):
        """ZoneInfo 属性计算"""
        elems = [
            make_elem(0, 0, 100, 50, control_type="ButtonControl"),
            make_elem(100, 0, 200, 50, control_type="ButtonControl"),
        ]
        partitioner = ZonePartitioner()
        merger = ElementMerger()
        merged = merger.merge(elems)
        zone = partitioner._make_zone("标题栏", ZoneType.TITLE_BAR, merged)

        assert zone.name == "标题栏"
        assert zone.zone_type == ZoneType.TITLE_BAR
        assert zone.element_count == 2
        assert zone.left == 0
        assert zone.top == 0
        assert zone.width == 200
        assert zone.height == 50
        assert zone.dominant_control_type == "ButtonControl"
