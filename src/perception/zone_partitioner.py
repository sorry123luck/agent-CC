"""
区域划分模块

将 UIA 元素按窗口功能区进行划分：
- 标题栏（title_bar）：窗口顶部，包含窗口控制按钮
- 菜单栏（menu_bar）：菜单项水平排列
- 工具栏（tool_bar）：工具按钮水平排列
- 侧边栏（side_bar）：窗口左侧或右侧的面板
- 内容区（content_area）：主要工作区域
- 状态栏（status_bar）：窗口底部的状态信息

区域划分基于 UIA 元素的 position + control_type 启发式判断。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from src.perception.element_merger import ElementMerger, MergedElement
from src.perception.uia_client import UIAElementInfo


class ZoneType(Enum):
    """区域类型"""
    TITLE_BAR = "title_bar"
    MENU_BAR = "menu_bar"
    TOOL_BAR = "tool_bar"
    SIDE_BAR = "side_bar"
    CONTENT_AREA = "content_area"
    STATUS_BAR = "status_bar"
    UNKNOWN = "unknown"


@dataclass
class ZoneInfo:
    """区域信息"""
    name: str  # 区域名称（人类可读）
    zone_type: ZoneType
    elements: list[MergedElement] = field(default_factory=list)
    bounding_rect: tuple[int, int, int, int] | None = None  # 区域边界框
    element_count: int = 0
    dominant_control_type: str | None = None  # 主导控制类型

    @property
    def left(self) -> int | None:
        return self.bounding_rect[0] if self.bounding_rect else None

    @property
    def top(self) -> int | None:
        return self.bounding_rect[1] if self.bounding_rect else None

    @property
    def width(self) -> int | None:
        if not self.bounding_rect:
            return None
        return self.bounding_rect[2] - self.bounding_rect[0]

    @property
    def height(self) -> int | None:
        if not self.bounding_rect:
            return None
        return self.bounding_rect[3] - self.bounding_rect[1]


@dataclass
class WindowZoneStructure:
    """窗口区域结构"""
    zones: list[ZoneInfo] = field(default_factory=list)
    window_width: int = 0
    window_height: int = 0

    def get_zone(self, zone_type: ZoneType) -> ZoneInfo | None:
        """获取指定类型的区域"""
        for zone in self.zones:
            if zone.zone_type == zone_type:
                return zone
        return None

    def get_all_zones_ordered(self) -> list[ZoneInfo]:
        """按从上到下、从左到右的顺序返回所有区域"""
        return sorted(self.zones, key=lambda z: (z.top or 0, z.left or 0))


class ZonePartitioner:
    """窗口区域划分器"""

    # 窗口区域比例阈值
    TITLE_BAR_HEIGHT_RATIO = 0.05  # 标题栏高度不超过窗口高度的 5%
    STATUS_BAR_HEIGHT_RATIO = 0.05  # 状态栏高度不超过窗口高度的 5%
    SIDE_BAR_WIDTH_RATIO = 0.3  # 侧边栏宽度不超过窗口宽度的 30%

    # Y 坐标容差（用于判断同一行）
    Y_TOLERANCE = 20

    def __init__(self, merger: ElementMerger | None = None) -> None:
        self.merger = merger or ElementMerger()
        self._element_id_counter = 0

    def _next_element_id(self) -> str:
        """生成唯一 element_id"""
        self._element_id_counter += 1
        return f"zp_{self._element_id_counter}"

    def partition(
        self,
        elements: list[UIAElementInfo],
        window_width: int,
        window_height: int,
    ) -> WindowZoneStructure:
        """
        对 UIA 元素进行区域划分

        Args:
            elements: UIA 元素列表
            window_width: 窗口内容区宽度
            window_height: 窗口内容区高度

        Returns:
            WindowZoneStructure 包含划分后的区域列表
        """
        if not elements:
            return WindowZoneStructure(window_width=window_width, window_height=window_height)

        # Step 1: 归并元素
        merged_elements = self.merger.merge(elements)

        # Step 2: 预分类元素
        classified = self._classify_elements(merged_elements, window_width, window_height)

        # Step 3: 构建区域
        zones = self._build_zones(classified, window_width, window_height)

        return WindowZoneStructure(
            zones=zones,
            window_width=window_width,
            window_height=window_height,
        )

    def partition_from_dicts(
        self,
        raw_elements: list[dict[str, Any]],
        window_width: int,
        window_height: int,
    ) -> WindowZoneStructure:
        """
        从原始元素字典列表进行区域划分（内部使用）

        将 dict 格式的 raw_elements 转换为 UIAElementInfo，
        并为每个元素生成唯一 element_id 以便后续追踪。

        Args:
            raw_elements: 原始元素字典列表（来自 IUIAProvider.find_all_elements()）
            window_width: 窗口内容区宽度
            window_height: 窗口内容区高度

        Returns:
            WindowZoneStructure 包含划分后的区域列表
        """
        if not raw_elements:
            return WindowZoneStructure(window_width=window_width, window_height=window_height)

        # 将 dict 转换为 UIAElementInfo，保留 element_id
        uia_elements: list[UIAElementInfo] = []
        for elem_dict in raw_elements:
            bounds = elem_dict.get("bounding_rect")
            uia_elem = UIAElementInfo(
                name=elem_dict.get("name"),
                automation_id=elem_dict.get("automation_id"),
                control_type=elem_dict.get("control_type"),
                bounding_rect=bounds,
                is_enabled=True,
                handle=None,
                element_id=elem_dict.get("element_id") or self._next_element_id(),
            )
            uia_elements.append(uia_elem)

        return self.partition(uia_elements, window_width, window_height)

    def _classify_elements(
        self,
        elements: list[MergedElement],
        window_width: int,
        window_height: int,
    ) -> dict[ZoneType, list[MergedElement]]:
        """对每个归并后元素进行区域类型预分类"""
        classified: dict[ZoneType, list[MergedElement]] = {zt: [] for zt in ZoneType}

        title_bar_height_max = int(window_height * self.TITLE_BAR_HEIGHT_RATIO) + self.Y_TOLERANCE
        status_bar_height_max = int(window_height * self.STATUS_BAR_HEIGHT_RATIO) + self.Y_TOLERANCE
        side_bar_width_max = int(window_width * self.SIDE_BAR_WIDTH_RATIO)

        for elem in elements:
            zone_type = self._classify_single_element(
                elem, window_width, window_height,
                title_bar_height_max, status_bar_height_max, side_bar_width_max
            )
            classified[zone_type].append(elem)

        return classified

    def _classify_single_element(
        self,
        elem: MergedElement,
        window_width: int,
        window_height: int,
        title_bar_height_max: int,
        status_bar_height_max: int,
        side_bar_width_max: int,
    ) -> ZoneType:
        """对单个归并元素进行区域类型分类"""
        ct = elem.control_type or ""
        is_button = "button" in ct.lower()
        is_menu = any(t in ct.lower() for t in ["menu", "item"])
        is_toolbar = "tool" in ct.lower()
        is_pane = "pane" in ct.lower()
        is_edit = any(t in ct.lower() for t in ["edit", "text", "document", "scroll"])
        is_list = any(t in ct.lower() for t in ["list", "tree"])
        is_status = "status" in ct.lower()

        top = elem.top
        bottom = elem.bottom
        left = elem.left
        right = elem.right

        # 状态栏：窗口底部，StatusBar 类型
        if is_status:
            return ZoneType.STATUS_BAR

        # 从底部往上找状态栏
        if bottom >= window_height - status_bar_height_max and top >= window_height - status_bar_height_max * 2:
            if is_status or any(t in ct.lower() for t in ["bar", "panel"]):
                return ZoneType.STATUS_BAR

        # 标题栏：窗口顶部，包含 Button 元素
        # 但 Toolbar/Pane 等非按钮类型不应误归为标题栏
        if top <= title_bar_height_max and bottom <= title_bar_height_max * 2:
            if is_button or any(t in ct.lower() for t in ["title", "window"]):
                return ZoneType.TITLE_BAR
            # ToolBarControl 专门处理，不归为标题栏
            if "toolbar" in ct.lower():
                return ZoneType.TOOL_BAR

        # 侧边栏：窗口左侧或右侧，高度接近窗口高度
        elem_height = elem.height
        if elem_height >= window_height * 0.6:
            if left <= side_bar_width_max:
                return ZoneType.SIDE_BAR
            if right >= window_width - side_bar_width_max:
                return ZoneType.SIDE_BAR

        # 菜单栏：标题栏下方，Menu 类型或水平排列的 MenuItem
        if top <= window_height * 0.15 and top > title_bar_height_max:
            if is_menu or any(t in ct.lower() for t in ["menu", "item"]):
                return ZoneType.MENU_BAR

        # 工具栏：标题栏或菜单栏下方，ToolBar 类型或水平 Button 组
        if top > title_bar_height_max and top <= window_height * 0.25:
            if is_toolbar or "toolbar" in ct.lower():
                return ZoneType.TOOL_BAR
            # 水平排列的 Button 组也视为工具栏
            if is_button and elem.width > elem.height * 2:
                return ZoneType.TOOL_BAR

        # 内容区：中间区域，Edit/Document/List 类型
        if is_edit or is_list and not is_pane:
            return ZoneType.CONTENT_AREA

        # 默认放到内容区（Pane 类型通常是内容区容器）
        if is_pane and not (top <= title_bar_height_max or bottom >= window_height - status_bar_height_max):
            # 如果不是标题栏区域或状态栏区域的 Pane，归为内容区
            if top > window_height * 0.1 and bottom < window_height * 0.95:
                return ZoneType.CONTENT_AREA

        return ZoneType.UNKNOWN

    def _build_zones(
        self,
        classified: dict[ZoneType, list[MergedElement]],
        window_width: int,
        window_height: int,
    ) -> list[ZoneInfo]:
        """根据预分类结果构建区域"""
        zones: list[ZoneInfo] = []

        # 内容区（必须包含所有未被分类的元素）
        content_elems: list[MergedElement] = []
        for zone_type, elems in classified.items():
            if zone_type in (ZoneType.CONTENT_AREA, ZoneType.UNKNOWN):
                content_elems.extend(elems)

        # 对 content_area 做坐标合法性过滤：
        # 过滤掉 bounding_rect 完全越界或 top>=bottom 的无效元素
        if content_elems:
            content_elems = self._filter_content_area_elements(content_elems, window_width, window_height)

        # 标题栏
        title_bar_elems = classified[ZoneType.TITLE_BAR]
        if title_bar_elems:
            zones.append(self._make_zone(
                "标题栏", ZoneType.TITLE_BAR, title_bar_elems
            ))

        # 菜单栏
        menu_bar_elems = classified[ZoneType.MENU_BAR]
        if menu_bar_elems:
            zones.append(self._make_zone(
                "菜单栏", ZoneType.MENU_BAR, menu_bar_elems
            ))

        # 工具栏
        tool_bar_elems = classified[ZoneType.TOOL_BAR]
        if tool_bar_elems:
            zones.append(self._make_zone(
                "工具栏", ZoneType.TOOL_BAR, tool_bar_elems
            ))

        # 侧边栏
        side_bar_elems = classified[ZoneType.SIDE_BAR]
        if side_bar_elems:
            zones.append(self._make_zone(
                "侧边栏", ZoneType.SIDE_BAR, side_bar_elems
            ))

        if content_elems:
            zones.append(self._make_zone(
                "内容区", ZoneType.CONTENT_AREA, content_elems
            ))

        # 状态栏
        status_bar_elems = classified[ZoneType.STATUS_BAR]
        if status_bar_elems:
            zones.append(self._make_zone(
                "状态栏", ZoneType.STATUS_BAR, status_bar_elems
            ))

        return zones

    def _filter_content_area_elements(
        self,
        elements: list[MergedElement],
        window_width: int,
        window_height: int,
    ) -> list[MergedElement]:
        """
        过滤 content_area 中坐标严重越界的元素

        WebView/iframe 子元素可能使用不同坐标原点，导致 bounding_rect 越界。
        过滤标准：
        1. bounding_rect 的 top 必须 >= 0（内容区不应在窗口上方）
        2. bounding_rect 的 bottom 必须 <= window_height（不应在窗口下方）
        3. bounding_rect 的 left >= 0 且 right <= window_width
        """
        valid_top = int(window_height * 0.05)  # 内容区 top 应在窗口 5% 高度以下
        valid: list[MergedElement] = []
        for elem in elements:
            if not elem.bounding_rect:
                continue
            l, t, r, b = elem.bounding_rect
            # 过滤：top < 0 或 bottom > window_height 或 top >= bottom
            if t < 0 or b > window_height or t >= b:
                continue
            # 额外安全过滤：坐标值过大（疑似坐标系统错误）
            if any(abs(v) > window_width * 5 for v in (l, t, r, b)):
                continue
            valid.append(elem)
        return valid

    def _make_zone(
        self,
        name: str,
        zone_type: ZoneType,
        elements: list[MergedElement],
    ) -> ZoneInfo:
        """创建区域对象"""
        if not elements:
            return ZoneInfo(name=name, zone_type=zone_type, element_count=0)

        # 计算边界框
        all_rects = [e.bounding_rect for e in elements if e.bounding_rect]
        if all_rects:
            left = min(r[0] for r in all_rects)
            top = min(r[1] for r in all_rects)
            right = max(r[2] for r in all_rects)
            bottom = max(r[3] for r in all_rects)
            bounding_rect = (left, top, right, bottom)
        else:
            bounding_rect = None

        # 主导控制类型
        control_types = [e.control_type for e in elements]
        from collections import Counter
        dominant = Counter(control_types).most_common(1)
        dominant_ct = dominant[0][0] if dominant else None

        return ZoneInfo(
            name=name,
            zone_type=zone_type,
            elements=elements,
            bounding_rect=bounding_rect,
            element_count=sum(e.original_count for e in elements),
            dominant_control_type=dominant_ct,
        )
