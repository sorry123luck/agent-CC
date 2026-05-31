"""
元素归并模块

将 UIA 元素树中的细碎元素归并为逻辑上更合理的组。

归并策略（v2 - 基于水平 Band 聚类）：
1. 水平 Band 聚类：将元素按 Y 轴中心坐标聚类到水平带（每带 30px 高度）
2. 同带同类型归并：在同一水平带内，相同类型的相邻元素合并为一个组
3. 跨带同类归并：相邻水平带内相同类型的元素，如果 Y 间距小也可以合并
4. 最小面积过滤：过滤极小元素（< 5px 宽或高）

核心设计原则：
- 优先保证归并后的元素有实际语义（同一 UI 行内的同类型元素）
- 不破坏 UIA 树层级结构（仅归并非容器元素）
"""
from dataclasses import dataclass, field
from collections import defaultdict

from src.perception.uia_client import UIAElementInfo


@dataclass
class MergedElement:
    """归并后的元素"""
    elements: list[UIAElementInfo]  # 被归并的原始元素列表
    bounding_rect: tuple[int, int, int, int]  # 归并后的边界框（所有元素的并集）
    control_type: str  # 统一控制类型
    name: str | None  # 如果所有元素 name 相同则保留，否则 None
    is_merged: bool  # 是否发生了归并（True）还是保持单元素（False）
    original_count: int  # 被归并的元素数量
    element_id: str | None = None  # 归并后元素的唯一标识
    original_element_ids: list[str] = field(default_factory=list)  # 原始 UIAElementInfo.element_id 列表

    @property
    def left(self) -> int:
        return self.bounding_rect[0]

    @property
    def top(self) -> int:
        return self.bounding_rect[1]

    @property
    def right(self) -> int:
        return self.bounding_rect[2]

    @property
    def bottom(self) -> int:
        return self.bounding_rect[3]

    @property
    def width(self) -> int:
        return self.bounding_rect[2] - self.bounding_rect[0]

    @property
    def height(self) -> int:
        return self.bounding_rect[3] - self.bounding_rect[1]


class ElementMerger:
    """
    UIA 元素归并器（v2 - 基于水平 Band 聚类）

    更激进的归并策略，适用于 VS Code 这类元素众多的 GUI 应用。
    """

    def __init__(
        self,
        min_element_width: int = 5,
        min_element_height: int = 5,
        band_height: int = 100,  # 水平带高度（px）- 较大的值可以跨行归并
        same_row_gap: int = 50,  # 同行元素最大间距
    ) -> None:
        self.min_element_width = min_element_width
        self.min_element_height = min_element_height
        self.band_height = band_height
        self.same_row_gap = same_row_gap
        self._element_id_counter = 0

    def _next_element_id(self) -> str:
        """生成下一个唯一 element_id"""
        self._element_id_counter += 1
        return f"merged_{self._element_id_counter}"

    def merge(self, elements: list[UIAElementInfo]) -> list[MergedElement]:
        """
        对 UIA 元素列表进行归并

        Args:
            elements: 原始 UIA 元素列表

        Returns:
            归并后的 MergedElement 列表
        """
        if not elements:
            return []

        # Step 1: 过滤极小元素
        filtered = self._filter_tiny_elements(elements)
        if not filtered:
            return []

        # Step 2: 按水平 Band 聚类
        bands = self._cluster_into_bands(filtered)

        # Step 3: 在每个 band 内按类型分组归并
        merged_groups = self._merge_within_bands(bands)

        # Step 4: 转换为 MergedElement
        result = [self._make_merged_element(group) for group in merged_groups]
        return result

    def _filter_tiny_elements(
        self, elements: list[UIAElementInfo]
    ) -> list[UIAElementInfo]:
        """过滤极小的元素"""
        result = []
        for elem in elements:
            if not elem.bounding_rect or not elem.width or not elem.height:
                continue
            if elem.width < self.min_element_width and elem.height < self.min_element_height:
                continue
            result.append(elem)
        return result

    def _cluster_into_bands(
        self, elements: list[UIAElementInfo]
    ) -> dict[int, list[UIAElementInfo]]:
        """
        将元素按 Y 轴中心坐标聚类到水平带

        Returns:
            {band_index: [elements in this band]}
        """
        bands: dict[int, list[UIAElementInfo]] = defaultdict(list)

        for elem in elements:
            if not elem.bounding_rect:
                continue
            center_y = elem.top + elem.height // 2
            band_idx = center_y // self.band_height
            bands[band_idx].append(elem)

        return bands

    def _merge_within_bands(
        self, bands: dict[int, list[UIAElementInfo]]
    ) -> list[list[UIAElementInfo]]:
        """
        在每个 band 内按类型分组，同类型的相邻元素合并

        同时处理跨 band 的合并（相邻 band 内的同类元素）
        """
        # 在 band 内按类型分组
        all_groups: list[list[UIAElementInfo]] = []

        for band_idx in sorted(bands.keys()):
            elems = bands[band_idx]
            # 按 X 坐标排序
            sorted_elems = sorted(elems, key=lambda e: e.left)

            # 在排序后的元素中，按类型和间距分组
            current_group: list[UIAElementInfo] = []

            for elem in sorted_elems:
                if not current_group:
                    current_group.append(elem)
                    continue

                last_elem = current_group[-1]

                # 判断是否应该与当前组归并
                # 条件：同类型，且 X 间距在阈值内
                if (elem.control_type == last_elem.control_type and
                    elem.left - last_elem.right <= self.same_row_gap):
                    current_group.append(elem)
                else:
                    all_groups.append(current_group)
                    current_group = [elem]

            if current_group:
                all_groups.append(current_group)

        # 处理跨 band 的合并（相邻 band 的同类元素）
        # 由于 band 已经是 30px 高度，相邻 band 意味着 Y 坐标相差 < 60px
        # 如果两个 band 的元素类型相同且水平位置接近，可以合并
        final_groups = self._merge_cross_band(all_groups)

        return final_groups

    def _merge_cross_band(
        self, groups: list[list[UIAElementInfo]]
    ) -> list[list[UIAElementInfo]]:
        """
        处理跨 band 的合并

        相邻 band 内类型相同的元素，如果水平重叠或接近，合并为一个组
        """
        if len(groups) <= 1:
            return groups

        # 按 top Y 坐标排序
        sorted_groups = sorted(groups, key=lambda g: min(e.top for e in g))
        result: list[list[UIAElementInfo]] = []

        for group in sorted_groups:
            if not result:
                result.append(group)
                continue

            last_group = result[-1]

            # 检查是否可以合并
            if self._can_merge_groups(last_group, group):
                result[-1] = last_group + group
            else:
                result.append(group)

        return result

    def _can_merge_groups(
        self,
        group1: list[UIAElementInfo],
        group2: list[UIAElementInfo],
    ) -> bool:
        """判断两个元素组是否可以合并"""
        if not group1 or not group2:
            return False

        # 必须是同类型
        ct1 = group1[0].control_type
        ct2 = group2[0].control_type
        if ct1 != ct2:
            return False

        # 检查 Y 轴是否接近（相邻 band：Y 差距 < band_height * 2）
        max_top1 = max(e.top for e in group1)
        min_top2 = min(e.top for e in group2)

        if min_top2 - max_top1 > self.band_height * 2:
            return False

        # 检查 X 轴是否有重叠或接近
        max_right1 = max(e.right for e in group1)
        min_left2 = min(e.left for e in group2)

        # 水平接近或重叠
        if min_left2 - max_right1 > self.same_row_gap * 2:
            return False

        return True

    def _make_merged_element(
        self, group: list[UIAElementInfo]
    ) -> MergedElement:
        """将一组元素归并为一个 MergedElement"""
        # 收集原始 element_ids（来自 UIAElementInfo）
        orig_ids = [e.element_id for e in group if e.element_id]
        merged_id = self._next_element_id()

        if len(group) == 1:
            elem = group[0]
            return MergedElement(
                elements=group,
                bounding_rect=elem.bounding_rect or (0, 0, 0, 0),
                control_type=elem.control_type or "Unknown",
                name=elem.name,
                is_merged=False,
                original_count=1,
                element_id=elem.element_id or merged_id,
                original_element_ids=orig_ids,
            )

        # 计算并集边界框
        left = min(e.bounding_rect[0] for e in group if e.bounding_rect)
        top = min(e.bounding_rect[1] for e in group if e.bounding_rect)
        right = max(e.bounding_rect[2] for e in group if e.bounding_rect)
        bottom = max(e.bounding_rect[3] for e in group if e.bounding_rect)

        # 检查 name 是否全部相同
        names = set(e.name for e in group if e.name)
        common_name = list(names)[0] if len(names) == 1 else None

        control_types = set(e.control_type for e in group if e.control_type)
        control_type = list(control_types)[0] if control_types else "Unknown"

        return MergedElement(
            elements=group,
            bounding_rect=(left, top, right, bottom),
            control_type=control_type,
            name=common_name,
            is_merged=True,
            original_count=len(group),
            element_id=merged_id,
            original_element_ids=orig_ids,
        )

    def get_merge_stats(
        self, original: list[UIAElementInfo], merged: list[MergedElement]
    ) -> dict:
        """获取归并统计信息"""
        merged_count = len(merged)
        original_count = len(original)
        merged_elems = sum(1 for m in merged if m.is_merged)

        # 总原始元素数（从 MergedElement 追溯）
        total_original = sum(m.original_count for m in merged)

        return {
            "original_count": original_count,
            "merged_count": merged_count,
            "merged_elements": merged_elems,
            "reduction_ratio": 1 - merged_count / original_count if original_count > 0 else 0,
            "avg_group_size": sum(m.original_count for m in merged if m.is_merged) / merged_elems if merged_elems > 0 else 1.0,
        }
