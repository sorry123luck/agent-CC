"""
ElementMerger 单元测试

测试元素归并模块 v2（基于水平 Band 聚类）
"""
import pytest

from src.perception.element_merger import ElementMerger, MergedElement
from src.perception.uia_client import UIAElementInfo


def make_elem(
    left: int, top: int, right: int, bottom: int,
    name: str | None = None,
    control_type: str = "ButtonControl",
    automation_id: str | None = None,
) -> UIAElementInfo:
    """创建测试用 UIA 元素"""
    return UIAElementInfo(
        name=name,
        automation_id=automation_id,
        control_type=control_type,
        bounding_rect=(left, top, right, bottom),
        is_enabled=True,
    )


class TestElementMergerFiltering:
    """元素过滤测试"""

    def setup_method(self):
        self.merger = ElementMerger()

    def test_filter_tiny_elements(self):
        """极小元素应被过滤"""
        elems = [
            make_elem(0, 0, 3, 3),   # 宽3<5 且 高3<5 → 过滤
            make_elem(10, 0, 50, 20), # 宽40>5 → 保留
        ]
        filtered = self.merger._filter_tiny_elements(elems)
        assert len(filtered) == 1


class TestBandClustering:
    """水平 Band 聚类测试"""

    def setup_method(self):
        self.merger = ElementMerger(band_height=30, same_row_gap=15)

    def test_band_assignment(self):
        """元素应被正确分配到 band"""
        elems = [
            make_elem(0, 0, 50, 20),   # center_y=10, band=0
            make_elem(60, 0, 110, 20),  # center_y=10, band=0
            make_elem(0, 50, 50, 70),   # center_y=60, band=2
        ]
        bands = self.merger._cluster_into_bands(elems)
        assert len(bands[0]) == 2
        assert len(bands[2]) == 1

    def test_merge_within_band(self):
        """同一 band 内的相邻同类型元素应归并"""
        elems = [
            make_elem(0, 0, 50, 20, control_type="ButtonControl"),
            make_elem(55, 0, 110, 20, control_type="ButtonControl"),  # 同一 band(0)，间距5<15
            make_elem(120, 0, 170, 20, control_type="ButtonControl"),  # 同一 band(0)，间距10<15
        ]
        result = self.merger.merge(elems)
        # 应该归并为一组
        assert len(result) == 1
        assert result[0].is_merged is True
        assert result[0].original_count == 3

    def test_no_merge_different_types(self):
        """不同类型不应归并"""
        elems = [
            make_elem(0, 0, 50, 20, control_type="ButtonControl"),
            make_elem(55, 0, 110, 20, control_type="EditControl"),  # 同 band 但不同类型
        ]
        result = self.merger.merge(elems)
        assert len(result) == 2

    def test_no_merge_far_apart(self):
        """间距过大不应归并"""
        elems = [
            make_elem(0, 0, 50, 20, control_type="ButtonControl"),
            make_elem(100, 0, 150, 20, control_type="ButtonControl"),  # 间距50 > same_row_gap(15)
        ]
        result = self.merger.merge(elems)
        assert len(result) == 2

    def test_merge_adjacent_same_row(self):
        """相邻同类型同行元素应归并"""
        elems = [
            make_elem(0, 0, 50, 20),
            make_elem(52, 0, 100, 20),
            make_elem(102, 0, 150, 20),
        ]
        result = self.merger.merge(elems)
        assert len(result) == 1
        assert result[0].is_merged is True
        assert result[0].original_count == 3


class TestMergedElement:
    """MergedElement 属性测试"""

    def test_single_element(self):
        """单元素 MergedElement"""
        elem = make_elem(10, 20, 110, 50)
        merger = ElementMerger()
        result = merger.merge([elem])
        assert len(result) == 1
        m = result[0]
        assert m.is_merged is False
        assert m.original_count == 1
        assert m.bounding_rect == (10, 20, 110, 50)
        assert m.left == 10
        assert m.top == 20
        assert m.right == 110
        assert m.bottom == 50
        assert m.width == 100
        assert m.height == 30

    def test_merged_element_union_rect(self):
        """归并元素的边界框是所有元素的并集"""
        elems = [
            make_elem(0, 0, 50, 20),
            make_elem(100, 10, 200, 40),
        ]
        merger = ElementMerger(same_row_gap=100)  # 允许100px间距归并
        result = merger.merge(elems)
        assert len(result) == 1
        m = result[0]
        assert m.left == 0
        assert m.top == 0
        assert m.right == 200
        assert m.bottom == 40

    def test_merged_common_name(self):
        """归并后 name 相同时保留"""
        elems = [
            make_elem(0, 0, 50, 20, name="OK", control_type="ButtonControl"),
            make_elem(55, 0, 100, 20, name="OK", control_type="ButtonControl"),
        ]
        merger = ElementMerger(same_row_gap=15)
        result = merger.merge(elems)
        assert result[0].name == "OK"

    def test_merged_different_names(self):
        """归并后 name 不同时为 None"""
        elems = [
            make_elem(0, 0, 50, 20, name="OK", control_type="ButtonControl"),
            make_elem(55, 0, 100, 20, name="Cancel", control_type="ButtonControl"),
        ]
        merger = ElementMerger(same_row_gap=15)
        result = merger.merge(elems)
        assert result[0].name is None


class TestMergeStats:
    """归并统计测试"""

    def test_stats_reduction(self):
        """归并统计能正确计算减少比例"""
        elems = [
            make_elem(0, 0, 50, 20),
            make_elem(52, 0, 100, 20),
            make_elem(102, 0, 150, 20),
        ]
        merger = ElementMerger()
        merged = merger.merge(elems)
        stats = merger.get_merge_stats(elems, merged)

        assert stats["original_count"] == 3
        assert stats["merged_count"] == 1
        assert stats["merged_elements"] == 1
        assert stats["reduction_ratio"] == pytest.approx(2/3, rel=0.01)

    def test_large_scale_merge(self):
        """大量元素的归并测试"""
        # 创建 100 个相邻的同类型元素
        elems = [make_elem(i * 52, 0, i * 52 + 50, 20) for i in range(100)]
        merger = ElementMerger(same_row_gap=10)
        merged = merger.merge(elems)
        stats = merger.get_merge_stats(elems, merged)

        # 应该归并为约 1-2 组（所有元素都在同一 band）
        assert stats["reduction_ratio"] > 0.8  # 至少减少 80%
