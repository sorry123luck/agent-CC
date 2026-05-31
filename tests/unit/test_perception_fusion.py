"""
PerceptionFusion 单元测试

测试融合逻辑和坐标变换
"""
import pytest
from unittest.mock import MagicMock, patch

from src.perception.perception_fusion import (
    PerceptionFusion,
    FusedElement,
    PageStructure,
)
from src.perception.uia_client import UIAElementInfo
from src.perception.ocr_service import OCRTextBlock
from src.windows.window_enum import WindowInfoExt, WindowState


class TestPerceptionFusionIOU:
    """IOU 计算测试"""

    def setup_method(self):
        self.fusion = PerceptionFusion()

    def test_iou_no_overlap(self):
        """两矩形无重叠时 IOU = 0"""
        rect1 = (0, 0, 10, 10)
        rect2 = (20, 20, 30, 30)
        iou = self.fusion._compute_iou(rect1, rect2)
        assert iou == 0.0

    def test_iou_complete_overlap(self):
        """两矩形完全重叠时 IOU = 1"""
        rect = (0, 0, 10, 10)
        iou = self.fusion._compute_iou(rect, rect)
        assert iou == 1.0

    def test_iou_partial_overlap(self):
        """两矩形部分重叠"""
        rect1 = (0, 0, 10, 10)  # 面积 100
        rect2 = (5, 5, 15, 15)   # 重叠面积 25, 并集面积 175
        iou = self.fusion._compute_iou(rect1, rect2)
        assert iou == pytest.approx(25 / 175, rel=0.01)

    def test_iou_edge_touching(self):
        """两矩形边相接触但不重叠"""
        rect1 = (0, 0, 10, 10)
        rect2 = (10, 0, 20, 10)
        iou = self.fusion._compute_iou(rect1, rect2)
        assert iou == 0.0


class TestPerceptionFusionUnion:
    """矩形并集计算测试"""

    def setup_method(self):
        self.fusion = PerceptionFusion()

    def test_union_no_overlap(self):
        """两矩形无重叠时并集"""
        rect1 = (0, 0, 10, 10)
        rect2 = (20, 20, 30, 30)
        union = self.fusion._union_rect(rect1, rect2)
        assert union == (0, 0, 30, 30)

    def test_union_partial_overlap(self):
        """两矩形部分重叠时并集"""
        rect1 = (0, 0, 10, 10)
        rect2 = (5, 5, 15, 15)
        union = self.fusion._union_rect(rect1, rect2)
        assert union == (0, 0, 15, 15)

    def test_union_same_rect(self):
        """两矩形相同时并集等于自身"""
        rect = (0, 0, 10, 10)
        union = self.fusion._union_rect(rect, rect)
        assert union == (0, 0, 10, 10)


class TestFusedElement:
    """融合元素数据模型测试"""

    def test_fused_element_with_both(self):
        """有 UIA 和 OCR 两种来源的元素"""
        elem = FusedElement(
            uia_element=UIAElementInfo(
                name="Test",
                automation_id="btn1",
                control_type="Button",
                bounding_rect=(10, 10, 50, 30),
                is_enabled=True,
            ),
            ocr_text="Test Button",
            ocr_bbox=(10, 10, 50, 30),
            ocr_confidence=0.95,
            fused_bbox=(10, 10, 50, 30),
            sources=["uia", "ocr"],
        )
        assert elem.ocr_text == "Test Button"
        assert elem.sources == ["uia", "ocr"]

    def test_fused_element_uia_only(self):
        """只有 UIA 来源的元素"""
        elem = FusedElement(
            uia_element=UIAElementInfo(
                name="Invisible",
                automation_id="inv1",
                control_type="Pane",
                bounding_rect=(0, 0, 0, 0),
                is_enabled=False,
            ),
            ocr_text=None,
            ocr_bbox=None,
            ocr_confidence=None,
            fused_bbox=(0, 0, 0, 0),
            sources=["uia"],
        )
        assert elem.ocr_text is None
        assert elem.sources == ["uia"]

    def test_fused_element_ocr_only(self):
        """只有 OCR 来源的元素（UIA 无法识别的文本）"""
        elem = FusedElement(
            uia_element=None,
            ocr_text="Detected Text",
            ocr_bbox=(100, 100, 200, 120),
            ocr_confidence=0.88,
            fused_bbox=(100, 100, 200, 120),
            sources=["ocr"],
        )
        assert elem.uia_element is None
        assert elem.ocr_text == "Detected Text"


class TestPageStructure:
    """页面结构模型测试"""

    def test_page_structure_empty(self):
        """空页面结构"""
        page = PageStructure(
            window_info=None,
            uia_elements=[],
            ocr_blocks=[],
            fused_elements=[],
            screenshot_size=None,
        )
        assert page.window_info is None
        assert len(page.fused_elements) == 0

    def test_page_structure_with_data(self):
        """有数据的页面结构"""
        window_info = WindowInfoExt(
            hwnd=12345,
            title="Test Window",
            process_name="test.exe",
            state=WindowState.NORMAL,
        )
        uia_elem = UIAElementInfo(
            name="Button",
            automation_id="btn",
            control_type="Button",
            bounding_rect=(10, 10, 50, 30),
            is_enabled=True,
        )
        ocr_block = OCRTextBlock(
            text="Click Me",
            bbox=(10, 10, 50, 30),
            confidence=0.95,
        )
        fused = FusedElement(
            uia_element=uia_elem,
            ocr_text="Click Me",
            ocr_bbox=(10, 10, 50, 30),
            ocr_confidence=0.95,
            fused_bbox=(10, 10, 50, 30),
            sources=["uia", "ocr"],
        )
        page = PageStructure(
            window_info=window_info,
            uia_elements=[uia_elem],
            ocr_blocks=[ocr_block],
            fused_elements=[fused],
            screenshot_size=(1280, 720),
        )
        assert page.window_info.title == "Test Window"
        assert len(page.uia_elements) == 1
        assert len(page.ocr_blocks) == 1
        assert len(page.fused_elements) == 1
        assert page.screenshot_size == (1280, 720)
