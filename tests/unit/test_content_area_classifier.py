"""
ContentAreaClassifier 单元测试

测试 content_area 子类型判定逻辑（CHAT/FORM/LIST_DETAIL/EDITOR/DASHBOARD/CANVAS_DOC_VIEWER）
"""

import pytest

from src.perception.content_area_classifier import (
    ContentAreaClassifier,
    ContentAreaClassification,
    ContentAreaEvidence,
)
from src.perception.page_compiler_models import ContentAreaSubtype


class TestContentAreaClassifier:
    """ContentAreaClassifier 单元测试"""

    def setup_method(self):
        self.classifier = ContentAreaClassifier()

    def _make_elem(
        self,
        control_type: str,
        name: str | None = None,
        value: str | None = None,
        text: str | None = None,
        left: int = 0,
        top: int = 0,
        right: int = 100,
        bottom: int = 50,
    ) -> dict:
        return {
            "control_type": control_type,
            "name": name,
            "value": value,
            "text": text,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "bounding_rect": (left, top, right, bottom),
        }

    def test_empty_elements_returns_unknown(self):
        """空元素列表返回 UNKNOWN"""
        result = self.classifier.classify([])
        assert result.subtype == ContentAreaSubtype.UNKNOWN
        assert len(result.evidence) == 1

    def test_chat_input_zone_classified_as_chat(self):
        """
        底部输入区（Edit + Button）+ 滚动区 → CHAT

        典型聊天面板：上方消息流（List/Scroll）+ 底部输入框 + 发送按钮
        """
        elements = [
            # 消息列表（中部）
            self._make_elem("List", left=0, top=50, right=400, bottom=400),
            self._make_elem("ScrollBar", left=390, top=50, right=400, bottom=400),
            # 底部输入区
            self._make_elem("Edit", left=0, top=420, right=320, bottom=450, name="输入框"),
            self._make_elem("Button", left=330, top=420, right=400, bottom=450, name="发送"),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.CHAT
        evidence_features = {e.feature for e in result.evidence}
        assert "chat_input_zone" in evidence_features

    def test_multi_edit_fields_classified_as_form(self):
        """
        多个 Edit + Button → FORM

        典型表单：label-edit 配对，提交按钮
        """
        elements = [
            self._make_elem("Text", name="用户名", left=10, top=50, right=100, bottom=70),
            self._make_elem("Edit", left=110, top=50, right=300, bottom=70, name="username"),
            self._make_elem("Text", name="密码", left=10, top=80, right=100, bottom=100),
            self._make_elem("Edit", left=110, top=80, right=300, bottom=100, name="password"),
            self._make_elem("Button", left=110, top=130, right=200, bottom=160, name="提交"),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.FORM
        evidence_features = {e.feature for e in result.evidence}
        assert "multi_edit_fields" in evidence_features or "label_edit_pairs" in evidence_features

    def test_vertical_split_classified_as_list_detail(self):
        """
        垂直分割布局（左列表 + 右详情）→ LIST_DETAIL

        典型列表详情：左侧可选择列表项，右侧显示详情
        """
        elements = [
            # 左侧列表区域（Tree + TreeItems，right edge = 190）
            self._make_elem("Tree", left=0, top=0, right=190, bottom=500),
            self._make_elem("TreeItem", left=10, top=10, right=180, bottom=30),
            self._make_elem("TreeItem", left=10, top=35, right=180, bottom=55),
            self._make_elem("TreeItem", left=10, top=60, right=180, bottom=80),
            # 中间有明显分隔空白（gap > 5% of 400 = 20px）
            # 右侧详情区域（left edge = 230，gap = 230 - 190 = 40px > 20px）
            self._make_elem("Pane", left=230, top=0, right=500, bottom=480),
            self._make_elem("Text", left=240, top=10, right=490, bottom=30, name="详情标题"),
            self._make_elem("Button", left=230, top=450, right=500, bottom=480, name="操作"),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.LIST_DETAIL
        evidence_features = {e.feature for e in result.evidence}
        assert "split_layout_vertical" in evidence_features

    def test_tab_with_toolbar_classified_as_editor(self):
        """
        Tab + ToolBar + Document → EDITOR

        典型编辑器：Tab 栏切换文档，ToolBar 工具栏，Document 编辑区
        """
        elements = [
            self._make_elem("Tab", left=0, top=0, right=400, bottom=30),
            self._make_elem("TabItem", left=10, top=5, right=80, bottom=25),
            self._make_elem("TabItem", left=85, top=5, right=150, bottom=25),
            self._make_elem("ToolBar", left=0, top=30, right=400, bottom=60),
            self._make_elem("Document", left=0, top=60, right=400, bottom=500),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.EDITOR
        evidence_features = {e.feature for e in result.evidence}
        assert "tab_with_toolbar" in evidence_features or "document_area" in evidence_features

    def test_multi_widget_diversity_classified_as_dashboard(self):
        """
        多种控件类型 + 多个 Pane → DASHBOARD

        典型仪表盘：多种控件混杂（按钮、图表、表格、输入框等）
        """
        elements = [
            self._make_elem("Pane", left=0, top=0, right=130, bottom=150),
            self._make_elem("Button", left=10, top=10, right=60, bottom=40),
            self._make_elem("Button", left=70, top=10, right=120, bottom=40),
            self._make_elem("Pane", left=140, top=0, right=300, bottom=150),
            self._make_elem("Image", left=150, top=10, right=290, bottom=140),
            self._make_elem("Pane", left=0, top=160, right=300, bottom=300),
            self._make_elem("DataGrid", left=10, top=170, right=290, bottom=290),
            self._make_elem("Edit", left=0, top=310, right=150, bottom=340),
            self._make_elem("Button", left=160, top=310, right=230, bottom=340),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.DASHBOARD
        evidence_features = {e.feature for e in result.evidence}
        assert "multi_widget_diversity" in evidence_features

    def test_pane_dominant_no_input_classified_as_canvas_doc_viewer(self):
        """
        Pane 主导 + 无底部输入区 → CANVAS_DOC_VIEWER

        典型文档查看器：大视口 + 翻页控件
        """
        elements = [
            self._make_elem("Pane", left=0, top=0, right=400, bottom=450),
            self._make_elem("Button", left=0, top=460, right=50, bottom=490, name="上一页"),
            self._make_elem("Button", left=60, top=460, right=110, bottom=490, name="下一页"),
            self._make_elem("Text", left=175, top=465, right=225, bottom=485, name="1/10"),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.CANVAS_DOC_VIEWER
        evidence_features = {e.feature for e in result.evidence}
        assert "pane_dominant_no_input" in evidence_features

    def test_confidence_is_bounded(self):
        """置信度在 [0, 1] 范围内"""
        elements = [
            self._make_elem("Edit", left=0, top=0, right=100, bottom=30),
            self._make_elem("Button", left=110, top=0, right=150, bottom=30),
        ]
        result = self.classifier.classify(elements)
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_always_has_description(self):
        """所有证据都有 description"""
        elements = [self._make_elem("Pane")]
        result = self.classifier.classify(elements)
        for e in result.evidence:
            assert e.description

    def test_form_with_label_edit_pairs(self):
        """表单 label-edit 配对检测"""
        elements = [
            self._make_elem("Text", name="Name", left=10, top=40, right=100, bottom=60),
            self._make_elem("Edit", left=110, top=40, right=300, bottom=60),
            self._make_elem("Text", name="Email", left=10, top=70, right=100, bottom=90),
            self._make_elem("Edit", left=110, top=70, right=300, bottom=90),
            self._make_elem("Text", name="Age", left=10, top=100, right=100, bottom=120),
            self._make_elem("Edit", left=110, top=100, right=300, bottom=120),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.FORM
        evidence_features = {e.feature for e in result.evidence}
        assert "label_edit_pairs" in evidence_features

    def test_list_detail_with_tree_and_edit(self):
        """LIST_DETAIL 检测：树控件 + Edit"""
        elements = [
            self._make_elem("Tree", left=0, top=0, right=180, bottom=500),
            self._make_elem("TreeItem", left=10, top=10, right=170, bottom=35),
            self._make_elem("TreeItem", left=10, top=40, right=170, bottom=65),
            self._make_elem("Edit", left=190, top=10, right=500, bottom=40),
            self._make_elem("Edit", left=190, top=50, right=500, bottom=80),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.LIST_DETAIL

    def test_chat_with_scroll_bottom_input(self):
        """CHAT 强化：底部输入 + Scroll"""
        elements = [
            self._make_elem("ScrollBar", left=390, top=0, right=400, bottom=420),
            self._make_elem("List", left=0, top=0, right=390, bottom=420),
            self._make_elem("Edit", left=0, top=430, right=340, bottom=460),
            self._make_elem("Button", left=350, top=430, right=400, bottom=460),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.CHAT

    def test_editor_document_enhances_editor_score(self):
        """Document 控件强化 EDITOR 分数"""
        elements = [
            self._make_elem("Document", left=0, top=60, right=500, bottom=500),
            self._make_elem("ToolBar", left=0, top=30, right=500, bottom=55),
            self._make_elem("Button", left=10, top=35, right=40, bottom=50),
        ]
        result = self.classifier.classify(elements)
        assert result.subtype == ContentAreaSubtype.EDITOR
