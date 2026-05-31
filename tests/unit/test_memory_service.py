"""
Phase 5 记忆系统单元测试
"""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Literal

# Mock ZoneType for testing
class ZoneType:
    TITLE_BAR = "title_bar"
    MENU_BAR = "menu_bar"
    TOOL_BAR = "tool_bar"
    SIDE_BAR = "side_bar"
    CONTENT_AREA = "content_area"
    STATUS_BAR = "status_bar"


class TestPageTemplateManager:
    """PageTemplateManager 测试"""

    def test_capture_generates_fingerprint(self):
        """capture() 应生成稳定的 fingerprint_hash"""
        from src.memory.page_template_manager import PageTemplateManager

        manager = PageTemplateManager()

        # Mock zone structure
        mock_zone = MagicMock()
        mock_zone.zone_type = MagicMock()
        mock_zone.zone_type.value = "content_area"
        mock_zone.elements = []

        mock_structure = MagicMock()
        mock_structure.zones = [mock_zone]

        result = manager.capture("vscode", "editor", mock_structure)

        assert result["app_id"] == "vscode"
        assert result["page_type"] == "editor"
        assert result["fingerprint_hash"] is not None
        assert len(result["fingerprint_hash"]) == 16

    def test_capture_same_structure_same_fingerprint(self):
        """相同结构应生成相同 fingerprint"""
        from src.memory.page_template_manager import PageTemplateManager

        manager = PageTemplateManager()

        mock_zone = MagicMock()
        mock_zone.zone_type = MagicMock()
        mock_zone.zone_type.value = "content_area"
        mock_zone.elements = []

        mock_structure1 = MagicMock()
        mock_structure1.zones = [mock_zone]
        mock_structure2 = MagicMock()
        mock_structure2.zones = [mock_zone]

        r1 = manager.capture("app", "page", mock_structure1)
        r2 = manager.capture("app", "page", mock_structure2)

        assert r1["fingerprint_hash"] == r2["fingerprint_hash"]

    def test_capture_elements_included_in_fingerprint(self):
        """capture() 应将元素信息包含在 fingerprint 中"""
        from src.memory.page_template_manager import PageTemplateManager

        manager = PageTemplateManager()

        # 有元素的 zone
        mock_elem = MagicMock()
        mock_elem.control_type = "Button"
        mock_elem.name = "OK"
        mock_elem.bounding_rect = (10, 10, 100, 50)

        mock_zone = MagicMock()
        mock_zone.zone_type = MagicMock()
        mock_zone.zone_type.value = "content_area"
        mock_zone.elements = [mock_elem]

        mock_structure = MagicMock()
        mock_structure.zones = [mock_zone]

        result = manager.capture("app", "page", mock_structure)

        assert result["fingerprint_hash"] is not None
        assert len(result["elements_data"]) == 1
        assert result["elements_data"][0]["type"] == "Button"
        assert result["elements_data"][0]["name"] == "OK"


class TestDriftDetector:
    """DriftDetector 测试"""

    def test_drift_when_elements_missing(self):
        """元素大量消失应报告漂移"""
        from src.memory.drift_detector import DriftDetector

        detector = DriftDetector()

        # 存储 3 个元素，当前只有 1 个（missing ratio > 30%）
        mock_template = MagicMock()
        mock_template.structure_json = (
            '[{"zone":"content_area","type":"Edit","name":"a"},'
            '{"zone":"content_area","type":"Edit","name":"b"},'
            '{"zone":"content_area","type":"Edit","name":"c"}]'
        )
        mock_template.confidence = 1.0

        mock_zone = MagicMock()
        mock_zone.zone_type = MagicMock()
        mock_zone.zone_type.value = "content_area"
        mock_zone.elements = [MagicMock(
            control_type="Edit",
            name=None,
            bounding_rect=(0, 0, 100, 50)
        )]

        mock_structure = MagicMock()
        mock_structure.zones = [mock_zone]

        report = detector.detect(mock_template, mock_structure)

        assert report.drifted is True
        assert len(report.missing_elements) > 0
        assert report.suggestion == "relearn"

    def test_drift_report_structure(self):
        """DriftReport 应有正确的属性"""
        from src.memory.drift_detector import DriftDetector, DriftReport

        detector = DriftDetector()
        mock_template = MagicMock()
        mock_template.structure_json = '[]'
        mock_template.confidence = 0.5

        mock_zone = MagicMock()
        mock_zone.zone_type = MagicMock()
        mock_zone.zone_type.value = "content_area"
        mock_zone.elements = []

        mock_structure = MagicMock()
        mock_structure.zones = [mock_zone]

        report = detector.detect(mock_template, mock_structure)

        assert hasattr(report, "drifted")
        assert hasattr(report, "confidence_drop")
        assert hasattr(report, "changed_zones")
        assert hasattr(report, "suggestion")


class TestActionRecipeData:
    """ActionRecipeData 测试"""

    def test_action_step_fields(self):
        """ActionStep 应有正确的字段"""
        from src.memory.action_recipe_manager import ActionStep, ActionRecipeData

        step = ActionStep(action="click", target="100,200", value=None)
        recipe = ActionRecipeData(
            app_id="notepad",
            page_type="editor",
            action_name="open_file",
            steps=[step],
        )

        assert recipe.app_id == "notepad"
        assert recipe.page_type == "editor"
        assert recipe.action_name == "open_file"
        assert len(recipe.steps) == 1
        assert recipe.steps[0].action == "click"
        assert recipe.steps[0].target == "100,200"

    def test_action_recipe_with_text_input(self):
        """带文本输入的动作配方"""
        from src.memory.action_recipe_manager import ActionStep, ActionRecipeData

        step = ActionStep(action="send_text", target="content_area", value="hello world")
        recipe = ActionRecipeData(
            app_id="notepad",
            page_type="editor",
            action_name="type_text",
            steps=[step],
        )

        assert recipe.steps[0].value == "hello world"
