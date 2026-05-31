"""
SurfaceClassifier 单元测试

测试四层证据打分的 surface_type 判定逻辑
"""

import pytest

from src.perception.surface_classifier import (
    SurfaceClassifier,
    SurfaceEvidence,
    KNOWN_BROWSER_PROCESSES,
    KNOWN_ELECTRON_PROCESS_PATTERNS,
)
from src.perception.page_compiler_models import SurfaceType


class TestSurfaceClassifier:
    """SurfaceClassifier 单元测试"""

    def setup_method(self):
        self.classifier = SurfaceClassifier()

    def test_unknown_process_defaults_to_native_uia(self):
        """未知进程默认原生 GUI"""
        result = self.classifier.classify(
            process_name="unknown_app.exe",
            uia_element_count=500,
            uia_control_types=["Button", "Edit", "Tree"],
        )
        assert result.surface_type == SurfaceType.NATIVE_UIA
        assert result.confidence > 0.0

    def test_chrome_process_classified_as_browser(self):
        """Chrome 进程分类为 browser"""
        result = self.classifier.classify(
            process_name="chrome.exe",
            uia_element_count=50,
            uia_control_types=["WebView", "Button"],
            has_dom_bridge=True,
            dom_ready=True,
        )
        assert result.surface_type == SurfaceType.BROWSER
        evidence_layers = {e.layer for e in result.evidence}
        assert "process" in evidence_layers
        assert "dom" in evidence_layers

    def test_vscode_process_classified_as_electron(self):
        """VS Code 进程分类为 electron_webview"""
        result = self.classifier.classify(
            process_name="Code.exe",
            uia_element_count=300,
            uia_control_types=["ToolBar", "MenuBar", "Tree"],
        )
        assert result.surface_type == SurfaceType.ELECTRON_WEBVIEW
        evidence_features = [e.feature for e in result.evidence]
        assert any("electron" in f for f in evidence_features)

    def test_wechat_process_classified_as_electron_webview(self):
        """微信进程分类为 electron_webview（WebView2 内核）"""
        result = self.classifier.classify(
            process_name="WeChat.exe",
            uia_element_count=200,
            uia_control_types=["ToolBar", "Edit", "Pane"],
        )
        assert result.surface_type == SurfaceType.ELECTRON_WEBVIEW

    def test_paint_process_classified_as_canvas(self):
        """画图进程分类为 canvas_self_drawn"""
        result = self.classifier.classify(
            process_name="paint.exe",
            uia_element_count=20,
            uia_control_types=["Pane"],
        )
        assert result.surface_type == SurfaceType.CANVAS_SELF_DRAWN

    def test_unknown_self_drawn_ultra_sparse_no_standard_controls(self):
        """
        未知进程 + 极稀疏 UIA 树 + 无标准控件 → canvas_self_drawn

        模拟场景：未知自绘应用（如某个游戏启动器、特殊工具），
        UIA 树元素极少（< 30），无标准控件，无 WebView 特征。
        这种情况应该被判定为 canvas_self_drawn，而不是默认 native_uia。
        """
        result = self.classifier.classify(
            process_name="unknown_game_launcher.exe",
            uia_element_count=15,
            uia_control_types=["Pane", "Custom"],  # 无标准控件
        )
        assert result.surface_type == SurfaceType.CANVAS_SELF_DRAWN
        evidence_features = {e.feature for e in result.evidence}
        assert "ultra_sparse_no_standard_controls" in evidence_features

    def test_rich_uia_tree_favors_native_uia(self):
        """丰富 UIA 树倾向原生 GUI"""
        result = self.classifier.classify(
            process_name="explorer.exe",
            uia_element_count=500,
            uia_control_types=[
                "MenuBar", "ToolBar", "Tree", "Button", "Edit", "StatusBar"
            ],
        )
        assert result.surface_type == SurfaceType.NATIVE_UIA

    def test_webview_evidence_detected(self):
        """WebView 特征被正确检测"""
        result = self.classifier.classify(
            process_name="someapp.exe",
            uia_element_count=40,
            uia_control_types=["WebView2", "Button", "ToolBar"],
        )
        # WebView 特征应该出现在证据中
        features = {e.feature for e in result.evidence}
        assert "browser_webview" in features or "electron_shell" in features

    def test_locator_priority_for_browser(self):
        """browser surface_type 的 locator 优先级正确"""
        result = self.classifier.classify(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
        )
        assert result.locator_priority[0] == "dom"

    def test_locator_priority_for_native_uia(self):
        """native_uia surface_type 的 locator 优先级正确"""
        result = self.classifier.classify(
            process_name="notepad.exe",
            uia_element_count=50,
            uia_control_types=["Edit", "MenuBar"],
        )
        assert result.locator_priority[0] == "uia"

    def test_no_uia_data_still_returns_classification(self):
        """无 UIA 数据时仍能返回分类结果"""
        result = self.classifier.classify(
            process_name="unknown.exe",
        )
        assert result.surface_type is not None
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.locator_priority) > 0

    def test_evidence_contains_layer_info(self):
        """证据包含层信息"""
        result = self.classifier.classify(
            process_name="chrome.exe",
            uia_element_count=30,
            uia_control_types=["WebView"],
        )
        layers = {e.layer for e in result.evidence}
        assert "process" in layers
        assert "uia_pattern" in layers
        assert "dom" in layers

    def test_visual_score_biases_toward_native_gui(self):
        """高视觉分数倾向原生 GUI"""
        result_high = self.classifier.classify(
            process_name="someapp.exe",
            uia_element_count=500,
            uia_control_types=["Button", "Edit", "Tree", "MenuBar"],
            visual_score=0.95,
        )
        result_low = self.classifier.classify(
            process_name="someapp.exe",
            uia_element_count=500,
            uia_control_types=["Button", "Edit", "Tree", "MenuBar"],
            visual_score=0.10,
        )
        # 高视觉分 → native_uia 置信度应更高
        high_native = [e for e in result_high.evidence if e.layer == "visual"][0]
        low_native = [e for e in result_low.evidence if e.layer == "visual"][0]
        assert high_native.value > low_native.value

    def test_empty_uia_control_types_handled(self):
        """空 UIA 控件类型列表被正确处理"""
        result = self.classifier.classify(
            process_name="notepad.exe",
            uia_element_count=0,
            uia_control_types=[],
        )
        assert result.surface_type is not None

    def test_dom_ready_promotes_browser(self):
        """DOM 就绪时提升 browser 置信度"""
        with_dom = self.classifier.classify(
            process_name="someapp.exe",
            uia_element_count=50,
            uia_control_types=["WebView"],
            has_dom_bridge=True,
            dom_ready=True,
        )
        without_dom = self.classifier.classify(
            process_name="someapp.exe",
            uia_element_count=50,
            uia_control_types=["WebView"],
            has_dom_bridge=False,
            dom_ready=False,
        )
        # 有 DOM 时 browser 置信度应更高
        assert with_dom.surface_type == SurfaceType.BROWSER
        assert with_dom.confidence > without_dom.confidence
