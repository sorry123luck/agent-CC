"""
PageClassClassifier 单元测试
"""

import pytest

from src.perception.page_class_classifier import (
    PageClassClassifier,
    PageClassResult,
    PageClassEvidence,
)
from src.perception.page_compiler_models import (
    InteractionCanvas,
    Candidate,
    Locator,
    LocatorKind,
    LocatorStatus,
    SurfaceType,
    SurfaceInfo,
    AppInfo,
    WindowInfoSnapshot,
    SemanticRole,
)


class TestPageClassClassifier:
    """PageClassClassifier 单元测试"""

    def setup_method(self):
        self.classifier = PageClassClassifier()

    def _make_snapshot(
        self,
        app_id: str | None = None,
        process_name: str | None = None,
        surface_type: SurfaceType = SurfaceType.NATIVE_UIA,
        elements: list[Candidate] | None = None,
        state_flags: dict[str, bool] | None = None,
    ) -> InteractionCanvas:
        from src.perception.page_compiler_models import PageInfo

        app = AppInfo(app_id=app_id, process_name=process_name)
        surface = SurfaceInfo(surface_type=surface_type, confidence=0.9)
        page_info = PageInfo(state_flags=state_flags or {})
        return InteractionCanvas(
            canvas_id="test",
            app=app,
            window=WindowInfoSnapshot(hwnd=12345, title="Test"),
            surface=surface,
            page=page_info,
            elements=elements or [],
        )

    def test_wechat_app_id(self):
        """微信进程映射到 wechat app"""
        snapshot = self._make_snapshot(process_name="WeChat.exe")
        result = self.classifier.classify(snapshot)
        assert result.app == "wechat"

    def test_vscode_app_id(self):
        """VS Code 进程映射到 vscode app"""
        snapshot = self._make_snapshot(process_name="Code.exe")
        result = self.classifier.classify(snapshot)
        assert result.app == "vscode"

    def test_unknown_process(self):
        """未知进程默认 unknown app"""
        snapshot = self._make_snapshot(process_name="unknown.exe")
        result = self.classifier.classify(snapshot)
        assert result.app == "unknown"

    def test_page_class_format(self):
        """page_class 格式正确"""
        snapshot = self._make_snapshot(process_name="WeChat.exe")
        result = self.classifier.classify(snapshot)
        parts = result.page_class.split("/")
        assert len(parts) == 4

    def test_workflow_from_surface_type(self):
        """workflow 从 surface_type 推断"""
        snapshot = self._make_snapshot(surface_type=SurfaceType.BROWSER)
        result = self.classifier.classify(snapshot)
        assert result.workflow == "web"

    def test_state_from_flags_editable(self):
        """state 从 state_flags 推断：editable → main"""
        snapshot = self._make_snapshot(state_flags={"editable": True})
        result = self.classifier.classify(snapshot)
        assert result.state == "main"

    def test_state_from_flags_dialog(self):
        """state 从 state_flags 推断：dialog_open → dialog"""
        snapshot = self._make_snapshot(state_flags={"dialog_open": True})
        result = self.classifier.classify(snapshot)
        assert result.state == "dialog"

    def test_state_empty_when_no_elements(self):
        """无元素时 state 为 empty"""
        snapshot = self._make_snapshot(elements=[])
        result = self.classifier.classify(snapshot)
        assert result.state == "empty"

    def test_confidence_bounded(self):
        """置信度在 [0, 1] 范围内"""
        snapshot = self._make_snapshot(process_name="WeChat.exe")
        result = self.classifier.classify(snapshot)
        assert 0.0 <= result.class_confidence <= 1.0

    def test_fingerprint_generated(self):
        """指纹生成"""
        snapshot = self._make_snapshot(process_name="WeChat.exe")
        result = self.classifier.classify(snapshot)
        assert "layout_hash" in result.fingerprint
        assert "element_count_hash" in result.fingerprint
        assert "surface_hash" in result.fingerprint

    def test_workflow_map_chat(self):
        """chat workflow 映射"""
        from src.perception.page_compiler_models import ContentAreaSubtype, Region

        snapshot = self._make_snapshot(surface_type=SurfaceType.NATIVE_UIA)
        snapshot.regions = [
            Region(
                region_id="r1",
                role="content_area",
                subtype=ContentAreaSubtype.CHAT,
            )
        ]
        result = self.classifier.classify(snapshot)
        assert result.workflow == "chat"

    def test_variant_default_for_normal_content(self):
        """正常内容区 variant 为 default"""
        from src.perception.page_compiler_models import ContentAreaSubtype, Region

        snapshot = self._make_snapshot(surface_type=SurfaceType.NATIVE_UIA)
        snapshot.regions = [
            Region(
                region_id="r1",
                role="content_area",
                subtype=ContentAreaSubtype.FORM,
                bounds=(0, 100, 600, 800),
            )
        ]
        result = self.classifier.classify(snapshot)
        assert result.variant == "default"
