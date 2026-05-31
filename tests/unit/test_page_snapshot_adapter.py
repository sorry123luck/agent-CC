"""
InteractionCanvasAdapter 单元测试

测试 InteractionCanvasAdapter 对 InteractionCanvas 的适配能力。
覆盖：UIA 路径、坐标路径、缺失 locator、缺失 hwnd、不支持方法路径。
"""

import pytest
from unittest.mock import MagicMock, patch

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
    CoordinateSpace,
)
from src.execution.action_executor import ActionResult, MouseButton
from src.execution.canvas_action_adapter import (
    InteractionCanvasAdapter,
    get_surface_execution_strategy,
)
from src.perception.page_compiler import InteractionCanvasEngine


class TestInteractionCanvasAdapter:
    """InteractionCanvasAdapter 单元测试"""

    def setup_method(self):
        self.adapter = InteractionCanvasAdapter()

    def _make_locator(
        self,
        locator_id: str,
        kind: LocatorKind,
        automation_id: str | None = None,
        bbox: tuple | None = None,
    ) -> Locator:
        selector = {}
        if automation_id:
            selector["automation_id"] = automation_id
        if bbox:
            selector["bbox"] = bbox
        return Locator(
            locator_id=locator_id,
            element_ref=f"elem_{locator_id}",
            kind=kind,
            priority=1,
            status=LocatorStatus.ACTIVE,
            selector=selector,
        )

    def _make_element(
        self,
        element_id: str,
        locator_ids: list[str],
        bounds: tuple | None = (100, 100, 200, 150),
        automation_id: str | None = None,
    ) -> Candidate:
        attrs = {}
        if automation_id:
            attrs["automation_id"] = automation_id
        return Candidate(
            element_id=element_id,
            semantic_role=SemanticRole.BUTTON,
            control_type="Button",
            bounds=bounds,
            locator_ids=locator_ids,
            attributes=attrs,
        )

    def _make_snapshot(
        self,
        elements: list[Candidate],
        locators: list[Locator],
        hwnd: int = 12345,
        surface_type: SurfaceType = SurfaceType.NATIVE_UIA,
    ) -> InteractionCanvas:
        window = WindowInfoSnapshot(hwnd=hwnd, title="Test Window")
        surface = SurfaceInfo(surface_type=surface_type, confidence=0.9)
        app = AppInfo(process_name="test.exe")
        return InteractionCanvas(
            canvas_id="test_snapshot",
            window=window,
            surface=surface,
            app=app,
            elements=elements,
            locators=locators,
        )

    def test_resolve_element_uia_path(self):
        """UIA locator 解析"""
        locator = self._make_locator("loc1", LocatorKind.UIA, automation_id="btn_ok")
        element = self._make_element("elem1", ["loc1"], automation_id="btn_ok")
        snapshot = self._make_snapshot([element], [locator])

        result = self.adapter.resolve_element(snapshot, "elem1")

        assert result is not None
        assert result["method"] == "uia"
        assert result["locator"].kind == LocatorKind.UIA

    def test_resolve_element_coordinate_path(self):
        """坐标类 locator 解析"""
        locator = self._make_locator("loc1", LocatorKind.VISION_BBOX)
        element = self._make_element("elem1", ["loc1"], bounds=(100, 100, 200, 150))
        snapshot = self._make_snapshot([element], [locator])

        result = self.adapter.resolve_element(snapshot, "elem1")

        assert result is not None
        assert result["method"] == "geometry"
        assert result["coordinates"] == (150, 125)  # center of (100,100,200,150)
        assert result["geometry"] == (100, 100, 100, 50)

    def test_resolve_element_not_found(self):
        """元素不存在"""
        locator = self._make_locator("loc1", LocatorKind.UIA)
        element = self._make_element("elem1", ["loc1"])
        snapshot = self._make_snapshot([element], [locator])

        result = self.adapter.resolve_element(snapshot, "nonexistent")

        assert result is None

    def test_resolve_element_no_matching_locator(self):
        """元素无匹配 locator"""
        element = self._make_element("elem1", [])  # 无 locator
        snapshot = self._make_snapshot([element], [])

        result = self.adapter.resolve_element(snapshot, "elem1")

        assert result is None

    def test_execute_element_no_hwnd(self):
        """无 hwnd 时返回错误"""
        locator = self._make_locator("loc1", LocatorKind.UIA, automation_id="btn")
        element = self._make_element("elem1", ["loc1"])
        snapshot = self._make_snapshot([element], [locator], hwnd=0)  # hwnd=0 无效

        result = self.adapter.execute_element_action(snapshot, "elem1", "click")

        assert result.success is False
        assert result.error == "no_hwnd"

    def test_execute_element_not_found(self):
        """元素不存在时返回错误"""
        locator = self._make_locator("loc1", LocatorKind.UIA)
        element = self._make_element("elem1", ["loc1"])
        snapshot = self._make_snapshot([element], [locator])

        result = self.adapter.execute_element_action(snapshot, "nonexistent", "click")

        assert result.success is False
        assert result.error == "element_not_found"

    def test_execute_uia_no_automation_id(self):
        """UIA locator 无 automation_id 时返回错误"""
        # locator 没有 automation_id
        locator = self._make_locator("loc1", LocatorKind.UIA)
        element = self._make_element("elem1", ["loc1"])
        snapshot = self._make_snapshot([element], [locator])

        result = self.adapter.execute_element_action(snapshot, "elem1", "click")

        assert result.success is False
        assert result.error == "no_automation_id"

    def test_execute_uia_success(self):
        """UIA 路径执行成功"""
        locator = self._make_locator("loc1", LocatorKind.UIA, automation_id="btn_ok")
        element = self._make_element("elem1", ["loc1"])
        snapshot = self._make_snapshot([element], [locator])

        with patch.object(
            self.adapter._executor,
            "invoke_uia_element",
            return_value=ActionResult(success=True, action="invoke_uia", details="ok"),
        ) as mock_invoke:
            result = self.adapter.execute_element_action(snapshot, "elem1", "click")

            assert result.success is True
            mock_invoke.assert_called_once_with(hwnd=12345, automation_id="btn_ok")

    def test_execute_geometry_no_geometry(self):
        """geometry 类 locator 无几何信息时返回错误"""
        # bounds=None 时无法计算坐标
        locator = self._make_locator("loc1", LocatorKind.VISION_BBOX)
        element = self._make_element("elem1", ["loc1"], bounds=None)
        snapshot = self._make_snapshot([element], [locator])

        result = self.adapter.execute_element_action(snapshot, "elem1", "click")

        assert result.success is False
        assert result.error == "no_geometry"

    def test_execute_geometry_click(self):
        """geometry 路径 click 执行"""
        locator = self._make_locator("loc1", LocatorKind.VISION_BBOX)
        element = self._make_element("elem1", ["loc1"], bounds=(100, 100, 200, 150))
        snapshot = self._make_snapshot([element], [locator])

        with patch.object(
            self.adapter._executor,
            "click_element",
            return_value=ActionResult(success=True, action="click", details="ok"),
        ) as mock_click_element:
            result = self.adapter.execute_element_action(snapshot, "elem1", "click")

            assert result.success is True
            mock_click_element.assert_called_once()
            call_args = mock_click_element.call_args
            assert call_args[0][0] == 12345  # hwnd
            assert call_args[0][1] == 100  # element left
            assert call_args[0][2] == 100  # element top
            assert call_args[0][3] == 100  # width
            assert call_args[0][4] == 50   # height

    def test_execute_geometry_double_click(self):
        """geometry 路径 double_click 执行"""
        locator = self._make_locator("loc1", LocatorKind.VISION_BBOX)
        element = self._make_element("elem1", ["loc1"], bounds=(100, 100, 200, 150))
        snapshot = self._make_snapshot([element], [locator])

        with patch.object(
            self.adapter._executor,
            "click_element",
            return_value=ActionResult(success=True, action="click", details="ok"),
        ) as mock_click:
            result = self.adapter.execute_element_action(snapshot, "elem1", "double_click")

            assert result.success is True
            call_args = mock_click.call_args
            assert call_args[1]["double"] is True

    def test_execute_geometry_right_click(self):
        """geometry 路径 right_click 执行"""
        locator = self._make_locator("loc1", LocatorKind.VISION_BBOX)
        element = self._make_element("elem1", ["loc1"], bounds=(100, 100, 200, 150))
        snapshot = self._make_snapshot([element], [locator])

        with patch.object(
            self.adapter._executor,
            "click_element",
            return_value=ActionResult(success=True, action="click", details="ok"),
        ) as mock_click:
            result = self.adapter.execute_element_action(snapshot, "elem1", "right_click")

            assert result.success is True
            call_args = mock_click.call_args
            assert call_args[1]["button"] == MouseButton.RIGHT  # button

    def test_execute_ephemeral_window_coordinate_uses_client_to_screen(self):
        """window-local ephemeral coord 执行前应转换为 screen 坐标"""
        locator = Locator(
            locator_id="loc_coord",
            element_ref="elem1",
            kind=LocatorKind.EPHEMERAL_COORD,
            priority=1,
            status=LocatorStatus.ACTIVE,
            coordinate_space=CoordinateSpace.WINDOW,
            selector={"x": 150, "y": 125},
        )
        element = self._make_element("elem1", ["loc_coord"], bounds=(100, 100, 200, 150))
        snapshot = self._make_snapshot([element], [locator])

        with patch.object(
            self.adapter._executor,
            "_client_to_screen",
            return_value=(450, 325),
        ) as mock_to_screen, patch.object(
            self.adapter._executor,
            "click",
            return_value=ActionResult(success=True, action="click", details="ok"),
        ) as mock_click:
            result = self.adapter.execute_element_action(snapshot, "elem1", "click")

            assert result.success is True
            mock_to_screen.assert_called_once_with(12345, 150, 125)
            mock_click.assert_called_once_with(12345, 450, 325)

    def test_execute_dom_action_uses_connected_provider(self):
        """DOM provider 已连接时应优先使用 DOM native action"""
        dom_provider = MagicMock()
        dom_provider.is_dom_ready.return_value = True
        dom_provider.perform_action.return_value = {
            "success": True,
            "details": "dom_click_ok",
            "rect": (100, 120, 80, 40),
        }
        adapter = InteractionCanvasAdapter(dom_provider=dom_provider)

        locator = Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=1,
            status=LocatorStatus.ACTIVE,
            selector={"name": "signInBtn"},
        )
        element = self._make_element("elem1", ["loc_dom"], bounds=(100, 120, 180, 160))
        snapshot = self._make_snapshot([element], [locator], surface_type=SurfaceType.BROWSER)

        with patch.object(
            adapter._executor,
            "click_element",
            return_value=ActionResult(success=True, action="click", details="ok"),
        ) as mock_click:
            result = adapter.execute_element_action(snapshot, "elem1", "click")

        assert result.success is True
        assert result.details == "dom_click_ok"
        dom_provider.perform_action.assert_called_once_with({"name": "signInBtn"}, "click")
        mock_click.assert_not_called()

    def test_execute_dom_action_falls_back_to_rect_click_when_native_action_not_successful(self):
        """DOM native action 不可用时应回退到基于 rect 的点击"""
        dom_provider = MagicMock()
        dom_provider.is_dom_ready.return_value = True
        dom_provider.perform_action.return_value = {
            "success": False,
            "details": "dom_click_failed",
            "rect": None,
        }
        dom_provider.get_element_by_locator.return_value = {"rect": (100, 120, 80, 40)}
        adapter = InteractionCanvasAdapter(dom_provider=dom_provider)

        locator = Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=1,
            status=LocatorStatus.ACTIVE,
            selector={"name": "signInBtn"},
        )
        element = self._make_element("elem1", ["loc_dom"], bounds=(100, 120, 180, 160))
        snapshot = self._make_snapshot([element], [locator], surface_type=SurfaceType.BROWSER)

        with patch.object(
            adapter._executor,
            "click_element",
            return_value=ActionResult(success=True, action="click", details="ok"),
        ) as mock_click:
            result = adapter.execute_element_action(snapshot, "elem1", "click")

        assert result.success is True
        dom_provider.perform_action.assert_called_once_with({"name": "signInBtn"}, "click")
        dom_provider.get_element_by_locator.assert_called_once_with({"name": "signInBtn"})
        mock_click.assert_called_once_with(12345, 140, 140, 80, 40)

    def test_execute_dom_action_reports_not_ready_provider(self):
        """DOM provider 未 ready 时应明确返回 dom_not_ready"""
        dom_provider = MagicMock()
        dom_provider.is_dom_ready.return_value = False
        adapter = InteractionCanvasAdapter(dom_provider=dom_provider)

        locator = Locator(
            locator_id="loc_dom",
            element_ref="elem1",
            kind=LocatorKind.DOM,
            priority=1,
            status=LocatorStatus.ACTIVE,
            selector={"name": "signInBtn"},
        )
        element = self._make_element("elem1", ["loc_dom"], bounds=(100, 120, 180, 160))
        snapshot = self._make_snapshot([element], [locator], surface_type=SurfaceType.BROWSER)

        result = adapter.execute_element_action(snapshot, "elem1", "click")

        assert result.success is False
        assert result.error == "dom_not_ready"


class TestLocatorPriorityBySurfaceType:
    """验证 InteractionCanvasAdapter 对不同 surface_type 正确选择 locator"""

    def setup_method(self):
        self.compiler = InteractionCanvasEngine()
        self.adapter = InteractionCanvasAdapter()

    def _make_snapshot(
        self,
        elements: list[Candidate],
        locators: list[Locator],
        hwnd: int = 12345,
        surface_type: SurfaceType = SurfaceType.NATIVE_UIA,
    ) -> InteractionCanvas:
        window = WindowInfoSnapshot(hwnd=hwnd, title="Test Window")
        surface = SurfaceInfo(surface_type=surface_type, confidence=0.9)
        app = AppInfo(process_name="test.exe")
        return InteractionCanvas(
            canvas_id="test_snapshot",
            window=window,
            surface=surface,
            app=app,
            elements=elements,
            locators=locators,
        )

    def _make_locator(
        self,
        locator_id: str,
        kind: LocatorKind,
        priority: int,
        automation_id: str | None = None,
        name: str | None = None,
    ) -> Locator:
        selector = {}
        if automation_id:
            selector["automation_id"] = automation_id
        if name:
            selector["name"] = name
        return Locator(
            locator_id=locator_id,
            element_ref="elem_test",
            kind=kind,
            priority=priority,
            status=LocatorStatus.ACTIVE,
            selector=selector,
        )

    def _make_element(
        self,
        element_id: str,
        locator_ids: list[str],
        automation_id: str | None = None,
        name: str | None = None,
    ) -> Candidate:
        attrs = {}
        if automation_id:
            attrs["automation_id"] = automation_id
        return Candidate(
            element_id=element_id,
            semantic_role=SemanticRole.BUTTON,
            control_type="Button",
            bounds=(100, 100, 200, 150),
            name=name,
            locator_ids=locator_ids,
            attributes=attrs,
        )

    def test_native_uia_selects_uia_locator(self):
        """NATIVE_UIA: UIA locator (automation_id) 优先于坐标"""
        uia_loc = self._make_locator("loc_uia", LocatorKind.UIA, 0, automation_id="btn_ok")
        vision_loc = self._make_locator("loc_vis", LocatorKind.VISION_BBOX, 3)
        elem = self._make_element("e1", ["loc_uia", "loc_vis"], automation_id="btn_ok")
        snap = self._make_snapshot(
            [elem], [uia_loc, vision_loc],
            hwnd=12345, surface_type=SurfaceType.NATIVE_UIA
        )
        resolved = self.adapter.resolve_element(snap, "e1")
        assert resolved is not None
        assert resolved["method"] == "uia"
        assert resolved["locator"].kind == LocatorKind.UIA

    def test_browser_selects_dom_locator(self):
        """BROWSER: DOM locator 优先于坐标"""
        dom_loc = self._make_locator("loc_dom", LocatorKind.DOM, 0, name="signInBtn")
        vision_loc = self._make_locator("loc_vis", LocatorKind.VISION_BBOX, 2)
        elem = self._make_element("e2", ["loc_dom", "loc_vis"], name="signInBtn")
        snap = self._make_snapshot(
            [elem], [dom_loc, vision_loc],
            hwnd=12345, surface_type=SurfaceType.BROWSER
        )
        resolved = self.adapter.resolve_element(snap, "e2")
        assert resolved is not None
        assert resolved["method"] == "dom"
        assert resolved["locator"].kind == LocatorKind.DOM

    def test_electron_webview_uia_first(self):
        """ELECTRON_WEBVIEW: UIA 优先于 DOM"""
        uia_loc = self._make_locator("loc_uia", LocatorKind.UIA, 0, automation_id="send_btn")
        dom_loc = self._make_locator("loc_dom", LocatorKind.DOM, 1, name="sendButton")
        elem = self._make_element("e3", ["loc_uia", "loc_dom"], automation_id="send_btn")
        snap = self._make_snapshot(
            [elem], [uia_loc, dom_loc],
            hwnd=12345, surface_type=SurfaceType.ELECTRON_WEBVIEW
        )
        resolved = self.adapter.resolve_element(snap, "e3")
        assert resolved is not None
        assert resolved["method"] == "uia"
        assert resolved["locator"].kind == LocatorKind.UIA

    def test_canvas_self_drawn_selects_vision_bbox(self):
        """CANVAS_SELF_DRAWN: VISION_BBOX 优先"""
        vision_loc = self._make_locator("loc_vis", LocatorKind.VISION_BBOX, 0)
        uia_loc = self._make_locator("loc_uia", LocatorKind.UIA, 1)
        elem = self._make_element("e4", ["loc_vis", "loc_uia"])
        snap = self._make_snapshot(
            [elem], [vision_loc, uia_loc],
            hwnd=12345, surface_type=SurfaceType.CANVAS_SELF_DRAWN
        )
        resolved = self.adapter.resolve_element(snap, "e4")
        assert resolved is not None
        assert resolved["method"] == "geometry"
        assert resolved["locator"].kind == LocatorKind.VISION_BBOX


class TestGetSurfaceExecutionStrategy:
    """get_surface_execution_strategy 测试"""

    def test_native_uia_strategy(self):
        result = get_surface_execution_strategy(SurfaceType.NATIVE_UIA)
        assert result["preferred_executor"] == "UIAExecutor"
        assert result["verification_level"] == 2
        assert "uia" in result["locator_chain"]

    def test_browser_strategy(self):
        result = get_surface_execution_strategy(SurfaceType.BROWSER)
        assert result["preferred_executor"] == "DOMExecutor"
        assert result["locator_chain"][0] == "dom"
        assert result["locator_chain"][:4] == ["dom", "shortcut", "relative", "ocr"]

    def test_electron_strategy(self):
        result = get_surface_execution_strategy(SurfaceType.ELECTRON_WEBVIEW)
        assert result["verification_level"] == 3
        assert result["locator_chain"][:6] == ["uia", "dom", "shortcut", "relative", "ocr", "vision_bbox"]

    def test_canvas_strategy(self):
        result = get_surface_execution_strategy(SurfaceType.CANVAS_SELF_DRAWN)
        assert result["locator_chain"][0] == "vision_bbox"

    def test_unknown_strategy(self):
        result = get_surface_execution_strategy(SurfaceType.UNKNOWN)
        assert result["verification_level"] == 4
