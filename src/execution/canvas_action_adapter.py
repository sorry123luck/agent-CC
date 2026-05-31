"""
P4 执行层 InteractionCanvas 适配器

使现有 P4 执行层能够消费新版 InteractionCanvas 输出。
仅做适配层，不重构 P4 本体。

适配内容：
- 将 element_id + InteractionCanvas 转换为执行器可执行的定位信息
- 将 locator 优先级转换为实际执行路径
- 提供 SurfaceType 驱动的执行策略选择

注意：本适配器仅使用 ActionExecutor 已有的方法，不引入不存在的方法。
当前 ActionExecutor 支持：
- click(hwnd, x, y, button, double) — 坐标点击
- click_element(hwnd, elem_x, elem_y, elem_w, elem_h, button, double) — 元素中心点击
- invoke_uia_element(hwnd, automation_id, name, control_type) — UIA invoke
- select_uia_item(hwnd, item_name, automation_id) — UIA 列表选择
- send_keys(hwnd, text) — 键盘输入
"""

from typing import Any, Literal

from src.perception.page_compiler_models import (
    InteractionCanvas,
    Candidate,
    Locator,
    LocatorKind,
    SurfaceType,
)
from src.execution.action_executor import ActionExecutor, ActionResult, MouseButton


class InteractionCanvasAdapter:
    """
    InteractionCanvas 到 P4 执行器的适配器

    使用流程：
    1. 接收 InteractionCanvas 和目标 element_id
    2. 根据 surface_type 选择执行策略
    3. 按 locator_priority 尝试定位并执行
    4. 返回 ActionResult
    """

    def __init__(self, dom_provider=None) -> None:
        self._executor = ActionExecutor()
        self._dom_provider = dom_provider

    def set_dom_provider(self, dom_provider) -> None:
        self._dom_provider = dom_provider

    def clear_dom_provider(self) -> None:
        self._dom_provider = None

    def resolve_element(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        locator_kind: LocatorKind | None = None,
    ) -> dict[str, Any] | None:
        """
        将 element_id 解析为可执行信息

        Args:
            snapshot: InteractionCanvas
            element_id: 目标元素 ID
            locator_kind: 指定 locator 类型（不指定则按优先级）

        Returns:
            dict 包含执行所需信息：
            - element: Candidate
            - locator: 将要使用的 Locator
            - coordinates: (x, y) 坐标（如果是点位类 locator）
            - geometry: (x, y, width, height) 几何信息（如果是 bbox/relative 类 locator）
            - method: 执行方法
        """
        element = snapshot.get_element(element_id)
        if not element:
            return None

        # 选择 locator
        locator: Locator | None = None
        if locator_kind:
            locator = self._find_locator_by_kind(element, locator_kind, snapshot)
        else:
            # 按 surface_type 对应的 locator_priority 选择
            locator = self._select_locator_by_priority(element, snapshot, snapshot.surface_type)

        if not locator:
            return None

        # 根据 locator kind 确定执行方法
        method = self._get_execution_method(locator.kind)

        # 计算坐标和几何信息
        coordinates: tuple[int, int] | None = None
        geometry: tuple[int, int, int, int] | None = None
        bounds = element.bounds
        selector = locator.selector or {}

        if locator.kind == LocatorKind.EPHEMERAL_COORD:
            x = selector.get("x")
            y = selector.get("y")
            if x is not None and y is not None:
                coordinates = (x, y)
        elif locator.kind in (
            LocatorKind.VISION_BBOX,
            LocatorKind.OCR,
            LocatorKind.TEMPLATE_ICON,
            LocatorKind.RELATIVE,
        ):
            bbox = selector.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                left, top, right, bottom = bbox
                geometry = (left, top, right - left, bottom - top)
            elif bounds:
                left, top, right, bottom = bounds
                geometry = (left, top, right - left, bottom - top)

            if geometry:
                x, y, width, height = geometry
                coordinates = (x + width // 2, y + height // 2)

        return {
            "element": element,
            "locator": locator,
            "coordinates": coordinates,
            "geometry": geometry,
            "method": method,
            "surface_type": snapshot.surface_type,
        }

    def execute_element_action(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: Literal["click", "double_click", "right_click"],
        verify: bool = True,
        locator_kind: LocatorKind | None = None,
    ) -> ActionResult:
        """
        对 InteractionCanvas 中的元素执行动作

        当前仅支持 click / double_click / right_click。

        Args:
            snapshot: InteractionCanvas
            element_id: 目标元素 ID
            action: 动作类型 ("click" | "double_click" | "right_click")
            verify: 是否验证（当前未实现）

        Returns:
            ActionResult 执行结果
        """
        # 检查 hwnd
        hwnd = snapshot.window.hwnd if snapshot.window else None
        if not hwnd:
            return ActionResult(
                success=False,
                action=action,
                details="No window hwnd in snapshot",
                error="no_hwnd",
            )

        resolved = self.resolve_element(snapshot, element_id, locator_kind=locator_kind)
        if not resolved:
            return ActionResult(
                success=False,
                action=action,
                details=f"Element {element_id} not found in snapshot",
                error="element_not_found",
            )

        element = resolved["element"]
        locator = resolved["locator"]
        coordinates = resolved["coordinates"]
        geometry = resolved["geometry"]
        method = resolved["method"]

        # 根据执行方法和动作类型执行
        try:
            if method == "uia":
                return self._execute_uia_action(hwnd, locator, action)
            elif method == "dom":
                return self._execute_dom_action(hwnd, locator, action, snapshot)
            elif method in {"geometry", "relative"}:
                if not geometry:
                    return ActionResult(
                        success=False,
                        action=action,
                        details="No geometry for geometry-based action",
                        error="no_geometry",
                    )
                return self._execute_geometry_action(hwnd, geometry, action)
            elif method == "coordinate":
                if not coordinates:
                    return ActionResult(
                        success=False,
                        action=action,
                        details="No coordinates for coordinate-based action",
                        error="no_coordinates",
                    )
                return self._execute_coordinate_action(
                    hwnd,
                    coordinates,
                    action,
                    locator.coordinate_space.value,
                )
            else:
                return ActionResult(
                    success=False,
                    action=action,
                    details=f"Unknown execution method: {method}",
                    error="unknown_method",
                )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                details=str(e),
                error="execution_error",
            )

    def _find_locator_by_kind(
        self,
        element: Candidate,
        kind: LocatorKind,
        snapshot: InteractionCanvas,
    ) -> Locator | None:
        """查找指定类型的 locator"""
        for locator_id in element.locator_ids:
            locator = snapshot.get_locator(locator_id)
            if locator and locator.kind == kind:
                return locator
        return None

    def _select_locator_by_priority(
        self,
        element: Candidate,
        snapshot: InteractionCanvas,
        surface_type: SurfaceType,
    ) -> Locator | None:
        """按 surface_type 对应的 locator_priority 选择最佳 locator"""
        from src.perception.surface_classifier import SurfaceClassifier

        priority = SurfaceClassifier()._get_locator_priority(surface_type)

        # 构建 kind -> priority 映射
        kind_priority: dict[LocatorKind, int] = {}
        for i, kind_name in enumerate(priority):
            try:
                kind = LocatorKind[kind_name.upper()]
                kind_priority[kind] = i
            except KeyError:
                pass

        # 按优先级排序所有 locators
        locators: list[tuple[int, Locator]] = []
        for locator_id in element.locator_ids:
            locator = snapshot.get_locator(locator_id)
            if locator:
                p = kind_priority.get(locator.kind, 999)
                locators.append((p, locator))

        locators.sort(key=lambda x: x[0])
        return locators[0][1] if locators else None

    def _get_execution_method(self, locator_kind: LocatorKind) -> str:
        """将 LocatorKind 映射为执行方法"""
        method_map = {
            LocatorKind.UIA: "uia",
            LocatorKind.DOM: "dom",
            LocatorKind.OCR: "geometry",
            LocatorKind.VISION_BBOX: "geometry",
            LocatorKind.RELATIVE: "relative",
            LocatorKind.TEMPLATE_ICON: "geometry",
            LocatorKind.EPHEMERAL_COORD: "coordinate",
        }
        return method_map.get(locator_kind, "unknown")

    def _execute_geometry_action(
        self,
        hwnd: int,
        geometry: tuple[int, int, int, int],
        action: str,
    ) -> ActionResult:
        """Execute action against window-local geometry."""
        x, y, width, height = geometry

        if action == "click":
            return self._executor.click_element(hwnd, x, y, width, height)
        elif action == "double_click":
            return self._executor.click_element(hwnd, x, y, width, height, double=True)
        elif action == "right_click":
            return self._executor.click_element(
                hwnd,
                x,
                y,
                width,
                height,
                button=MouseButton.RIGHT,
            )
        else:
            return ActionResult(
                success=False,
                action=action,
                details=f"Unsupported action for geometry: {action}",
                error="unsupported_action",
            )

    def _execute_uia_action(
        self,
        hwnd: int,
        locator: Locator,
        action: str,
    ) -> ActionResult:
        """通过 UIA 执行动作

        使用 ActionExecutor.invoke_uia_element() 方法。
        方法签名：invoke_uia_element(hwnd, automation_id=None, name=None, control_type=None)
        """
        # 提取 automation_id
        automation_id = (
            locator.selector.get("automation_id")
            or locator.selector.get("id")
        )

        if not automation_id:
            return ActionResult(
                success=False,
                action=action,
                details="No automation_id for UIA action",
                error="no_automation_id",
            )

        # 调用 UIA invoke
        result = self._executor.invoke_uia_element(
            hwnd=hwnd,
            automation_id=automation_id,
        )
        # 包装为 ActionResult
        return ActionResult(
            success=result.success,
            action=action,
            details=result.details,
            error=result.error,
        )

    def _execute_coordinate_action(
        self,
        hwnd: int,
        coordinates: tuple[int, int],
        action: str,
        coordinate_space: str = "screen",
    ) -> ActionResult:
        """通过坐标执行动作

        使用 ActionExecutor.click() 方法。
        方法签名：click(hwnd, x, y, button=MouseButton.LEFT, double=False)
        """
        x, y = coordinates
        if coordinate_space != "screen":
            x, y = self._executor._client_to_screen(hwnd, x, y)

        if action == "click":
            return self._executor.click(hwnd, x, y)
        elif action == "double_click":
            return self._executor.click(hwnd, x, y, double=True)
        elif action == "right_click":
            return self._executor.click(hwnd, x, y, MouseButton.RIGHT)
        else:
            return ActionResult(
                success=False,
                action=action,
                details=f"Unsupported action for coordinates: {action}",
                error="unsupported_action",
            )

    def _execute_dom_action(
        self,
        hwnd: int,
        locator: Locator,
        action: str,
        snapshot: InteractionCanvas,
    ) -> ActionResult:
        """
        通过 DOM Provider 执行动作

        使用 IDOMProvider.get_element_by_locator() 查找元素，
        然后用 ActionExecutor.click_element() 在元素中心点击。

        当前 DOM Provider 尚未实现（USABLE_SKELETON），
        本方法在 provider 不可用时返回明确的不可用状态，
        而非泛化 not_implemented。
        """
        selector = locator.selector or {}
        locator_name = selector.get("name")
        locator_text = selector.get("text")

        # 尝试使用 DOM Provider
        dom_provider = self._get_dom_provider()
        if dom_provider is None:
            return ActionResult(
                success=False,
                action=action,
                details="DOM execution unavailable: no DOM Provider connected. "
                        "DOM locator targets element by "
                        f"name={locator_name!r} or text={locator_text!r}. "
                        "A DOM Provider (e.g. Playwright-based) must be connected to execute DOM actions.",
                error="dom_provider_unavailable",
            )

        try:
            if hasattr(dom_provider, "is_dom_ready") and not dom_provider.is_dom_ready():
                return ActionResult(
                    success=False,
                    action=action,
                    details="DOM execution blocked: provider connected but DOM is not ready.",
                    error="dom_not_ready",
                )
            if hasattr(dom_provider, "perform_action"):
                dom_result = dom_provider.perform_action(selector, action)
                if dom_result.get("success"):
                    return ActionResult(
                        success=True,
                        action=action,
                        details=dom_result.get("details", f"dom_{action}_ok"),
                    )
            dom_element = dom_provider.get_element_by_locator(selector)
        except NotImplementedError:
            return ActionResult(
                success=False,
                action=action,
                details="DOM execution blocked: IDOMProvider.get_element_by_locator() "
                        "is not yet implemented. "
                        f"DOM locator targets name={locator_name!r}, text={locator_text!r}. "
                        "Implement IDOMProvider to enable DOM execution.",
                error="dom_provider_not_implemented",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                details=f"DOM execution failed: {e}",
                error="dom_execution_error",
            )

        if dom_element is None:
            return ActionResult(
                success=False,
                action=action,
                details=f"DOM element not found for selector: {selector}",
                error="dom_element_not_found",
            )

        # 从 DOM 元素提取坐标
        rect = dom_element.get("rect")
        if not rect:
            return ActionResult(
                success=False,
                action=action,
                details="DOM element found but has no rect (position) information",
                error="dom_no_rect",
            )

        x, y, w, h = rect
        cx, cy = x + w // 2, y + h // 2

        # 使用 click_element 执行点击
        try:
            if action == "click":
                return self._executor.click_element(hwnd, cx, cy, w, h)
            elif action == "double_click":
                return self._executor.click_element(hwnd, cx, cy, w, h, double=True)
            elif action == "right_click":
                from src.execution.action_executor import MouseButton
                return self._executor.click_element(hwnd, cx, cy, w, h, button=MouseButton.RIGHT)
            else:
                return ActionResult(
                    success=False,
                    action=action,
                    details=f"Unsupported DOM action: {action}",
                    error="unsupported_action",
                )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                details=f"DOM click execution failed: {e}",
                error="dom_click_error",
            )

    def _get_dom_provider(self):
        """
        获取当前已连接的 DOM Provider。

        尝试导入并返回 IDOMProvider 实例。
        如果没有已连接的 provider，返回 None。

        注意：当前 DOM Provider 为 USABLE_SKELETON，
        即使返回实例，其方法也可能 raise NotImplementedError。
        """
        try:
            from src.perception.providers.dom_provider import IDOMProvider
            provider = getattr(self, "_dom_provider", None)
            if provider is not None:
                return provider
            return None
        except ImportError:
            return None

def get_surface_execution_strategy(surface_type: SurfaceType) -> dict[str, Any]:
    """
    获取 surface_type 对应的执行策略

    Returns:
        dict 包含执行策略信息：
        - preferred_executor: str
        - verification_level: int (1-4)
        - recovery_enabled: bool
    """
    strategies = {
        SurfaceType.NATIVE_UIA: {
            "preferred_executor": "UIAExecutor",
            "verification_level": 2,
            "recovery_enabled": True,
            "locator_chain": ["uia", "shortcut", "relative", "ocr", "template_icon", "vision_bbox", "ephemeral_coord"],
        },
        SurfaceType.BROWSER: {
            "preferred_executor": "DOMExecutor",
            "verification_level": 2,
            "recovery_enabled": True,
            "locator_chain": ["dom", "shortcut", "relative", "ocr", "vision_bbox", "template_icon", "ephemeral_coord"],
        },
        SurfaceType.ELECTRON_WEBVIEW: {
            "preferred_executor": "HybridExecutor",
            "verification_level": 3,
            "recovery_enabled": True,
            "locator_chain": ["uia", "dom", "shortcut", "relative", "ocr", "vision_bbox", "template_icon", "ephemeral_coord"],
        },
        SurfaceType.CANVAS_SELF_DRAWN: {
            "preferred_executor": "InputExecutor",
            "verification_level": 3,
            "recovery_enabled": True,
            "locator_chain": ["vision_bbox", "ocr", "template_icon", "relative", "shortcut", "ephemeral_coord"],
        },
        SurfaceType.UNKNOWN: {
            "preferred_executor": "InputExecutor",
            "verification_level": 4,
            "recovery_enabled": True,
            "locator_chain": ["uia", "shortcut", "relative", "ocr", "vision_bbox", "ephemeral_coord"],
        },
    }
    return strategies.get(surface_type, strategies[SurfaceType.UNKNOWN])
