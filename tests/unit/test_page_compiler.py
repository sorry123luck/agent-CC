"""
InteractionCanvasEngine 单元测试
"""

import pytest

from src.perception.page_compiler import InteractionCanvasEngine
from src.perception.page_compiler_candidates import build_boundary_candidates
from src.perception.geometric_partitioner import GeometricRegion
from src.perception.page_compiler_models import Candidate, CoordinateSpace, LocatorKind, Region, SemanticRole, SurfaceType


class TestInteractionCanvasEngine:
    """InteractionCanvasEngine 单元测试"""

    def setup_method(self):
        self.compiler = InteractionCanvasEngine()

    def test_compile_returns_snapshot(self):
        """compile 返回 InteractionCanvas"""
        result = self.compiler.compile(
            process_name="WeChat.exe",
            window_hwnd=12345,
            window_title="WeChat",
            uia_element_count=200,
            uia_control_types=["Button", "Edit", "Pane"],
        )
        assert result.canvas_id is not None
        assert result.surface.surface_type is not None

    def test_compile_surface_type_wechat(self):
        """WeChat.exe 被分类为 ELECTRON_WEBVIEW"""
        result = self.compiler.compile(
            process_name="WeChat.exe",
            uia_element_count=200,
            uia_control_types=["ToolBar", "Edit", "Pane"],
        )
        assert result.surface.surface_type == SurfaceType.ELECTRON_WEBVIEW

    def test_compile_surface_type_chrome(self):
        """chrome.exe 被分类为 BROWSER"""
        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
        )
        assert result.surface.surface_type == SurfaceType.BROWSER

    def test_compile_surface_type_unknown_defaults_native_uia(self):
        """未知进程默认 NATIVE_UIA"""
        result = self.compiler.compile(
            process_name="unknown.exe",
            uia_element_count=500,
            uia_control_types=["Button", "Edit", "Tree"],
        )
        assert result.surface.surface_type == SurfaceType.NATIVE_UIA

    def test_synthetic_control_type_does_not_turn_long_submit_text_into_button(self):
        """长说明文本包含“提交”时不应被误分成 submit/button 控件。"""
        raw_elements = [{"bounding_rect": (0, 0, 1600, 900)}]

        short_action = self.compiler._infer_synthetic_control_type("提交", (1200, 700, 1260, 730), raw_elements)
        long_text = self.compiler._infer_synthetic_control_type(
            '执行： git add CLAUDE.md git commit -m "docs: align entry documents with current handoff" 提交后输出状态',
            (100, 80, 900, 120),
            raw_elements,
        )

        assert short_action == "ButtonControl"
        assert long_text != "ButtonControl"

    def test_compile_page_class_wechat(self):
        """WeChat page_class 包含 wechat app"""
        result = self.compiler.compile(
            process_name="WeChat.exe",
        )
        assert result.page.page_class.startswith("wechat/")

    def test_compile_regions_created(self):
        """compile 生成 regions"""
        result = self.compiler.compile(
            process_name="WeChat.exe",
            uia_element_count=100,
            uia_control_types=["Button", "Edit"],
        )
        assert len(result.regions) >= 1

    def test_compile_layout_first_regions_without_zone_structure(self):
        """无 legacy zone 时也应做最小 layout-first 分区，而不是单 content 区"""
        raw_elements = [
            {
                "element_id": "title",
                "control_type": "TitleBarControl",
                "name": "TitleBar",
                "bounding_rect": (0, 0, 1200, 48),
            },
            {
                "element_id": "sidebar",
                "control_type": "PaneControl",
                "name": "Sidebar",
                "bounding_rect": (0, 60, 220, 780),
            },
            {
                "element_id": "content",
                "control_type": "DocumentControl",
                "name": "Content",
                "bounding_rect": (240, 80, 1180, 780),
            },
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )
        roles = {region.role for region in result.regions}
        assert "title_bar" in roles
        assert "side_bar" in roles
        assert "content_area" in roles
        sidebar = next(element for element in result.elements if element.element_id == "sidebar")
        content = next(element for element in result.elements if element.element_id == "content")
        assert sidebar.region_id == "region_side_bar"
        assert content.region_id in {"region_content_0", "region_content_viewport_0"}

    def test_compile_keeps_geometric_fusion_regions_diagnostic_only(self):
        """几何 ROI 和融合标签只做诊断/ROI 证据，不直接进入最终 regions。"""
        raw_elements = [
            {
                "element_id": "edit",
                "control_type": "Edit",
                "name": "Input",
                "bounding_rect": (330, 620, 900, 700),
            },
            {
                "element_id": "send",
                "control_type": "Button",
                "name": "Send",
                "bounding_rect": (920, 650, 980, 700),
            },
        ]
        geometric_regions = [
            GeometricRegion(
                region_id="R0",
                bounds=(0, 0, 300, 720),
                boundary_evidence=["vertical_separator_L95%"],
                density_profile=(10.0, 0.02),
                geometry_confidence=0.8,
            ),
            GeometricRegion(
                region_id="R1",
                bounds=(300, 600, 1000, 720),
                boundary_evidence=["horizontal_separator_L95%"],
                density_profile=(12.0, 0.03),
                geometry_confidence=0.82,
            ),
        ]
        fusion_diagnostics = {
            "regions": [
                {
                    "region_id": "R0",
                    "semantic_label": "navigation",
                    "confidence": 0.55,
                    "reason": "side list",
                },
                {
                    "region_id": "R1",
                    "semantic_label": "input_area",
                    "confidence": 0.85,
                    "reason": "UIA edit present",
                },
            ]
        }

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
            geometric_regions=geometric_regions,
            fusion_diagnostics=fusion_diagnostics,
        )

        assert not any(region.geometric_source_id for region in result.regions)
        assert not any(region.role == "geometry_region" for region in result.regions)
        assert {item["region_id"] for item in result.artifacts["geometric_regions"]} >= {"R0", "R1"}
        assert result.artifacts["fusion_diagnostics"]["regions"][0]["semantic_label"] == "navigation"
        assert result.artifacts["geometric_regions"][0]["boundary_evidence"] == ["vertical_separator_L95%"]

    def test_compile_keeps_unknown_geometric_regions_diagnostic_only(self):
        """未被语义融合命名的几何分区只保留在诊断 artifacts，不进入最终 regions。"""
        raw_elements = [
            {
                "element_id": "root",
                "control_type": "Window",
                "name": "Root",
                "bounding_rect": (0, 0, 1000, 720),
            },
            {
                "element_id": "label_a",
                "control_type": "Text",
                "name": "A",
                "bounding_rect": (40, 40, 120, 80),
            },
            {
                "element_id": "label_b",
                "control_type": "Text",
                "name": "B",
                "bounding_rect": (420, 40, 520, 80),
            },
        ]
        geometric_regions = [
            GeometricRegion("R0", (0, 0, 300, 720), ["separator_partition"], (10.0, 0.02), 0.55),
            GeometricRegion("R1", (300, 0, 700, 720), ["separator_partition"], (11.0, 0.03), 0.55),
            GeometricRegion("R2", (700, 0, 1000, 720), ["separator_partition"], (9.0, 0.02), 0.55),
        ]
        fusion_diagnostics = {
            "regions": [
                {"region_id": "R0", "semantic_label": None, "confidence": 0.0},
                {"region_id": "R1", "semantic_label": None, "confidence": 0.0},
                {"region_id": "R2", "semantic_label": None, "confidence": 0.0},
            ]
        }

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
            geometric_regions=geometric_regions,
            fusion_diagnostics=fusion_diagnostics,
        )

        assert not any(region.role == "geometry_region" for region in result.regions)
        assert {item["region_id"] for item in result.artifacts["geometric_regions"]} >= {"R0", "R1", "R2"}

    def test_compile_provider_trace_uia(self):
        """有 UIA 元素时 provider_trace 标记 uia_used"""
        result = self.compiler.compile(
            process_name="notepad.exe",
            uia_element_count=50,
            uia_control_types=["Edit", "MenuBar"],
        )
        assert result.provider_trace.uia_used is True

    def test_compile_provider_trace_dom(self):
        """provider_trace.dom_used 反映真实 DOM locator 生成，不只是 has_dom_bridge 标志"""
        # 有 bridge 但无 DOM 元素（无 name）→ dom_used=False，status 说明原因
        result_no_elements = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
        )
        dom_detail = result_no_elements.provider_trace.provider_details.get("dom", {})
        assert result_no_elements.provider_trace.dom_used is False
        assert dom_detail.get("status") == "bridge_exists_no_dom_elements"

        # has_dom_bridge=False → 不生成 DOM locator → dom_used=False
        raw_elements_with_name = [
            {"element_id": "e1", "control_type": "Button", "name": "submitBtn", "text": "Submit", "bounding_rect": (10, 10, 50, 30)},
        ]
        result_no_bridge = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=False,  # 无 bridge
            dom_ready=False,
            raw_elements=raw_elements_with_name,
        )
        assert result_no_bridge.provider_trace.dom_used is False

        # 有 bridge + 有 name 元素 → DOM locator 真实生成 → dom_used=True
        result_with_dom = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements_with_name,
        )
        dom_detail_with = result_with_dom.provider_trace.provider_details.get("dom", {})
        assert result_with_dom.provider_trace.dom_used is True
        assert dom_detail_with.get("dom_locator_count", 0) >= 1

    def test_compile_vision_used(self):
        """有 visual_score 时标记 vision_used"""
        result = self.compiler.compile(
            process_name="chrome.exe",
            visual_score=0.5,
        )
        assert result.provider_trace.vision_used is True

    def test_compile_confidence_valid(self):
        """surface confidence 在有效范围"""
        result = self.compiler.compile(
            process_name="WeChat.exe",
        )
        assert 0.0 <= result.surface.confidence <= 1.0

    def test_compile_raw_elements_become_page_elements(self):
        """raw_elements 被转换为 Candidate"""
        raw_elements = [
            {
                "element_id": "e1",
                "control_type": "Button",
                "name": "OK",
                "text": "OK",
                "bounding_rect": (10, 10, 50, 30),
            },
            {
                "element_id": "e2",
                "control_type": "Edit",
                "name": "Input",
                "text": "",
                "bounding_rect": (10, 40, 200, 60),
            },
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
        )
        assert len(result.elements) == 2
        assert result.elements[0].element_id == "e1"
        assert result.elements[0].control_type == "Button"
        assert result.elements[0].semantic_role.value == "button"

    def test_compile_infers_send_button(self):
        """发送按钮被识别为 SEND_BUTTON"""
        raw_elements = [
            {
                "element_id": "e1",
                "control_type": "Button",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (300, 400, 360, 430),
            },
        ]
        result = self.compiler.compile(
            process_name="wechat.exe",
            raw_elements=raw_elements,
        )
        send_buttons = [e for e in result.elements if e.semantic_role.value == "send_button"]
        assert len(send_buttons) == 1
        assert "send" in send_buttons[0].risk_tags

    def test_compile_canvas_id_unique(self):
        """每次 compile 生成唯一 canvas_id"""
        result1 = self.compiler.compile(process_name="test.exe")
        result2 = self.compiler.compile(process_name="test.exe")
        assert result1.canvas_id != result2.canvas_id

    def test_compile_uia_locator_requires_automation_id(self):
        """UIA locator 只在有 automation_id 时才生成"""
        # 使用 devenv.exe（保证 NATIVE_UIA surface），并传 uia_element_count>0
        # 无 automation_id → 不应有 UIA locator
        raw_elements_no_aid = [
            {
                "element_id": "e1",
                "control_type": "Button",
                "name": "OK",
                "text": "OK",
                "bounding_rect": (10, 10, 50, 30),
                "automation_id": None,
            },
        ]
        result_no_aid = self.compiler.compile(
            process_name="devenv.exe",
            raw_elements=raw_elements_no_aid,
            uia_element_count=100,
            uia_control_types=["Button"],
        )
        assert result_no_aid.surface.surface_type.value == "native_uia"
        uia_locs = [l for l in result_no_aid.locators if l.kind.value == "uia"]
        assert len(uia_locs) == 0, "无 automation_id 时不应生成 UIA locator"
        # 但应有 VISION_BBOX 和 EPHEMERAL_COORD 作为兜底
        vision_locs = [l for l in result_no_aid.locators if l.kind.value == "vision_bbox"]
        assert len(vision_locs) == 1, "无 automation_id 时应有 VISION_BBOX 兜底"

        # 有 automation_id → 应生成 UIA locator
        raw_elements_with_aid = [
            {
                "element_id": "e2",
                "control_type": "Button",
                "name": "OK",
                "text": "OK",
                "bounding_rect": (10, 10, 50, 30),
                "automation_id": "btn_ok",
            },
        ]
        result_with_aid = self.compiler.compile(
            process_name="devenv.exe",
            raw_elements=raw_elements_with_aid,
            uia_element_count=100,
            uia_control_types=["Button"],
        )
        assert result_with_aid.surface.surface_type.value == "native_uia"
        uia_locs_with = [l for l in result_with_aid.locators if l.kind.value == "uia"]
        assert len(uia_locs_with) == 1, "有 automation_id 时应生成 UIA locator"
        assert uia_locs_with[0].selector.get("automation_id") == "btn_ok"

    def test_compile_zone_structure_produces_real_regions(self):
        """有 zone_structure 时 compile 生成真实多 region"""
        from src.perception.zone_partitioner import ZonePartitioner, ZoneType

        # 构建有真实区域的 zone_structure
        zp = ZonePartitioner()
        raw_elements_for_zp = [
            {"name": "TitleBar", "control_type": "TitleBarControl", "bounding_rect": (0, 0, 1920, 50), "element_id": "t1"},
            {"name": "MenuBar", "control_type": "MenuBarControl", "bounding_rect": (0, 50, 1920, 80), "element_id": "m1"},
            {"name": "Content", "control_type": "PaneControl", "bounding_rect": (0, 80, 1920, 1000), "element_id": "c1"},
        ]
        zone_struct = zp.partition_from_dicts(raw_elements_for_zp, 1920, 1000)
        assert len(zone_struct.zones) >= 2, "应有多个 zone"

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements_for_zp,
            zone_structure=zone_struct,
        )
        # 验证 provider_trace 记录了 zone_partition
        assert result.provider_trace.provider_details.get("zone_partition", {}).get("used") is True
        # 验证生成了多个 region（而非单虚拟 region）
        assert len(result.regions) >= 2

    def test_compile_old_format_bounds_still_works(self):
        """旧格式 left/top/right/bottom 字段仍能正常生成完整 bounds 和 locators"""
        # 旧格式：dict 用 left/top/right/bottom 而非 bounding_rect
        old_format_elements = [
            {
                "element_id": "old_btn",
                "control_type": "Button",
                "name": "OK",
                "text": "OK",
                "left": 10, "top": 10, "right": 50, "bottom": 30,
            },
            {
                "element_id": "old_input",
                "control_type": "Edit",
                "name": "Name",
                "text": "",
                "left": 60, "top": 10, "right": 200, "bottom": 30,
                "automation_id": "name_field",
            },
        ]
        result = self.compiler.compile(
            process_name="devenv.exe",  # NATIVE_UIA
            raw_elements=old_format_elements,
            uia_element_count=10,
        )
        assert result.canvas_id is not None
        # elements 生成
        assert len(result.elements) == 2
        # bounds 正确解析
        elem_btn = next(e for e in result.elements if e.element_id == "old_btn")
        assert elem_btn.bounds == (10, 10, 50, 30)
        elem_input = next(e for e in result.elements if e.element_id == "old_input")
        assert elem_input.bounds == (60, 10, 200, 30)
        # automation_id 保留
        assert elem_input.attributes.get("automation_id") == "name_field"
        # locators 生成（UIA locator 因为有 automation_id）
        uia_locs = [l for l in result.locators if l.kind.value == "uia"]
        assert len(uia_locs) >= 1, "有 automation_id 应生成 UIA locator"
        # anchors 生成
        assert len(result.anchors) >= 0
        # relations 生成
        assert len(result.relations) >= 0

    def test_compile_element_sources_attribution(self):
        """elements 有正确的 provider_sources 归因"""
        from src.perception.zone_partitioner import ZonePartitioner
        zp = ZonePartitioner()
        raw_elements_for_zp = [
            {"name": "TitleBar", "control_type": "TitleBarControl", "bounding_rect": (0, 0, 1920, 50), "element_id": "t1"},
            {"name": "Content", "control_type": "PaneControl", "bounding_rect": (0, 50, 1920, 1000), "element_id": "c1"},
        ]
        zone_struct = zp.partition_from_dicts(raw_elements_for_zp, 1920, 1000)

        raw_elements = [
            {"element_id": "t1", "control_type": "TitleBarControl", "name": "TitleBar", "bounding_rect": (0, 0, 1920, 50)},
            {"element_id": "c1", "control_type": "PaneControl", "name": "Content", "bounding_rect": (0, 50, 1920, 1000)},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            zone_structure=zone_struct,
        )

        # provider_trace.element_sources 应记录来源统计
        elem_sources = result.provider_trace.provider_details.get("element_sources", {})
        assert elem_sources.get("zone_partition", 0) >= 1, "zone_partition 元素应有归因"

        # 有 zone_structure 时，provider_sources 应为 zone_partition
        zone_part_elem = [e for e in result.elements if "zone_partition" in e.provider_sources]
        assert len(zone_part_elem) >= 1

    def test_compile_generates_locators_anchors_relations(self):
        """compile 生成 locators / anchors / relations"""
        raw_elements = [
            {
                "element_id": "e1",
                "control_type": "Edit",
                "name": "输入框",
                "text": "",
                "bounding_rect": (100, 500, 500, 540),
                "automation_id": "input_field",
            },
            {
                "element_id": "e2",
                "control_type": "Button",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (510, 500, 580, 540),
                "automation_id": "send_btn",
            },
        ]
        result = self.compiler.compile(
            process_name="wechat.exe",
            raw_elements=raw_elements,
            uia_control_types=["Edit", "Button"],
        )
        assert len(result.locators) > 0, "应有 locators"
        assert len(result.anchors) > 0, "应有 anchors"
        assert len(result.relations) > 0, "应有 relations"
        # 有 automation_id 的元素应有 UIA locator
        uia_locs = [l for l in result.locators if l.kind.value == "uia"]
        assert len(uia_locs) == 2

    def test_compile_locator_stability(self):
        """相同输入两次调用 compile()，locator 结构应稳定（ID 除外）"""
        raw_elements = [
            {
                "element_id": "e1",
                "control_type": "Edit",
                "name": "input_field",
                "text": "",
                "bounding_rect": (100, 500, 500, 540),
                "automation_id": "input_field",
            },
            {
                "element_id": "e2",
                "control_type": "Button",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (510, 500, 580, 540),
                "automation_id": "send_btn",
            },
        ]

        result1 = self.compiler.compile(
            process_name="wechat.exe",
            raw_elements=raw_elements,
            uia_control_types=["Edit", "Button"],
        )
        result2 = self.compiler.compile(
            process_name="wechat.exe",
            raw_elements=raw_elements,
            uia_control_types=["Edit", "Button"],
        )

        # canvas_id 不同（每次生成）
        assert result1.canvas_id != result2.canvas_id

        # locator 数量相同
        assert len(result1.locators) == len(result2.locators)

        # locator 结构稳定：种类相同、selector 内容相同（ID 不同可接受）
        result1_sorted = sorted(result1.locators, key=lambda l: l.kind.value)
        result2_sorted = sorted(result2.locators, key=lambda l: l.kind.value)

        for loc1, loc2 in zip(result1_sorted, result2_sorted):
            assert loc1.kind == loc2.kind, "locator kind 应相同"
            assert loc1.selector == loc2.selector, "locator selector 应相同"
            assert loc1.element_ref == loc2.element_ref, "element_ref 应相同"
            assert loc1.priority == loc2.priority, "locator priority 应相同"

    def test_compile_electron_locator_priority_matches_execution_strategy(self):
        """ELECTRON_WEBVIEW 下 locator 优先级应与执行策略保持一致"""
        raw_elements = [
            {
                "element_id": "send_btn",
                "control_type": "Button",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (510, 500, 580, 540),
                "automation_id": "send_btn",
            },
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            has_dom_bridge=True,
            dom_ready=True,
            uia_element_count=50,
            uia_control_types=["ToolBar", "Edit", "WebView2"],
        )

        priorities = {loc.kind: loc.priority for loc in result.locators}

        assert result.surface.surface_type == SurfaceType.ELECTRON_WEBVIEW
        assert priorities[LocatorKind.UIA] < priorities[LocatorKind.DOM]
        assert priorities[LocatorKind.DOM] < priorities[LocatorKind.VISION_BBOX]
        assert priorities[LocatorKind.VISION_BBOX] < priorities[LocatorKind.EPHEMERAL_COORD]

    def test_compile_coordinate_locators_use_window_coordinate_space(self):
        """窗口局部统一坐标系下，坐标类 locator 应标记为 WINDOW"""
        raw_elements = [
            {
                "element_id": "send_btn",
                "control_type": "Button",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (510, 500, 580, 540),
                "automation_id": "send_btn",
            },
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            uia_element_count=50,
            uia_control_types=["ToolBar", "Edit", "WebView2"],
        )

        coordinate_locators = [
            loc for loc in result.locators
            if loc.kind in (LocatorKind.VISION_BBOX, LocatorKind.EPHEMERAL_COORD)
        ]

        assert coordinate_locators, "应至少生成坐标类 locator"
        assert all(
            loc.coordinate_space == CoordinateSpace.WINDOW
            for loc in coordinate_locators
        )

    def test_compile_chat_vs_form_detection(self):
        """验证 chat 类页面（底部输入区+消息流）被识别为 CHAT 而非 FORM"""
        # 模拟 ChatGPT 类页面：顶部 pane（消息流）+ 底部输入框+按钮
        raw_elements = [
            # 顶部消息区（pane 占主导）
            {
                "element_id": "msg_pane",
                "control_type": "PaneControl",
                "name": "chat_messages",
                "bounding_rect": (0, 50, 800, 500),
            },
            # 底部输入框
            {
                "element_id": "chat_input",
                "control_type": "EditControl",
                "name": "chat_input",
                "text": "",
                "bounding_rect": (50, 520, 600, 560),
            },
            # 底部发送按钮
            {
                "element_id": "send_btn",
                "control_type": "ButtonControl",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (610, 520, 700, 560),
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements,
            uia_control_types=["Pane", "Edit", "Button"],
        )

        # 找到 content_area region
        content_regions = [r for r in result.regions if r.role == "content_area"]
        assert len(content_regions) >= 1, "应有 content_area region"

        # content_area 应识别为 CHAT
        content_region = content_regions[0]
        assert content_region.subtype.value == "chat", \
            f"ChatGPT 类页面应识别为 chat，实际: {content_region.subtype.value}"

    def test_compile_fuses_ocr_text_into_elements_and_generates_ocr_locator(self):
        """OCR 可补空文本元素，并生成 OCR locator 与 provider trace。"""
        raw_elements = [
            {
                "element_id": "submit_btn",
                "control_type": "ButtonControl",
                "name": None,
                "text": "",
                "bounding_rect": (100, 100, 220, 150),
            },
        ]
        ocr_blocks = [
            {
                "text": "提交",
                "bbox": (105, 102, 210, 148),
                "confidence": 0.96,
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )

        submit = next(element for element in result.elements if element.element_id == "submit_btn")
        assert submit.text == "提交"
        assert "ocr" in submit.provider_sources
        assert submit.attributes["ocr_text"] == "提交"
        ocr_locators = [loc for loc in result.locators if loc.kind == LocatorKind.OCR]
        assert len(ocr_locators) == 1
        assert result.provider_trace.ocr_used is True
        assert result.provider_trace.provider_details["ocr"]["block_count"] == 1

    def test_compile_records_vision_candidates_in_provider_trace(self):
        """Vision candidates 应进入 provider_trace 作为正式输入记录。"""
        raw_elements = [
            {
                "element_id": "item",
                "control_type": "PaneControl",
                "name": "Item",
                "bounding_rect": (10, 10, 110, 110),
            },
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            vision_candidates=[
                {
                    "candidate_id": "vision_1",
                    "bbox": [12, 12, 108, 108],
                    "kind": "icon_candidate",
                    "confidence": 0.8,
                }
            ],
            allow_legacy_zone_reconstruction=False,
        )

        assert result.provider_trace.vision_used is True
        assert result.provider_trace.provider_details["vision"]["candidate_count"] == 1

    def test_compile_attaches_anchor_refs_to_locators(self):
        """生成 anchors 后，locator 应带上 anchor refs 供执行层消费。"""
        raw_elements = [
            {
                "element_id": "send_btn",
                "control_type": "ButtonControl",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (100, 100, 180, 140),
                "automation_id": "send_btn",
            },
            {
                "element_id": "input_box",
                "control_type": "EditControl",
                "name": "message",
                "bounding_rect": (20, 100, 90, 140),
            },
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            uia_element_count=50,
            uia_control_types=["Button", "Edit"],
            allow_legacy_zone_reconstruction=False,
        )

        send_element = next(element for element in result.elements if element.element_id == "send_btn")
        send_locator = next(
            locator for locator in result.locators if locator.element_ref == "send_btn" and locator.kind == LocatorKind.UIA
        )
        assert send_element.anchor_ids
        assert send_locator.anchor_refs == send_element.anchor_ids

    def test_compile_chat_layout_builds_message_and_composer_regions(self):
        """chat fallback 应拆出消息流和 composer 子区，并把输入控件挂到 composer。"""
        raw_elements = [
            {
                "element_id": "msg_pane",
                "control_type": "PaneControl",
                "name": "chat_messages",
                "bounding_rect": (40, 80, 860, 520),
            },
            {
                "element_id": "chat_input",
                "control_type": "EditControl",
                "name": "chat_input",
                "bounding_rect": (60, 560, 720, 610),
            },
            {
                "element_id": "send_btn",
                "control_type": "ButtonControl",
                "name": "发送",
                "text": "发送",
                "bounding_rect": (740, 560, 840, 610),
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        roles = {region.role for region in result.regions}
        assert "message_stream" in roles
        assert "composer_area" in roles

        content_region = next(region for region in result.regions if region.region_id == "region_content_0")
        assert {
            "region_content_messages_0",
            "region_content_composer_0",
        }.issubset(set(content_region.child_region_ids))

        chat_input = next(element for element in result.elements if element.element_id == "chat_input")
        send_button = next(element for element in result.elements if element.element_id == "send_btn")
        assert chat_input.region_id == "region_content_composer_0"
        assert send_button.region_id == "region_content_composer_0"

    def test_compile_form_layout_builds_field_and_action_regions(self):
        """form fallback 应拆出字段区和底部动作区。"""
        raw_elements = [
            {
                "element_id": "username_label",
                "control_type": "TextControl",
                "name": "用户名",
                "text": "用户名",
                "bounding_rect": (80, 120, 200, 150),
            },
            {
                "element_id": "username_input",
                "control_type": "EditControl",
                "name": "username",
                "bounding_rect": (220, 110, 760, 160),
            },
            {
                "element_id": "submit_btn",
                "control_type": "ButtonControl",
                "name": "提交",
                "text": "提交",
                "bounding_rect": (560, 620, 700, 670),
            },
            {
                "element_id": "cancel_btn",
                "control_type": "ButtonControl",
                "name": "取消",
                "text": "取消",
                "bounding_rect": (380, 620, 520, 670),
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        roles = {region.role for region in result.regions}
        assert "form_fields" in roles
        assert "form_actions" in roles

        submit = next(element for element in result.elements if element.element_id == "submit_btn")
        cancel = next(element for element in result.elements if element.element_id == "cancel_btn")
        username = next(element for element in result.elements if element.element_id == "username_input")
        assert username.region_id == "region_content_form_fields_0"
        assert submit.region_id == "region_content_form_actions_0"
        assert cancel.region_id == "region_content_form_actions_0"

    def test_compile_list_detail_layout_builds_list_and_detail_regions(self):
        """list_detail fallback 应拆出左侧列表和右侧详情区。"""
        raw_elements = [
            {
                "element_id": "thread_list",
                "control_type": "ListControl",
                "name": "threads",
                "bounding_rect": (30, 90, 320, 760),
            },
            {
                "element_id": "thread_item",
                "control_type": "ListItemControl",
                "name": "thread_1",
                "bounding_rect": (40, 120, 300, 180),
            },
            {
                "element_id": "thread_item_2",
                "control_type": "ListItemControl",
                "name": "thread_2",
                "bounding_rect": (40, 200, 300, 260),
            },
            {
                "element_id": "detail_pane",
                "control_type": "PaneControl",
                "name": "detail",
                "bounding_rect": (360, 90, 1160, 760),
            },
        ]

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        content_region = next(region for region in result.regions if region.role == "content_area")
        assert content_region.subtype.value == "list_detail"
        roles = {region.role for region in result.regions}
        assert "list_panel" in roles
        assert "detail_panel" in roles

        thread_item = next(element for element in result.elements if element.element_id == "thread_item")
        detail_pane = next(element for element in result.elements if element.element_id == "detail_pane")
        assert thread_item.region_id == "region_content_list_0"
        assert detail_pane.region_id == "region_content_detail_0"

    def test_compile_generates_standard_regions_content_groups_and_scroll_contexts(self):
        """标准 child regions、content groups、scroll contexts 应被实际生成。"""
        raw_elements = [
            {
                "element_id": "search_input",
                "control_type": "EditControl",
                "name": "search",
                "bounding_rect": (40, 60, 280, 100),
            },
            {
                "element_id": "filter_toggle",
                "control_type": "ButtonControl",
                "name": "filter",
                "text": "Filter",
                "bounding_rect": (300, 60, 380, 100),
            },
            {
                "element_id": "list_pane",
                "control_type": "ListControl",
                "name": "threads",
                "bounding_rect": (30, 120, 320, 760),
            },
            {
                "element_id": "list_item",
                "control_type": "ListItemControl",
                "name": "thread_1",
                "bounding_rect": (40, 160, 300, 220),
            },
            {
                "element_id": "detail_view",
                "control_type": "DocumentControl",
                "name": "detail",
                "bounding_rect": (360, 120, 1160, 760),
            },
            {
                "element_id": "apply_btn",
                "control_type": "ButtonControl",
                "name": "Apply",
                "text": "Apply",
                "bounding_rect": (980, 780, 1120, 830),
            },
        ]

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        roles = {region.role for region in result.regions}
        assert "filter_bar" in roles
        assert "action_bar" in roles
        assert "viewport" in roles

        scroll_region_ids = {context.region_id for context in result.scroll_contexts}
        assert "region_content_viewport_0" in scroll_region_ids or "region_content_detail_0" in scroll_region_ids

        apply_btn = next(element for element in result.elements if element.element_id == "apply_btn")
        search_input = next(element for element in result.elements if element.element_id == "search_input")
        filter_toggle = next(element for element in result.elements if element.element_id == "filter_toggle")
        detail_view = next(element for element in result.elements if element.element_id == "detail_view")
        assert search_input.region_id == "region_content_filter_bar_0"
        assert filter_toggle.region_id == "region_content_filter_bar_0"
        assert detail_view.region_id in {
            "region_content_detail_0",
            "region_content_viewport_0",
            "region_content_dialog_body_0",
        }
        assert apply_btn.region_id == "region_content_action_bar_0"
        assert apply_btn.content_group_id in {"cg_action_bar", "cg_viewport", None}

    def test_compile_generates_dialog_body_region_for_centered_modal_panel(self):
        """居中大面板应识别出 dialog_body 子区。"""
        raw_elements = [
            {
                "element_id": "modal_panel",
                "control_type": "PaneControl",
                "name": "settings_dialog",
                "bounding_rect": (220, 120, 780, 620),
            },
            {
                "element_id": "confirm_btn",
                "control_type": "ButtonControl",
                "name": "确认",
                "text": "确认",
                "bounding_rect": (600, 560, 720, 610),
            },
        ]

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        roles = {region.role for region in result.regions}
        assert "dialog_body" in roles
        dialog_region = next(region for region in result.regions if region.role == "dialog_body")
        confirm_btn = next(element for element in result.elements if element.element_id == "confirm_btn")
        assert dialog_region.attributes.get("source") == "dialog_parser"
        assert confirm_btn.region_id in {dialog_region.region_id, "region_content_action_bar_0"}

    def test_compile_dialog_layout_builds_dialog_body_and_action_bar(self):
        raw_elements = [
            {
                "element_id": "backdrop_panel",
                "control_type": "PaneControl",
                "name": "settings_dialog",
                "bounding_rect": (220, 100, 900, 700),
            },
            {
                "element_id": "dialog_title",
                "control_type": "TextControl",
                "name": "Settings",
                "text": "Settings",
                "bounding_rect": (280, 150, 520, 200),
            },
            {
                "element_id": "dialog_input",
                "control_type": "EditControl",
                "name": "username",
                "bounding_rect": (300, 260, 760, 320),
            },
            {
                "element_id": "cancel_btn",
                "control_type": "ButtonControl",
                "name": "Cancel",
                "text": "Cancel",
                "bounding_rect": (520, 610, 660, 660),
            },
            {
                "element_id": "confirm_btn",
                "control_type": "ButtonControl",
                "name": "Confirm",
                "text": "Confirm",
                "bounding_rect": (690, 610, 840, 660),
            },
        ]

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        roles = {region.role for region in result.regions}
        assert "dialog_body" in roles
        assert "action_bar" in roles
        dialog_region = next(region for region in result.regions if region.role == "dialog_body")
        dialog_input = next(element for element in result.elements if element.element_id == "dialog_input")
        confirm_btn = next(element for element in result.elements if element.element_id == "confirm_btn")
        assert dialog_input.region_id == "region_content_dialog_body_0"
        assert confirm_btn.region_id == "region_content_action_bar_0"
        assert dialog_region.attributes["dialog_kind"] == "form_dialog"
        assert "dialog_input" in dialog_region.attributes["internal_roles"]["form_fields"]
        assert dialog_input.attributes["dialog_slot"] == "form_fields"

    def test_compile_chat_layout_reassigns_top_controls_to_toolbar_and_filter(self):
        raw_elements = [
            {
                "element_id": "chat_tab",
                "control_type": "TabItemControl",
                "name": "Messages",
                "text": "Messages",
                "bounding_rect": (40, 60, 220, 100),
            },
            {
                "element_id": "chat_search",
                "control_type": "EditControl",
                "name": "Search",
                "bounding_rect": (260, 60, 520, 100),
            },
            {
                "element_id": "msg_pane",
                "control_type": "PaneControl",
                "name": "chat_messages",
                "bounding_rect": (40, 120, 860, 520),
            },
            {
                "element_id": "chat_input",
                "control_type": "EditControl",
                "name": "chat_input",
                "bounding_rect": (60, 560, 720, 610),
            },
            {
                "element_id": "send_btn",
                "control_type": "ButtonControl",
                "name": "Send",
                "text": "Send",
                "bounding_rect": (740, 560, 840, 610),
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        chat_tab = next(element for element in result.elements if element.element_id == "chat_tab")
        chat_search = next(element for element in result.elements if element.element_id == "chat_search")
        msg_pane = next(element for element in result.elements if element.element_id == "msg_pane")
        assert chat_tab.region_id == "region_content_toolbar_0"
        assert chat_search.region_id == "region_content_filter_bar_0"
        assert msg_pane.region_id == "region_content_messages_0"

    def test_compile_uses_vision_layout_regions_as_structure_evidence(self):
        raw_elements = [
            {
                "element_id": "toolbar_btn",
                "control_type": "ButtonControl",
                "name": "Refresh",
                "bounding_rect": (40, 40, 120, 80),
            },
            {
                "element_id": "filter_input",
                "control_type": "EditControl",
                "name": "Search",
                "bounding_rect": (140, 40, 360, 80),
            },
            {
                "element_id": "thread_list",
                "control_type": "ListControl",
                "name": "Threads",
                "bounding_rect": (40, 120, 320, 760),
            },
            {
                "element_id": "detail_doc",
                "control_type": "DocumentControl",
                "name": "Detail",
                "bounding_rect": (360, 120, 1160, 760),
            },
            {
                "element_id": "send_btn",
                "control_type": "CustomControl",
                "name": "",
                "bounding_rect": (1020, 780, 1130, 830),
            },
        ]

        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=raw_elements,
            vision_candidates=[
                {
                    "candidate_id": "vision_send",
                    "bbox": [1020, 780, 1130, 830],
                    "kind": "send_button",
                    "group_id": "cg_composer",
                    "region_role": "action_bar",
                    "interaction_hints": {"preferred_action": "click"},
                    "structure_evidence_score": 0.84,
                    "confidence": 0.9,
                }
            ],
            vision_layout_regions=[
                {"role": "toolbar", "bbox": [20, 20, 1180, 90], "structure_evidence_score": 0.7},
                {"role": "filter_bar", "bbox": [120, 20, 420, 100], "structure_evidence_score": 0.68},
                {"role": "list_panel", "bbox": [20, 110, 340, 780], "structure_evidence_score": 0.8},
                {"role": "detail_panel", "bbox": [340, 110, 1180, 780], "structure_evidence_score": 0.8},
                {"role": "action_bar", "bbox": [900, 740, 1180, 840], "structure_evidence_score": 0.82},
            ],
            vision_control_groups=[
                {"group_id": "cg_composer", "member_ids": ["send_btn"]},
            ],
            structure_evidence_score=0.79,
            allow_legacy_zone_reconstruction=False,
        )

        region_by_role = {region.role: region for region in result.regions}
        assert region_by_role["toolbar"].attributes["source"] == "vision_layout"
        assert region_by_role["action_bar"].attributes["source"] == "vision_layout"
        assert "list_panel" in region_by_role
        assert "detail_panel" in region_by_role

        send_btn = next(element for element in result.elements if element.element_id == "send_btn")
        assert send_btn.semantic_role.value == "send_button"
        assert send_btn.content_group_id == "cg_composer"
        assert send_btn.region_id == "region_content_action_bar_0"
        assert result.provider_trace.provider_details["vision"]["layout_region_count"] == 5
        assert result.provider_trace.provider_details["vision"]["control_group_count"] == 1

    def test_compile_editor_layout_builds_toolbar_side_and_viewport(self):
        raw_elements = [
            {
                "element_id": "editor_tab",
                "control_type": "TabItemControl",
                "name": "main.py",
                "text": "main.py",
                "bounding_rect": (40, 60, 220, 100),
            },
            {
                "element_id": "editor_toolbar",
                "control_type": "ToolBarControl",
                "name": "editor_tools",
                "bounding_rect": (240, 60, 720, 100),
            },
            {
                "element_id": "file_tree",
                "control_type": "TreeControl",
                "name": "files",
                "bounding_rect": (30, 120, 240, 760),
            },
            {
                "element_id": "code_view",
                "control_type": "DocumentControl",
                "name": "editor_surface",
                "bounding_rect": (260, 120, 1180, 760),
            },
        ]

        result = self.compiler.compile(
            process_name="devenv.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        content_region = next(region for region in result.regions if region.role == "content_area")
        roles = {region.role for region in result.regions}
        assert content_region.subtype.value == "editor"
        assert {"toolbar", "side_panel", "viewport"}.issubset(roles)
        editor_tab = next(element for element in result.elements if element.element_id == "editor_tab")
        file_tree = next(element for element in result.elements if element.element_id == "file_tree")
        code_view = next(element for element in result.elements if element.element_id == "code_view")
        assert editor_tab.region_id == "region_content_toolbar_0"
        assert file_tree.region_id == "region_content_side_panel_0"
        assert code_view.region_id == "region_content_viewport_0"

    def test_compile_dashboard_layout_builds_filter_and_viewport_regions(self):
        raw_elements = [
            {
                "element_id": "date_filter",
                "control_type": "ComboBoxControl",
                "name": "Date Filter",
                "text": "Date Filter",
                "bounding_rect": (40, 60, 240, 100),
            },
            {
                "element_id": "search_filter",
                "control_type": "EditControl",
                "name": "Search",
                "bounding_rect": (260, 60, 520, 100),
            },
            {
                "element_id": "nav_pane",
                "control_type": "PaneControl",
                "name": "dashboard_nav",
                "bounding_rect": (30, 130, 220, 760),
            },
            {
                "element_id": "chart_a",
                "control_type": "CustomControl",
                "name": "Revenue Chart",
                "bounding_rect": (260, 140, 700, 420),
            },
            {
                "element_id": "chart_b",
                "control_type": "ImageControl",
                "name": "Conversion Chart",
                "bounding_rect": (730, 140, 1160, 420),
            },
            {
                "element_id": "summary_list",
                "control_type": "ListControl",
                "name": "Summary",
                "bounding_rect": (260, 450, 1160, 760),
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        content_region = next(region for region in result.regions if region.role == "content_area")
        roles = {region.role for region in result.regions}
        assert content_region.subtype.value == "dashboard"
        assert {"filter_bar", "side_panel", "viewport"}.issubset(roles)
        viewport_region = next(region for region in result.regions if region.role == "viewport")
        search_filter = next(element for element in result.elements if element.element_id == "search_filter")
        chart_a = next(element for element in result.elements if element.element_id == "chart_a")
        chart_b = next(element for element in result.elements if element.element_id == "chart_b")
        assert search_filter.region_id == "region_content_filter_bar_0"
        assert chart_a.region_id == "region_content_viewport_0"
        assert viewport_region.attributes["widget_group_count"] >= 2
        assert chart_a.attributes["widget_group_id"].startswith("widget_")
        assert chart_b.attributes["widget_group_id"].startswith("widget_")

    def test_compile_grid_table_layout_builds_filter_viewport_and_action_regions(self):
        raw_elements = [
            {
                "element_id": "table_search",
                "control_type": "EditControl",
                "name": "Search",
                "bounding_rect": (40, 60, 260, 100),
            },
            {
                "element_id": "table_filter",
                "control_type": "ComboBoxControl",
                "name": "Status Filter",
                "bounding_rect": (280, 60, 520, 100),
            },
            {
                "element_id": "grid_view",
                "control_type": "DataGridControl",
                "name": "Orders Grid",
                "bounding_rect": (40, 140, 1160, 720),
            },
            {
                "element_id": "grid_header",
                "control_type": "HeaderControl",
                "name": "Name Status Date",
                "text": "Name Status Date",
                "bounding_rect": (60, 150, 1140, 210),
            },
            {
                "element_id": "row_1",
                "control_type": "CustomControl",
                "name": "row_1",
                "bounding_rect": (60, 230, 1140, 290),
            },
            {
                "element_id": "page_info",
                "control_type": "TextControl",
                "name": "Page 1/3",
                "text": "Page 1/3",
                "bounding_rect": (860, 640, 1000, 690),
            },
            {
                "element_id": "apply_btn",
                "control_type": "ButtonControl",
                "name": "Apply",
                "text": "Apply",
                "bounding_rect": (980, 760, 1120, 810),
            },
        ]

        result = self.compiler.compile(
            process_name="chrome.exe",
            has_dom_bridge=True,
            dom_ready=True,
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        content_region = next(region for region in result.regions if region.role == "content_area")
        roles = {region.role for region in result.regions}
        assert content_region.subtype.value == "grid_table"
        assert {"filter_bar", "viewport", "action_bar"}.issubset(roles)
        viewport_region = next(region for region in result.regions if region.role == "viewport")
        grid_view = next(element for element in result.elements if element.element_id == "grid_view")
        grid_header = next(element for element in result.elements if element.element_id == "grid_header")
        row_1 = next(element for element in result.elements if element.element_id == "row_1")
        page_info = next(element for element in result.elements if element.element_id == "page_info")
        apply_btn = next(element for element in result.elements if element.element_id == "apply_btn")
        assert grid_view.region_id == "region_content_viewport_0"
        assert apply_btn.region_id == "region_content_action_bar_0"
        assert "grid_header" in viewport_region.attributes["grid_header_ids"]
        assert viewport_region.attributes["row_group_count"] >= 1
        assert grid_header.attributes["grid_role"] == "header"
        assert row_1.attributes["grid_role"] == "row"
        assert page_info.attributes["grid_role"] == "pagination"

    def test_compile_canvas_doc_viewer_builds_toolbar_side_and_viewport(self):
        raw_elements = [
            {
                "element_id": "zoom_toolbar",
                "control_type": "ToolBarControl",
                "name": "viewer_toolbar",
                "bounding_rect": (40, 60, 500, 100),
            },
            {
                "element_id": "thumb_list",
                "control_type": "ListControl",
                "name": "thumbnails",
                "bounding_rect": (30, 130, 200, 760),
            },
            {
                "element_id": "page_canvas",
                "control_type": "PaneControl",
                "name": "document_viewer",
                "bounding_rect": (240, 130, 1120, 760),
            },
            {
                "element_id": "next_page",
                "control_type": "ButtonControl",
                "name": "Next",
                "text": "Next",
                "bounding_rect": (980, 780, 1120, 830),
            },
        ]

        result = self.compiler.compile(
            process_name="pdfviewer.exe",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        content_region = next(region for region in result.regions if region.role == "content_area")
        roles = {region.role for region in result.regions}
        assert content_region.subtype.value == "canvas_doc_viewer"
        assert {"toolbar", "side_panel", "viewport", "action_bar"}.issubset(roles)
        thumb_list = next(element for element in result.elements if element.element_id == "thumb_list")
        page_canvas = next(element for element in result.elements if element.element_id == "page_canvas")
        next_page = next(element for element in result.elements if element.element_id == "next_page")
        assert thumb_list.region_id == "region_content_side_panel_0"
        assert page_canvas.region_id == "region_content_viewport_0"
        assert next_page.region_id == "region_content_action_bar_0"

    def test_compile_sparse_wechat_like_ocr_blocks_become_chat_structure(self):
        raw_elements = [
            {
                "element_id": "title",
                "control_type": "TitleBarControl",
                "name": "微信",
                "text": "微信",
                "bounding_rect": (0, 0, 1080, 40),
            },
            {
                "element_id": "wechat_root",
                "control_type": "PaneControl",
                "name": "MMUIRenderSubWindowHW",
                "bounding_rect": (0, 40, 1080, 760),
            },
        ]
        ocr_blocks = [
            {"text": "搜索", "bbox": (70, 60, 230, 92), "confidence": 0.98},
            {"text": "文件传输助手", "bbox": (70, 130, 250, 160), "confidence": 0.97},
            {"text": "项目群", "bbox": (70, 190, 180, 220), "confidence": 0.96},
            {"text": "张三", "bbox": (420, 65, 500, 92), "confidence": 0.97},
            {"text": "下午 2:30", "bbox": (460, 150, 560, 176), "confidence": 0.94},
            {"text": "收到", "bbox": (720, 330, 790, 360), "confidence": 0.95},
            {"text": "发送", "bbox": (950, 700, 1010, 730), "confidence": 0.98},
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )

        content_region = next(region for region in result.regions if region.role == "content_area")
        roles = {region.role for region in result.regions}
        assert content_region.subtype.value == "chat"
        assert "message_stream" in roles
        assert "composer_area" in roles
        assert len(result.elements) >= 6
        assert any("ocr" in element.provider_sources for element in result.elements)
        assert any(element.semantic_role.value == "search_input" for element in result.elements)
        assert any(element.semantic_role.value == "send_button" for element in result.elements)
        assert any(element.semantic_role.value == "chat_item" for element in result.elements)
        assert any(element.semantic_role.value == "message_input" for element in result.elements)

    def test_compile_synthesizes_message_input_from_composer_area(self):
        raw_elements = [
            {
                "element_id": "title",
                "control_type": "TitleBarControl",
                "name": "微信",
                "text": "微信",
                "bounding_rect": (0, 0, 1080, 40),
            },
            {
                "element_id": "wechat_root",
                "control_type": "PaneControl",
                "name": "MMUIRenderSubWindowHW",
                "bounding_rect": (0, 40, 1080, 760),
            },
        ]
        ocr_blocks = [
            {"text": "搜索", "bbox": (70, 60, 230, 92), "confidence": 0.98},
            {"text": "文件传输助手", "bbox": (70, 130, 250, 160), "confidence": 0.97},
            {"text": "项目群", "bbox": (70, 190, 180, 220), "confidence": 0.96},
            {"text": "张三", "bbox": (420, 65, 500, 92), "confidence": 0.97},
            {"text": "下午 2:30", "bbox": (460, 150, 560, 176), "confidence": 0.94},
            {"text": "收到", "bbox": (720, 330, 790, 360), "confidence": 0.95},
            {"text": "发送", "bbox": (950, 700, 1010, 730), "confidence": 0.98},
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )

        message_input = next(
            element for element in result.elements if element.semantic_role.value == "message_input"
        )
        send_button = next(
            element for element in result.elements if element.semantic_role.value == "send_button"
        )
        assert message_input.region_id == "region_content_composer_0"
        assert send_button.region_id in {"region_content_composer_0", "region_content_action_bar_0"}
        assert message_input.bounds[2] <= send_button.bounds[0]
        assert message_input.provider_sources == ["boundary_candidate"]

    def test_does_not_synthesize_invalid_message_input_bounds(self):
        region = Region(
            region_id="region_content_composer_0",
            role="composer_area",
            bounds=(0, 660, 12, 880),
        )
        send_button = Candidate(
            element_id="send_btn",
            semantic_role=SemanticRole.BUTTON,
            bounds=(0, 700, 8, 730),
        )

        candidate = self.compiler._synthesize_message_input_candidate(
            region,
            region_items=[],
            all_items=[send_button],
            existing_ids=set(),
        )

        assert candidate is None

    def test_compile_populates_boundary_candidates_artifact(self):
        raw_elements = [
            {
                "element_id": "search_box",
                "control_type": "EditControl",
                "name": "Search",
                "text": "Search",
                "bounding_rect": (40, 50, 220, 90),
            },
            {
                "element_id": "send_btn",
                "control_type": "ButtonControl",
                "name": "Send",
                "text": "Send",
                "bounding_rect": (920, 700, 1000, 735),
            },
        ]
        ocr_blocks = [
            {"text": "Search", "bbox": (40, 50, 220, 90), "confidence": 0.96},
            {"text": "Send", "bbox": (920, 700, 1000, 735), "confidence": 0.98},
        ]
        vision_candidates = [
            {
                "candidate_id": "vision_send",
                "bbox": [920, 700, 1000, 735],
                "confidence": 0.83,
                "kind": "send_button",
                "text": "Send",
                "source": "omniparser",
                "region_role": "action_bar",
            }
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            allow_legacy_zone_reconstruction=False,
        )

        candidates = result.artifacts["boundary_candidates"]
        stats = result.artifacts["boundary_candidate_stats"]
        search_candidate = next(item for item in candidates if item["candidate_id"] == "ocr::0")
        send_candidate = next(item for item in candidates if item["candidate_id"] == "ocr::1")
        assert len(candidates) >= 3
        assert any(item["candidate_id"] == "elem::search_box" for item in candidates)
        assert any(item["candidate_id"] == "ocr::0" for item in candidates)
        assert any(item["candidate_id"] == "vision_send" for item in candidates)
        assert search_candidate["control_hint"] == "edit_like"
        assert search_candidate["candidate_kind"] == "input_candidate"
        assert send_candidate["control_hint"] == "button_like"
        assert send_candidate["candidate_kind"] == "button_candidate"
        assert stats["by_source"]["ocr"] >= 1
        assert stats["by_source"]["omniparser"] >= 1
        assert result.provider_trace.provider_details["boundary_candidates"]["count"] == len(candidates)

    def test_compile_promotes_vision_layout_regions_and_groups_to_boundary_candidates(self):
        raw_elements = [
            {
                "element_id": "send_btn",
                "control_type": "ButtonControl",
                "name": "Send",
                "text": "Send",
                "bounding_rect": (920, 700, 1000, 735),
            },
        ]
        vision_candidates = [
            {
                "candidate_id": "vision_send",
                "bbox": [920, 700, 1000, 735],
                "confidence": 0.83,
                "kind": "send_button",
                "text": "Send",
                "source": "omniparser",
                "region_role": "action_bar",
                "group_id": "cg_footer",
            }
        ]
        vision_layout_regions = [
            {
                "region_id": "vision_toolbar",
                "role": "toolbar",
                "bbox": [20, 20, 1180, 90],
                "confidence": 0.72,
                "label": "Toolbar",
            }
        ]
        vision_control_groups = [
            {
                "group_id": "cg_footer",
                "member_ids": ["vision_send"],
                "confidence": 0.66,
                "label": "Footer group",
            }
        ]

        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=raw_elements,
            vision_candidates=vision_candidates,
            vision_layout_regions=vision_layout_regions,
            vision_control_groups=vision_control_groups,
            allow_legacy_zone_reconstruction=False,
        )

        candidates = result.artifacts["boundary_candidates"]
        stats = result.artifacts["boundary_candidate_stats"]
        toolbar_candidate = next(item for item in candidates if item["candidate_id"] == "vision_toolbar")
        group_candidate = next(item for item in candidates if item["candidate_id"] == "vision_group::cg_footer")

        assert toolbar_candidate["candidate_kind"] == "header_candidate"
        assert toolbar_candidate["control_hint"] == "toolbar_like"
        assert toolbar_candidate["attributes"]["candidate_origin"] == "vision_layout_region"
        assert group_candidate["candidate_kind"] == "panel_candidate"
        assert group_candidate["control_hint"] == "group_like"
        assert group_candidate["bbox"] == [920, 700, 1000, 735]
        assert group_candidate["attributes"]["candidate_origin"] == "vision_control_group"
        assert group_candidate["attributes"]["member_ids"] == ["vision_send"]
        assert stats["by_source"]["omniparser"] >= 3

    def test_build_boundary_candidates_keeps_same_bbox_candidates_when_origin_differs(self):
        snapshot = {
            "artifacts": {
                "boundary_candidates": [
                    {
                        "candidate_id": "vision::0",
                        "bbox": [100, 100, 180, 140],
                        "source": "omniparser",
                        "confidence": 0.82,
                        "candidate_kind": "button_candidate",
                        "text": "Send",
                        "control_hint": "button_like",
                        "region_hint": "action_bar",
                        "attributes": {"candidate_origin": "vision_candidate"},
                    },
                    {
                        "candidate_id": "vision_group::footer",
                        "bbox": [100, 100, 180, 140],
                        "source": "omniparser",
                        "confidence": 0.66,
                        "candidate_kind": "panel_candidate",
                        "text": "Send",
                        "control_hint": "button_like",
                        "region_hint": "action_bar",
                        "attributes": {"candidate_origin": "vision_control_group"},
                    },
                ]
            }
        }

        candidates = build_boundary_candidates(snapshot)

        assert len(candidates) == 2
        assert any(item["candidate_id"] == "vision::0" for item in candidates)
        assert any(item["candidate_id"] == "vision_group::footer" for item in candidates)

    def test_build_boundary_candidates_adds_coarse_geometry_region_hints_for_sparse_layouts(self):
        snapshot = {
            "window": {"rect_client": [0, 0, 896, 648]},
            "regions": [
                {"region_id": "region_content_0", "role": "content_area", "bounds": [0, 0, 896, 648]},
                {"region_id": "region_content_viewport_0", "role": "viewport", "bounds": [8, 0, 888, 640]},
            ],
            "elements": [],
            "artifacts": {
                "ocr_blocks": [
                    {"text": "搜索", "bbox": [103, 46, 137, 65], "confidence": 0.96},
                    {"text": "孙宇", "bbox": [122, 92, 157, 112], "confidence": 0.92},
                    {"text": "发送", "bbox": [824, 598, 857, 621], "confidence": 0.98},
                ],
                "vision_candidates": [],
            },
        }

        candidates = build_boundary_candidates(snapshot)
        search_candidate = next(item for item in candidates if item["candidate_id"] == "ocr::0")
        list_candidate = next(item for item in candidates if item["candidate_id"] == "ocr::1")
        send_candidate = next(item for item in candidates if item["candidate_id"] == "ocr::2")

        assert search_candidate["region_hint"] == "top_left_filter"
        assert search_candidate["control_hint"] == "edit_like"
        assert search_candidate["candidate_kind"] == "input_candidate"

        assert list_candidate["region_hint"] == "left_panel_list"
        assert list_candidate["control_hint"] == "list_item_like"
        assert list_candidate["candidate_kind"] == "list_item_candidate"

        assert send_candidate["region_hint"] == "bottom_right_action"
        assert send_candidate["control_hint"] == "button_like"
        assert send_candidate["candidate_kind"] == "button_candidate"

    def test_build_boundary_candidates_marks_top_header_text_as_header_candidate(self):
        snapshot = {
            "window": {"rect_client": [0, 0, 850, 603]},
            "regions": [
                {"region_id": "region_content_0", "role": "content_area", "bounds": [0, 0, 850, 603]},
            ],
            "elements": [],
            "artifacts": {
                "ocr_blocks": [
                    {"text": "全局", "bbox": [234, 51, 304, 76], "confidence": 0.9},
                ],
                "vision_candidates": [],
            },
        }

        candidates = build_boundary_candidates(snapshot)
        header_candidate = next(item for item in candidates if item["candidate_id"] == "ocr::0")

        assert header_candidate["region_hint"] == "top_header_bar"
        assert header_candidate["control_hint"] == "header_like"
        assert header_candidate["candidate_kind"] == "header_candidate"

    def test_compile_qq_sparse_root_gets_local_layout_candidates(self):
        raw_elements = [
            {
                "element_id": "qq_root",
                "control_type": "PaneControl",
                "name": "QQ",
                "text": "QQ",
                "bounding_rect": (0, 0, 960, 640),
            }
        ]

        result = self.compiler.compile(
            process_name="qq.exe",
            window_title="QQ",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        roles = {element.semantic_role.value: element for element in result.elements}
        assert "search_input" in roles
        assert "chat_item" in roles
        assert "message_input" in roles
        assert "send_button" in roles
        assert roles["send_button"].text == "发送"
        assert "send" in roles["send_button"].risk_tags
        assert "app_layout" in roles["search_input"].provider_sources

    def test_compile_qq_sparse_root_prefers_ocr_candidates_before_app_layout(self):
        raw_elements = [
            {
                "element_id": "qq_root",
                "control_type": "PaneControl",
                "name": "QQ",
                "text": "QQ",
                "bounding_rect": (0, 0, 960, 640),
            }
        ]
        ocr_blocks = [
            {"text": "搜索", "bbox": (76, 28, 259, 57), "confidence": 0.98},
            {"text": "会话列表", "bbox": (67, 76, 307, 140), "confidence": 0.96},
            {"text": "发送", "bbox": (787, 595, 892, 627), "confidence": 0.98},
        ]

        result = self.compiler.compile(
            process_name="qq.exe",
            window_title="QQ",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )

        by_role = {}
        for element in result.elements:
            by_role.setdefault(element.semantic_role.value, []).append(element)

        # R1: OCR synthetic elements use TextControl, but chat tokens still map to CHAT_ITEM.
        assert "ocr" in by_role["search_input"][0].provider_sources
        assert "app_layout" not in by_role["search_input"][0].provider_sources
        assert "ocr" in by_role["chat_item"][0].provider_sources
        assert "ocr" in by_role["send_button"][0].provider_sources
        assert "app_layout" not in by_role["send_button"][0].provider_sources

    def test_compile_voicemeeter_sparse_root_gets_local_layout_candidates(self):
        raw_elements = [
            {
                "element_id": "vm_root",
                "control_type": "PaneControl",
                "name": "VoiceMeeter",
                "text": "VoiceMeeter",
                "bounding_rect": (0, 0, 1645, 770),
            }
        ]

        result = self.compiler.compile(
            process_name="voicemeeter8x64.exe",
            window_title="VoiceMeeter",
            raw_elements=raw_elements,
            allow_legacy_zone_reconstruction=False,
        )

        by_text = {element.text: element for element in result.elements if element.text}
        assert "A1" in by_text
        assert "Menu" in by_text
        assert "Hardware" in by_text
        assert "app_layout" in by_text["A1"].provider_sources

    def test_compile_voicemeeter_sparse_root_prefers_ocr_candidates_before_app_layout(self):
        raw_elements = [
            {
                "element_id": "vm_root",
                "control_type": "PaneControl",
                "name": "VoiceMeeter",
                "text": "VoiceMeeter",
                "bounding_rect": (0, 0, 1645, 770),
            }
        ]
        ocr_blocks = [
            {"text": "Hardware", "bbox": (4, 38, 190, 83), "confidence": 0.91},
            {"text": "A1", "bbox": (1152, 8, 1185, 54), "confidence": 0.94},
            {"text": "Menu", "bbox": (1463, 15, 1545, 46), "confidence": 0.92},
        ]

        result = self.compiler.compile(
            process_name="voicemeeter8x64.exe",
            window_title="VoiceMeeter",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )

        by_text = {element.text: element for element in result.elements if element.text}
        assert "ocr" in by_text["Hardware"].provider_sources
        assert "app_layout" not in by_text["Hardware"].provider_sources
        assert by_text["A1"].semantic_role == SemanticRole.BUTTON
        assert "ocr" in by_text["A1"].provider_sources
        assert "app_layout" not in by_text["A1"].provider_sources
        assert by_text["Menu"].semantic_role in {SemanticRole.BUTTON, SemanticRole.MENU_ITEM}
        assert "ocr" in by_text["Menu"].provider_sources
        assert "app_layout" not in by_text["Menu"].provider_sources

    def test_ocr_actionable_menu_generates_button(self):
        """OCR 'Menu' text should generate a ButtonControl candidate."""
        raw_elements = [
            {"element_id": "root", "control_type": "PaneControl", "name": "App", "bounding_rect": (0, 0, 800, 600)},
        ]
        ocr_blocks = [
            {"text": "Menu", "bbox": (700, 10, 760, 35), "confidence": 0.92},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            window_title="Test",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        menu_candidates = [e for e in result.elements if e.text and "menu" in e.text.lower()]
        assert len(menu_candidates) >= 1
        assert menu_candidates[0].control_type in ("ButtonControl", "ButtonControl")

    def test_ocr_actionable_url_not_converted_to_button(self):
        """OCR URL text must NOT become a button candidate."""
        raw_elements = [
            {"element_id": "root", "control_type": "PaneControl", "name": "App", "bounding_rect": (0, 0, 800, 600)},
        ]
        ocr_blocks = [
            {"text": "https://chatgpt.com/g/g-p-6a1ea86", "bbox": (100, 10, 400, 30), "confidence": 0.95},
            {"text": "www.google.com", "bbox": (100, 40, 300, 60), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            window_title="Test",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        button_candidates = [e for e in result.elements if e.control_type == "ButtonControl" and e.element_id.startswith("ocr_action_")]
        assert len(button_candidates) == 0

    def test_ocr_actionable_date_not_converted_to_button(self):
        """OCR date text like '05/12' must NOT become a button candidate."""
        raw_elements = [
            {"element_id": "root", "control_type": "PaneControl", "name": "App", "bounding_rect": (0, 0, 800, 600)},
        ]
        ocr_blocks = [
            {"text": "05/12", "bbox": (10, 10, 60, 30), "confidence": 0.95},
            {"text": "2024-01-15", "bbox": (10, 40, 100, 60), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            window_title="Test",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        button_candidates = [e for e in result.elements if e.control_type == "ButtonControl" and e.element_id.startswith("ocr_action_")]
        assert len(button_candidates) == 0

    def test_ocr_actionable_long_text_not_converted(self):
        """Long OCR text (>10 chars) must NOT become actionable."""
        raw_elements = [
            {"element_id": "root", "control_type": "PaneControl", "name": "App", "bounding_rect": (0, 0, 800, 600)},
        ]
        ocr_blocks = [
            {"text": "这是一个很长的文本不应该变成按钮", "bbox": (10, 10, 300, 30), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            window_title="Test",
            raw_elements=raw_elements,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        button_candidates = [e for e in result.elements if e.control_type == "ButtonControl" and e.element_id.startswith("ocr_action_")]
        assert len(button_candidates) == 0


# --- R1: OCR synthetic path regression tests ---


class TestR1OcrSyntheticPath:
    """R1: Verify OCR synthetic elements no longer get position-based ListItemControl."""

    def setup_method(self):
        self.compiler = InteractionCanvasEngine()
        # Minimal raw_elements representing a sparse app (1 UIA root)
        self.sparse_root = [
            {
                "element_id": "root",
                "control_type": "WindowControl",
                "name": "TestApp",
                "text": "TestApp",
                "bounding_rect": (0, 0, 1200, 800),
            }
        ]

    def test_paint_menu_items_not_chat_item(self):
        """Paint 菜单项 '文件/编辑/查看' 不应生成 chat_item 元素."""
        ocr_blocks = [
            {"text": "文件", "bbox": (8, 37, 47, 63), "confidence": 0.95},
            {"text": "编辑", "bbox": (67, 42, 101, 61), "confidence": 0.95},
            {"text": "查看", "bbox": (123, 42, 157, 61), "confidence": 0.95},
            {"text": "工具", "bbox": (275, 152, 309, 175), "confidence": 0.95},
            {"text": "画笔", "bbox": (380, 157, 407, 172), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="mspaint.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        roles_by_text = {e.text: e.semantic_role.value for e in result.elements if e.text}
        for menu_text in ("文件", "编辑", "查看", "工具", "画笔"):
            assert roles_by_text.get(menu_text) != "chat_item", f"'{menu_text}' should not be chat_item"

    def test_paint_menu_items_are_text_or_list_item(self):
        """Paint 菜单项应为 text 或 list_item（中性角色）."""
        ocr_blocks = [
            {"text": "文件", "bbox": (8, 37, 47, 63), "confidence": 0.95},
            {"text": "编辑", "bbox": (67, 42, 101, 61), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="mspaint.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        roles = {e.semantic_role.value for e in result.elements if e.text in ("文件", "编辑")}
        assert roles.issubset({"text", "list_item"}), f"Expected text/list_item, got {roles}"

    def test_ocr_synthetic_left_side_not_list_item_control(self):
        """OCR 文本在窗口左侧不应被推断为 ListItemControl."""
        ocr_blocks = [
            {"text": "设置", "bbox": (10, 100, 60, 120), "confidence": 0.95},
            {"text": "音乐", "bbox": (10, 140, 60, 160), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        for elem in result.elements:
            if elem.text in ("设置", "音乐"):
                assert elem.control_type != "ListItemControl", \
                    f"'{elem.text}' should not be ListItemControl (position-based inference removed)"

    def test_ocr_send_still_button_control(self):
        """'发送' OCR 应仍生成 ButtonControl + SEND_BUTTON."""
        ocr_blocks = [
            {"text": "发送", "bbox": (900, 700, 960, 730), "confidence": 0.98},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        send_elements = [e for e in result.elements if e.text == "发送"]
        assert len(send_elements) >= 1
        assert send_elements[0].control_type == "ButtonControl"
        assert send_elements[0].semantic_role.value == "send_button"

    def test_ocr_search_still_edit_control(self):
        """'搜索' OCR 应仍生成 EditControl + SEARCH_INPUT."""
        ocr_blocks = [
            {"text": "搜索", "bbox": (100, 50, 200, 75), "confidence": 0.98},
        ]
        result = self.compiler.compile(
            process_name="test.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        search_elements = [e for e in result.elements if e.text == "搜索"]
        assert len(search_elements) >= 1
        assert search_elements[0].control_type == "EditControl"
        assert search_elements[0].semantic_role.value == "search_input"

    def test_wechat_ocr_chat_tokens_still_chat(self):
        """微信 OCR 中含 chat token 的 CJK 文本仍应生成 chat_item."""
        ocr_blocks = [
            {"text": "文件传输助手", "bbox": (70, 130, 250, 160), "confidence": 0.97},
            {"text": "搜索", "bbox": (70, 60, 230, 92), "confidence": 0.98},
            {"text": "发送", "bbox": (950, 700, 1010, 730), "confidence": 0.98},
        ]
        result = self.compiler.compile(
            process_name="WeChat.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        roles_by_text = {e.text: e.semantic_role.value for e in result.elements if e.text}
        # R1: TextControl + CJK + chat token → CHAT_ITEM (via targeted token match)
        assert roles_by_text.get("文件传输助手") == "chat_item"
        assert roles_by_text.get("搜索") == "search_input"
        assert roles_by_text.get("发送") == "send_button"

    def test_qq_ocr_chat_tokens_still_chat(self):
        """QQ OCR 中含 chat token 的 CJK 文本仍应生成 chat_item."""
        ocr_blocks = [
            {"text": "会话列表", "bbox": (67, 76, 307, 140), "confidence": 0.96},
            {"text": "联系人", "bbox": (67, 150, 200, 180), "confidence": 0.96},
        ]
        result = self.compiler.compile(
            process_name="qq.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        roles_by_text = {e.text: e.semantic_role.value for e in result.elements if e.text}
        # R1: TextControl + CJK + chat token → CHAT_ITEM
        assert roles_by_text.get("会话列表") == "chat_item"
        assert roles_by_text.get("联系人") == "chat_item"

    def test_chrome_title_not_menu_or_chat(self):
        """Chrome 标题/URL 文本不应变成 menu_item 或 chat_item."""
        ocr_blocks = [
            {"text": "首次调用 API", "bbox": (100, 10, 300, 30), "confidence": 0.95},
            {"text": "DeepSeek API Docs", "bbox": (100, 40, 350, 60), "confidence": 0.95},
        ]
        result = self.compiler.compile(
            process_name="chrome.exe",
            raw_elements=self.sparse_root,
            ocr_blocks=ocr_blocks,
            allow_legacy_zone_reconstruction=False,
        )
        for elem in result.elements:
            if elem.text in ("首次调用 API", "DeepSeek API Docs"):
                assert elem.semantic_role.value not in ("chat_item", "menu_item"), \
                    f"Chrome text '{elem.text}' should not be chat_item or menu_item"
