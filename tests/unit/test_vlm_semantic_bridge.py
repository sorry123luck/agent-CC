"""VLM Semantic Bridge 单元测试。"""

from __future__ import annotations

import pytest
from copy import deepcopy

from src.perception.vlm_semantic_bridge import VLMSemanticBridge, _iou, _vlm_bounds_to_tuple
from src.vlm.schema import (
    AppIdentity,
    CandidateCorrection,
    Control,
    DynamicZone,
    PageSemanticModel,
    PageState,
    Region as VLMRegion,
    Transition,
)
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    PageInfo,
    Region,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
)


def _make_canvas(
    app_id: str = "unknown",
    page_class: str = "",
    elements: list[Candidate] | None = None,
    regions: list[Region] | None = None,
) -> InteractionCanvas:
    canvas = InteractionCanvas()
    canvas.app = AppInfo(app_id=app_id)
    canvas.page = PageInfo(page_class=page_class)
    canvas.surface = SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA)
    canvas.elements = elements or []
    canvas.regions = regions or []
    return canvas


def _make_vlm_model(
    app_identity: AppIdentity | None = None,
    page_state: PageState | None = None,
    regions: list[VLMRegion] | None = None,
    controls: list[Control] | None = None,
    corrections: list[CandidateCorrection] | None = None,
    dynamic_zones: list[DynamicZone] | None = None,
) -> PageSemanticModel:
    return PageSemanticModel(
        app_identity=app_identity or AppIdentity(),
        page_state=page_state or PageState(),
        regions=regions or [],
        fixed_controls=controls or [],
        candidate_corrections=corrections or [],
        dynamic_zones=dynamic_zones or [],
    )


class TestIou:
    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_full_overlap(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_partial_overlap(self):
        result = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < result < 1.0

    def test_adjacent_no_overlap(self):
        assert _iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


class TestVlmBounds:
    def test_valid(self):
        assert _vlm_bounds_to_tuple([1, 2, 3, 4]) == (1, 2, 3, 4)

    def test_too_short(self):
        assert _vlm_bounds_to_tuple([1, 2]) is None

    def test_empty(self):
        assert _vlm_bounds_to_tuple([]) is None


class TestApplyAppIdentity:
    def test_updates_unknown_app_id(self):
        canvas = _make_canvas(app_id="unknown")
        model = _make_vlm_model(app_identity=AppIdentity(app_name="微信"))
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.app.app_id == "微信"

    def test_does_not_overwrite_existing(self):
        canvas = _make_canvas(app_id="wechat")
        model = _make_vlm_model(app_identity=AppIdentity(app_name="WeChat Pro"))
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.app.app_id == "wechat"

    def test_empty_app_name_noop(self):
        canvas = _make_canvas(app_id="unknown")
        model = _make_vlm_model(app_identity=AppIdentity(app_name=""))
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.app.app_id == "unknown"


class TestApplyPageState:
    def test_sets_page_class(self):
        canvas = _make_canvas()
        model = _make_vlm_model(page_state=PageState(page_class="chat", state_label="聊天窗口"))
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.page.page_class == "chat"

    def test_sets_state_flags(self):
        canvas = _make_canvas()
        model = _make_vlm_model(page_state=PageState(state_flags=["has_input_focus"]))
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.attributes.get("vlm_state_flags") == ["has_input_focus"]


class TestMergeRegions:
    def test_adds_new_region(self):
        canvas = _make_canvas()
        vlm_region = VLMRegion(region_id="r1", role="toolbar", bounds=[0, 0, 100, 50], purpose="顶部工具栏")
        model = _make_vlm_model(regions=[vlm_region])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert len(result.regions) == 1
        assert result.regions[0].role == "toolbar"
        assert result.regions[0].attributes.get("source") == "vlm"

    def test_merges_overlapping_region(self):
        existing = Region(region_id="er1", role="unknown", bounds=(10, 10, 100, 50))
        canvas = _make_canvas(regions=[existing])
        vlm_region = VLMRegion(region_id="vr1", role="toolbar", bounds=[10, 10, 100, 50], purpose="工具栏")
        model = _make_vlm_model(regions=[vlm_region])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert len(result.regions) == 1
        assert result.regions[0].role == "toolbar"
        assert result.regions[0].attributes.get("vlm_purpose") == "工具栏"

    def test_preserves_existing_role_on_overlap(self):
        existing = Region(region_id="er1", role="title_bar", bounds=(10, 10, 100, 50))
        canvas = _make_canvas(regions=[existing])
        vlm_region = VLMRegion(region_id="vr1", role="toolbar", bounds=[10, 10, 100, 50], purpose="标题栏")
        model = _make_vlm_model(regions=[vlm_region])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.regions[0].role == "title_bar"


class TestAddVlmCandidates:
    def test_adds_new_candidate(self):
        canvas = _make_canvas()
        ctrl = Control(
            control_id="c1", region_id="r1", control_type="Button",
            text="发送", bounds=[100, 200, 150, 230],
            semantic_role="send_button", interactable=True, confidence=0.9,
        )
        model = _make_vlm_model(controls=[ctrl])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert len(result.elements) == 1
        assert result.elements[0].text == "发送"
        assert result.elements[0].provider_sources == ["vlm"]

    def test_merges_overlapping_candidate(self):
        existing = Candidate(
            element_id="e1", bounds=(100, 200, 150, 230),
            semantic_role=SemanticRole.UNKNOWN, provider_sources=["uia"],
        )
        canvas = _make_canvas(elements=[existing])
        ctrl = Control(
            control_id="c1", bounds=[100, 200, 150, 230],
            semantic_role="send_button", confidence=0.9,
        )
        model = _make_vlm_model(controls=[ctrl])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert len(result.elements) == 1
        assert "vlm" in result.elements[0].provider_sources
        assert result.elements[0].attributes.get("vlm_semantic_role") == "send_button"


class TestApplyCorrections:
    def test_writes_correction_to_attributes(self):
        elem = Candidate(element_id="e1", attributes={})
        canvas = _make_canvas(elements=[elem])
        corr = CandidateCorrection(
            candidate_id="e1", corrected_role="send_button",
            corrected_confidence=0.95, reason="VLM 识别为发送按钮",
        )
        model = _make_vlm_model(corrections=[corr])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        vlm_corr = result.elements[0].attributes.get("vlm_correction")
        assert vlm_corr is not None
        assert vlm_corr["corrected_role"] == "send_button"
        assert vlm_corr["reason"] == "VLM 识别为发送按钮"


class TestMarkDynamicZones:
    def test_marks_overlapping_region(self):
        region = Region(region_id="r1", role="content_area", bounds=(0, 0, 500, 400))
        canvas = _make_canvas(regions=[region])
        zone = DynamicZone(zone_id="z1", content_type="chat_messages", bounds=[0, 0, 500, 400])
        model = _make_vlm_model(dynamic_zones=[zone])
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.regions[0].attributes.get("dynamic_zone") is True
        assert result.regions[0].attributes.get("vlm_content_type") == "chat_messages"


class TestPreservesOriginal:
    def test_original_not_mutated(self):
        original = _make_canvas(app_id="test_app")
        original.elements.append(Candidate(element_id="e1"))
        model = _make_vlm_model(app_identity=AppIdentity(app_name="Changed"))
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(original, model)
        assert original.app.app_id == "test_app"
        assert result.app.app_id == "test_app"


class TestEmptyModel:
    def test_empty_model_noop(self):
        canvas = _make_canvas(app_id="test")
        model = _make_vlm_model()
        bridge = VLMSemanticBridge()
        result = bridge.apply_to_canvas(canvas, model)
        assert result.app.app_id == "test"
        assert len(result.elements) == 0
        assert len(result.regions) == 0
