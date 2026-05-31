"""Tests for page_identity — 页面模型与状态模板核心逻辑."""

import pytest

from src.memory.page_identity import (
    LayoutFingerprint,
    build_layout_fingerprint,
    build_layout_signature,
    classify_state_label,
    compute_fingerprint_similarity,
    extract_page_class_prefix,
    fingerprint_to_signature,
    normalize_page_class_for_memory,
)
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    ElementState,
    InteractionCanvas,
    PageInfo,
    Region,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_candidate(
    role: SemanticRole = SemanticRole.BUTTON,
    text: str = "",
    bounds: tuple[int, int, int, int] = (100, 100, 200, 150),
    element_id: str = "test_1",
    region_id: str = "",
) -> Candidate:
    return Candidate(
        element_id=element_id,
        text=text,
        semantic_role=role,
        bounds=bounds,
        region_id=region_id,
        visual_type="button",
        confidence=0.8,
        provider_sources=["uia"],
    )


def _make_canvas(
    elements: list[Candidate] | None = None,
    regions: list[Region] | None = None,
    state_flags: dict | None = None,
    surface_type: SurfaceType = SurfaceType.NATIVE_UIA,
    app_id: str = "test_app",
    process_name: str = "test.exe",
) -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="test_canvas",
        app=AppInfo(app_id=app_id, process_name=process_name),
        window=WindowInfoSnapshot(hwnd=12345, rect_client=(0, 0, 896, 648)),
        surface=SurfaceInfo(surface_type=surface_type),
        page=PageInfo(
            page_class="test_app/gui/main/default",
            state_flags=state_flags or {},
        ),
        regions=regions or [],
        elements=elements or [],
    )


def _make_fp(
    fixed_roles: tuple[str, ...] = ("button", "menu_item"),
    region_structure: tuple[tuple[str, int], ...] = (("content_area", 10),),
    state_flags: tuple[tuple[str, bool], ...] = (("editable", True),),
    bucket: str = "5-20",
    input_state: str = "unknown",
    has_dialog: bool = False,
    has_menu: bool = False,
) -> LayoutFingerprint:
    return LayoutFingerprint(
        fixed_roles=fixed_roles,
        region_structure=region_structure,
        state_flags=state_flags,
        element_count_bucket=bucket,
        input_state=input_state,
        has_dialog=has_dialog,
        has_menu=has_menu,
    )


# =============================================================================
# Test: extract_page_class_prefix
# =============================================================================


class TestExtractPageClassPrefix:
    def test_normal_case(self):
        assert extract_page_class_prefix("wechat/chat/main/default") == "wechat/chat"

    def test_two_segments(self):
        assert extract_page_class_prefix("notepad/editor") == "notepad/editor"

    def test_unknown_with_process(self):
        result = extract_page_class_prefix(
            "unknown/document/main/wide",
            process_name="notepad.exe",
            surface_type="native_uia",
        )
        # workflow 从 page_class 第二段取（"document"），不再回退到 surface_type
        assert result == "notepad/document"

    def test_unknown_without_process(self):
        result = extract_page_class_prefix("unknown/document/main/wide")
        # workflow 从 page_class 第二段取（"document"），不再回退到 surface_type
        assert result == "unknown/document"

    def test_empty_string(self):
        result = extract_page_class_prefix("", process_name="chrome.exe")
        assert result == "chrome/gui"


class TestNormalizePageClassForMemory:
    def test_qq_chat_visual_mode_uses_stable_chat_class(self):
        assert normalize_page_class_for_memory(
            "qq/viewer/main/wide",
            process_name="qq.exe",
            visual_mode="chat_workspace",
        ) == "qq/chat/main/wide"
        assert normalize_page_class_for_memory(
            "qq/dashboard/main/wide",
            process_name="qq.exe",
            visual_mode="chat_search_results",
        ) == "qq/chat/main/wide"

    def test_feishu_collaboration_visual_mode_uses_stable_collaboration_class(self):
        assert normalize_page_class_for_memory(
            "lark/form/main/wide",
            process_name="feishu.exe",
            visual_mode="collaboration_search_overlay",
        ) == "feishu/collaboration/main/wide"


# =============================================================================
# Test: build_layout_fingerprint
# =============================================================================


class TestBuildLayoutFingerprint:
    def test_same_canvas_same_fingerprint(self):
        elements = [
            _make_candidate(SemanticRole.BUTTON, "发送"),
            _make_candidate(SemanticRole.TEXT_INPUT, ""),
        ]
        canvas = _make_canvas(elements=elements)
        fp1 = build_layout_fingerprint(canvas)
        fp2 = build_layout_fingerprint(canvas)
        assert fp1 == fp2

    def test_input_empty_vs_filled(self):
        """输入框空 vs 有内容 → 不同 input_state。"""
        empty_elements = [_make_candidate(SemanticRole.TEXT_INPUT, "")]
        filled_elements = [_make_candidate(SemanticRole.TEXT_INPUT, "hello")]

        fp_empty = build_layout_fingerprint(_make_canvas(elements=empty_elements))
        fp_filled = build_layout_fingerprint(_make_canvas(elements=filled_elements))

        assert fp_empty.input_state == "empty"
        assert fp_filled.input_state == "filled"
        assert fp_empty != fp_filled

    def test_input_content_change_same_state(self):
        """输入框内容 123 → 456 → 同一 input_state（都是 filled）。"""
        elem_a = [_make_candidate(SemanticRole.TEXT_INPUT, "123")]
        elem_b = [_make_candidate(SemanticRole.TEXT_INPUT, "456")]

        fp_a = build_layout_fingerprint(_make_canvas(elements=elem_a))
        fp_b = build_layout_fingerprint(_make_canvas(elements=elem_b))

        assert fp_a.input_state == "filled"
        assert fp_b.input_state == "filled"
        assert fp_a.input_state == fp_b.input_state

    def test_synthetic_message_input_uses_overlapping_ocr_text(self):
        """合成输入框没有 text 时，可用框内 OCR 文本区分 filled 状态。"""
        input_element = _make_candidate(
            SemanticRole.MESSAGE_INPUT,
            "",
            bounds=(300, 610, 880, 705),
            element_id="synthetic_chat_composer_input",
            region_id="composer",
        )
        input_element.attributes["candidate_kind"] = "message_input_candidate"
        canvas = _make_canvas(elements=[input_element])
        canvas.artifacts["ocr_blocks"] = [
            {"bbox": [326, 630, 360, 656], "text": "123", "confidence": 0.94},
        ]

        fp = build_layout_fingerprint(canvas)

        assert fp.input_state == "filled"

    def test_input_state_ignores_ocr_text_outside_input_bounds(self):
        """消息区 OCR 文本不能污染底部输入框状态。"""
        input_element = _make_candidate(
            SemanticRole.MESSAGE_INPUT,
            "",
            bounds=(300, 610, 880, 705),
            element_id="synthetic_chat_composer_input",
            region_id="composer",
        )
        input_element.attributes["candidate_kind"] = "message_input_candidate"
        message = _make_candidate(
            SemanticRole.TEXT,
            "聊天消息 123",
            bounds=(360, 220, 620, 260),
            element_id="msg_1",
            region_id="messages",
        )
        canvas = _make_canvas(elements=[input_element, message])
        canvas.artifacts["ocr_blocks"] = [
            {"bbox": [360, 220, 620, 260], "text": "聊天消息 123", "confidence": 0.94},
        ]

        fp = build_layout_fingerprint(canvas)

        assert fp.input_state == "empty"

    def test_input_state_ignores_control_label_ocr_inside_input_bounds(self):
        """图标/控件标签被 OCR 到输入区内时，不应误判为已输入。"""
        input_element = _make_candidate(
            SemanticRole.MESSAGE_INPUT,
            "",
            bounds=(300, 610, 880, 705),
            element_id="synthetic_chat_composer_input",
            region_id="composer",
        )
        input_element.attributes["candidate_kind"] = "message_input_candidate"
        canvas = _make_canvas(elements=[input_element])
        canvas.artifacts["ocr_blocks"] = [
            {"bbox": [820, 650, 860, 680], "text": "发送", "confidence": 0.90},
        ]

        fp = build_layout_fingerprint(canvas)

        assert fp.input_state == "empty"

    def test_send_button_enabled_state_enters_fingerprint_flags(self):
        send_enabled = _make_candidate(
            SemanticRole.SEND_BUTTON,
            "发送",
            element_id="send",
        )
        send_enabled.state = ElementState(enabled=True, visible=True)
        send_disabled = _make_candidate(
            SemanticRole.SEND_BUTTON,
            "发送",
            element_id="send",
        )
        send_disabled.state = ElementState(enabled=False, visible=True)

        fp_enabled = build_layout_fingerprint(_make_canvas(elements=[send_enabled]))
        fp_disabled = build_layout_fingerprint(_make_canvas(elements=[send_disabled]))

        assert ("send_enabled", True) in fp_enabled.state_flags
        assert ("send_enabled", False) in fp_disabled.state_flags
        assert fp_enabled != fp_disabled

    def test_dialog_open_changes_fingerprint(self):
        """弹窗打开 → 不同指纹。"""
        fp_no = build_layout_fingerprint(_make_canvas(state_flags={}))
        fp_yes = build_layout_fingerprint(_make_canvas(state_flags={"dialog_open": True}))

        assert fp_no.has_dialog is False
        assert fp_yes.has_dialog is True
        assert fp_no != fp_yes

    def test_menu_open_changes_fingerprint(self):
        """菜单展开 → 不同指纹。"""
        fp_no = build_layout_fingerprint(_make_canvas(state_flags={}))
        fp_yes = build_layout_fingerprint(_make_canvas(state_flags={"menu_open": True}))

        assert fp_no.has_menu is False
        assert fp_yes.has_menu is True
        assert fp_no != fp_yes

    def test_dynamic_message_growth_does_not_change_fingerprint(self):
        """聊天消息新增不应改变 StateTemplate 指纹。"""
        fixed = [
            _make_candidate(SemanticRole.BUTTON, "发送", element_id="send", region_id="composer"),
            _make_candidate(SemanticRole.MESSAGE_INPUT, "", element_id="input", region_id="composer"),
        ]
        messages_a = [
            _make_candidate(SemanticRole.TEXT, f"msg {idx}", element_id=f"msg_{idx}", region_id="thread")
            for idx in range(10)
        ]
        messages_b = [
            _make_candidate(SemanticRole.TEXT, f"msg {idx}", element_id=f"msg_{idx}", region_id="thread")
            for idx in range(45)
        ]
        regions_a = [
            Region(region_id="thread", role="content_area", element_ids=[item.element_id for item in messages_a]),
            Region(region_id="composer", role="tool_bar", element_ids=["send", "input"]),
        ]
        regions_b = [
            Region(region_id="thread", role="content_area", element_ids=[item.element_id for item in messages_b]),
            Region(region_id="composer", role="tool_bar", element_ids=["send", "input"]),
        ]

        fp_a = build_layout_fingerprint(_make_canvas(elements=fixed + messages_a, regions=regions_a))
        fp_b = build_layout_fingerprint(_make_canvas(elements=fixed + messages_b, regions=regions_b))

        assert fp_a == fp_b
        assert fp_a.element_count_bucket == "0-5"
        assert fp_a.region_structure == (("content_area", 0), ("tool_bar", 2))


# =============================================================================
# Test: fingerprint_to_signature
# =============================================================================


class TestFingerprintToSignature:
    def test_same_fingerprint_same_signature(self):
        fp = _make_fp()
        assert fingerprint_to_signature(fp) == fingerprint_to_signature(fp)

    def test_different_fingerprint_different_signature(self):
        fp_a = _make_fp(input_state="empty")
        fp_b = _make_fp(input_state="filled")
        assert fingerprint_to_signature(fp_a) != fingerprint_to_signature(fp_b)


# =============================================================================
# Test: compute_fingerprint_similarity
# =============================================================================


class TestFingerprintSimilarity:
    def test_identical_returns_one(self):
        fp = _make_fp()
        assert compute_fingerprint_similarity(fp, fp) == pytest.approx(1.0, abs=0.01)

    def test_small_change_high_similarity(self):
        """输入框状态变化 → 相似度 = 0.70（状态相关字段是硬差异）。"""
        a = _make_fp(input_state="empty")
        b = _make_fp(input_state="filled")
        sim = compute_fingerprint_similarity(a, b)
        # input_state 不同 → 相似度被限制在 0.70（低于 0.80 的复用阈值）
        assert sim == pytest.approx(0.70, abs=0.01), f"Expected 0.70, got {sim:.3f}"

    def test_dialog_open_similarity(self):
        """弹窗打开 → 相似度 = 0.70（状态相关字段是硬差异）。"""
        a = _make_fp(has_dialog=False, state_flags=(("editable", True),))
        b = _make_fp(has_dialog=True, state_flags=(("editable", True), ("dialog_open", True),))
        sim = compute_fingerprint_similarity(a, b)
        # has_dialog 不同 → 相似度被限制在 0.70（低于 0.80 的复用阈值）
        assert sim == pytest.approx(0.70, abs=0.01), f"Expected 0.70, got {sim:.3f}"

    def test_completely_different_low_similarity(self):
        """完全不同的指纹 → 相似度 < 0.60。"""
        a = _make_fp(
            fixed_roles=("button",),
            region_structure=(("content_area", 5),),
            state_flags=(("editable", True),),
            bucket="5-20",
            input_state="empty",
        )
        b = _make_fp(
            fixed_roles=("menu_item", "toolbar", "sidebar"),
            region_structure=(("sidebar", 20), ("toolbar", 10),),
            state_flags=(("dialog_open", True),),
            bucket="50+",
            input_state="filled",
            has_dialog=True,
        )
        sim = compute_fingerprint_similarity(a, b)
        assert sim < 0.60, f"Expected < 0.60, got {sim:.3f}"

    def test_role_subset_high_similarity(self):
        """角色子集 → 相似度较高。"""
        a = _make_fp(fixed_roles=("button", "menu_item", "text_input"))
        b = _make_fp(fixed_roles=("button", "menu_item"))
        sim = compute_fingerprint_similarity(a, b)
        # Jaccard = 2/3 ≈ 0.67, 其他维度相同 → 总体 >= 0.80
        assert sim >= 0.75, f"Expected >= 0.75, got {sim:.3f}"


# =============================================================================
# Test: classify_state_label
# =============================================================================


class TestClassifyStateLabel:
    def test_dialog_open(self):
        canvas = _make_canvas(state_flags={"dialog_open": True})
        assert classify_state_label(canvas) == "弹窗打开"

    def test_menu_open(self):
        canvas = _make_canvas(state_flags={"menu_open": True})
        assert classify_state_label(canvas) == "菜单展开"

    def test_loading(self):
        canvas = _make_canvas(state_flags={"loading": True})
        assert classify_state_label(canvas) == "加载中"

    def test_empty_input(self):
        elements = [_make_candidate(SemanticRole.TEXT_INPUT, "")]
        canvas = _make_canvas(elements=elements)
        assert classify_state_label(canvas) == "输入框为空"

    def test_filled_input(self):
        elements = [_make_candidate(SemanticRole.MESSAGE_INPUT, "hello")]
        canvas = _make_canvas(elements=elements)
        assert classify_state_label(canvas) == "输入框有内容"

    def test_empty_page(self):
        canvas = _make_canvas(elements=[])
        assert classify_state_label(canvas) == "空页面"

    def test_main_page(self):
        elements = [_make_candidate(SemanticRole.BUTTON, "OK")] * 5
        canvas = _make_canvas(elements=elements)
        assert classify_state_label(canvas) == "主页面"


# =============================================================================
# Test: build_layout_signature
# =============================================================================


class TestBuildLayoutSignature:
    def test_same_canvas_same_signature(self):
        elements = [_make_candidate(SemanticRole.BUTTON, "OK")]
        canvas = _make_canvas(elements=elements)
        assert build_layout_signature(canvas) == build_layout_signature(canvas)

    def test_state_change_same_signature(self):
        """弹窗打开不应改变 layout_signature（关注页面结构，不关注状态）。"""
        elements = [_make_candidate(SemanticRole.BUTTON, "OK")]
        sig_a = build_layout_signature(_make_canvas(elements=elements, state_flags={}))
        sig_b = build_layout_signature(_make_canvas(elements=elements, state_flags={"dialog_open": True}))
        # layout_signature 基于固定控件角色和区域角色，不包含 state_flags
        assert sig_a == sig_b
