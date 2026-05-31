"""Tests for VLM Semantic Modeler schema — 数据类 + 校验。"""

import pytest

from src.vlm.schema import (
    AppIdentity,
    CandidateCorrection,
    Control,
    DynamicZone,
    PageSemanticModel,
    PageState,
    Region,
    Transition,
    ValidationError,
    VLMSemanticModelerResult,
    validate_page_semantic_model,
)


# =============================================================================
# Helpers
# =============================================================================


def _valid_raw() -> dict:
    """返回一份合法的 raw JSON dict。"""
    return {
        "schema_version": "1.0",
        "app_identity": {"app_name": "微信", "app_id": "wechat", "surface_type": "native_uia"},
        "page_state": {"page_class": "chat", "state_label": "聊天窗口-文本输入态", "state_flags": ["has_input_focus"]},
        "regions": [
            {"region_id": "top_bar", "role": "navigation", "bounds": [0, 0, 800, 40], "purpose": "顶部导航栏"},
            {"region_id": "chat_area", "role": "content", "bounds": [0, 40, 800, 600], "purpose": "消息列表区"},
            {"region_id": "input_area", "role": "input", "bounds": [0, 600, 800, 700], "purpose": "文本输入区"},
        ],
        "fixed_controls": [
            {
                "control_id": "ctrl_1",
                "region_id": "top_bar",
                "control_type": "button",
                "text": "+",
                "bounds": [10, 5, 35, 35],
                "semantic_role": "add_button",
                "visual_type": "icon",
                "interactable": True,
                "confidence": 0.9,
                "source_candidate_ids": ["uia_1"],
                "matches_candidate_id": "uia_1",
            },
            {
                "control_id": "ctrl_2",
                "region_id": "input_area",
                "control_type": "button",
                "text": "发送",
                "bounds": [700, 610, 790, 690],
                "semantic_role": "send_button",
                "visual_type": "button",
                "interactable": True,
                "confidence": 0.95,
                "source_candidate_ids": [],
                "matches_candidate_id": None,
            },
        ],
        "dynamic_zones": [
            {"zone_id": "chat_messages", "region_id": "chat_area", "content_type": "chat_messages", "bounds": [0, 40, 800, 600], "note": "消息列表，内容动态变化"},
        ],
        "candidate_corrections": [
            {"candidate_id": "ocr_5", "corrected_type": "icon", "corrected_role": "emoji_picker", "corrected_confidence": 0.8, "reason": "VLM 识别为表情图标而非文本"},
        ],
        "transitions": [
            {"from_state": "chat-文本输入态", "trigger": "click:发送按钮", "to_state": "chat-已发送态", "confidence": 0.7},
        ],
        "confidence": 0.85,
        "needs_review": False,
    }


# =============================================================================
# to_dict / from_dict roundtrip
# =============================================================================


class TestRoundtrip:
    def test_page_semantic_model_roundtrip(self):
        raw = _valid_raw()
        model, _ = validate_page_semantic_model(raw)
        d = model.to_dict()
        model2 = PageSemanticModel.from_dict(d)
        assert model2 == model

    def test_result_roundtrip(self):
        raw = _valid_raw()
        model, _ = validate_page_semantic_model(raw)
        result = VLMSemanticModelerResult(
            status="success",
            model=model,
            provider_name="openai",
            model_name="gpt-4o",
            token_input=1000,
            token_output=500,
            latency_ms=2000,
        )
        d = result.to_dict()
        result2 = VLMSemanticModelerResult.from_dict(d)
        assert result2.status == "success"
        assert result2.model is not None
        assert result2.model.app_identity.app_name == "微信"
        assert result2.token_input == 1000

    def test_sub_structures_roundtrip(self):
        ai = AppIdentity(app_name="Chrome", app_id="chrome", surface_type="browser")
        assert AppIdentity.from_dict(ai.to_dict()) == ai

    def test_app_identity_display_name_roundtrip(self):
        """Invariant 2: display_name survives from_dict/to_dict."""
        ai = AppIdentity(app_name="WeChat", app_id="wechat", surface_type="native_uia", display_name="微信")
        d = ai.to_dict()
        assert d["display_name"] == "微信"
        ai2 = AppIdentity.from_dict(d)
        assert ai2.display_name == "微信"
        assert ai2 == ai

    def test_app_identity_display_name_default_empty(self):
        """display_name defaults to empty string when absent from dict."""
        ai = AppIdentity.from_dict({"app_name": "Test"})
        assert ai.display_name == ""

        ps = PageState(page_class="editor", state_label="编辑态", state_flags=["has_input_focus"])
        assert PageState.from_dict(ps.to_dict()) == ps

        r = Region(region_id="toolbar", role="toolbar", bounds=[0, 0, 800, 40], purpose="工具栏")
        assert Region.from_dict(r.to_dict()) == r

        c = Control(control_id="c1", region_id="toolbar", control_type="button", text="Save", bounds=[10, 5, 60, 35], semantic_role="save_button", visual_type="button", interactable=True, confidence=0.9, source_candidate_ids=["a"], matches_candidate_id="a")
        assert Control.from_dict(c.to_dict()) == c

        z = DynamicZone(zone_id="body", region_id="content", content_type="article", bounds=[0, 40, 800, 600], note="文档内容")
        assert DynamicZone.from_dict(z.to_dict()) == z

        cc = CandidateCorrection(candidate_id="ocr_1", corrected_type="icon", corrected_role="emoji", corrected_confidence=0.8, reason="视觉识别")
        assert CandidateCorrection.from_dict(cc.to_dict()) == cc

        t = Transition(from_state="home", trigger="click:设置", to_state="settings", confidence=0.6)
        assert Transition.from_dict(t.to_dict()) == t


# =============================================================================
# validate_page_semantic_model
# =============================================================================


class TestValidation:
    def test_valid_full_model(self):
        model, warnings = validate_page_semantic_model(_valid_raw())
        assert model.app_identity.app_name == "微信"
        assert len(model.regions) == 3
        assert len(model.fixed_controls) == 2
        assert model.confidence == 0.85
        assert isinstance(warnings, list)

    def test_missing_required_field_regions(self):
        raw = _valid_raw()
        del raw["regions"]
        with pytest.raises(ValidationError, match="regions"):
            validate_page_semantic_model(raw)

    def test_empty_regions(self):
        raw = _valid_raw()
        raw["regions"] = []
        with pytest.raises(ValidationError, match="non-empty"):
            validate_page_semantic_model(raw)

    def test_invalid_bounds_not_4_elements(self):
        raw = _valid_raw()
        raw["regions"][0]["bounds"] = [0, 0, 800]
        with pytest.raises(ValidationError, match="bounds must be"):
            validate_page_semantic_model(raw)

    def test_invalid_bounds_left_gte_right(self):
        raw = _valid_raw()
        raw["regions"][0]["bounds"] = [200, 0, 100, 40]
        with pytest.raises(ValidationError, match="left < right"):
            validate_page_semantic_model(raw)

    def test_invalid_bounds_negative(self):
        raw = _valid_raw()
        raw["regions"][0]["bounds"] = [-1, 0, 800, 40]
        with pytest.raises(ValidationError, match="non-negative"):
            validate_page_semantic_model(raw)

    def test_bounds_exceed_image_size(self):
        raw = _valid_raw()
        # bounds [0, 0, 800, 700] but image_size=(400, 400) → strict failure
        with pytest.raises(ValidationError, match="exceeded image size"):
            validate_page_semantic_model(raw, image_size=(400, 400))

    def test_bounds_within_image_size(self):
        raw = _valid_raw()
        model, _ = validate_page_semantic_model(raw, image_size=(1000, 1000))
        assert model is not None

    def test_confidence_out_of_range(self):
        raw = _valid_raw()
        raw["confidence"] = 1.5
        with pytest.raises(ValidationError, match="confidence"):
            validate_page_semantic_model(raw)

    def test_schema_version_mismatch(self):
        raw = _valid_raw()
        raw["schema_version"] = "9.0"
        with pytest.raises(ValidationError, match="schema_version mismatch"):
            validate_page_semantic_model(raw)

    def test_transition_no_to_state(self):
        raw = _valid_raw()
        raw["transitions"][0]["to_state"] = ""
        model, warnings = validate_page_semantic_model(raw)
        assert len(model.transitions) == 0
        assert any("missing to_state" in w for w in warnings)

    def test_optional_shared_regions(self):
        raw = _valid_raw()
        raw["shared_regions"] = None
        raw["global_controls"] = None
        model, _ = validate_page_semantic_model(raw)
        assert model.shared_regions is None
        assert model.global_controls is None

    def test_source_candidate_ids_type(self):
        raw = _valid_raw()
        raw["fixed_controls"][0]["source_candidate_ids"] = [123]
        with pytest.raises(ValidationError, match="source_candidate_ids"):
            validate_page_semantic_model(raw)

    def test_unknown_candidate_id_warning(self):
        raw = _valid_raw()
        raw["fixed_controls"][0]["source_candidate_ids"] = ["unknown_xyz"]
        model, warnings = validate_page_semantic_model(raw, known_candidate_ids={"uia_1"})
        assert any("unknown candidate" in w for w in warnings)

    def test_unknown_matches_candidate_id_warning(self):
        raw = _valid_raw()
        raw["fixed_controls"][0]["matches_candidate_id"] = "nonexistent"
        model, warnings = validate_page_semantic_model(raw, known_candidate_ids={"uia_1"})
        assert any("nonexistent" in w for w in warnings)

    def test_dynamic_overlap_in_chat_warned(self):
        raw = _valid_raw()
        # 添加一个在 chat_messages 区域内的 fixed_control
        raw["fixed_controls"].append({
            "control_id": "ctrl_in_chat",
            "region_id": "chat_messages",
            "control_type": "button",
            "text": "回复",
            "bounds": [100, 200, 150, 230],
            "semantic_role": "reply_button",
            "visual_type": "button",
            "interactable": True,
            "confidence": 0.5,
        })
        _, warnings = validate_page_semantic_model(raw)
        assert any("dynamic content zone" in w for w in warnings)

    def test_dynamic_overlap_in_toolbar_allowed(self):
        raw = _valid_raw()
        # toolbar 区域内的 fixed_control 不应 warning
        raw["dynamic_zones"] = [
            {"zone_id": "toolbar", "region_id": "top_bar", "content_type": "toolbar", "bounds": [0, 0, 800, 40], "note": ""},
        ]
        raw["fixed_controls"][0]["region_id"] = "toolbar"
        _, warnings = validate_page_semantic_model(raw)
        assert not any("dynamic content zone" in w for w in warnings)
