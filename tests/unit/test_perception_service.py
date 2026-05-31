"""Unit tests for the current Pro-baseline PerceptionService flow."""

from unittest.mock import MagicMock, patch

from PIL import Image

from src.perception.element_merger import MergedElement
from src.perception.perception_service import PerceptionService, ZonePageStructure
from src.perception.uia_client import UIAElementInfo
from src.perception.zone_partitioner import ZoneInfo, ZoneType


def make_elem(
    left: int,
    top: int,
    right: int,
    bottom: int,
    name: str | None = None,
    control_type: str = "PaneControl",
    automation_id: str | None = None,
    element_id: str | None = None,
) -> UIAElementInfo:
    return UIAElementInfo(
        name=name,
        automation_id=automation_id,
        control_type=control_type,
        bounding_rect=(left, top, right, bottom),
        is_enabled=True,
        element_id=element_id,
    )


def make_merged(
    left: int,
    top: int,
    right: int,
    bottom: int,
    control_type: str = "PaneControl",
    name: str | None = None,
    automation_id: str | None = None,
    element_id: str = "elem_0",
) -> MergedElement:
    elem = make_elem(
        left,
        top,
        right,
        bottom,
        name=name,
        control_type=control_type,
        automation_id=automation_id,
        element_id=f"{element_id}_uia",
    )
    return MergedElement(
        elements=[elem],
        bounding_rect=(left, top, right, bottom),
        control_type=control_type,
        name=name,
        is_merged=False,
        original_count=1,
        element_id=element_id,
        original_element_ids=[elem.element_id],
    )


def test_zone_page_structure_defaults():
    structure = ZonePageStructure()
    assert structure.zones == []
    assert structure.all_elements_merged == []
    assert structure.ocr_blocks == []
    assert structure.vision_candidates == []


def test_get_content_size_prefers_window_rect():
    svc = PerceptionService()
    uia_elements = [
        make_elem(-8, -8, 1928, 1040, control_type="WindowControl"),
        make_elem(0, 0, 1920, 1032, control_type="PaneControl"),
    ]
    window_info = type("WindowInfo", (), {"rect": (-8, -8, 1928, 1040)})()

    width, height = svc._get_content_size(uia_elements, window_info)

    assert width == 1936
    assert height == 1048


def test_create_page_snapshot_records_legacy_zone_as_hint_only():
    svc = PerceptionService()
    merged = make_merged(510, 500, 580, 540, control_type="ButtonControl", name="发送", automation_id="btnSend", element_id="merged_btn")
    zone = ZoneInfo(
        name="内容区",
        zone_type=ZoneType.CONTENT_AREA,
        elements=[merged],
        bounding_rect=(0, 80, 1920, 1000),
        element_count=1,
    )
    zone_page = ZonePageStructure(
        zones=[zone],
        all_elements_merged=[merged],
        screenshot_size=(1920, 1080),
    )

    snapshot = svc.create_page_snapshot(zone_page, process_name="WeChat.exe")

    assert "zone_partition" not in snapshot.provider_trace.provider_details
    assert snapshot.artifacts["legacy_zone_hints"]["used_as_hint"] is True
    assert snapshot.provider_trace.provider_details["legacy_zone_hints"]["zone_count"] == 1
    uia_locs = [locator for locator in snapshot.locators if locator.kind.value == "uia"]
    assert uia_locs
    assert uia_locs[0].selector["automation_id"] == "btnSend"


def test_synthesizes_review_chat_input_for_collaboration_detail_pane():
    svc = PerceptionService()
    zone_page = ZonePageStructure(
        zones=[],
        all_elements_merged=[],
        screenshot_size=(1018, 768),
        vision_candidates=[
            {"candidate_id": "voice", "bbox": [731, 702, 759, 730], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "send", "bbox": [958, 704, 982, 732], "kind": "icon", "confidence": 0.72},
        ],
    )
    snapshot = svc.create_page_snapshot(zone_page, process_name="feishu.exe")

    svc._synthesize_chat_input_candidate(snapshot, mode="collaboration_inbox")

    candidate = next(item for item in snapshot.elements if item.element_id == "synthetic_collaboration_composer_input")
    assert candidate.semantic_role.value == "message_input"
    assert candidate.bounds == (509, 658, 990, 738)
    assert candidate.attributes["actionability"] == "review"
    assert candidate.attributes["safe_to_type"] is False


def test_synthesizes_review_chat_input_for_chat_workspace_toolbar_icons():
    svc = PerceptionService()
    zone_page = ZonePageStructure(
        zones=[],
        all_elements_merged=[],
        screenshot_size=(1002, 731),
        vision_candidates=[
            {"candidate_id": "emoji", "bbox": [322, 692, 340, 710], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "attachment", "bbox": [358, 692, 376, 710], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "file", "bbox": [394, 692, 414, 710], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "send", "bbox": [922, 690, 979, 714], "kind": "button", "confidence": 0.72},
        ],
    )
    snapshot = svc.create_page_snapshot(zone_page, process_name="WeChat.exe")

    candidate = next(item for item in snapshot.elements if item.element_id == "synthetic_chat_composer_input")
    assert candidate.semantic_role.value == "message_input"
    assert candidate.bounds[1] >= 600
    assert candidate.bounds[0] > 300
    assert candidate.bounds[2] <= 914
    assert candidate.attributes["actionability"] == "review"
    assert candidate.attributes["safe_to_type"] is False
    assert candidate.attributes["needs_manual_label"] is True


def test_synthesizes_review_chat_input_when_screenshot_size_is_missing():
    svc = PerceptionService()
    zone_page = ZonePageStructure(
        zones=[],
        all_elements_merged=[],
        vision_candidates=[
            {"candidate_id": "emoji", "bbox": [315, 681, 350, 713], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "attachment", "bbox": [351, 680, 385, 713], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "file", "bbox": [389, 682, 419, 711], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "send", "bbox": [918, 683, 978, 709], "kind": "icon", "confidence": 0.72},
        ],
    )
    snapshot = svc.create_page_snapshot(zone_page, process_name="weixin.exe")

    candidate = next(item for item in snapshot.elements if item.element_id == "synthetic_chat_composer_input")
    assert candidate.bounds[0] >= 300
    assert candidate.bounds[2] <= 910
    assert candidate.attributes["actionability"] == "review"
    assert candidate.attributes["safe_to_type"] is False


def test_synthesizes_review_chat_input_with_right_side_voice_control():
    svc = PerceptionService()
    zone_page = ZonePageStructure(
        zones=[],
        all_elements_merged=[],
        vision_candidates=[
            {"candidate_id": "emoji", "bbox": [315, 681, 350, 713], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "attachment", "bbox": [351, 680, 385, 713], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "file", "bbox": [389, 682, 419, 711], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "voice", "bbox": [865, 681, 896, 714], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "send", "bbox": [918, 683, 978, 709], "kind": "icon", "confidence": 0.72},
        ],
    )
    snapshot = svc.create_page_snapshot(zone_page, process_name="weixin.exe")

    candidate = next(item for item in snapshot.elements if item.element_id == "synthetic_chat_composer_input")
    assert candidate.bounds[0] <= 315
    assert candidate.bounds[2] <= 857
    assert candidate.bounds[3] <= 675
    assert candidate.bounds[2] - candidate.bounds[0] >= 320


def test_synthesizes_review_chat_input_for_qq_mid_lower_toolbar():
    svc = PerceptionService()
    zone_page = ZonePageStructure(
        zones=[],
        all_elements_merged=[
            make_merged(472, 28, 481, 55, control_type="EditControl", name="", element_id="top_false_input"),
            make_merged(665, 599, 729, 625, control_type="ButtonControl", name="发送", element_id="send_btn"),
        ],
        screenshot_size=(960, 640),
        vision_candidates=[
            {"candidate_id": "emoji", "bbox": [330, 441, 354, 465], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "file", "bbox": [413, 441, 437, 465], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "image", "bbox": [456, 441, 480, 465], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "voice", "bbox": [536, 441, 560, 465], "kind": "icon", "confidence": 0.72},
            {"candidate_id": "send", "bbox": [665, 599, 729, 625], "kind": "button", "confidence": 0.72},
        ],
    )
    snapshot = svc.create_page_snapshot(zone_page, process_name="qq.exe")

    assert snapshot.artifacts["visual_pattern"]["mode"] == "chat_workspace"
    candidate = next(item for item in snapshot.elements if item.element_id == "synthetic_chat_composer_input")
    assert candidate.semantic_role.value == "message_input"
    assert candidate.bounds[0] >= 300
    assert candidate.bounds[1] >= 520
    assert candidate.bounds[2] <= 657
    assert candidate.attributes["actionability"] == "review"
    assert candidate.attributes["safe_to_type"] is False


def test_analyze_populates_ocr_blocks_and_fallback_vision_candidates():
    svc = PerceptionService()
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.side_effect = RuntimeError("vision offline")
    uia_elements = [
        make_elem(0, 0, 1920, 50, control_type="TitleBarControl"),
        make_elem(600, 150, 1300, 980, control_type="PaneControl"),
    ]
    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (1280, 692)
    mock_ocr_result = type(
        "OCRResult",
        (),
        {
            "blocks": [
                type("Block", (), {"text": "发送", "bbox": (100, 120, 160, 150), "confidence": 0.9})(),
                type("Block", (), {"text": "联系人", "bbox": (40, 140, 120, 170), "confidence": 0.88})(),
            ],
            "provider": "paddleocr_bridge",
            "success": True,
            "error": None,
            "elapsed_seconds": 0.12,
            "used_region": (0, 50, 1280, 692),
            "worker_command": ["python", "worker.py"],
            "worker_mode": "persistent",
            "worker_reused": True,
            "startup_seconds": 0.0,
            "fallback_reason": None,
        },
    )()

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = uia_elements
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr_service:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = mock_ocr_result
                    mock_get_ocr_service.return_value = mock_ocr
                    result = svc.analyze(12345)

    assert len(result.ocr_blocks) == 2
    assert result.ocr_provider_details["block_count"] == 2
    assert len(result.vision_candidates) == 2
    assert result.vision_candidates[0]["source"] == "paddleocr_bridge"


def test_analyze_without_screenshot_returns_empty_ocr_and_vision():
    svc = PerceptionService()

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = []
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.side_effect = Exception("no screenshot")
                mock_ss_cls.return_value = mock_ss
                result = svc.analyze(12345)

    assert result.ocr_auxiliary_texts == []
    assert result.ocr_blocks == []
    assert result.vision_candidates == []
    assert result.screenshot_provider_details["success"] is False
    assert result.screenshot_provider_details["provider"] == "window_capture"
    assert "no screenshot" in result.screenshot_provider_details["error"]


def test_analyze_screen_region_capture_uses_window_frame_region():
    svc = PerceptionService()
    screenshot = Image.new("RGB", (100, 80), "white")

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.get_root_element.return_value = make_elem(10, 20, 110, 100, control_type="WindowControl")
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.windows.window_bounds.get_capture_frame_bounds", return_value=(10, 20, 110, 100)):
                    result = svc.analyze(12345, fast_mode=True, screenshot_capture_mode="screen_region")

    mock_ss.capture.assert_called_once_with(mode="region", rect=(10, 20, 110, 100))
    assert result.screenshot_provider_details["provider"] == "screen_region_capture"
    assert result.screenshot_provider_details["screen_region"] == [10, 20, 110, 100]
    assert result.screenshot_size == (100, 80)


def test_create_page_snapshot_preserves_screenshot_capture_failure_artifact():
    svc = PerceptionService()
    zone_page = ZonePageStructure(
        all_elements_merged=[],
        screenshot_provider_details={
            "provider": "window_capture",
            "success": False,
            "error": "窗口截图失败: no screenshot",
        },
    )

    snapshot = svc.create_page_snapshot(zone_page, process_name="qq.exe")

    assert snapshot.artifacts["screenshot_provider"]["success"] is False
    assert "no screenshot" in snapshot.artifacts["screenshot_provider"]["error"]
    assert snapshot.provider_trace.provider_details["screenshot_provider"]["success"] is False


def test_create_page_snapshot_preserves_vision_structure_artifacts():
    svc = PerceptionService()
    merged = make_merged(100, 100, 220, 150, control_type="CustomControl", name=None)
    zone_page = ZonePageStructure(
        all_elements_merged=[merged],
        screenshot_size=(800, 600),
        ocr_blocks=[type("Block", (), {"text": "提交", "bbox": (105, 102, 210, 148), "confidence": 0.95})()],
        vision_candidates=[
            {
                "candidate_id": "vision_1",
                "bbox": [100, 100, 220, 150],
                "confidence": 0.8,
                "kind": "submit_button",
                "group_id": "cg_submit",
            }
        ],
        vision_layout_regions=[{"role": "action_bar", "bbox": [80, 90, 260, 170]}],
        vision_control_groups=[{"group_id": "cg_submit", "member_ids": ["vision_1"]}],
        vision_interaction_hints=[{"candidate_id": "vision_1", "preferred_action": "click"}],
        ocr_provider_details={"provider": "paddleocr_bridge", "block_count": 1},
        vision_provider_details={"provider": "omniparser", "layout_region_count": 1},
        structure_evidence_score=0.67,
    )

    snapshot = svc.create_page_snapshot(zone_page, process_name="chrome.exe")

    assert len(snapshot.artifacts["vision_layout_regions"]) == 1
    assert len(snapshot.artifacts["vision_control_groups"]) == 1
    assert len(snapshot.artifacts["vision_interaction_hints"]) == 1
    assert snapshot.artifacts["structure_evidence_score"] == 0.67
    assert "boundary_candidates" in snapshot.artifacts
    assert snapshot.artifacts["boundary_candidate_stats"]["count"] >= 2
    assert snapshot.artifacts["boundary_candidate_stats"]["by_source"]["ocr"] >= 1
    assert snapshot.artifacts["boundary_candidate_stats"]["by_source"]["omniparser"] >= 1
    assert any(item["candidate_id"] == "vision_group::cg_submit" for item in snapshot.artifacts["boundary_candidates"])
    assert any(item["candidate_id"].startswith("vision_region::") for item in snapshot.artifacts["boundary_candidates"])
    assert snapshot.provider_trace.provider_details["vision_layout_regions"]["count"] == 1
    assert snapshot.provider_trace.provider_details["structure_evidence_score"] == 0.67
    assert snapshot.provider_trace.provider_details["boundary_candidates"]["count"] >= 2


def test_analyze_prefers_remote_vision_provider_candidates_and_structure_metadata():
    svc = PerceptionService()
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.return_value = type(
        "VisionResult",
        (),
        {
            "provider": "omniparser",
            "success": True,
            "error": None,
            "visual_score": 0.73,
            "layout_regions": [{"role": "viewport", "bbox": [0, 0, 1200, 680]}],
            "control_groups": [{"group_id": "cg_composer", "member_ids": ["vision_send"]}],
            "interaction_hints": [{"candidate_id": "vision_send", "preferred_action": "click"}],
            "structure_evidence_score": 0.82,
            "candidates": [
                type(
                    "Candidate",
                    (),
                    {
                        "element_id": "vision_send",
                        "bounding_box": (10, 10, 80, 36),
                        "semantic_label": "send_button",
                        "confidence": 0.88,
                        "text": "发送",
                        "icon_type": None,
                        "region_role": "action_bar",
                        "group_id": "cg_composer",
                        "interaction_hints": {"preferred_action": "click"},
                        "structure_evidence_score": 0.82,
                        "attributes": {"source": "omniparser"},
                    },
                )()
            ],
        },
    )()
    uia_elements = [make_elem(600, 150, 1300, 980, control_type="PaneControl")]
    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (1280, 692)

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = uia_elements
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr_service:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = type(
                        "OCRResult",
                        (),
                        {
                            "blocks": [],
                            "provider": "paddleocr_bridge",
                            "success": True,
                            "error": None,
                            "elapsed_seconds": 0.2,
                            "used_region": None,
                            "worker_command": [],
                        },
                    )()
                    mock_get_ocr_service.return_value = mock_ocr
                    result = svc.analyze(12345)

    assert result.visual_score == 0.73
    assert result.vision_candidates[0]["candidate_id"] == "vision_send"
    assert result.vision_layout_regions[0]["role"] == "viewport"
    assert result.vision_control_groups[0]["group_id"] == "cg_composer"
    assert result.vision_interaction_hints[0]["candidate_id"] == "vision_send"
    assert result.structure_evidence_score == 0.82


def test_analyze_does_not_fallback_to_ocr_sidecar_when_omniparser_succeeds_without_candidates():
    svc = PerceptionService()
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.return_value = type(
        "VisionResult",
        (),
        {
            "provider": "omniparser",
            "success": True,
            "error": None,
            "visual_score": 0.5,
            "layout_regions": [{"role": "viewport", "bbox": [0, 0, 1200, 680]}],
            "control_groups": [],
            "interaction_hints": [],
            "structure_evidence_score": 0.61,
            "candidates": [],
        },
    )()
    uia_elements = [make_elem(600, 150, 1300, 980, control_type="PaneControl")]
    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (1280, 692)
    mock_ocr_result = type(
        "OCRResult",
        (),
        {
            "blocks": [
                type("Block", (), {"text": "搜索", "bbox": (100, 120, 160, 150), "confidence": 0.9})(),
            ],
            "provider": "paddleocr_bridge",
            "success": True,
            "error": None,
            "elapsed_seconds": 0.1,
            "used_region": None,
            "worker_command": [],
        },
    )()

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = uia_elements
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr_service:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = mock_ocr_result
                    mock_get_ocr_service.return_value = mock_ocr
                    result = svc.analyze(12345)

    assert result.vision_provider_details["success"] is True
    assert result.vision_candidates == []
    assert result.vision_provider_details.get("fallback") is None


# ===== _should_call_vlm tests =====


class TestShouldCallVlm:
    """Test the _should_call_vlm pure function with 5 conditions."""

    def _make_svc(self, vlm_available: bool = True) -> PerceptionService:
        svc = PerceptionService()
        svc._vlm_provider = MagicMock()
        svc._vlm_provider.available = vlm_available
        return svc

    def test_force_vlm_calls_regardless(self):
        svc = self._make_svc(vlm_available=True)
        assert svc._should_call_vlm(
            allow_vlm=False, force_vlm=True, vision_candidates=[], vision_success=True
        ) is True

    def test_disabled_when_vlm_not_available(self):
        svc = self._make_svc(vlm_available=False)
        assert svc._should_call_vlm(
            allow_vlm=True, force_vlm=True, vision_candidates=[], vision_success=True
        ) is False

    def test_disabled_when_allow_vlm_false(self):
        svc = self._make_svc(vlm_available=True)
        assert svc._should_call_vlm(
            allow_vlm=False, force_vlm=False, vision_candidates=[], vision_success=True
        ) is False

    def test_calls_when_few_local_candidates(self):
        svc = self._make_svc(vlm_available=True)
        few_candidates = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
        assert svc._should_call_vlm(
            allow_vlm=True, force_vlm=False, vision_candidates=few_candidates, vision_success=True
        ) is True

    def test_calls_when_local_provider_failed(self):
        svc = self._make_svc(vlm_available=True)
        assert svc._should_call_vlm(
            allow_vlm=True, force_vlm=False, vision_candidates=[{"c": i} for i in range(10)], vision_success=False
        ) is True

    def test_no_call_when_sufficient_local_candidates(self):
        svc = self._make_svc(vlm_available=True)
        enough = [{"candidate_id": f"c{i}", "confidence": 0.8} for i in range(5)]
        assert svc._should_call_vlm(
            allow_vlm=True, force_vlm=False, vision_candidates=enough, vision_success=True
        ) is False


# ===== VLM integration tests =====


def test_vlm_candidates_merged_into_vision_candidates():
    """When VLM is called, its candidates appear in vision_candidates with source='vlm'."""
    svc = PerceptionService()
    mock_vlm_provider = MagicMock()
    mock_vlm_provider.available = True
    from src.perception.providers.vlm_provider import VLMCandidate

    mock_vlm_provider.analyze_screenshot.return_value = [
        VLMCandidate(bbox=(10, 20, 60, 40), control_type="button", text="VLM Button", confidence=0.85),
    ]
    svc._vlm_provider = mock_vlm_provider
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.side_effect = RuntimeError("offline")

    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (800, 600)

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = []
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = type(
                        "OCRResult", (), {"blocks": [], "provider": "paddleocr_bridge", "success": True,
                                          "error": None, "elapsed_seconds": 0.1, "used_region": None,
                                          "worker_command": []}
                    )()
                    mock_get_ocr.return_value = mock_ocr
                    result = svc.analyze(12345, allow_vlm=True)

    vlm_candidates = [c for c in result.vision_candidates if c.get("source") == "vlm"]
    assert len(vlm_candidates) == 1
    assert vlm_candidates[0]["text"] == "VLM Button"
    assert vlm_candidates[0]["confidence"] == 0.85
    assert result.vision_provider_details.get("vlm_candidate_count") == 1


def test_vlm_failure_recorded_in_provider_details():
    """When VLM raises, the error is recorded in vision_provider_details."""
    svc = PerceptionService()
    mock_vlm_provider = MagicMock()
    mock_vlm_provider.available = True
    mock_vlm_provider.analyze_screenshot.side_effect = RuntimeError("API timeout")
    svc._vlm_provider = mock_vlm_provider
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.side_effect = RuntimeError("offline")

    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (800, 600)

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = []
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = type(
                        "OCRResult", (), {"blocks": [], "provider": "paddleocr_bridge", "success": True,
                                          "error": None, "elapsed_seconds": 0.1, "used_region": None,
                                          "worker_command": []}
                    )()
                    mock_get_ocr.return_value = mock_ocr
                    result = svc.analyze(12345, force_vlm=True)

    assert "API timeout" in result.vision_provider_details.get("vlm_error", "")


# ===== C-1: low confidence trigger tests =====


def test_calls_when_local_confidence_low():
    """VLM should be called when local candidates have low max confidence."""
    svc = PerceptionService()
    svc._vlm_provider = MagicMock()
    svc._vlm_provider.available = True
    low_conf_candidates = [
        {"candidate_id": "c1", "confidence": 0.3},
        {"candidate_id": "c2", "confidence": 0.4},
        {"candidate_id": "c3", "confidence": 0.5},
    ]
    assert svc._should_call_vlm(
        allow_vlm=True, force_vlm=False,
        vision_candidates=low_conf_candidates, vision_success=True,
    ) is True


def test_no_call_when_local_confidence_high():
    """VLM should NOT be called when local candidates have high confidence."""
    svc = PerceptionService()
    svc._vlm_provider = MagicMock()
    svc._vlm_provider.available = True
    high_conf_candidates = [
        {"candidate_id": "c1", "confidence": 0.9},
        {"candidate_id": "c2", "confidence": 0.85},
        {"candidate_id": "c3", "confidence": 0.7},
    ]
    assert svc._should_call_vlm(
        allow_vlm=True, force_vlm=False,
        vision_candidates=high_conf_candidates, vision_success=True,
    ) is False


def test_calls_when_mixed_confidence_below_threshold():
    """VLM called when max confidence is below threshold even with many candidates."""
    svc = PerceptionService()
    svc._vlm_provider = MagicMock()
    svc._vlm_provider.available = True
    candidates = [
        {"candidate_id": f"c{i}", "confidence": 0.55} for i in range(10)
    ]
    assert svc._should_call_vlm(
        allow_vlm=True, force_vlm=False,
        vision_candidates=candidates, vision_success=True,
    ) is True


# ===== C-2: VLM status in providers_used/providers_failed =====


def test_vlm_success_shows_in_providers_used():
    """When VLM succeeds, 'vlm' appears in providers_used."""
    svc = PerceptionService()
    mock_vlm_provider = MagicMock()
    mock_vlm_provider.available = True
    from src.perception.providers.vlm_provider import VLMCandidate

    mock_vlm_provider.analyze_screenshot.return_value = [
        VLMCandidate(bbox=(10, 20, 60, 40), control_type="button", text="OK", confidence=0.9),
    ]
    svc._vlm_provider = mock_vlm_provider
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.side_effect = RuntimeError("offline")

    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (800, 600)

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = []
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = type(
                        "OCRResult", (), {"blocks": [], "provider": "paddleocr_bridge", "success": True,
                                          "error": None, "elapsed_seconds": 0.1, "used_region": None,
                                          "worker_command": []}
                    )()
                    mock_get_ocr.return_value = mock_ocr
                    zone_page = svc.analyze(12345, allow_vlm=True)

    snapshot = svc.create_page_snapshot(zone_page)
    assert "vlm" in snapshot.providers_used
    assert snapshot.provider_trace.vlm_used is True


def test_vlm_failure_shows_in_providers_failed():
    """When VLM fails, 'vlm' appears in providers_failed."""
    svc = PerceptionService()
    mock_vlm_provider = MagicMock()
    mock_vlm_provider.available = True
    mock_vlm_provider.analyze_screenshot.side_effect = RuntimeError("API down")
    svc._vlm_provider = mock_vlm_provider
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.side_effect = RuntimeError("offline")

    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (800, 600)

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = []
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = type(
                        "OCRResult", (), {"blocks": [], "provider": "paddleocr_bridge", "success": True,
                                          "error": None, "elapsed_seconds": 0.1, "used_region": None,
                                          "worker_command": []}
                    )()
                    mock_get_ocr.return_value = mock_ocr
                    zone_page = svc.analyze(12345, force_vlm=True)

    snapshot = svc.create_page_snapshot(zone_page)
    assert "vlm" in snapshot.providers_failed


# ===== C-4: skeleton provider graceful degradation =====


def test_observe_graceful_degradation_with_skeleton_vlm_provider():
    """When VLM provider raises NotImplementedError, observe does not crash."""
    svc = PerceptionService()
    mock_vlm_provider = MagicMock()
    mock_vlm_provider.available = True
    mock_vlm_provider.analyze_screenshot.side_effect = NotImplementedError("local VLM not implemented")
    svc._vlm_provider = mock_vlm_provider
    svc._vision_provider = MagicMock()
    svc._vision_provider.parse_screenshot.side_effect = RuntimeError("offline")

    mock_screenshot = MagicMock(spec=Image.Image)
    mock_screenshot.size = (800, 600)

    with patch("src.perception.perception_service.UIAClient") as mock_uia_cls:
        mock_uia = MagicMock()
        mock_uia.find_all.return_value = []
        mock_uia_cls.return_value = mock_uia
        with patch("src.perception.perception_service.WindowEnumService") as mock_enum_cls:
            mock_enum = MagicMock()
            mock_enum.enumerate_all.return_value = []
            mock_enum.get_foreground_window.return_value = None
            mock_enum_cls.return_value = mock_enum
            with patch("src.windows.screenshot_service.ScreenshotService") as mock_ss_cls:
                mock_ss = MagicMock()
                mock_ss.capture.return_value = mock_screenshot
                mock_ss_cls.return_value = mock_ss
                with patch("src.perception.perception_service.get_ocr_service") as mock_get_ocr:
                    mock_ocr = MagicMock()
                    mock_ocr.extract_with_metadata.return_value = type(
                        "OCRResult", (), {"blocks": [], "provider": "paddleocr_bridge", "success": True,
                                          "error": None, "elapsed_seconds": 0.1, "used_region": None,
                                          "worker_command": []}
                    )()
                    mock_get_ocr.return_value = mock_ocr
                    # Should not raise — graceful degradation
                    zone_page = svc.analyze(12345, force_vlm=True)

    # VLM error recorded, no crash
    assert "not implemented" in zone_page.vision_provider_details.get("vlm_error", "").lower()
    # VLM failure tracked in providers
    snapshot = svc.create_page_snapshot(zone_page)
    assert "vlm" in snapshot.providers_failed
