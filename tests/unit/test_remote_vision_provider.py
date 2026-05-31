from unittest.mock import MagicMock, patch

from PIL import Image

from src.perception.providers.remote_vision_provider import OmniParserRemoteVisionProvider


def test_default_local_omniparser_paths_are_project_owned(tmp_path):
    provider = OmniParserRemoteVisionProvider(config_path=tmp_path / "missing-models.yaml")

    config = provider._config

    assert "X:\\OmniParser" not in config.project_root
    assert "X:\\OmniParser" not in config.som_model_path
    assert "X:\\OmniParser" not in config.caption_model_path
    assert config.project_root.endswith("vendor\\omniparser_runtime")
    assert config.som_model_path.endswith("models\\omniparser\\weights\\icon_detect\\model.pt")


def test_parse_screenshot_normalizes_omniparser_payload():
    provider = OmniParserRemoteVisionProvider()
    image = Image.new("RGB", (100, 80), color="white")
    response = MagicMock()
    response.json.return_value = {
        "parsed_elements": [
            {
                "id": "send_button",
                "bbox": [10, 12, 60, 32],
                "type": "send_button",
                "text": "发送",
                "confidence": 0.94,
                "region_role": "action_bar",
                "group_id": "cg_composer",
            }
        ],
        "layout_regions": [{"role": "viewport", "bbox": [0, 0, 100, 80]}],
        "control_groups": [{"group_id": "cg_composer", "member_ids": ["send_button"]}],
        "interaction_hints": [{"candidate_id": "send_button", "preferred_action": "click"}],
        "structure_evidence_score": 0.72,
        "visual_score": 0.61,
    }
    response.raise_for_status.return_value = None

    with patch.object(provider, "_ensure_service_available", return_value=True):
        with patch("requests.post", return_value=response):
            result = provider.parse_screenshot(image)

    assert result.success is True
    assert result.visual_score == 0.61
    assert result.structure_evidence_score == 0.72
    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_label == "send_button"
    assert result.candidates[0].bounding_box == (10, 12, 60, 32)
    assert result.candidates[0].text == "发送"
    assert result.candidates[0].group_id == "cg_composer"
    assert result.candidates[0].interaction_hints == {
        "candidate_id": "send_button",
        "preferred_action": "click",
    }
    assert result.control_groups[0]["group_id"] == "cg_composer"


def test_parse_screenshot_normalizes_parsed_content_ratio_payload():
    provider = OmniParserRemoteVisionProvider()
    image = Image.new("RGB", (200, 100), color="white")
    response = MagicMock()
    response.json.return_value = {
        "parsed_content_list": [
            {
                "id": 7,
                "type": "icon_button",
                "content": "发送",
                "bbox": [0.5, 0.2, 0.75, 0.5],
                "interactivity": True,
                "source": "box_yolo_content_yolo",
            }
        ],
        "latency": 1.23,
    }
    response.raise_for_status.return_value = None

    with patch.object(provider, "_ensure_service_available", return_value=True):
        with patch("requests.post", return_value=response):
            result = provider.parse_screenshot(image)

    assert result.success is True
    assert len(result.candidates) == 1
    assert result.candidates[0].element_id == "7"
    assert result.candidates[0].semantic_label == "icon_button"
    assert result.candidates[0].text == "发送"
    assert result.candidates[0].bounding_box == (100, 20, 150, 50)
    assert result.candidates[0].confidence >= 0.7


def test_parse_screenshot_sends_detector_only_flags():
    provider = OmniParserRemoteVisionProvider()
    image = Image.new("RGB", (100, 80), color="white")
    response = MagicMock()
    response.json.return_value = {"parsed_content_list": []}
    response.raise_for_status.return_value = None

    with patch.object(provider, "_ensure_service_available", return_value=True):
        with patch("requests.post", return_value=response) as post:
            result = provider.parse_screenshot(image)

    assert result.success is True
    request_json = post.call_args.kwargs["json"]
    assert request_json["include_caption"] is False
    assert request_json["include_ocr"] is False


def test_parse_screenshot_returns_provider_unavailable_when_service_cannot_start():
    provider = OmniParserRemoteVisionProvider()
    image = Image.new("RGB", (100, 80), color="white")

    with patch.object(provider, "_ensure_service_available", return_value=False):
        result = provider.parse_screenshot(image)

    assert result.success is False
    assert result.error == "omniparser_provider_unavailable"
