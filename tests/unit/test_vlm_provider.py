"""Unit tests for VLMProvider — cloud API call, parsing, availability."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.common.config_manager import VLMConfig
from src.perception.providers.vlm_provider import VLMCandidate, VLMProvider


def _make_config(**overrides) -> VLMConfig:
    defaults = dict(provider="disabled", api_key="", endpoint="", model="", timeout_seconds=30, max_retries=1)
    defaults.update(overrides)
    return VLMConfig(**defaults)


@contextmanager
def _mock_vlm_session(response=None, side_effect=None):
    session = MagicMock()
    if side_effect is not None:
        session.post.side_effect = side_effect
    else:
        session.post.return_value = response
    with patch("src.perception.providers.vlm_provider.create_vlm_session", return_value=session):
        yield session


class TestAvailability:
    def test_not_available_when_disabled(self):
        provider = VLMProvider(_make_config(provider="disabled"))
        assert provider.available is False

    def test_not_available_when_cloud_no_api_key(self):
        provider = VLMProvider(_make_config(provider="cloud", api_key=""))
        assert provider.available is False

    def test_available_when_cloud_with_key(self):
        provider = VLMProvider(_make_config(provider="cloud", api_key="test-key"))
        assert provider.available is True

    def test_available_when_local(self):
        provider = VLMProvider(_make_config(provider="local"))
        assert provider.available is True


class TestAnalyzeScreenshot:
    def test_returns_empty_when_unavailable(self):
        provider = VLMProvider(_make_config(provider="disabled"))
        image = Image.new("RGB", (100, 80), "white")
        result = provider.analyze_screenshot(image)
        assert result == []

    def test_calls_openai_api_with_correct_payload(self):
        config = _make_config(
            provider="cloud",
            api_key="test-key",
            endpoint="https://api.openai.com/v1/chat/completions",
            model="gpt-4o",
        )
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"bbox": [10, 20, 60, 40], "type": "button", "text": "OK", "confidence": 0.9}
            ])}}]
        }

        with _mock_vlm_session(mock_resp) as session:
            result = provider.analyze_screenshot(image, window_title="Test Window")

        assert len(result) == 1
        assert result[0].bbox == (10, 20, 60, 40)
        assert result[0].text == "OK"
        assert result[0].control_type == "button"
        assert result[0].confidence == 0.9

        # Verify API call
        call_kwargs = session.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "gpt-4o"
        assert "Authorization" in call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))

    def test_uses_model_from_config_not_hardcoded(self):
        config = _make_config(
            provider="cloud",
            api_key="test-key",
            model="qwen-vl-max",
        )
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with _mock_vlm_session(mock_resp) as session:
            provider.analyze_screenshot(image)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert payload["model"] == "qwen-vl-max"

    def test_parses_multiple_candidates(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)
        image = Image.new("RGB", (200, 100), "white")

        vlm_response = [
            {"bbox": [10, 10, 50, 30], "type": "button", "text": "Send", "confidence": 0.92},
            {"bbox": [60, 10, 190, 30], "type": "textbox", "text": "", "confidence": 0.85},
            {"bbox": [10, 50, 80, 70], "type": "link", "text": "Help", "confidence": 0.78},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(vlm_response)}}]}

        with _mock_vlm_session(mock_resp):
            result = provider.analyze_screenshot(image)

        assert len(result) == 3
        assert result[0].control_type == "button"
        assert result[1].control_type == "textbox"
        assert result[2].confidence == 0.78

    def test_handles_api_error_gracefully(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        import requests as req
        with _mock_vlm_session(side_effect=req.ConnectionError("timeout")):
            result = provider.analyze_screenshot(image)

        assert result == []

    def test_handles_invalid_json_response(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "not valid json"}}]}

        with _mock_vlm_session(mock_resp):
            result = provider.analyze_screenshot(image)

        assert result == []

    def test_handles_markdown_wrapped_json(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        wrapped = "```json\n" + json.dumps([{"bbox": [1, 2, 3, 4], "type": "btn", "text": "X", "confidence": 0.6}]) + "\n```"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": wrapped}}]}

        with _mock_vlm_session(mock_resp):
            result = provider.analyze_screenshot(image)

        assert len(result) == 1
        assert result[0].bbox == (1, 2, 3, 4)

    def test_skips_items_with_invalid_bbox(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        vlm_response = [
            {"bbox": [10, 20, 60, 40], "type": "button", "text": "OK", "confidence": 0.9},
            {"bbox": "invalid", "type": "button", "text": "Bad", "confidence": 0.5},
            {"type": "button", "text": "No bbox", "confidence": 0.5},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(vlm_response)}}]}

        with _mock_vlm_session(mock_resp):
            result = provider.analyze_screenshot(image)

        assert len(result) == 1

    def test_uses_pil_image_not_file_path(self):
        """Verify the method accepts PIL.Image and doesn't try to read from disk."""
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with _mock_vlm_session(mock_resp) as session:
            result = provider.analyze_screenshot(image)

        # Verify base64 data is in the payload (image was encoded in memory)
        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        content = payload["messages"][0]["content"]
        image_part = [p for p in content if p.get("type") == "image_url"][0]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_uses_explicit_proxy_port_for_cloud_vlm(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o", proxy_port=7890)
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with patch("src.perception.providers.vlm_provider.create_vlm_session") as session_factory:
            session = MagicMock()
            session.post.return_value = mock_resp
            session_factory.return_value = session

            provider.analyze_screenshot(image)

        session_factory.assert_called_once_with(proxy_url="", proxy_port=7890)

    def test_local_provider_raises_not_implemented(self):
        config = _make_config(provider="local")
        provider = VLMProvider(config)
        image = Image.new("RGB", (100, 80), "white")

        with pytest.raises(NotImplementedError):
            provider.analyze_screenshot(image)


class TestSuggestCandidates:
    def test_converts_vlm_candidates_to_candidate_objects(self):
        config = _make_config(provider="cloud", api_key="test-key", model="gpt-4o")
        provider = VLMProvider(config)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps([
                {"bbox": [10, 20, 60, 40], "type": "button", "text": "Submit", "confidence": 0.88}
            ])}}]
        }

        canvas = MagicMock()
        canvas.window.title = "Test"

        with _mock_vlm_session(mock_resp):
            results = provider.suggest_candidates(canvas, Image.new("RGB", (100, 80), "white"))

        assert len(results) == 1
        assert results[0].element_id == "vlm_0"
        assert results[0].text == "Submit"
        assert results[0].confidence == 0.88
        assert results[0].provider_sources == ["vlm"]
        assert results[0].bounds == (10, 20, 60, 40)

