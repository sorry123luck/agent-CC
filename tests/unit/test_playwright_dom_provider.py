from unittest.mock import MagicMock, patch
import builtins

from src.perception.providers.playwright_dom_provider import PlaywrightCDPDOMProvider


def test_connect_returns_false_when_no_endpoint():
    provider = PlaywrightCDPDOMProvider()

    assert provider.connect("chromium") is False


def test_connect_returns_false_when_playwright_missing():
    provider = PlaywrightCDPDOMProvider(endpoint_url="http://127.0.0.1:9222")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "playwright.sync_api":
            raise ImportError
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert provider.connect("chromium") is False


def test_perform_action_clicks_locator_first():
    provider = PlaywrightCDPDOMProvider(endpoint_url="ws://unused")
    locator = MagicMock()
    locator.count.return_value = 1
    locator.first = locator
    locator.bounding_box.return_value = {"x": 10, "y": 20, "width": 30, "height": 40}
    provider._page = MagicMock()

    with patch.object(provider, "_resolve_locator", return_value=locator):
        result = provider.perform_action({"css": "#send"}, "click")

    assert result["success"] is True
    assert result["details"] == "dom_click_ok"
    locator.click.assert_called_once_with(timeout=2000)


def test_get_element_by_locator_returns_normalized_rect():
    provider = PlaywrightCDPDOMProvider(endpoint_url="ws://unused")
    locator = MagicMock()
    locator.count.return_value = 1
    locator.first = locator
    locator.bounding_box.return_value = {"x": 10, "y": 20, "width": 30, "height": 40}
    locator.inner_text.return_value = "Send"
    provider._page = MagicMock()

    with patch.object(provider, "_resolve_locator", return_value=locator):
        result = provider.get_element_by_locator({"text": "Send"})

    assert result == {"rect": (10, 20, 30, 40), "text": "Send"}
