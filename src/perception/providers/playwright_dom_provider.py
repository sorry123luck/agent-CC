"""Optional Playwright CDP-backed DOM provider."""

from __future__ import annotations

import os
from typing import Any

from src.perception.page_compiler_models import (
    InteractionCanvas,
    ProviderTrace,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)
from src.perception.providers.dom_provider import IDOMProvider


class PlaywrightCDPDOMProvider(IDOMProvider):
    """Use Playwright CDP to drive browser or webview DOM when available."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        target_url: str | None = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._target_url = target_url
        self._playwright = None
        self._browser = None
        self._page = None

    @property
    def name(self) -> str:
        return "PlaywrightCDPDOMProvider"

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        snapshot = existing_structure or InteractionCanvas(
            canvas_id="dom_provider_snapshot",
            window=window_info,
            surface=SurfaceInfo(surface_type=SurfaceType.BROWSER, confidence=0.7),
        )
        snapshot.provider_trace.dom_used = True
        snapshot.artifacts["dom_tree"] = self.get_dom_tree()
        return snapshot

    def connect(self, browser_type: str, url: str | None = None) -> bool:
        endpoint_url = self._resolve_endpoint_url(browser_type)
        if not endpoint_url:
            return False

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return False

        self.disconnect()
        self._playwright = sync_playwright().start()
        browser_type_impl = getattr(self._playwright, browser_type, None)
        if browser_type_impl is None:
            self.disconnect()
            return False

        try:
            self._browser = browser_type_impl.connect_over_cdp(endpoint_url)
            self._page = self._select_page(url or self._target_url)
            return self._page is not None
        except Exception:
            self.disconnect()
            return False

    def disconnect(self) -> None:
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._browser = None
        self._playwright = None
        self._page = None

    def get_dom_tree(self) -> dict[str, Any]:
        if self._page is None:
            raise RuntimeError("DOM page is not connected")

        return self._page.evaluate(
            """
            () => {
              const walk = (node) => ({
                tag_name: node.tagName ? node.tagName.toLowerCase() : "#text",
                text: node.nodeType === Node.TEXT_NODE ? node.textContent : null,
                attributes: node.attributes
                  ? Object.fromEntries(Array.from(node.attributes).map(attr => [attr.name, attr.value]))
                  : {},
                children: Array.from(node.childNodes || []).slice(0, 50).map(walk),
              });
              return walk(document.body);
            }
            """
        )

    def get_element_by_locator(
        self,
        selector: dict[str, Any],
    ) -> dict[str, Any] | None:
        locator = self._resolve_locator(selector)
        if locator is None:
            return None

        if locator.count() == 0:
            return None

        first = locator.first
        rect = first.bounding_box()
        text = first.inner_text(timeout=1000) if rect is not None else ""
        return {
            "rect": self._normalize_rect(rect),
            "text": text,
        }

    def perform_action(
        self,
        selector: dict[str, Any],
        action: str,
    ) -> dict[str, Any]:
        locator = self._resolve_locator(selector)
        if locator is None or locator.count() == 0:
            return {
                "success": False,
                "details": f"DOM element not found for selector: {selector}",
                "rect": None,
            }

        first = locator.first
        if action == "click":
            first.click(timeout=2000)
        elif action == "double_click":
            first.dblclick(timeout=2000)
        elif action == "right_click":
            first.click(timeout=2000, button="right")
        else:
            return {
                "success": False,
                "details": f"Unsupported DOM action: {action}",
                "rect": None,
            }

        return {
            "success": True,
            "details": f"dom_{action}_ok",
            "rect": self._normalize_rect(first.bounding_box()),
        }

    def is_dom_ready(self) -> bool:
        if self._page is None:
            return False
        try:
            ready_state = self._page.evaluate("document.readyState")
            return ready_state in {"interactive", "complete"}
        except Exception:
            return False

    def take_screenshot(self, full_page: bool = False) -> bytes:
        if self._page is None:
            raise RuntimeError("DOM page is not connected")
        return self._page.screenshot(full_page=full_page)

    def _resolve_endpoint_url(self, browser_type: str) -> str | None:
        if self._endpoint_url:
            return self._endpoint_url

        env_names = [
            f"OPENCLAW_{browser_type.upper()}_CDP_URL",
            "OPENCLAW_BROWSER_CDP_URL",
        ]
        for env_name in env_names:
            value = os.getenv(env_name)
            if value:
                return value
        return None

    def _select_page(self, target_url: str | None):
        if self._browser is None:
            return None

        pages = []
        for context in self._browser.contexts:
            pages.extend(context.pages)
        if not pages:
            return None

        if target_url:
            for page in pages:
                if target_url in page.url:
                    return page
        return pages[0]

    def _resolve_locator(self, selector: dict[str, Any]):
        if self._page is None:
            raise RuntimeError("DOM page is not connected")

        if "css" in selector:
            return self._page.locator(selector["css"])
        if "xpath" in selector:
            return self._page.locator(f"xpath={selector['xpath']}")
        if "role" in selector:
            kwargs = {}
            if selector.get("name"):
                kwargs["name"] = selector["name"]
            return self._page.get_by_role(selector["role"], **kwargs)
        if "test_id" in selector:
            return self._page.get_by_test_id(selector["test_id"])
        if "label" in selector:
            return self._page.get_by_label(selector["label"])
        if "text" in selector:
            return self._page.get_by_text(selector["text"], exact=selector.get("exact", True))
        if "name" in selector:
            name = str(selector["name"]).replace('"', '\\"')
            return self._page.locator(
                f'[name="{name}"], [aria-label="{name}"], [data-testid="{name}"]'
            )
        return None

    def _normalize_rect(self, rect: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
        if not rect:
            return None
        return (
            int(rect["x"]),
            int(rect["y"]),
            int(rect["width"]),
            int(rect["height"]),
        )
