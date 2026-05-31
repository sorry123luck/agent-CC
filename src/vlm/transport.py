"""HTTP transport helpers for external VLM APIs."""

from __future__ import annotations

import requests


def normalize_proxy_url(proxy_url: str = "", proxy_port: int | str = 0) -> str:
    """Return a proxy URL, or an empty string for direct transport."""
    value = str(proxy_url or "").strip()
    if value:
        if "://" not in value:
            return f"http://{value}"
        return value
    try:
        port = int(proxy_port or 0)
    except (TypeError, ValueError):
        port = 0
    if port <= 0:
        return ""
    return f"http://127.0.0.1:{port}"


def create_vlm_session(proxy_url: str = "", proxy_port: int | str = 0) -> requests.Session:
    """Create a requests session for VLM APIs.

    VLM calls default to direct transport by disabling environment proxy lookup.
    A proxy is used only when explicitly configured for the VLM provider.
    """
    session = requests.Session()
    session.trust_env = False
    proxy = normalize_proxy_url(proxy_url=proxy_url, proxy_port=proxy_port)
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session
