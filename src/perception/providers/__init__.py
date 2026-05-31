"""
Perception Provider 接口层

本模块定义各类感知 Provider 的抽象接口：
- IUIAProvider      — Windows UIA 元素提取
- IOCRProvider      — OCR 文本提取
- IDOMProvider      — 浏览器 DOM 提取
- IRemoteVisionProvider — 远程视觉解析
- ITemplateProvider — 图标模板匹配

所有 Provider 均实现 PerceptionProvider 协议（见 page_compiler_models.py）。

Provider 的职责边界（2026-04-05 确认）：
- OCR 只负责文本提取，不负责页面主划区
- 视觉模型只做增强，不做最终动作裁决
- 主链路：UIA + PaddleOCR + Template + DOM
"""

from src.perception.providers.base import PerceptionProviderBase
from src.perception.providers.uia_provider import IUIAProvider
from src.perception.providers.ocr_provider import IOCRProvider
from src.perception.providers.dom_provider import IDOMProvider
from src.perception.providers.playwright_dom_provider import PlaywrightCDPDOMProvider
from src.perception.providers.remote_vision_provider import (
    IRemoteVisionProvider,
    OmniParserRemoteVisionProvider,
)
from src.perception.providers.template_provider import ITemplateProvider
from src.perception.providers.vlm_provider import VLMProvider, create_vlm_provider

__all__ = [
    "PerceptionProviderBase",
    "IUIAProvider",
    "IOCRProvider",
    "IDOMProvider",
    "PlaywrightCDPDOMProvider",
    "IRemoteVisionProvider",
    "OmniParserRemoteVisionProvider",
    "ITemplateProvider",
    "VLMProvider",
    "create_vlm_provider",
]
