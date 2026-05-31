"""VLM (Vision Language Model) internal provider for enhanced perception.

VLM is an internal enhancement provider within observe/query/refine,
NOT an independent API exposed to agents.

Controlled by allow_vlm/force_vlm flags in observe/query requests.
VLM only outputs candidate suggestions, never suggested actions.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import requests
from PIL import Image

from src.common.config_manager import VLMConfig, load_config
from src.perception.page_compiler_models import (
    Candidate,
    ConfidenceLevel,
    InteractionCanvas,
    SemanticRole,
    WindowInfoSnapshot,
)
from src.vlm.transport import create_vlm_session

logger = logging.getLogger(__name__)

# Prompt for structured UI element identification.
# Asks "what interactive elements are visible", never "what should I do".
_VLM_PROMPT = """Analyze this screenshot and identify all interactive UI elements.

Return a JSON array. Each element must have:
- "bbox": [left, top, right, bottom] in pixels
- "type": control type (button, textbox, link, menu_item, tab, checkbox, dropdown, icon, etc.)
- "text": visible text label (empty string if none)
- "confidence": 0.0-1.0

Only include elements a user could interact with (click, type, select).
Do NOT include static text or decorative elements.
Return ONLY the JSON array, no other text."""


class VLMCandidate:
    """A candidate suggested by VLM."""

    __slots__ = ("bbox", "text", "control_type", "confidence", "description")

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        text: str = "",
        control_type: str = "",
        confidence: float = 0.5,
        description: str = "",
    ) -> None:
        self.bbox = bbox
        self.text = text
        self.control_type = control_type
        self.confidence = confidence
        self.description = description


class VLMProvider:
    """VLM internal provider for enhanced perception.

    This provider is called internally when:
    - force_vlm=true
    - allow_vlm=true + low local confidence / few candidates / provider failure

    It returns candidate suggestions that go through the same
    fusion pipeline as local candidates. It never suggests actions.
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self._config = config or load_config().vlm

    @property
    def name(self) -> str:
        return "vlm"

    @property
    def available(self) -> bool:
        """Check if VLM is configured and available."""
        if self._config.provider == "disabled":
            return False
        if self._config.provider == "cloud" and not self._config.api_key:
            return False
        return True

    def analyze_screenshot(
        self,
        image: Image.Image,
        window_title: str = "",
        task_hint: str = "",
    ) -> list[VLMCandidate]:
        """Analyze a screenshot and return candidate suggestions.

        Accepts PIL.Image (in-memory), never writes to disk.
        Returns VLMCandidate list — only candidates, never suggested actions.
        """
        if not self.available:
            logger.debug("VLM provider not available, skipping")
            return []

        logger.info(
            "VLM analyze: provider=%s, model=%s, window=%s",
            self._config.provider,
            self._config.model,
            window_title,
        )

        if self._config.provider == "cloud":
            return self._call_cloud_vlm(image, window_title, task_hint)
        if self._config.provider == "local":
            return self._call_local_vlm(image, window_title, task_hint)

        logger.warning("Unknown VLM provider: %s", self._config.provider)
        return []

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL.Image to base64 string in memory (no temp file)."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _call_cloud_vlm(
        self,
        image: Image.Image,
        window_title: str,
        task_hint: str,
    ) -> list[VLMCandidate]:
        """Call OpenAI-compatible cloud VLM API."""
        endpoint = self._config.endpoint or "https://api.openai.com/v1/chat/completions"
        model = self._config.model
        if not model:
            logger.warning("VLM model not configured, skipping")
            return []

        b64_image = self._encode_image(image)

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": _VLM_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
        ]
        if window_title:
            user_content.insert(0, {"type": "text", "text": f"Window: {window_title}"})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 4096,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            session = create_vlm_session(
                proxy_url=self._config.proxy_url,
                proxy_port=self._config.proxy_port,
            )
            resp = session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("VLM API call failed: %s", exc)
            return []

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return self._parse_vlm_response(text)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.warning("VLM response parse error: %s", exc)
            return []

    def _call_local_vlm(
        self,
        image: Image.Image,
        window_title: str,
        task_hint: str,
    ) -> list[VLMCandidate]:
        """Local VLM call — skeleton. Implement for Ollama/MiniCPM-V."""
        raise NotImplementedError(
            "Local VLM provider not yet implemented. "
            "Use provider='cloud' with an OpenAI-compatible endpoint."
        )

    def _parse_vlm_response(self, text: str) -> list[VLMCandidate]:
        """Parse VLM response text into VLMCandidate list."""
        # Extract JSON array from response (may be wrapped in markdown code block)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("VLM response is not valid JSON: %s", text[:200])
            return []

        if not isinstance(items, list):
            logger.warning("VLM response is not a JSON array")
            return []

        candidates: list[VLMCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                bbox_int = tuple(int(v) for v in bbox)
            except (ValueError, TypeError):
                continue
            candidates.append(VLMCandidate(
                bbox=bbox_int,  # type: ignore[arg-type]
                text=str(item.get("text", "")),
                control_type=str(item.get("type", "")),
                confidence=float(item.get("confidence", 0.5)),
                description=str(item.get("description", "")),
            ))

        return candidates

    def suggest_candidates(
        self,
        canvas: InteractionCanvas,
        image: Image.Image,
        query_hint: str = "",
    ) -> list[Candidate]:
        """Suggest candidates that local providers may have missed.

        Returns candidates that go through the same fusion pipeline.
        """
        if not self.available:
            return []

        window_title = canvas.window.title if canvas.window else ""
        vlm_candidates = self.analyze_screenshot(
            image=image,
            window_title=window_title,
            task_hint=query_hint,
        )

        results: list[Candidate] = []
        for i, vc in enumerate(vlm_candidates):
            candidate = Candidate(
                element_id=f"vlm_{i}",
                semantic_role=SemanticRole.UNKNOWN,
                control_type=vc.control_type,
                bounds=vc.bbox,
                text=vc.text,
                confidence=vc.confidence,
                provider_sources=["vlm"],
                interactable=True,
                attributes={"vlm_description": vc.description},
            )
            results.append(candidate)

        return results


def create_vlm_provider() -> VLMProvider:
    """Create a VLM provider from application config."""
    config = load_config()
    return VLMProvider(config.vlm)
