"""Effective candidate service.

This is the single read-side entrypoint for consumers that need the final
candidate layer after manual/agent/VLM semantic corrections are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.memory.candidate_override_apply import apply_candidate_overrides
from src.memory.candidate_override_store import CandidateOverrideData
from src.perception.page_compiler_models import InteractionCanvas


@dataclass(frozen=True)
class EffectiveCandidatePolicy:
    """Read-side priority contract for candidate overlays."""

    priority_order: tuple[str, ...] = (
        "manual",
        "agent",
        "vlm_semantic",
        "memory",
        "uia",
        "ocr",
        "omni",
        "vision",
    )
    vlm_only_default_actionability: str = "review"
    inferred_zone_default_actionability: str = "semantic_only"


class EffectiveCandidateService:
    """Build effective canvases without mutating raw perception records."""

    def __init__(self, policy: EffectiveCandidatePolicy | None = None):
        self.policy = policy or EffectiveCandidatePolicy()

    def apply(
        self,
        canvas: InteractionCanvas,
        overrides: Iterable[CandidateOverrideData] | None = None,
    ) -> InteractionCanvas:
        """Return an effective canvas copy with correction overlays applied."""
        effective = apply_candidate_overrides(canvas, overrides=overrides)
        self._annotate_actionability(effective)
        return effective

    def _annotate_actionability(self, canvas: InteractionCanvas) -> None:
        for candidate in canvas.elements:
            attrs = dict(candidate.attributes or {})
            sources = set(candidate.provider_sources or [])
            if attrs.get("actionability"):
                continue
            if "manual" in sources or "agent" in sources:
                attrs["actionability"] = "safe"
            elif sources == {"vlm"} or attrs.get("vision_only"):
                attrs["actionability"] = self.policy.vlm_only_default_actionability
            elif candidate.interactable:
                attrs["actionability"] = "safe"
            else:
                attrs["actionability"] = "review"
            candidate.attributes = attrs
