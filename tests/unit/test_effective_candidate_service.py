"""EffectiveCandidateService tests."""

from __future__ import annotations

from src.memory.effective_candidate_service import EffectiveCandidateService
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    ElementState,
    InteractionCanvas,
    PageInfo,
    ProviderTrace,
    Region,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def _canvas_with(candidate: Candidate) -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="canvas_1",
        app=AppInfo(app_id="app", process_name="app.exe"),
        window=WindowInfoSnapshot(hwnd=1, title="app"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
        page=PageInfo(page_class="app/main", class_confidence=0.9),
        regions=[Region(region_id="main", role="content", bounds=(0, 0, 800, 600), element_ids=[candidate.element_id])],
        elements=[candidate],
        provider_trace=ProviderTrace(uia_used=True),
        providers_used=candidate.provider_sources,
    )


def test_interactable_local_candidate_is_safe():
    candidate = Candidate(
        element_id="btn",
        semantic_role=SemanticRole.BUTTON,
        bounds=(10, 10, 80, 40),
        interactable=True,
        state=ElementState(enabled=True, visible=True),
        provider_sources=["uia"],
    )
    effective = EffectiveCandidateService().apply(_canvas_with(candidate))
    assert effective.elements[0].attributes["actionability"] == "safe"


def test_vlm_only_candidate_defaults_to_review():
    candidate = Candidate(
        element_id="vlm_btn",
        semantic_role=SemanticRole.BUTTON,
        bounds=(10, 10, 80, 40),
        interactable=True,
        state=ElementState(enabled=True, visible=True),
        provider_sources=["vlm"],
        attributes={"vision_only": True},
    )
    effective = EffectiveCandidateService().apply(_canvas_with(candidate))
    assert effective.elements[0].attributes["actionability"] == "review"


def test_existing_actionability_is_preserved():
    candidate = Candidate(
        element_id="zone",
        semantic_role=SemanticRole.UNKNOWN,
        bounds=(10, 10, 80, 40),
        interactable=False,
        state=ElementState(enabled=True, visible=True),
        provider_sources=["vlm"],
        attributes={"actionability": "semantic_only"},
    )
    effective = EffectiveCandidateService().apply(_canvas_with(candidate))
    assert effective.elements[0].attributes["actionability"] == "semantic_only"
