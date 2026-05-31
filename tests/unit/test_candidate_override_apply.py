from src.memory.candidate_override_apply import apply_candidate_overrides, candidate_override_keys
from src.memory.candidate_override_store import CandidateOverrideData
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    PageInfo,
    SemanticRole,
    WindowInfoSnapshot,
)


def _canvas(page_model_id: str = "pm-a") -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id="snap_a",
        app=AppInfo(app_id="test", process_name="test.exe"),
        window=WindowInfoSnapshot(hwnd=1, rect_client=(0, 0, 100, 100)),
        page=PageInfo(page_class="test/main/default"),
        page_model_id=page_model_id,
        state_template_id="st-a",
        elements=[
            Candidate(
                element_id="btn",
                semantic_role=SemanticRole.BUTTON,
                text="Old",
                bounds=(10, 10, 30, 30),
                confidence=0.8,
                stable_key_id="stable-1",
                provider_sources=["uia"],
            ),
            Candidate(
                element_id="msg",
                semantic_role=SemanticRole.TEXT,
                text="Message",
                bounds=(40, 40, 80, 60),
                confidence=0.7,
                provider_sources=["ocr"],
            ),
        ],
    )


def _override(scope_key: str, **kwargs) -> CandidateOverrideData:
    return CandidateOverrideData(
        override_id="ovr",
        scope_key=scope_key,
        created_at="2026-05-12T00:00:00+00:00",
        updated_at="2026-05-12T00:00:00+00:00",
        **kwargs,
    )


def test_stable_override_is_page_model_scoped():
    canvas = _canvas(page_model_id="pm-a")
    override = _override("stable:pm-a:stable-1", label="保存", semantic_role="menu_item")

    effective = apply_candidate_overrides(canvas, [override])
    assert effective.elements[0].role_label == "保存"
    assert effective.elements[0].semantic_role == SemanticRole.MENU_ITEM

    other_canvas = _canvas(page_model_id="pm-b")
    other_effective = apply_candidate_overrides(other_canvas, [override])
    assert other_effective.elements[0].role_label is None
    assert other_effective.elements[0].semantic_role == SemanticRole.BUTTON


def test_canvas_override_can_ignore_transient_candidate():
    canvas = _canvas()
    override = _override("canvas:snap_a:msg", kind="ignored")

    effective = apply_candidate_overrides(canvas, [override])
    assert [element.element_id for element in effective.elements] == ["btn"]


def test_relative_bounds_override_updates_bounds_and_click_point():
    canvas = _canvas()
    override = _override("canvas:snap_a:msg", relative_bounds=[0.1, 0.2, 0.5, 0.6])

    effective = apply_candidate_overrides(canvas, [override])
    updated = effective.get_element("msg")
    assert updated is not None
    assert updated.bounds == (10, 20, 50, 60)
    assert updated.click_point == (30, 40)


def test_candidate_override_keys_include_scoped_stable_and_legacy_fallback():
    canvas = _canvas(page_model_id="pm-a")
    keys = candidate_override_keys(canvas.elements[0], canvas)

    assert keys[0] == "stable:pm-a:stable-1"
    assert "stable:stable-1" in keys
    assert "canvas:snap_a:btn" in keys
