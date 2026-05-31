from PIL import Image

from src.vlm.coordinate_space import (
    normalize_model_to_original,
    original_to_vlm_bounds,
    prepare_vlm_prompt_input,
    vlm_to_original_bounds,
)
from src.vlm.prompt_builder import PromptInput
from src.vlm.schema import AppIdentity, Control, PageSemanticModel, PageState, Region


def test_prepare_vlm_prompt_input_resizes_image_and_scales_candidates():
    image = Image.new("RGB", (1920, 1080), "white")
    prompt_input = PromptInput(
        screenshot=image,
        omni_candidates=[
            {"element_id": "c1", "bounds": [960, 540, 1920, 1080], "text": "right"},
            {"element_id": "outside", "bounds": [2000, 10, 2010, 20]},
        ],
        uia_elements=[{"element_id": "u1", "bounds": [0, 0, 192, 108]}],
    )

    prepared, meta = prepare_vlm_prompt_input(prompt_input, max_width=1280)

    assert prepared.screenshot.size == (1280, 720)
    assert meta.original_size == (1920, 1080)
    assert meta.vlm_size == (1280, 720)
    assert meta.candidate_count_before_filter == 3
    assert meta.candidate_count_after_filter == 2
    assert prepared.omni_candidates == [
        {"element_id": "c1", "bounds": [640, 360, 1280, 720], "text": "right"},
    ]
    assert prepared.uia_elements == [{"element_id": "u1", "bounds": [0, 0, 128, 72]}]


def test_vlm_to_original_bounds_maps_back_to_original_space():
    assert vlm_to_original_bounds([640, 360, 1280, 720], scale_x=1280 / 1920, scale_y=720 / 1080) == [
        960,
        540,
        1920,
        1080,
    ]


def test_bounds_projection_preserves_coverage_with_floor_ceil():
    assert original_to_vlm_bounds([1, 1, 10, 10], scale_x=0.5, scale_y=0.5) == [0, 0, 5, 5]
    assert vlm_to_original_bounds([1, 1, 10, 10], scale_x=0.5, scale_y=0.5) == [2, 2, 20, 20]
    assert vlm_to_original_bounds([1, 1, 10, 10], scale_x=0.66, scale_y=0.66) == [1, 1, 16, 16]


def test_normalize_model_to_original_updates_bounds_and_image_size():
    image = Image.new("RGB", (1920, 1080), "white")
    _, meta = prepare_vlm_prompt_input(PromptInput(screenshot=image), max_width=1280)
    model = PageSemanticModel(
        app_identity=AppIdentity(app_name="Test"),
        page_state=PageState(page_class="main", state_label="main"),
        image_size=[1280, 720],
        regions=[Region(region_id="r1", role="content", bounds=[0, 0, 1280, 720])],
        fixed_controls=[Control(control_id="c1", bounds=[640, 360, 1280, 720])],
        confidence=0.9,
    )

    normalized = normalize_model_to_original(model, meta)

    assert normalized.image_size == [1920, 1080]
    assert normalized.regions[0].bounds == [0, 0, 1920, 1080]
    assert normalized.fixed_controls[0].bounds == [960, 540, 1920, 1080]
