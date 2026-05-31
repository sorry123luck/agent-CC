from dataclasses import dataclass

from PIL import Image

from src.vlm.icon_crop import crop_candidate_icon, image_dhash


@dataclass
class _Elem:
    element_id: str
    bounds: list[int]


def test_image_dhash_is_stable_for_same_image():
    image = Image.new("RGB", (16, 16), "white")

    assert image_dhash(image) == image_dhash(image.copy())


def test_crop_candidate_icon_returns_crop_with_relative_bounds():
    screenshot = Image.new("RGB", (100, 80), "white")
    elem = _Elem("icon_1", [10, 20, 30, 40])

    crop = crop_candidate_icon(screenshot, elem, padding=4)

    assert crop is not None
    assert crop.candidate_id == "icon_1"
    assert crop.image.size == (28, 28)
    assert crop.bounds == [4, 4, 24, 24]
    assert len(crop.dhash) == 16


def test_crop_candidate_icon_rejects_invalid_bounds():
    screenshot = Image.new("RGB", (100, 80), "white")
    elem = _Elem("bad", [20, 20, 20, 30])

    assert crop_candidate_icon(screenshot, elem) is None
