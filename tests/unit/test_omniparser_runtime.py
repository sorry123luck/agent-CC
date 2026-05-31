import sys
from pathlib import Path

from PIL import Image


RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "omniparser_runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from util.utils import get_som_labeled_img, remove_overlap_new


def test_remove_overlap_new_returns_dicts_without_ocr():
    boxes = [
        {
            "type": "icon",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "interactivity": True,
            "content": None,
            "source": "box_yolo_content_yolo",
        }
    ]

    filtered = remove_overlap_new(boxes=boxes, iou_threshold=0.7, ocr_bbox=[])

    assert len(filtered) == 1
    assert isinstance(filtered[0], dict)
    assert filtered[0]["bbox"] == [0.1, 0.1, 0.2, 0.2]
    assert filtered[0]["content"] is None


def test_get_som_labeled_img_accepts_empty_ocr(monkeypatch):
    class FakeTensor:
        device = "cpu"

        def __truediv__(self, _other):
            return self

        def to(self, _device):
            return self

        def tolist(self):
            return [[0.1, 0.1, 0.2, 0.2]]

    monkeypatch.setattr("util.utils.predict_yolo", lambda **_kwargs: (FakeTensor(), [0.9], ["0"]))
    monkeypatch.setattr(
        "util.utils.annotate",
        lambda image_source, boxes, logits, phrases, **_kwargs: (image_source, {0: [10, 10, 20, 20]}),
    )

    encoded_image, label_coordinates, parsed_content = get_som_labeled_img(
        image_source=Image.new("RGB", (100, 100), color="white"),
        model=object(),
        ocr_bbox=[],
        ocr_text=[],
        use_local_semantics=False,
        output_coord_in_ratio=True,
    )

    assert encoded_image
    assert label_coordinates
    assert parsed_content[0]["content"] is None
