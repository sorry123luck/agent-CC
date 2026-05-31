from scripts.run_roi_icon_matrix import (
    build_icon_matrix_cases,
    load_detail_and_screenshot,
    labels_match,
    parse_expected_labels,
    score_annotations,
)


def test_parse_expected_labels_requires_candidate_mapping() -> None:
    assert parse_expected_labels(["vision_15=microphone", "vision_16=folder"]) == {
        "vision_15": "microphone",
        "vision_16": "folder",
    }


def test_build_icon_matrix_cases_varies_count_crop_and_marker_policy() -> None:
    elements = {
        "vision_15": {"bounds": [474, 685, 504, 715]},
        "vision_16": {"bounds": [390, 686, 419, 714]},
        "vision_12": {"bounds": [315, 684, 350, 717]},
    }

    cases = build_icon_matrix_cases(
        roi_bounds=[180, 629, 1002, 731],
        candidate_ids=["vision_15", "vision_16", "vision_12"],
        elements_by_id=elements,
        group_sizes=[1, 2],
        variants=["roi_marked", "tight_unmarked"],
        padding=10,
    )

    assert [case.case_id for case in cases] == [
        "roi_marked_n1",
        "tight_unmarked_n1",
        "roi_marked_n2",
        "tight_unmarked_n2",
    ]
    assert cases[0].bounds == [180, 629, 1002, 731]
    assert cases[0].draw_markers is True
    assert cases[1].bounds == [464, 675, 514, 725]
    assert cases[1].draw_markers is False
    assert cases[3].candidate_ids == ["vision_15", "vision_16"]


def test_score_annotations_matches_common_icon_aliases() -> None:
    score = score_annotations(
        [
            {"candidate_id": "vision_15", "role": "voice", "label": "record audio"},
            {"candidate_id": "vision_16", "label": "file"},
            {"candidate_id": "vision_12", "label": "send"},
        ],
        {
            "vision_15": "microphone",
            "vision_16": "folder",
            "vision_12": "emoji",
            "vision_14": "attachment",
        },
    )

    assert score["correct"] == 2
    assert score["wrong"] == 1
    assert score["missing"] == 1
    assert score["accuracy"] == 0.5


def test_labels_match_normalizes_synonyms() -> None:
    assert labels_match("smile emoji", "emoji")
    assert labels_match("smiley emoji", "emoji")
    assert labels_match("smiley face emoji", "emoji")
    assert labels_match("emoji_picker", "emoji")
    assert labels_match("attach_file", "attachment")
    assert labels_match("send_button", "send")
    assert labels_match("scissors", "cut")
    assert not labels_match("send", "microphone")


def test_load_detail_and_screenshot_can_use_offline_files(tmp_path) -> None:
    from PIL import Image

    detail_path = tmp_path / "detail.json"
    image_path = tmp_path / "sample.png"
    detail_path.write_text('{"canvas_id":"offline_canvas","elements":[]}', encoding="utf-8")
    Image.new("RGB", (12, 8), "white").save(image_path)

    detail, screenshot = load_detail_and_screenshot(
        base_url="http://unused",
        canvas_id="ignored",
        detail_json=detail_path,
        screenshot_file=image_path,
    )

    assert detail["canvas_id"] == "offline_canvas"
    assert screenshot.size == (12, 8)
