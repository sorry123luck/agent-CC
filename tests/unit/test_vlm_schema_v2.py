import pytest

from src.vlm.schema import ValidationError, validate_page_semantic_model


def _v2_payload():
    return {
        "schema_version": "2.0",
        "task_mode": "candidate_annotation",
        "coordinate_space": "vlm_image",
        "image_size": [1280, 720],
        "app_identity": {"app_name": "WeChat", "surface_type": "native_uia"},
        "page_state": {"page_class": "chat", "state_label": "chat"},
        "layout_regions": [
            {
                "region_id": "r_sidebar",
                "role": "sidebar",
                "bounds": [0, 0, 300, 720],
                "purpose": "navigation",
            }
        ],
        "candidate_annotations": [
            {
                "candidate_id": "local_1",
                "semantic_role": "search_input",
                "control_type": "input",
                "interactable": True,
                "actionability": "safe",
                "coordinate_quality": "from_candidate",
                "confidence": 0.95,
                "reason": "search placeholder",
            }
        ],
        "confidence": 0.9,
        "needs_review": False,
    }


def test_schema_v2_candidate_annotations_convert_to_corrections():
    model, warnings = validate_page_semantic_model(
        _v2_payload(),
        image_size=(1280, 720),
        known_candidate_ids={"local_1"},
    )

    assert warnings == []
    assert model.schema_version == "2.0"
    assert model.fixed_controls == []
    assert model.regions[0].bounds == [0, 0, 300, 720]
    assert len(model.candidate_corrections) == 1
    corr = model.candidate_corrections[0]
    assert corr.candidate_id == "local_1"
    assert corr.corrected_role == "search_input"
    assert corr.corrected_type == "input"
    assert corr.corrected_confidence == 0.95


def test_schema_v2_rejects_region_bounds_outside_vlm_image():
    payload = _v2_payload()
    payload["layout_regions"][0]["bounds"] = [0, 0, 1281, 720]

    with pytest.raises(ValidationError):
        validate_page_semantic_model(payload, image_size=(1280, 720), known_candidate_ids={"local_1"})


def test_schema_v2_unknown_candidate_is_skipped_not_safe():
    model, warnings = validate_page_semantic_model(
        _v2_payload(),
        image_size=(1280, 720),
        known_candidate_ids={"different"},
    )

    assert model.candidate_corrections == []
    assert any("unknown candidate" in warning for warning in warnings)


def test_schema_v2_preserves_missing_suggestions_as_non_safe_artifacts():
    payload = _v2_payload()
    payload["task_mode"] = "missing_audit"
    payload["candidate_annotations"] = []
    payload["missing_suggestions"] = [
        {
            "suggestion_id": "m1",
            "type": "possible_missing_control",
            "description": "right toolbar icon may need crop classification",
            "rough_area": [1200, 20, 1270, 80],
            "recommended_next_step": "crop_and_classify",
            "confidence": 0.7,
            "actionability": "safe",
        }
    ]

    model, warnings = validate_page_semantic_model(
        payload,
        image_size=(1280, 720),
        known_candidate_ids={"local_1"},
    )

    assert len(model.missing_suggestions) == 1
    assert model.missing_suggestions[0]["rough_area"] == [1200, 20, 1270, 80]
    assert model.missing_suggestions[0]["actionability"] == "review"
    assert any("downgraded" in warning for warning in warnings)


def test_schema_v2_rejects_missing_suggestion_rough_area_outside_image():
    payload = _v2_payload()
    payload["task_mode"] = "missing_audit"
    payload["candidate_annotations"] = []
    payload["missing_suggestions"] = [
        {
            "suggestion_id": "m1",
            "type": "possible_missing_control",
            "rough_area": [1200, 20, 1300, 80],
        }
    ]

    with pytest.raises(ValidationError):
        validate_page_semantic_model(payload, image_size=(1280, 720), known_candidate_ids={"local_1"})


def test_schema_v2_icon_crop_mode_is_supported():
    payload = _v2_payload()
    payload["task_mode"] = "icon_crop_understanding"
    payload["layout_regions"] = []

    model, warnings = validate_page_semantic_model(
        payload,
        image_size=(96, 96),
        known_candidate_ids={"local_1"},
    )

    assert warnings == []
    assert model.task_mode == "icon_crop_understanding"
    assert model.candidate_corrections[0].candidate_id == "local_1"
