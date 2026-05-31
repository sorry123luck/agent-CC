from scripts.audit_local_evidence import analyze_canvas_detail, classify_candidate_evidence, select_detail_files


def test_classify_candidate_evidence_prefers_element_text() -> None:
    element = {
        "element_id": "button_refresh",
        "bounds": [10, 10, 60, 40],
        "control_type": "ButtonControl",
        "text": "刷新",
        "provider_sources": ["uia"],
    }

    evidence = classify_candidate_evidence(element, ocr_blocks=[], vision_candidates=[])

    assert evidence["has_local_text"] is True
    assert evidence["text_sources"] == ["element"]
    assert evidence["needs_vlm"] is False


def test_classify_candidate_evidence_uses_overlapping_ocr_text() -> None:
    element = {
        "element_id": "ocr_button",
        "bounds": [10, 10, 60, 40],
        "control_type": "ButtonControl",
        "text": "",
    }
    ocr_blocks = [{"bbox": [12, 12, 58, 38], "text": "发送", "confidence": 0.92}]

    evidence = classify_candidate_evidence(element, ocr_blocks=ocr_blocks, vision_candidates=[])

    assert evidence["has_local_text"] is True
    assert evidence["text_sources"] == ["ocr"]
    assert evidence["local_text"] == "发送"
    assert evidence["needs_vlm"] is False


def test_classify_candidate_evidence_keeps_icon_with_ocr_noise_for_vlm() -> None:
    element = {
        "element_id": "icon_ocr_noise",
        "bounds": [10, 10, 40, 40],
        "control_type": "icon",
        "text": "",
    }
    ocr_blocks = [{"bbox": [12, 12, 38, 38], "text": "口", "confidence": 0.9}]

    evidence = classify_candidate_evidence(element, ocr_blocks=ocr_blocks, vision_candidates=[])

    assert evidence["has_local_text"] is False
    assert evidence["weak_ocr_text"] == "口"
    assert evidence["needs_vlm"] is True


def test_classify_candidate_evidence_flags_plain_icon_for_vlm() -> None:
    element = {
        "element_id": "vision_icon",
        "bounds": [10, 10, 30, 30],
        "control_type": "icon",
        "text": "",
        "provider_sources": ["omniparser"],
    }

    evidence = classify_candidate_evidence(element, ocr_blocks=[], vision_candidates=[])

    assert evidence["has_local_text"] is False
    assert evidence["needs_vlm"] is True
    assert evidence["candidate_kind"] == "icon"


def test_analyze_canvas_detail_summarizes_roi_local_evidence() -> None:
    detail = {
        "canvas_id": "sample",
        "page_class": "demo/page",
        "roi_selection_plan": {
            "mode": "list_management",
            "rois": [
                {
                    "roi_id": "roi_toolbar",
                    "purpose": "toolbar_search_filters",
                    "bounds": [0, 0, 200, 80],
                    "candidate_ids": ["text_button", "plain_icon", "ocr_button"],
                }
            ],
        },
        "elements": [
            {"element_id": "text_button", "bounds": [10, 10, 60, 40], "control_type": "ButtonControl", "text": "刷新"},
            {"element_id": "plain_icon", "bounds": [70, 10, 95, 35], "control_type": "icon", "text": ""},
            {"element_id": "ocr_button", "bounds": [110, 10, 170, 40], "control_type": "ButtonControl", "text": ""},
        ],
        "artifacts": {
            "ocr_blocks": [{"bbox": [112, 12, 168, 38], "text": "新增", "confidence": 0.9}],
            "vision_candidates": [],
        },
    }

    summary = analyze_canvas_detail(detail, source_file="sample.detail.json")

    roi = summary["rois"][0]
    assert roi["candidate_count"] == 3
    assert roi["local_text_count"] == 2
    assert roi["needs_vlm_count"] == 1
    assert roi["local_text_ratio"] == 0.6667
    assert roi["needs_vlm_ids"] == ["plain_icon"]


def test_analyze_canvas_detail_accepts_top_level_canvas_detail_evidence() -> None:
    detail = {
        "canvas_id": "api_detail",
        "page_class": "demo/page",
        "roi_selection_plan": {
            "mode": "list_management",
            "rois": [
                {
                    "roi_id": "roi_toolbar",
                    "purpose": "toolbar_search_filters",
                    "bounds": [0, 0, 200, 80],
                    "candidate_ids": ["ocr_button"],
                }
            ],
        },
        "elements": [
            {"element_id": "ocr_button", "bounds": [110, 10, 170, 40], "control_type": "ButtonControl", "text": ""},
        ],
        "ocr_blocks": [{"bbox": [112, 12, 168, 38], "text": "新增", "confidence": 0.9}],
        "vision_candidates": [],
    }

    summary = analyze_canvas_detail(detail, source_file="api_detail.json")

    roi = summary["rois"][0]
    assert roi["local_text_count"] == 1
    assert roi["source_counts"]["ocr"] == 1
    assert roi["needs_vlm_count"] == 0


def test_select_detail_files_prefers_pre_vlm_detail_over_post_vlm(tmp_path) -> None:
    plain = tmp_path / "App_snap_1.detail.json"
    post = tmp_path / "App_snap_1.post_vlm.detail.json"
    other = tmp_path / "Other_snap_2.detail.json"
    plain.write_text("{}", encoding="utf-8")
    post.write_text("{}", encoding="utf-8")
    other.write_text("{}", encoding="utf-8")

    selected = select_detail_files([plain, post, other])

    assert selected == [plain, other]
