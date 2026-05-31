"""Tests for region-first ROI VLM supplement contracts."""

from __future__ import annotations

from src.vlm.roi_contract import (
    build_roi_vlm_jobs,
    merge_roi_vlm_response,
    record_roi_vlm_timeout,
    validate_roi_vlm_response,
)


def _canvas():
    class Canvas:
        canvas_id = "canvas_1"

        def __init__(self):
            self.app = type("App", (), {"process_name": "qq.exe"})()
            self.window = type("Window", (), {"title": "QQ"})()
            self.elements = [
                type("Element", (), {"element_id": "elem_1"})(),
                type("Element", (), {"element_id": "elem_2"})(),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_0",
                            "bounds": [0, 0, 400, 300],
                            "purpose": "message_stream",
                            "candidate_ids": ["elem_1"],
                            "allowed_outputs": [
                                "region_semantics",
                                "candidate_annotations",
                                "review_only_hints",
                            ],
                        }
                    ],
                }
            }

    return Canvas()


def _refinable_canvas():
    class Element:
        def __init__(self, element_id):
            self.element_id = element_id
            self.role_label = None
            self.role_source = ""
            self.role_confidence = 0.0
            self.role_evidence = []
            self.semantic_tags = []
            self.refine_status = "unreviewed"

    canvas = _canvas()
    canvas.elements = [Element("elem_1"), Element("elem_2")]
    return canvas


def test_build_roi_vlm_jobs_uses_local_roi_and_candidate_ids_only():
    jobs = build_roi_vlm_jobs(_canvas())

    assert len(jobs) == 1
    assert jobs[0]["canvas_id"] == "canvas_1"
    assert jobs[0]["roi_id"] == "roi_0"
    assert jobs[0]["bounds"] == [0, 0, 400, 300]
    assert jobs[0]["app_process"] == "qq.exe"
    assert jobs[0]["window_title"] == "QQ"
    assert jobs[0]["candidate_ids"] == ["elem_1"]
    assert jobs[0]["allowed_outputs"] == [
        "region_semantics",
        "candidate_annotations",
        "review_only_hints",
    ]
    assert jobs[0]["candidate_refs"] == [{"marker": "C1", "candidate_id": "elem_1"}]


def test_build_roi_vlm_jobs_sorts_candidate_refs_by_visual_position():
    class Element:
        def __init__(self, element_id, bounds):
            self.element_id = element_id
            self.bounds = bounds
            self.text = ""
            self.name = ""
            self.control_type = "Button"

    class Canvas:
        canvas_id = "canvas_refs"

        def __init__(self):
            self.elements = [
                Element("bottom", (10, 210, 40, 240)),
                Element("top_right", (110, 20, 140, 50)),
                Element("top_left", (10, 20, 40, 50)),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "collaboration_inbox",
                    "rois": [
                        {
                            "roi_id": "roi_rail",
                            "bounds": [0, 0, 180, 300],
                            "purpose": "app_rail",
                            "candidate_ids": ["bottom", "top_right", "top_left"],
                        }
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["top_left", "top_right", "bottom"]
    assert jobs[0]["candidate_refs"] == [
        {"marker": "C1", "candidate_id": "top_left", "bounds": [10, 20, 40, 50], "control_type": "Button"},
        {"marker": "C2", "candidate_id": "top_right", "bounds": [110, 20, 140, 50], "control_type": "Button"},
        {"marker": "C3", "candidate_id": "bottom", "bounds": [10, 210, 40, 240], "control_type": "Button"},
    ]


def test_build_roi_vlm_jobs_assigns_overlapping_candidates_to_first_roi_only():
    class Canvas:
        canvas_id = "canvas_overlap"

        def __init__(self):
            self.elements = [
                type("Element", (), {"element_id": "elem_shared"})(),
                type("Element", (), {"element_id": "elem_feature"})(),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "security_dashboard",
                    "rois": [
                        {
                            "roi_id": "roi_status",
                            "bounds": [0, 0, 400, 200],
                            "purpose": "security_status",
                            "priority": 1,
                            "candidate_ids": ["elem_shared"],
                        },
                        {
                            "roi_id": "roi_feature",
                            "bounds": [0, 120, 400, 500],
                            "purpose": "feature_grid",
                            "priority": 2,
                            "candidate_ids": ["elem_shared", "elem_feature"],
                        },
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["elem_shared"]
    assert jobs[1]["candidate_ids"] == ["elem_feature"]


def test_build_roi_vlm_jobs_filters_large_container_candidates_from_vlm_payload():
    class Element:
        def __init__(self, element_id, bounds, semantic_role="unknown", control_type=""):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = semantic_role
            self.control_type = control_type
            self.text = ""
            self.name = ""

    class Canvas:
        canvas_id = "canvas_dirty_candidates"

        def __init__(self):
            self.elements = [
                Element("root_window", (-7, 0, 1009, 738), "container", "WindowControl"),
                Element("render_pane", (1, 0, 1001, 730), "layout", "PaneControl"),
                Element("send_icon", (474, 685, 504, 715), "unknown", "icon"),
                Element("emoji_icon", (315, 684, 350, 717), "unknown", "icon"),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [180, 629, 1002, 731],
                            "purpose": "composer",
                            "candidate_ids": ["root_window", "render_pane", "send_icon", "emoji_icon"],
                        }
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["emoji_icon", "send_icon"]
    assert [ref["marker"] for ref in jobs[0]["candidate_refs"]] == ["C1", "C2"]


def test_build_roi_vlm_jobs_excludes_text_backed_candidates_from_vlm_payload():
    class Element:
        def __init__(self, element_id, bounds, control_type="", text="", attrs=None):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = control_type
            self.text = text
            self.name = ""
            self.value = ""
            self.attributes = attrs or {}

    class Canvas:
        canvas_id = "canvas_text_backed"

        def __init__(self):
            self.elements = [
                Element("refresh_button", (10, 10, 70, 40), "ButtonControl", "刷新"),
                Element("ocr_button", (80, 10, 140, 40), "ButtonControl", "", {"ocr_text": "新增"}),
                Element("plain_icon", (150, 10, 180, 40), "icon", ""),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "list_management",
                    "rois": [
                        {
                            "roi_id": "roi_toolbar",
                            "bounds": [0, 0, 200, 80],
                            "purpose": "toolbar_search_filters",
                            "candidate_ids": ["refresh_button", "ocr_button", "plain_icon"],
                        }
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["plain_icon"]
    assert jobs[0]["local_text_candidate_ids"] == ["refresh_button", "ocr_button"]


def test_build_roi_vlm_jobs_does_not_trust_ocr_text_on_icon_candidates():
    class Element:
        def __init__(self, element_id, bounds, control_type="", text="", attrs=None):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = control_type
            self.text = text
            self.name = ""
            self.value = ""
            self.attributes = attrs or {}

    class Canvas:
        canvas_id = "canvas_icon_ocr_noise"

        def __init__(self):
            self.elements = [
                Element("icon_ocr_noise", (10, 10, 40, 40), "icon", "", {"ocr_text": "口"}),
                Element("button_ocr_text", (50, 10, 120, 40), "ButtonControl", "", {"ocr_text": "刷新"}),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "list_management",
                    "rois": [
                        {
                            "roi_id": "roi_toolbar",
                            "bounds": [0, 0, 140, 60],
                            "purpose": "toolbar_search_filters",
                            "candidate_ids": ["icon_ocr_noise", "button_ocr_text"],
                        }
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["icon_ocr_noise"]
    assert jobs[0]["local_text_candidate_ids"] == ["button_ocr_text"]


def test_build_roi_vlm_jobs_excludes_titlebar_icons_from_content_roi():
    class Element:
        def __init__(self, element_id, bounds, control_type="icon"):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = control_type
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_detail_pane"

        def __init__(self):
            self.elements = [
                Element("close_icon", (871, 0, 899, 28)),
                Element("minimize_icon", (915, 0, 949, 28)),
                Element("detail_action", (720, 160, 760, 200)),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "collaboration_inbox",
                    "rois": [
                        {
                            "roi_id": "roi_detail",
                            "bounds": [590, 0, 1018, 768],
                            "purpose": "detail_pane",
                            "candidate_ids": ["close_icon", "minimize_icon", "detail_action"],
                        }
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["detail_action"]


def test_build_roi_vlm_jobs_uses_canvas_ocr_blocks_as_local_text_evidence():
    class Element:
        def __init__(self, element_id, bounds, control_type="", text=""):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = control_type
            self.text = text
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_top_level_ocr"

        def __init__(self):
            self.elements = [
                Element("ocr_button", (10, 10, 80, 40), "ButtonControl"),
                Element("icon_ocr_noise", (90, 10, 120, 40), "icon"),
                Element("plain_icon", (130, 10, 160, 40), "icon"),
            ]
            self.artifacts = {
                "ocr_blocks": [
                    {"bbox": [12, 12, 78, 38], "text": "保存", "confidence": 0.9},
                    {"bbox": [92, 12, 118, 38], "text": "口", "confidence": 0.9},
                ],
                "roi_selection_plan": {
                    "mode": "list_management",
                    "rois": [
                        {
                            "roi_id": "roi_toolbar",
                            "bounds": [0, 0, 180, 60],
                            "purpose": "toolbar_search_filters",
                            "candidate_ids": ["ocr_button", "icon_ocr_noise", "plain_icon"],
                        }
                    ],
                },
            }

    jobs = build_roi_vlm_jobs(Canvas())

    assert jobs[0]["candidate_ids"] == ["icon_ocr_noise", "plain_icon"]
    assert jobs[0]["local_text_candidate_ids"] == ["ocr_button"]


def test_build_roi_vlm_jobs_adds_chat_composer_position_hints():
    class Element:
        def __init__(self, element_id, bounds):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = "icon"
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_composer_hints"

        def __init__(self):
            self.elements = [
                Element("emoji", (220, 686, 250, 716)),
                Element("attachment", (260, 686, 290, 716)),
                Element("file", (300, 686, 330, 716)),
                Element("screenshot", (340, 686, 370, 716)),
                Element("voice", (390, 686, 420, 716)),
                Element("send", (900, 686, 970, 716)),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [180, 629, 1000, 731],
                            "purpose": "composer",
                            "candidate_ids": ["emoji", "attachment", "file", "screenshot", "voice", "send"],
                        }
                    ],
                }
            }

    jobs = build_roi_vlm_jobs(Canvas())

    hints = {ref["candidate_id"]: ref.get("local_hint") for ref in jobs[0]["candidate_refs"]}
    assert hints == {
        "emoji": "emoji_button",
        "attachment": "attachment_or_app_button",
        "file": "file_button",
        "screenshot": "screenshot_button",
        "voice": "voice_input_button",
        "send": "send_button",
    }


def test_build_roi_vlm_jobs_prioritizes_rightmost_send_in_chat_composer():
    class Element:
        def __init__(self, element_id, bounds):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = "icon"
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_composer_send_priority"

        def __init__(self):
            self.elements = [
                Element("emoji", (315, 681, 350, 713)),
                Element("attachment", (351, 680, 385, 713)),
                Element("file", (389, 682, 419, 711)),
                Element("screenshot", (425, 682, 469, 710)),
                Element("voice", (475, 682, 504, 711)),
                Element("more", (865, 681, 896, 714)),
                Element("send", (918, 683, 978, 709)),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [180, 629, 1000, 731],
                            "purpose": "composer",
                            "candidate_ids": [
                                "attachment",
                                "emoji",
                                "more",
                                "file",
                                "screenshot",
                                "voice",
                                "send",
                            ],
                        }
                    ],
                }
            }

    job = build_roi_vlm_jobs(Canvas())[0]
    hints = {ref["candidate_id"]: ref.get("local_hint") for ref in job["candidate_refs"]}

    assert hints["send"] == "send_button"
    assert "more" not in hints
    assert job["candidate_ids"] == ["emoji", "attachment", "file", "send", "screenshot", "voice"]
    assert [ref["marker"] for ref in job["candidate_refs"]] == ["C1", "C2", "C3", "C4", "C5", "C6"]


def test_build_roi_vlm_jobs_prefers_bottom_right_send_over_top_toolbar_edge():
    class Element:
        def __init__(self, element_id, bounds):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = "icon"
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_qq_composer_send_priority"

        def __init__(self):
            self.elements = [
                Element("emoji", (323, 433, 363, 475)),
                Element("attachment", (364, 434, 409, 475)),
                Element("file", (408, 435, 452, 474)),
                Element("top_right_tool", (950, 434, 960, 477)),
                Element("send", (844, 598, 942, 627)),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [172, 425, 960, 640],
                            "purpose": "composer",
                            "candidate_ids": ["emoji", "attachment", "file", "top_right_tool", "send"],
                        }
                    ],
                }
            }

    job = build_roi_vlm_jobs(Canvas())[0]
    hints = {ref["candidate_id"]: ref.get("local_hint") for ref in job["candidate_refs"]}

    assert hints["send"] == "send_button"
    assert "top_right_tool" not in hints
    assert job["candidate_ids"] == ["emoji", "attachment", "file", "send"]


def test_build_roi_vlm_jobs_keeps_qq_group_send_inside_middle_chat_width():
    class Element:
        def __init__(self, element_id, bounds):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = "icon"
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_qq_group_composer_send_priority"

        def __init__(self):
            self.elements = [
                Element("emoji", (324, 433, 362, 475)),
                Element("attachment", (365, 434, 407, 474)),
                Element("file", (408, 436, 452, 472)),
                Element("screenshot", (454, 440, 481, 467)),
                Element("voice", (489, 434, 529, 474)),
                Element("clock", (733, 439, 764, 471)),
                Element("send", (664, 598, 762, 626)),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [172, 425, 960, 640],
                            "purpose": "composer",
                            "candidate_ids": ["emoji", "attachment", "file", "screenshot", "voice", "clock", "send"],
                        }
                    ],
                }
            }

    job = build_roi_vlm_jobs(Canvas())[0]
    hints = {ref["candidate_id"]: ref.get("local_hint") for ref in job["candidate_refs"]}

    assert hints["send"] == "send_button"
    assert "clock" not in hints
    assert job["candidate_ids"] == ["emoji", "attachment", "file", "send", "screenshot", "voice"]


def test_build_roi_vlm_jobs_dedupes_collaboration_composer_controls():
    class Element:
        def __init__(self, element_id, bounds, control_type):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = control_type
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_collab_composer_dedupe"

        def __init__(self):
            self.elements = [
                Element("group_emoji", (762, 702, 788, 728), "GroupControl"),
                Element("button_emoji", (762, 702, 788, 728), "ButtonControl"),
                Element("group_attach", (792, 702, 818, 728), "GroupControl"),
                Element("button_attach", (792, 702, 818, 728), "ButtonControl"),
                Element("decorative_line", (736, 706, 951, 724), "ImageControl"),
                Element("send_icon", (958, 703, 982, 732), "icon"),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "collaboration_inbox",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [500, 602, 1000, 760],
                            "purpose": "composer",
                            "candidate_ids": [
                                "group_emoji",
                                "button_emoji",
                                "group_attach",
                                "button_attach",
                                "decorative_line",
                                "send_icon",
                            ],
                        }
                    ],
                }
            }

    job = build_roi_vlm_jobs(Canvas())[0]

    assert job["candidate_ids"] == ["button_emoji", "button_attach", "send_icon"]
    assert "decorative_line" not in job["candidate_ids"]
    assert [ref["marker"] for ref in job["candidate_refs"]] == ["C1", "C2", "C3"]
    assert job["candidate_refs"][-1]["local_hint"] == "send_button"


def test_build_roi_vlm_jobs_prefers_wide_bottom_send_parent_over_dropdown_icon():
    class Element:
        def __init__(self, element_id, bounds, control_type):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = "unknown"
            self.control_type = control_type
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_collab_send_parent"

        def __init__(self):
            self.elements = [
                Element("emoji", (762, 702, 788, 728), "ButtonControl"),
                Element("attach", (792, 702, 818, 728), "ButtonControl"),
                Element("file", (822, 702, 856, 728), "ButtonControl"),
                Element("send_parent", (890, 702, 980, 728), "ButtonControl"),
                Element("send_dropdown_icon", (958, 704, 982, 732), "icon"),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "collaboration_inbox",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [509, 661, 1018, 768],
                            "purpose": "composer",
                            "candidate_ids": [
                                "emoji",
                                "attach",
                                "file",
                                "send_parent",
                                "send_dropdown_icon",
                            ],
                        }
                    ],
                }
            }

    job = build_roi_vlm_jobs(Canvas())[0]
    hints = {ref["candidate_id"]: ref.get("local_hint") for ref in job["candidate_refs"]}

    assert hints["send_parent"] == "send_button"
    assert "send_dropdown_icon" not in hints


def test_build_roi_vlm_jobs_excludes_synthetic_message_input_from_composer_payload():
    class Element:
        def __init__(self, element_id, bounds, semantic_role, control_type):
            self.element_id = element_id
            self.bounds = bounds
            self.semantic_role = semantic_role
            self.control_type = control_type
            self.text = ""
            self.name = ""
            self.value = ""
            self.attributes = {}

    class Canvas:
        canvas_id = "canvas_composer_synthetic_input"

        def __init__(self):
            self.elements = [
                Element("synthetic_chat_composer_input", (316, 530, 655, 622), "message_input", "SyntheticInputControl"),
                Element("emoji", (665, 598, 692, 626), "unknown", "icon"),
                Element("send", (930, 598, 960, 626), "unknown", "icon"),
            ]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_composer",
                            "bounds": [172, 551, 960, 640],
                            "purpose": "composer",
                            "candidate_ids": ["synthetic_chat_composer_input", "emoji", "send"],
                        }
                    ],
                }
            }

    job = build_roi_vlm_jobs(Canvas())[0]

    assert job["candidate_ids"] == ["emoji", "send"]
    assert "synthetic_chat_composer_input" not in job["local_text_candidate_ids"]


def test_validate_roi_vlm_response_rejects_unknown_roi_and_candidates():
    canvas = _canvas()

    invalid = validate_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_missing",
            "candidate_annotations": [{"candidate_id": "elem_missing", "label": "Send"}],
        },
    )

    assert invalid.accepted is False
    assert "unknown_roi_id" in invalid.errors
    assert "unknown_candidate_id:elem_missing" in invalid.errors


def test_merge_roi_vlm_response_accepts_late_semantics_but_strips_bounds():
    canvas = _canvas()

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "message_stream", "bounds": [1, 2, 3, 4]},
            "candidate_annotations": [
                {"candidate_id": "elem_1", "label": "latest message", "bounds": [5, 6, 7, 8]},
            ],
            "review_only_hints": ["may contain unread messages"],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    saved = canvas.artifacts["roi_vlm_semantic_supplements"][0]
    assert saved["status"] == "timeout_late_success"
    assert saved["late_result_accepted"] is True
    assert saved["region_semantics"] == {"role": "message_stream"}
    assert saved["candidate_annotations"] == [{"candidate_id": "elem_1", "label": "latest message"}]


def test_timeout_keeps_job_context_so_late_response_can_merge():
    canvas = _canvas()
    job = build_roi_vlm_jobs(canvas)[0]

    record_roi_vlm_timeout(canvas, job, timeout_seconds=2.0)
    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "composer"},
            "candidate_annotations": [{"candidate_id": "elem_1", "label": "input area"}],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    assert canvas.artifacts["roi_vlm_timeouts"][0]["roi_id"] == "roi_0"
    assert canvas.artifacts["roi_vlm_timeouts"][0]["discard_late_response"] is False
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["late_result_accepted"] is True


def test_merge_roi_vlm_response_projects_candidate_annotations_to_elements():
    canvas = _refinable_canvas()

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "toolbar"},
            "candidate_annotations": [
                {"candidate_id": "elem_1", "role": "send", "label": "send button"},
            ],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    elem_1 = canvas.elements[0]
    elem_2 = canvas.elements[1]
    assert elem_1.role_label == "send button"
    assert elem_1.role_source == "roi_vlm"
    assert elem_1.role_confidence == 0.72
    assert elem_1.refine_status == "refined"
    assert "roi_vlm:toolbar" in elem_1.role_evidence
    assert "roi_vlm.role.send" in elem_1.semantic_tags
    assert elem_2.role_label is None


def test_merge_roi_vlm_response_does_not_overwrite_manual_or_memory_labels():
    canvas = _refinable_canvas()
    canvas.elements[0].role_label = "manual send"
    canvas.elements[0].role_source = "manual"
    canvas.elements[0].refine_status = "refined"

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "candidate_annotations": [
                {"candidate_id": "elem_1", "role": "send", "label": "send button"},
            ],
        },
        status="success",
    )

    assert merged.accepted is True
    assert canvas.elements[0].role_label == "manual send"
    assert canvas.elements[0].role_source == "manual"


def test_merge_roi_vlm_response_does_not_project_low_information_generic_labels():
    canvas = _refinable_canvas()

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "message_stream"},
            "candidate_annotations": [
                {"candidate_id": "elem_1", "role": "action", "label": "icon"},
                {"candidate_id": "elem_2", "role": "action"},
            ],
        },
        status="success",
    )

    assert merged.accepted is True
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["candidate_annotations"] == [
        {"candidate_id": "elem_1", "role": "action", "label": "icon"},
        {"candidate_id": "elem_2", "role": "action"},
    ]
    assert canvas.elements[0].role_label is None
    assert canvas.elements[0].role_source == ""
    assert canvas.elements[1].role_label is None
    assert canvas.elements[1].role_source == ""


def test_merge_roi_vlm_response_does_not_project_actions_inside_dynamic_content_roi():
    canvas = _refinable_canvas()

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "message_stream"},
            "candidate_annotations": [
                {"candidate_id": "elem_1", "role": "send", "label": "send"},
                {"candidate_id": "elem_2", "role": "input", "label": "input"},
            ],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    assert canvas.elements[0].role_label is None
    assert canvas.elements[0].role_source == ""
    assert canvas.elements[1].role_label is None
    assert canvas.elements[1].role_source == ""


def test_merge_roi_vlm_response_does_not_project_descriptive_icons_inside_dynamic_content_roi():
    canvas = _refinable_canvas()

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "detail_pane"},
            "candidate_annotations": [
                {"candidate_id": "elem_1", "role": "icon", "label": "angry face emoji"},
            ],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    assert canvas.elements[0].role_label is None
    assert canvas.elements[0].role_source == ""


def test_merge_roi_vlm_response_does_not_project_actions_inside_message_thread_roi():
    canvas = _refinable_canvas()

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "message_thread"},
            "candidate_annotations": [
                {"candidate_id": "elem_1", "role": "send", "label": "send"},
            ],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    assert canvas.elements[0].role_label is None
    assert canvas.elements[0].role_source == ""


def test_merge_roi_vlm_response_does_not_project_composer_icon_only_guess_labels():
    class Element:
        def __init__(self, element_id):
            self.element_id = element_id
            self.control_type = "icon"
            self.text = ""
            self.name = ""
            self.value = ""
            self.role_label = None
            self.role_source = ""
            self.role_confidence = 0.0
            self.role_evidence = []
            self.semantic_tags = []
            self.refine_status = "unreviewed"

    canvas = _canvas()
    canvas.elements = [Element("mic_icon"), Element("folder_icon")]

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "composer"},
            "candidate_annotations": [
                {"candidate_id": "mic_icon", "role": "action", "label": "send"},
                {"candidate_id": "folder_icon", "role": "action", "label": "voice"},
            ],
        },
        status="success",
    )

    assert merged.accepted is True
    assert canvas.elements[0].role_label is None
    assert canvas.elements[0].role_source == ""
    assert canvas.elements[1].role_label is None
    assert canvas.elements[1].role_source == ""


def test_merge_roi_vlm_response_does_not_project_review_only_message_input_candidate():
    class Element:
        element_id = "synthetic_chat_composer_input"
        semantic_role = "message_input"
        control_type = "SyntheticInputControl"
        text = ""
        name = ""
        value = ""
        role_label = "消息输入区候选"
        role_source = "geometry_fallback"
        role_confidence = 0.35
        role_evidence = ["chat composer has bottom toolbar/send controls but no input candidate"]
        semantic_tags = []
        refine_status = "uncertain"
        attributes = {
            "candidate_kind": "message_input_candidate",
            "actionability": "review",
            "safe_to_type": False,
            "needs_manual_label": True,
        }

    canvas = _canvas()
    canvas.elements = [Element()]
    canvas.artifacts["roi_selection_plan"]["rois"][0].update(
        {
            "purpose": "composer",
            "candidate_ids": ["synthetic_chat_composer_input"],
        }
    )

    merged = merge_roi_vlm_response(
        canvas,
        {
            "roi_id": "roi_0",
            "region_semantics": {"role": "composer"},
            "candidate_annotations": [
                {
                    "candidate_id": "synthetic_chat_composer_input",
                    "role": "input",
                    "label": "message input",
                },
            ],
        },
        status="timeout_late_success",
    )

    assert merged.accepted is True
    element = canvas.elements[0]
    assert element.role_label == "消息输入区候选"
    assert element.role_source == "geometry_fallback"
    assert element.refine_status == "uncertain"
    assert element.attributes["actionability"] == "review"
    assert element.attributes["safe_to_type"] is False
