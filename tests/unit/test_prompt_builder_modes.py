from PIL import Image

from src.vlm.prompt_builder import PromptInput, build_page_understanding_prompt


def test_candidate_annotation_prompt_uses_schema_v2_and_vlm_image_size():
    req = build_page_understanding_prompt(
        PromptInput(
            screenshot=Image.new("RGB", (1280, 720), "white"),
            omni_candidates=[{"element_id": "local_1", "bounds": [0, 0, 100, 40], "text": "搜索"}],
            task_mode="candidate_annotation",
        ),
    )

    assert '"schema_version": "2.0"' in req.system_prompt
    assert '"task_mode": "candidate_annotation"' in req.system_prompt
    assert '"image_size": [1280, 720]' in req.system_prompt
    assert "candidate_annotations 中不要输出 bounds" in req.system_prompt
    user_text = req.messages[0]["content"][0]["text"]
    assert "[local_1]" in user_text
    assert "bounds=[0, 0, 100, 40]" in user_text


def test_full_page_prompt_keeps_schema_v1_for_fallback():
    req = build_page_understanding_prompt(
        PromptInput(
            screenshot=Image.new("RGB", (800, 600), "white"),
            task_mode="full_page_recognition",
        ),
    )

    assert '"schema_version": "1.0"' in req.system_prompt
    assert "尽可能多地识别控件" in req.system_prompt


def test_region_understanding_prompt_uses_schema_v2_regions_not_controls():
    req = build_page_understanding_prompt(
        PromptInput(
            screenshot=Image.new("RGB", (960, 540), "white"),
            omni_candidates=[{"element_id": "local_1", "bounds": [0, 0, 100, 540]}],
            task_mode="region_understanding",
        ),
    )

    assert '"schema_version": "2.0"' in req.system_prompt
    assert '"task_mode": "region_understanding"' in req.system_prompt
    assert '"image_size": [960, 540]' in req.system_prompt
    assert "这个模式不负责控件检测" in req.system_prompt


def test_missing_audit_prompt_outputs_suggestions_not_safe_controls():
    req = build_page_understanding_prompt(
        PromptInput(
            screenshot=Image.new("RGB", (1280, 720), "white"),
            omni_candidates=[{"element_id": "local_1", "bounds": [20, 20, 60, 60]}],
            task_mode="missing_audit",
        ),
    )

    assert '"task_mode": "missing_audit"' in req.system_prompt
    assert '"missing_suggestions"' in req.system_prompt
    assert "不直接新增可点击控件" in req.system_prompt
    assert "不能输出 safe" in req.system_prompt


def test_icon_crop_prompt_outputs_candidate_annotations_without_bounds():
    req = build_page_understanding_prompt(
        PromptInput(
            screenshot=Image.new("RGB", (96, 96), "white"),
            omni_candidates=[{"element_id": "icon_1", "bounds": [4, 4, 92, 92]}],
            task_mode="icon_crop_understanding",
        ),
    )

    assert '"task_mode": "icon_crop_understanding"' in req.system_prompt
    assert '"candidate_annotations"' in req.system_prompt
    assert "不要重新输出 bounds" in req.system_prompt
    assert "[icon_1]" in req.messages[0]["content"][0]["text"]
