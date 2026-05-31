"""Tests for ROI VLM provider worker."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.vlm.provider import ProviderCapabilities, VLMSemanticRawResponse
from src.vlm.roi_provider_worker import create_configured_roi_provider, run_roi_vlm_provider_job


class FakeRoiProvider:
    name = "fake"
    model_id = "fake/roi"
    cost_tier = 1
    capabilities = ProviderCapabilities(max_tokens_limit=512)

    def __init__(self):
        self.requests = []
        self.expected_size = (200, 100)

    def is_available(self):
        return True

    def analyze_page(self, request):
        self.requests.append(request)
        if self.expected_size is not None:
            assert request.screenshot.size == self.expected_size
        assert "ROI semantic supplement" in request.system_prompt
        prompt_payload = json.loads(request.messages[0]["content"][0]["text"])
        return VLMSemanticRawResponse(
            raw_text=json.dumps({
                "roi_id": prompt_payload.get("roi_id", "roi_0"),
                "region_semantics": {"role": "message_stream", "bounds": [1, 2, 3, 4]},
                "candidate_annotations": [
                    {"candidate_id": "elem_1", "label": "latest message", "bounds": [5, 6, 7, 8]},
                ],
                "review_only_hints": ["contains chat history"],
            }),
            provider_name="fake",
            model_name="roi",
            finish_reason="stop",
            latency_ms=123,
            token_input=10,
            token_output=20,
        )


class MarkerOnlyProvider(FakeRoiProvider):
    def analyze_page(self, request):
        self.requests.append(request)
        prompt_payload = json.loads(request.messages[0]["content"][0]["text"])
        assert prompt_payload["candidate_refs"][0]["marker"] == "C1"
        return VLMSemanticRawResponse(
            raw_text=json.dumps({
                "roi_id": prompt_payload.get("roi_id", "roi_0"),
                "candidate_annotations": [
                    {"marker": "C1", "label": "日历", "role": "tab"},
                ],
            }),
            provider_name="fake",
            model_name="roi",
            finish_reason="stop",
            latency_ms=88,
            token_input=8,
            token_output=9,
        )


class MalformedProvider(FakeRoiProvider):
    def analyze_page(self, request):
        self.requests.append(request)
        return VLMSemanticRawResponse(
            raw_text='{"roi_id":"roi_rail","candidate_annotations":[{"marker":"C1"',
            provider_name="fake",
            model_name="roi",
            finish_reason="stop",
            latency_ms=77,
            token_input=7,
            token_output=8,
        )


def test_run_roi_vlm_provider_job_returns_contract_response():
    screenshot = Image.new("RGB", (400, 300), "white")
    provider = FakeRoiProvider()
    job = {
        "roi_id": "roi_0",
        "bounds": [0, 0, 200, 100],
        "purpose": "message_stream",
        "candidate_ids": ["elem_1"],
    }

    result = run_roi_vlm_provider_job(
        job,
        screenshot=screenshot,
        provider=provider,
        timeout_seconds=1.25,
    )

    assert result["roi_id"] == "roi_0"
    assert result["region_semantics"]["role"] == "message_stream"
    assert result["candidate_annotations"][0]["candidate_id"] == "elem_1"
    assert result["_provider"]["latency_ms"] == 123
    assert provider.requests[0].provider_options["timeout"] == 1.25
    assert provider.requests[0].provider_options["response_contract"] == "roi_semantic_supplement"


def test_run_roi_vlm_provider_job_resolves_marker_annotations_to_candidate_ids():
    screenshot = Image.new("RGB", (120, 120), "white")
    provider = MarkerOnlyProvider()
    provider.expected_size = (80, 80)
    job = {
        "roi_id": "roi_rail",
        "bounds": [0, 0, 80, 80],
        "mode": "collaboration_inbox",
        "purpose": "app_rail",
        "candidate_ids": ["vision_6"],
        "candidate_refs": [
            {"marker": "C1", "candidate_id": "vision_6", "bounds": [7, 6, 49, 53]},
        ],
    }

    result = run_roi_vlm_provider_job(job, screenshot=screenshot, provider=provider)

    assert result["candidate_annotations"] == [
        {"marker": "C1", "label": "日历", "role": "tab", "candidate_id": "vision_6"}
    ]


def test_run_roi_vlm_provider_job_sends_marker_overlay_image_to_provider():
    screenshot = Image.new("RGB", (120, 120), "white")
    provider = MarkerOnlyProvider()
    provider.expected_size = (80, 80)
    job = {
        "roi_id": "roi_rail",
        "bounds": [0, 0, 80, 80],
        "mode": "collaboration_inbox",
        "purpose": "app_rail",
        "candidate_ids": ["vision_6"],
        "candidate_refs": [
            {"marker": "C1", "candidate_id": "vision_6", "bounds": [7, 6, 49, 53]},
        ],
    }

    run_roi_vlm_provider_job(job, screenshot=screenshot, provider=provider)

    marked = provider.requests[0].screenshot.convert("RGB")
    assert marked.getbbox() is not None
    assert any(pixel != (255, 255, 255) for pixel in marked.getdata())


def test_chat_composer_icon_jobs_use_tight_candidate_union_crop():
    screenshot = Image.new("RGB", (400, 240), "white")
    provider = MarkerOnlyProvider()
    provider.expected_size = (112, 52)
    job = {
        "roi_id": "roi_composer",
        "bounds": [100, 100, 350, 180],
        "mode": "chat_workspace",
        "purpose": "composer",
        "candidate_ids": ["emoji", "attachment"],
        "candidate_refs": [
            {"marker": "C1", "candidate_id": "emoji", "bounds": [120, 130, 140, 150], "control_type": "icon"},
            {
                "marker": "C2",
                "candidate_id": "attachment",
                "bounds": [180, 130, 200, 150],
                "control_type": "icon",
            },
        ],
    }

    run_roi_vlm_provider_job(job, screenshot=screenshot, provider=provider, profile="fast", max_candidate_ids=4)

    prompt_payload = json.loads(provider.requests[0].messages[0]["content"][0]["text"])
    assert prompt_payload["region_options"] == ["composer", "toolbar"]
    assert provider.requests[0].screenshot.size == (112, 52)


def test_run_roi_vlm_provider_job_writes_debug_bundle_for_sent_image(tmp_path, monkeypatch):
    screenshot = Image.new("RGB", (120, 120), "white")
    provider = MarkerOnlyProvider()
    provider.expected_size = (80, 80)
    monkeypatch.setenv("OPENCLAW_ROI_VLM_DEBUG_DIR", str(tmp_path))
    job = {
        "canvas_id": "canvas_debug",
        "roi_id": "roi_rail",
        "bounds": [0, 0, 80, 80],
        "mode": "collaboration_inbox",
        "purpose": "app_rail",
        "candidate_ids": ["vision_6"],
        "candidate_refs": [
            {"marker": "C1", "candidate_id": "vision_6", "bounds": [7, 6, 49, 53]},
        ],
    }

    result = run_roi_vlm_provider_job(job, screenshot=screenshot, provider=provider)

    debug_dir = Path(result["_debug_artifacts"]["dir"])
    assert debug_dir.is_dir()
    assert (debug_dir / "sent.png").is_file()
    assert (debug_dir / "prompt.json").is_file()
    assert (debug_dir / "raw.txt").is_file()
    assert (debug_dir / "parsed.json").is_file()
    assert (debug_dir / "candidate_refs.json").is_file()
    assert json.loads((debug_dir / "candidate_refs.json").read_text(encoding="utf-8")) == [
        {"marker": "C1", "candidate_id": "vision_6", "bounds": [7, 6, 49, 53]}
    ]


def test_run_roi_vlm_provider_job_keeps_debug_bundle_when_json_parse_fails(tmp_path, monkeypatch):
    screenshot = Image.new("RGB", (120, 120), "white")
    provider = MalformedProvider()
    provider.expected_size = None
    monkeypatch.setenv("OPENCLAW_ROI_VLM_DEBUG_DIR", str(tmp_path))
    job = {
        "canvas_id": "canvas_debug",
        "roi_id": "roi_rail",
        "bounds": [0, 0, 80, 80],
        "mode": "collaboration_inbox",
        "purpose": "app_rail",
        "candidate_ids": ["vision_6"],
        "candidate_refs": [
            {"marker": "C1", "candidate_id": "vision_6", "bounds": [7, 6, 49, 53]},
        ],
    }

    try:
        run_roi_vlm_provider_job(job, screenshot=screenshot, provider=provider)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected malformed JSON to raise")

    assert "debug_dir=" in message
    debug_dir = Path(message.split("debug_dir=", 1)[1])
    assert (debug_dir / "sent.png").is_file()
    assert (debug_dir / "raw.txt").read_text(encoding="utf-8").startswith('{"roi_id"')
    parsed = json.loads((debug_dir / "parsed.json").read_text(encoding="utf-8"))
    assert parsed["error"] == "roi_vlm_json_parse_failed"


def test_configured_roi_provider_forces_thinking_off(monkeypatch):
    class SemanticModeler:
        enabled = True
        provider = "qwen"
        api_key = "secret"
        endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen3-vl-flash-2026-01-22"
        provider_variant = "qwen"
        free_model_only = False
        thinking_mode = "on"
        image_max_width = 1280
        proxy_url = ""
        proxy_port = 7890

    class Config:
        semantic_modeler = SemanticModeler()

    captured = {}

    def fake_load_config():
        return Config()

    def fake_create_modeler_provider(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.common.config_manager.load_config", fake_load_config)
    monkeypatch.setattr("src.vlm.roi_provider_worker.create_modeler_provider", fake_create_modeler_provider)

    provider = create_configured_roi_provider()

    assert provider is not None
    assert captured["thinking_mode"] == "off"
    assert captured["proxy_port"] == 7890


def test_run_roi_vlm_provider_job_fast_profile_limits_prompt_and_image():
    screenshot = Image.new("RGB", (1600, 1200), "white")
    provider = FakeRoiProvider()
    provider.capabilities = ProviderCapabilities(max_tokens_limit=2048)
    provider.expected_size = None
    job = {
        "roi_id": "roi_fast",
        "bounds": [0, 0, 1200, 800],
        "mode": "list_management",
        "purpose": "table_or_list",
        "candidate_ids": [f"elem_{i}" for i in range(30)],
    }

    result = run_roi_vlm_provider_job(
        job,
        screenshot=screenshot,
        provider=provider,
        profile="fast",
        max_candidate_ids=6,
        image_max_edge=512,
        max_tokens=384,
    )

    request = provider.requests[0]
    prompt_text = request.messages[0]["content"][0]["text"]
    assert result["roi_id"] == "roi_fast"
    assert max(request.screenshot.size) == 512
    assert request.max_tokens == 384
    assert '"elem_0"' in prompt_text
    assert '"elem_5"' in prompt_text
    assert '"elem_6"' not in prompt_text
    assert "review_only_hints" not in prompt_text
    assert "table" in prompt_text.lower()


def test_fast_prompt_includes_candidate_refs_for_visual_binding():
    from src.vlm.roi_provider_worker import _build_user_prompt

    prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_rail",
            "mode": "collaboration_inbox",
            "purpose": "app_rail",
            "candidate_ids": ["vision_6"],
            "candidate_refs": [
                {"marker": "C1", "candidate_id": "vision_6", "bounds": [7, 6, 49, 53]},
            ],
        },
        profile="fast",
    ))

    assert prompt["candidate_refs"] == [{"marker": "C1", "candidate_id": "vision_6"}]
    assert "Use visible marker labels, not list order" in prompt["rules"]
    assert "Do not annotate a candidate if the best label is only icon/action/item/button" in prompt["rules"]


def test_fast_prompt_includes_local_hints_as_weak_priors():
    from src.vlm.roi_provider_worker import _build_user_prompt

    prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_composer",
            "mode": "chat_workspace",
            "purpose": "composer",
            "candidate_ids": ["emoji"],
            "candidate_refs": [
                {
                    "marker": "C1",
                    "candidate_id": "emoji",
                    "control_type": "icon",
                    "local_hint": "emoji_button",
                },
            ],
        },
        profile="fast",
    ))

    assert prompt["candidate_refs"] == [
        {"marker": "C1", "candidate_id": "emoji", "control_type": "icon", "local_hint": "emoji_button"}
    ]
    assert "local_hint is only a weak prior; override it when visual evidence disagrees" in prompt["rules"]


def test_fast_prompt_avoids_placeholder_values():
    from src.vlm.roi_provider_worker import _build_user_prompt

    prompt = _build_user_prompt(
        {
            "roi_id": "roi_0",
            "mode": "chat_workspace",
            "purpose": "message_stream",
            "candidate_ids": ["elem_0", "elem_1"],
        },
        profile="fast",
    )

    assert '"role":"short"' not in prompt
    assert '"label":"short"' not in prompt
    assert "role_options" in prompt
    assert "Copy no placeholder values" in prompt


def test_fast_prompt_for_empty_candidate_ids_disables_candidate_annotations():
    from src.vlm.roi_provider_worker import _build_user_prompt

    prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_empty",
            "mode": "chat_workspace",
            "purpose": "navigation_and_list",
            "candidate_ids": [],
        },
        profile="fast",
    ))

    assert prompt["candidate_ids"] == []
    assert prompt["candidate_annotation_policy"] == "return_empty_array"
    assert "candidate_annotations" not in prompt["json_keys"]


def test_fast_prompt_sets_annotation_budget_from_candidate_ids():
    from src.vlm.roi_provider_worker import _build_user_prompt

    prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_dense",
            "mode": "control_matrix",
            "purpose": "middle_control_columns",
            "candidate_ids": [f"elem_{idx}" for idx in range(8)],
        },
        profile="fast",
    ))

    assert prompt["candidate_annotation_policy"] == "annotate_up_to_8_candidate_ids"
    assert prompt["max_candidate_annotations"] == 8
    assert "Return up to max_candidate_annotations candidate annotations" in prompt["rules"]


def test_fast_prompt_scopes_region_examples_by_mode():
    from src.vlm.roi_provider_worker import _build_user_prompt

    file_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_file",
            "mode": "file_search",
            "purpose": "result_list",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    chat_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_chat",
            "mode": "chat_workspace",
            "purpose": "message_stream",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))

    assert "chat_area" not in file_prompt["region_options"]
    assert "result_list" in file_prompt["region_options"]
    assert "file" in file_prompt["role_options"]
    assert "message" not in file_prompt["role_options"]
    assert "chat_area" in chat_prompt["region_options"]
    assert "message" in chat_prompt["role_options"]


def test_chat_workspace_fast_prompt_scopes_options_by_roi_purpose():
    from src.vlm.roi_provider_worker import _build_user_prompt

    navigation_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_nav",
            "mode": "chat_workspace",
            "purpose": "navigation_and_list",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    message_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_messages",
            "mode": "chat_workspace",
            "purpose": "message_stream",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    composer_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_composer",
            "mode": "chat_workspace",
            "purpose": "composer",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))

    assert "navigation_list" in navigation_prompt["region_options"]
    assert "chat_area" not in navigation_prompt["region_options"]
    assert "message_stream" in message_prompt["region_options"]
    assert "navigation_list" not in message_prompt["region_options"]
    assert composer_prompt["region_options"] == ["composer", "toolbar"]


def test_qq_message_stream_fast_prompt_can_classify_private_or_group_area():
    from src.vlm.roi_provider_worker import _build_user_prompt

    prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_messages",
            "mode": "chat_workspace",
            "purpose": "message_stream",
            "app_process": "qq.exe",
            "window_title": "QQ",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))

    assert prompt["app_process"] == "qq.exe"
    assert "region_semantics.summary" in prompt["json_keys"]
    assert "private_chat_area" in prompt["region_options"]
    assert "group_chat_area" in prompt["region_options"]
    assert "classify active chat variant" in prompt["task"]
    assert any(
        "region_semantics.role must be one of private_chat_area" in rule
        for rule in prompt["rules"]
    )


def test_security_and_control_prompts_scope_options_by_roi_purpose():
    from src.vlm.roi_provider_worker import _build_user_prompt

    feature_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_features",
            "mode": "security_dashboard",
            "purpose": "feature_grid",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    bottom_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_bottom",
            "mode": "security_dashboard",
            "purpose": "bottom_actions",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    middle_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_middle",
            "mode": "control_matrix",
            "purpose": "middle_control_columns",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    master_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_master",
            "mode": "control_matrix",
            "purpose": "right_master_section",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))

    assert feature_prompt["region_options"] == ["feature_grid", "security_feature_list"]
    assert "security_status" not in feature_prompt["region_options"]
    assert bottom_prompt["region_options"] == ["bottom_actions", "toolbar"]
    assert "control_columns" in middle_prompt["region_options"]
    assert "master_section" not in middle_prompt["region_options"]
    assert "master_section" in master_prompt["region_options"]


def test_collaboration_inbox_prompt_scopes_options_by_roi_purpose():
    from src.vlm.roi_provider_worker import _build_user_prompt

    app_rail_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_rail",
            "mode": "collaboration_inbox",
            "purpose": "app_rail",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    inbox_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_inbox",
            "mode": "collaboration_inbox",
            "purpose": "inbox_list",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    thread_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_thread",
            "mode": "collaboration_inbox",
            "purpose": "message_thread",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))
    composer_prompt = json.loads(_build_user_prompt(
        {
            "roi_id": "roi_composer",
            "mode": "collaboration_inbox",
            "purpose": "composer",
            "candidate_ids": ["elem_0"],
        },
        profile="fast",
    ))

    assert app_rail_prompt["region_options"] == ["app_rail", "workspace_navigation", "toolbar"]
    assert "inbox_list" in inbox_prompt["region_options"]
    assert "conversation" in inbox_prompt["role_options"]
    assert "message_thread" in thread_prompt["region_options"]
    assert "message" in thread_prompt["role_options"]
    assert composer_prompt["region_options"] == ["composer", "toolbar"]
    assert "send" in composer_prompt["role_options"]
