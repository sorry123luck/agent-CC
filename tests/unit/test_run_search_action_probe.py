"""Tests for controlled search action probe planning/reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_search_action_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_search_action_probe", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_probe_targets_attaches_default_queries_and_skips_blocked_rows():
    module = _load_module()
    plan = {
        "rows": [
            {"sample": "qq", "hwnd": 11, "process_name": "qq.exe", "status": "ready", "search_click": [163, 42]},
            {"sample": "wechat", "hwnd": 12, "process_name": "weixin.exe", "status": "ready", "search_click": [158, 55]},
            {"sample": "feishu", "hwnd": 13, "process_name": "feishu.exe", "status": "ready", "search_click": [94, 72]},
            {"sample": "bad", "hwnd": 14, "process_name": "qq.exe", "status": "blocked", "search_click": []},
        ]
    }

    targets = module.build_probe_targets(plan, queries={})

    assert [(item["process_name"], item["query"]) for item in targets] == [
        ("qq.exe", "sample_contact"),
        ("weixin.exe", "示例联系人"),
        ("feishu.exe", "示例联系人"),
    ]
    assert targets[0]["probe_strategy"] == "click_search"
    assert targets[2]["probe_strategy"] == "global_search_hotkey"


def test_build_probe_targets_allows_query_override():
    module = _load_module()
    plan = {"rows": [{"sample": "qq", "hwnd": 11, "process_name": "qq.exe", "status": "ready", "search_click": [1, 2]}]}

    targets = module.build_probe_targets(plan, queries={"qq.exe": "override"})

    assert targets[0]["query"] == "override"


def test_build_probe_targets_refreshes_hwnd_from_current_windows():
    module = _load_module()
    module._window_is_actionable = lambda hwnd: hwnd == 456
    plan = {
        "rows": [
            {
                "sample": "wechat",
                "hwnd": 123,
                "process_name": "weixin.exe",
                "status": "ready",
                "search_click": [158, 55],
            }
        ]
    }

    targets = module.build_probe_targets(
        plan,
        queries={},
        current_windows=[
            {"hwnd": 456, "process_name": "weixin.exe", "is_minimized": False},
        ],
    )

    assert targets[0]["hwnd"] == 456
    assert targets[0]["original_hwnd"] == 123
    assert targets[0]["hwnd_refreshed"] is True


def test_fresh_hwnd_skips_non_actionable_current_window(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: hwnd == 456)

    hwnd = module._fresh_hwnd_for_process(
        "weixin.exe",
        [
            {"hwnd": 999, "process_name": "weixin.exe", "is_minimized": False},
            {"hwnd": 456, "process_name": "weixin.exe", "is_minimized": False},
        ],
    )

    assert hwnd == 456


def test_build_probe_targets_keeps_original_hwnd_when_actionable(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: hwnd in {123, 456})
    plan = {
        "rows": [
            {
                "sample": "wechat",
                "hwnd": 123,
                "process_name": "weixin.exe",
                "status": "ready",
                "search_click": [158, 55],
            }
        ]
    }

    targets = module.build_probe_targets(
        plan,
        queries={},
        current_windows=[
            {"hwnd": 456, "process_name": "weixin.exe", "is_minimized": False},
        ],
    )

    assert targets[0]["hwnd"] == 123
    assert "hwnd_refreshed" not in targets[0]


def test_missing_ready_processes_only_includes_targets_without_visible_window():
    module = _load_module()
    module._window_is_actionable = lambda hwnd: hwnd == 222
    plan = {
        "rows": [
            {"process_name": "weixin.exe", "status": "ready", "search_click": [1, 1]},
            {"process_name": "qq.exe", "status": "ready", "search_click": [1, 1]},
            {"process_name": "feishu.exe", "status": "blocked", "search_click": [1, 1]},
        ]
    }

    missing = module._missing_ready_processes(
        plan,
        current_windows=[
            {"process_name": "qq.exe", "hwnd": 222, "is_minimized": False, "width": 800, "height": 600},
        ],
    )

    assert missing == ["weixin.exe"]


def test_tray_control_matching_is_process_specific():
    module = _load_module()

    assert module._tray_control_matches("weixin.exe", " 微信")
    assert module._tray_control_matches("wechat.exe", "WeChat")
    assert module._tray_control_matches("qq.exe", " QQ: 1077239875\n声音: 开启")
    assert not module._tray_control_matches("weixin.exe", " QQ: 1077239875")
    assert not module._tray_control_matches("feishu.exe", " 微信")


def test_run_probe_plan_dry_run_does_not_restore_from_tray(monkeypatch, tmp_path):
    module = _load_module()
    called: list[str] = []
    monkeypatch.setattr(module, "_fetch_windows", lambda _base_url: [])
    monkeypatch.setattr(module, "_restore_process_from_tray", lambda process: called.append(process) or True)
    plan = {
        "rows": [
            {"sample": "wechat", "hwnd": 123, "process_name": "weixin.exe", "status": "ready", "search_click": [1, 1]},
        ]
    }

    report = module.run_probe_plan(
        plan=plan,
        output_dir=tmp_path,
        base_url="http://127.0.0.1:8000",
        queries={},
        execute=False,
        wait_seconds=0,
    )

    assert called == []
    assert report["restored_from_tray"] == []


def test_run_probe_plan_execute_restores_missing_ready_process(monkeypatch, tmp_path):
    module = _load_module()
    calls = {"fetch": 0, "restore": []}

    def fake_fetch(_base_url):
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            return []
        return [{"hwnd": 456, "process_name": "weixin.exe", "is_minimized": False}]

    monkeypatch.setattr(module, "_fetch_windows", fake_fetch)
    monkeypatch.setattr(module, "_restore_process_from_tray", lambda process: calls["restore"].append(process) or True)
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: hwnd == 456)
    monkeypatch.setattr(module, "_run_one_target", lambda **kwargs: {"hwnd": kwargs["target"]["hwnd"], "status": "captured"})
    plan = {
        "rows": [
            {"sample": "wechat", "hwnd": 123, "process_name": "weixin.exe", "status": "ready", "search_click": [1, 1]},
        ]
    }

    report = module.run_probe_plan(
        plan=plan,
        output_dir=tmp_path,
        base_url="http://127.0.0.1:8000",
        queries={},
        execute=True,
        wait_seconds=0,
    )

    assert calls["restore"] == ["weixin.exe"]
    assert report["restored_from_tray"] == ["weixin.exe"]
    assert report["rows"][0]["hwnd"] == 456


def test_ensure_window_actionable_restores_hidden_tray_window(monkeypatch):
    module = _load_module()
    states = [False, True]
    restored: list[str] = []

    monkeypatch.setattr(module.action_ctx, "window_is_actionable", lambda _hwnd: states.pop(0))
    monkeypatch.setattr(module.action_ctx, "restore_hidden_window", lambda _hwnd: False)
    monkeypatch.setattr(module.action_ctx, "restore_process_from_tray", lambda process: restored.append(process) or True)

    module._ensure_window_actionable(123, "weixin.exe")

    assert restored == ["weixin.exe"]


def test_ensure_window_actionable_restores_existing_hidden_hwnd_before_tray(monkeypatch):
    module = _load_module()
    states = [False, True]
    calls = {"hidden": [], "tray": []}

    monkeypatch.setattr(module.action_ctx, "window_is_actionable", lambda _hwnd: states.pop(0))
    monkeypatch.setattr(module.action_ctx, "restore_hidden_window", lambda hwnd: calls["hidden"].append(hwnd) or True)
    monkeypatch.setattr(module.action_ctx, "restore_process_from_tray", lambda process: calls["tray"].append(process) or True)

    module._ensure_window_actionable(123, "weixin.exe")

    assert calls == {"hidden": [123], "tray": []}


def test_ensure_window_actionable_polls_after_tray_restore(monkeypatch):
    module = _load_module()
    states = [False, False, False, True]
    sleeps: list[float] = []

    monkeypatch.setattr(module.action_ctx, "window_is_actionable", lambda _hwnd: states.pop(0))
    monkeypatch.setattr(module.action_ctx, "restore_hidden_window", lambda _hwnd: False)
    monkeypatch.setattr(module.action_ctx, "restore_process_from_tray", lambda _process: True)
    monkeypatch.setattr(module.action_ctx.time, "sleep", lambda seconds: sleeps.append(seconds))

    module._ensure_window_actionable(123, "weixin.exe")

    assert sleeps == [0.25, 0.25]


def test_ensure_window_actionable_fails_when_restore_does_not_show_window(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(module.action_ctx, "window_is_actionable", lambda _hwnd: False)
    monkeypatch.setattr(module.action_ctx, "restore_hidden_window", lambda _hwnd: False)
    monkeypatch.setattr(module.action_ctx, "restore_process_from_tray", lambda _process: True)

    try:
        module._ensure_window_actionable(123, "weixin.exe")
    except ValueError as exc:
        assert "not actionable" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_plan_accepts_utf8_bom(tmp_path):
    module = _load_module()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text('\ufeff{"rows": []}', encoding="utf-8")

    plan = module.load_plan(plan_path)

    assert plan == {"rows": []}


def test_extract_canvas_state_summarizes_model_and_mode():
    module = _load_module()
    detail = {
        "canvas_id": "snap_1",
        "page_model_id": "pm_chat",
        "state_template_id": "st_search",
        "page_class": "chat_workspace",
        "state_label": "search results",
        "window_title": "QQ",
        "roi_selection_plan": {"mode": "chat_search_results"},
        "visual_pattern": {"mode": "chat_search_results"},
    }

    state = module.extract_canvas_state(detail)

    assert state == {
        "canvas_id": "snap_1",
        "page_model_id": "pm_chat",
        "state_template_id": "st_search",
        "page_class": "chat_workspace",
        "state_label": "search results",
        "mode": "chat_search_results",
        "window_title": "QQ",
    }


def test_build_state_transition_audit_records_stage_changes():
    module = _load_module()
    before = {"page_model_id": "pm_chat", "state_template_id": "st_chat", "mode": "chat_workspace"}
    search = {"page_model_id": "pm_chat", "state_template_id": "st_search", "mode": "chat_search_results"}
    selected = {"page_model_id": "pm_chat", "state_template_id": "st_chat", "mode": "chat_workspace"}

    audit = module.build_state_transition_audit(before=before, search=search, selected=selected)

    assert audit["before_to_search"] == {
        "page_model_changed": False,
        "state_template_changed": True,
        "mode_changed": True,
    }
    assert audit["search_to_selected"]["state_template_changed"] is True
    assert audit["selected_returned_to_before_state"] is True
    assert audit["path"] == "chat_workspace -> chat_search_results -> chat_workspace"


def test_global_search_input_click_uses_top_center_overlay_region():
    module = _load_module()

    click = module._global_search_input_click([100, 200, 1100, 900])

    assert click == [500, 84]


def test_pre_probe_escape_count_for_chat_search_apps():
    module = _load_module()

    assert module._pre_probe_escape_count("feishu.exe") == 2
    assert module._pre_probe_escape_count("qq.exe") == 1
    assert module._pre_probe_escape_count("weixin.exe") == 0
    assert module._pre_probe_escape_count("notepad.exe") == 0


def test_post_selection_escape_count_for_wechat_search_overlay():
    module = _load_module()

    assert module._post_selection_escape_count("weixin.exe") == 0
    assert module._post_selection_escape_count("wechat.exe") == 0
    assert module._post_selection_escape_count("qq.exe") == 0
    assert module._post_selection_escape_count("feishu.exe") == 0


def test_write_probe_report_outputs_json_and_markdown(tmp_path):
    module = _load_module()
    report = {
        "overall_status": "review",
        "rows": [
            {
                "sample": "qq",
                "process_name": "qq.exe",
                "query": "sample_contact",
                "status": "captured",
                "after_canvas_id": "snap_after",
                "after_screenshot": str(tmp_path / "qq_after.png"),
                "result_candidate_count": 2,
                "selected_result": {"click": [185, 122], "text": "sample_contact"},
                "selected_canvas_id": "snap_selected",
                "stage_states": {
                    "before": {"mode": "chat_workspace"},
                    "search": {"mode": "chat_search_results"},
                    "selected": {"mode": "chat_workspace"},
                },
                "state_transition": {"path": "chat_workspace -> chat_search_results -> chat_workspace"},
                "notes": ["search_state_changed"],
            }
        ],
    }

    module.write_probe_report(report, tmp_path)

    saved = json.loads((tmp_path / "search_probe_report.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "search_probe_report.md").read_text(encoding="utf-8")
    assert saved["overall_status"] == "review"
    assert "Search Probe Report" in markdown
    assert "result_count" in markdown
    assert "state_path" in markdown
    assert "| qq | qq.exe | sample_contact | captured | snap_after | snap_selected | 2 | [185, 122] | chat_workspace -> chat_search_results -> chat_workspace |" in markdown
    assert "selected_result" in saved["rows"][0]


def test_probe_overall_status_passes_when_all_searches_roundtrip():
    module = _load_module()
    rows = [
        {"status": "captured", "notes": ["search_state_captured", "selected_first_result"], "state_transition": {"path": "chat_workspace -> chat_search_results -> chat_workspace"}},
        {"status": "captured", "notes": ["search_state_captured", "selected_first_result"], "state_transition": {"path": "collaboration_inbox -> collaboration_search_overlay -> collaboration_inbox"}},
    ]

    assert module._probe_overall_status(rows, execute=True) == "pass"


def test_probe_overall_status_reviews_missing_search_capture():
    module = _load_module()
    rows = [{"status": "captured", "notes": ["search_state_not_captured"], "state_transition": {"path": "chat_workspace -> chat_workspace"}}]

    assert module._probe_overall_status(rows, execute=True) == "review"


def test_extract_result_options_prefers_qq_first_matching_contact_over_global_search():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_search_results"},
        "elements": [
            {"element_id": "tab", "control_type": "TextControl", "bounds": [80, 72, 116, 87], "text": "联系人"},
            {"element_id": "first", "control_type": "ListItemControl", "bounds": [60, 94, 310, 151], "text": "sample_contact(心有余) 来自: 我的好友"},
            {"element_id": "second", "control_type": "ListItemControl", "bounds": [60, 151, 310, 206], "text": "sample_contact(……) 来自: 我的好友"},
            {"element_id": "global", "control_type": "ListItemControl", "bounds": [60, 324, 310, 376], "text": "进入全网搜索sample_contact 查找用户、群聊等"},
        ],
    }

    options = module.extract_result_options(detail, process_name="qq.exe", query="sample_contact")

    assert len(options) == 2
    assert options[0]["element_id"] == "first"
    assert options[0]["click"] == [185, 122]
    assert options[0]["selection_policy"] == "first_matching_result"


def test_extract_result_options_keeps_qq_compact_text_match_rows():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_search_results"},
        "elements": [
            {"element_id": "search_query", "control_type": "TextControl", "bounds": [82, 34, 98, 50], "text": "Q sample_contact"},
            {"element_id": "first_text", "control_type": "TextControl", "bounds": [125, 84, 136, 102], "text": "sample_contact"},
            {"element_id": "global", "control_type": "ListItemControl", "bounds": [60, 324, 310, 376], "text": "进入全网搜索sample_contact 查找用户、群聊等"},
        ],
    }

    options = module.extract_result_options(detail, process_name="qq.exe", query="sample_contact")

    assert len(options) == 1
    assert options[0]["element_id"] == "first_text"
    assert options[0]["click"] == [130, 93]


def test_extract_result_options_uses_wechat_visual_rows_when_text_is_missing():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_search_results"},
        "elements": [
            {"element_id": "search_icon", "control_type": "icon", "bounds": [76, 44, 99, 68], "text": ""},
            {"element_id": "first_row", "control_type": "icon", "bounds": [59, 275, 299, 342], "text": ""},
            {"element_id": "second_row", "control_type": "icon", "bounds": [60, 341, 302, 408], "text": ""},
        ],
    }

    options = module.extract_result_options(detail, process_name="weixin.exe", query="示例联系人")

    assert len(options) == 2
    assert options[0]["element_id"] == "first_row"
    assert options[0]["click"] == [179, 308]
    assert options[0]["source"] == "visual_row_fallback"


def test_extract_result_options_does_not_use_wechat_sidebar_rows_when_mode_stays_workspace():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_workspace"},
        "elements": [
            {"element_id": "search_box", "control_type": "icon", "bounds": [76, 43, 1090, 68], "text": ""},
            {"element_id": "first_row", "control_type": "icon", "bounds": [59, 276, 299, 342], "text": ""},
        ],
    }

    options = module.extract_result_options(detail, process_name="weixin.exe", query="示例联系人")

    assert options == []


def test_search_mode_captured_requires_app_specific_search_mode():
    module = _load_module()

    assert module._search_mode_captured("weixin.exe", {"mode": "chat_search_results"}) is True
    assert module._search_mode_captured("weixin.exe", {"mode": "chat_workspace"}) is False
    assert module._search_mode_captured("qq.exe", {"mode": "chat_search_results"}) is True
    assert module._search_mode_captured("feishu.exe", {"mode": "collaboration_search_overlay"}) is True


def test_search_observe_capture_mode_uses_screen_region_for_wechat_popups():
    module = _load_module()

    assert module._search_observe_capture_mode("weixin.exe") == "screen_region"
    assert module._search_observe_capture_mode("wechat.exe") == "screen_region"
    assert module._search_observe_capture_mode("qq.exe") == "window"
    assert module._search_observe_capture_mode("feishu.exe") == "window"


def test_extract_result_options_prefers_wechat_contact_row_fragments_over_web_results():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_search_results"},
        "elements": [
            {"element_id": "contact_avatar", "control_type": "icon", "bounds": [67, 90, 116, 137], "text": ""},
            {"element_id": "contact_name", "control_type": "icon", "bounds": [115, 91, 202, 114], "text": ""},
            {"element_id": "web_result", "control_type": "icon", "bounds": [61, 213, 295, 277], "text": ""},
            {"element_id": "chat_record", "control_type": "icon", "bounds": [62, 276, 293, 342], "text": ""},
        ],
    }

    options = module.extract_result_options(detail, process_name="weixin.exe", query="示例联系人")

    assert len(options) >= 1
    assert options[0]["element_id"] == "contact_avatar"
    assert options[0]["click"] == [178, 113]
    assert options[0]["source"] == "contact_row_fragment"


def test_extract_result_options_uses_wechat_screen_region_popup_fragments():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_search_results"},
        "elements": [
            {"element_id": "search_icon", "control_type": "icon", "bounds": [76, 43, 98, 68], "text": ""},
            {"element_id": "query_text", "control_type": "icon", "bounds": [98, 43, 122, 68], "text": ""},
            {"element_id": "clear", "control_type": "icon", "bounds": [231, 45, 252, 69], "text": ""},
            {"element_id": "avatar", "control_type": "icon", "bounds": [80, 115, 126, 155], "text": ""},
            {"element_id": "name_fragment", "control_type": "icon", "bounds": [124, 118, 149, 154], "text": ""},
            {"element_id": "web_result", "control_type": "icon", "bounds": [84, 234, 319, 268], "text": ""},
        ],
    }

    options = module.extract_result_options(detail, process_name="weixin.exe", query="示例联系人")

    assert len(options) == 1
    assert options[0]["element_id"] == "avatar"
    assert options[0]["bounds"] == [60, 109, 396, 165]
    assert options[0]["source"] == "contact_row_fragment"


def test_extract_result_options_rejects_wechat_web_search_only_rows():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "chat_search_results"},
        "elements": [
            {"element_id": "search_icon", "control_type": "icon", "bounds": [76, 43, 98, 68], "text": ""},
            {"element_id": "query_text", "control_type": "icon", "bounds": [98, 43, 122, 68], "text": ""},
            {"element_id": "clear", "control_type": "icon", "bounds": [231, 45, 252, 69], "text": ""},
            {"element_id": "web_section", "control_type": "TextControl", "bounds": [82, 182, 180, 205], "text": "搜索网络结果"},
            {"element_id": "web_result_1", "control_type": "icon", "bounds": [61, 213, 295, 277], "text": ""},
            {"element_id": "web_result_2", "control_type": "icon", "bounds": [62, 276, 293, 342], "text": ""},
        ],
    }

    options = module.extract_result_options(detail, process_name="weixin.exe", query="示例联系人")

    assert options == []


def test_extract_result_options_prefers_feishu_overlay_result_text():
    module = _load_module()
    detail = {
        "roi_selection_plan": {"mode": "collaboration_search_overlay"},
        "elements": [
            {"element_id": "query", "control_type": "TextControl", "bounds": [163, 81, 353, 113], "text": ""},
            {"element_id": "first", "control_type": "TextControl", "bounds": [186, 200, 228, 217], "text": "示例联系人"},
            {"element_id": "second", "control_type": "TextControl", "bounds": [186, 263, 228, 308], "text": "示例联系人"},
        ],
    }

    options = module.extract_result_options(detail, process_name="feishu.exe", query="示例联系人")

    assert len(options) == 2
    assert options[0]["element_id"] == "first"
    assert options[0]["click"] == [207, 208]

