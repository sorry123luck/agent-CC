from __future__ import annotations

from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    ProviderTrace,
    RiskLevel,
    RiskTag,
    SemanticRole,
    WindowInfoSnapshot,
)

from scripts import p1_quality_report


def _candidate(
    element_id: str,
    *,
    text: str = "",
    role: SemanticRole = SemanticRole.UNKNOWN,
    bounds: tuple[int, int, int, int] = (10, 10, 80, 40),
    sources: list[str] | None = None,
    risk_tags: list[str] | None = None,
    risk_level: RiskLevel = RiskLevel.L0,
) -> Candidate:
    return Candidate(
        element_id=element_id,
        text=text,
        name=text,
        semantic_role=role,
        bounds=bounds,
        click_point=((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2),
        confidence=0.8,
        provider_sources=["uia"] if sources is None else sources,
        risk_tags=risk_tags or [],
        risk_level=risk_level,
    )


def test_canvas_quality_metrics_count_sources_bounds_risk_and_evidence():
    canvas = InteractionCanvas(
        canvas_id="canvas_quality",
        app=AppInfo(app_id="wechat"),
        window=WindowInfoSnapshot(hwnd=100, rect_screen=(0, 0, 500, 400), rect_client=(0, 0, 500, 400)),
        elements=[
            _candidate("search", text="搜索", role=SemanticRole.SEARCH_INPUT, bounds=(20, 20, 200, 48)),
            _candidate("unknown", text="", role=SemanticRole.UNKNOWN, bounds=(30, 80, 80, 110), sources=[]),
            _candidate(
                "send",
                text="发送",
                role=SemanticRole.SEND_BUTTON,
                bounds=(420, 350, 480, 382),
                risk_tags=[RiskTag.SEND.value],
                risk_level=RiskLevel.L2,
            ),
            _candidate("outside", text="越界", role=SemanticRole.BUTTON, bounds=(480, 360, 530, 410)),
        ],
        provider_trace=ProviderTrace(
            uia_used=True,
            ocr_used=True,
            provider_details={
                "uia": {"element_count": 7},
                "ocr": {"block_count": 2},
                "element_sources": {"uia": 3, "ocr": 1, "vision": 1},
            },
        ),
        artifacts={
            "ocr_blocks": [{"text": "搜索", "bbox": [20, 20, 120, 48], "confidence": 0.92}],
            "vision_candidates": [{"candidate_id": "vision_1", "source": "paddleocr_bridge"}],
        },
    )
    capture = {"capture_bounds": [0, 0, 500, 400], "screenshot_size": [500, 400], "valid": True}

    metrics = p1_quality_report.evaluate_canvas_quality(canvas, capture)

    assert metrics["canvas_created"] is True
    assert metrics["ocr_block_count"] == 1
    assert metrics["uia_candidate_count"] == 7
    assert metrics["fallback_vision_candidate_count"] == 1
    assert metrics["risky_candidate_count"] == 1
    assert metrics["candidates_outside_capture_bounds_count"] == 1
    assert metrics["outside_capture_bounds_candidates"][0]["element_id"] == "outside"
    assert metrics["unknown_candidate_ratio"] == 0.25
    assert metrics["evidence_coverage"] == 0.75


def test_canvas_quality_metrics_reports_app_layout_dependency():
    canvas = InteractionCanvas(
        canvas_id="layout_dependency",
        app=AppInfo(app_id="qq"),
        elements=[
            _candidate(
                "qq_layout_search",
                text="搜索",
                role=SemanticRole.SEARCH_INPUT,
                sources=["app_layout"],
            ),
            _candidate(
                "uia_button",
                text="设置",
                role=SemanticRole.BUTTON,
                sources=["uia"],
            ),
        ],
    )

    metrics = p1_quality_report.evaluate_canvas_quality(canvas, {"screenshot_size": [800, 600]})

    assert metrics["app_layout_dependency_count"] == 1
    assert metrics["app_layout_dependency_ratio"] == 0.5
    assert metrics["app_layout_dependent_candidates"][0]["element_id"] == "qq_layout_search"


def test_canvas_quality_metrics_reports_uia_provider_budget_details():
    canvas = InteractionCanvas(
        canvas_id="uia_budget",
        app=AppInfo(app_id="chrome"),
        elements=[],
        artifacts={
            "uia_provider": {
                "mode": "bounded",
                "truncated": True,
                "elapsed_seconds": 6.1,
                "element_count": 900,
            }
        },
    )

    metrics = p1_quality_report.evaluate_canvas_quality(canvas, {"screenshot_size": [800, 600]})

    assert metrics["uia_provider_mode"] == "bounded"
    assert metrics["uia_provider_truncated"] is True
    assert metrics["uia_provider_elapsed_seconds"] == 6.1
    assert metrics["uia_provider_element_count"] == 900


def test_canvas_quality_metrics_reports_ocr_provider_fallback_details():
    canvas = InteractionCanvas(
        canvas_id="ocr_fallback",
        app=AppInfo(app_id="voicemeeter"),
        elements=[],
        artifacts={
            "ocr_provider": {
                "provider": "paddleocr_bridge",
                "success": False,
                "error": "worker_timeout",
                "elapsed_seconds": 2.5,
                "worker_reused": False,
                "startup_seconds": 12.0,
                "fallback_reason": "sparse_uia_full_window",
                "block_count": 0,
            }
        },
    )

    metrics = p1_quality_report.evaluate_canvas_quality(canvas, {"screenshot_size": [800, 600]})

    assert metrics["ocr_provider_name"] == "paddleocr_bridge"
    assert metrics["ocr_provider_success"] is False
    assert metrics["ocr_provider_error"] == "worker_timeout"
    assert metrics["ocr_provider_elapsed_seconds"] == 2.5
    assert metrics["ocr_provider_worker_reused"] is False
    assert metrics["ocr_provider_startup_seconds"] == 12.0
    assert metrics["ocr_provider_fallback_reason"] == "sparse_uia_full_window"
    assert metrics["ocr_provider_block_count"] == 0


def test_canvas_quality_does_not_mark_readonly_text_as_dangerous():
    canvas = InteractionCanvas(
        canvas_id="readonly_text",
        app=AppInfo(app_id="chrome"),
        elements=[
            _candidate(
                "text_send",
                text="输入与发送安全",
                role=SemanticRole.TEXT,
                bounds=(10, 10, 200, 40),
            ),
            _candidate(
                "real_delete",
                text="删除",
                role=SemanticRole.BUTTON,
                bounds=(10, 60, 120, 92),
            ),
        ],
    )

    metrics = p1_quality_report.evaluate_canvas_quality(canvas, {"screenshot_size": [800, 600]})

    assert metrics["dangerous_unmarked_count"] == 1
    assert metrics["dangerous_unmarked_candidates"][0]["element_id"] == "real_delete"


def test_canvas_quality_does_not_mark_long_submit_text_as_dangerous_button():
    canvas = InteractionCanvas(
        canvas_id="long_submit_text",
        app=AppInfo(app_id="vscode"),
        elements=[
            _candidate(
                "long_submit",
                text='执行： git add CLAUDE.md git commit -m "docs: align entry documents" 提交后输出 commit hash',
                role=SemanticRole.SUBMIT_BUTTON,
                bounds=(10, 10, 600, 40),
            ),
            _candidate(
                "real_submit",
                text="提交",
                role=SemanticRole.SUBMIT_BUTTON,
                bounds=(10, 60, 120, 92),
            ),
        ],
    )

    metrics = p1_quality_report.evaluate_canvas_quality(canvas, {"screenshot_size": [800, 600]})

    assert metrics["dangerous_unmarked_count"] == 1
    assert metrics["dangerous_unmarked_candidates"][0]["element_id"] == "real_submit"


def test_query_diagnostic_explains_correct_and_false_positive_matches():
    query = {
        "label": "搜索框",
        "target": {"text": "搜索"},
        "expected": {"semantic_roles": ["search_input"], "text_any": ["搜索"]},
    }
    correct = _candidate("search", text="搜索", role=SemanticRole.SEARCH_INPUT, bounds=(20, 20, 200, 48))
    wrong = _candidate("title", text="设置", role=SemanticRole.BUTTON, bounds=(20, 80, 200, 110))

    correct_diag = p1_quality_report.diagnose_query_match(
        app_id="wechat",
        query_spec=query,
        candidate=correct,
        screenshot_path="reports/p1_quality/run/screenshots/wechat.png",
    )
    wrong_diag = p1_quality_report.diagnose_query_match(
        app_id="wechat",
        query_spec=query,
        candidate=wrong,
        screenshot_path="reports/p1_quality/run/screenshots/wechat.png",
    )

    assert correct_diag["matched"] is True
    assert correct_diag["verdict"] == "TP"
    assert correct_diag["center"] == [110, 34]
    assert any("role_match" in r for r in correct_diag["reasons"])

    assert wrong_diag["matched"] is True
    assert wrong_diag["verdict"] == "FP"
    assert any("no_role_or_text_match" in r or "negative" in r or "forbidden" in r for r in wrong_diag["reasons"])


def test_build_p1_quality_report_summarizes_query_correct_rate_and_acceptance():
    app_runs = [
        {
            "app_id": "chrome",
            "software_name": "Chrome",
            "observe_status": "canvas_created",
            "quality_metrics": {
                "canvas_created": True,
                "query_count": 3,
                "query_matched_count": 3,
                "query_correct_count": 3,
                "false_positive_count": 0,
                "dangerous_unmarked_count": 0,
                "candidates_outside_capture_bounds_count": 0,
            },
            "query_diagnostics": [{}, {}, {}],
        },
        {
            "app_id": "vscode",
            "software_name": "VSCode",
            "observe_status": "canvas_created",
            "quality_metrics": {
                "canvas_created": True,
                "query_count": 3,
                "query_matched_count": 2,
                "query_correct_count": 2,
                "false_positive_count": 0,
                "dangerous_unmarked_count": 0,
                "candidates_outside_capture_bounds_count": 0,
            },
            "query_diagnostics": [{}, {}, {}],
        },
    ]

    report = p1_quality_report.build_p1_quality_report([], app_runs, min_apps=2, min_correct_rate=0.8)

    assert report["schema_version"] == "deskcanvas_p1_quality_report.v1"
    assert report["summary"]["canvas_created"] == 2
    assert report["summary"]["total_query_count"] == 6
    assert report["summary"]["query_correct_rate"] == 5 / 6
    assert report["acceptance"]["status"] == "passed"


def test_build_p1_quality_report_records_coverage_gaps_and_layout_dependency_by_app():
    app_runs = [
        {
            "app_id": "vscode",
            "software_name": "VSCode",
            "discovery_status": "not_found",
            "failure_stage": "discovery",
            "failure_reason": "path_not_found",
            "next_fix_task": "Add software path discovery or target alias coverage for this app.",
            "quality_metrics": {},
            "query_diagnostics": [],
        },
        {
            "app_id": "qq",
            "software_name": "QQ",
            "discovery_status": "found",
            "observe_status": "canvas_created",
            "quality_metrics": {
                "canvas_created": True,
                "query_count": 4,
                "query_matched_count": 4,
                "query_correct_count": 4,
                "false_positive_count": 0,
                "dangerous_unmarked_count": 0,
                "candidates_outside_capture_bounds_count": 0,
                "app_layout_dependency_count": 4,
            },
            "query_diagnostics": [{}, {}, {}, {}],
        },
    ]

    report = p1_quality_report.build_p1_quality_report([], app_runs, min_apps=1, min_correct_rate=0.7)

    assert report["summary"]["coverage_gap_count"] == 1
    assert report["summary"]["coverage_gaps"] == [
        {
            "app_id": "vscode",
            "software_name": "VSCode",
            "stage": "discovery",
            "reason": "path_not_found",
            "next_fix_task": "Add software path discovery or target alias coverage for this app.",
        }
    ]
    assert report["summary"]["app_layout_dependency_by_app"] == {"qq": 4}
    assert report["acceptance"]["status"] == "passed"


def test_write_markdown_report_includes_p1_core_metrics(tmp_path):
    report = {
        "schema_version": "deskcanvas_p1_quality_report.v1",
        "summary": {
            "total_apps": 1,
            "canvas_created": 1,
            "total_query_count": 3,
            "query_correct_count": 2,
            "query_correct_rate": 2 / 3,
            "false_positive_count": 1,
            "dangerous_unmarked_count": 1,
            "candidates_outside_capture_bounds_count": 0,
            "app_layout_dependency_count": 2,
            "coverage_gap_count": 1,
            "coverage_gaps": [
                {
                    "app_id": "vscode",
                    "software_name": "VSCode",
                    "stage": "discovery",
                    "reason": "path_not_found",
                    "next_fix_task": "Add software path discovery or target alias coverage for this app.",
                }
            ],
            "app_layout_dependency_by_app": {"qq": 2},
        },
        "acceptance": {"status": "not_ready"},
        "apps": [
            {
                "app_id": "qq",
                "software_name": "QQ",
                "quality_metrics": {
                    "query_count": 3,
                    "query_correct_count": 2,
                    "false_positive_count": 1,
                    "dangerous_unmarked_count": 1,
                    "candidates_outside_capture_bounds_count": 0,
                    "app_layout_dependency_count": 2,
                    "ocr_provider_name": "paddleocr_bridge",
                    "ocr_provider_success": False,
                    "ocr_provider_error": "worker_timeout",
                    "ocr_provider_elapsed_seconds": 2.5,
                    "ocr_provider_worker_reused": False,
                    "ocr_provider_startup_seconds": 12.0,
                    "ocr_provider_fallback_reason": "sparse_uia_full_window",
                    "ocr_provider_block_count": 0,
                },
                "query_diagnostics": [
                    {
                        "query_label": "发送按钮",
                        "matched": True,
                        "heuristic_correct": False,
                        "false_positive_reason": "top_candidate_does_not_match_expected_role_or_text",
                    }
                ],
            }
        ],
    }

    out_path = tmp_path / "p1_quality_report.md"
    p1_quality_report.write_markdown_report(report, out_path)

    md = out_path.read_text(encoding="utf-8")
    assert "query_correct_rate" in md
    assert "false_positive_count" in md
    assert "dangerous_unmarked_count" in md
    assert "outside_bounds" in md
    assert "app_layout_dependency_count" in md
    assert "ocr" in md
    assert "paddleocr_bridge 2.5s startup:12.0s cold 0blk sparse_uia_full_window error:worker_timeout" in md
    assert "发送按钮" in md
    assert "Coverage Gaps" in md
    assert "VSCode" in md
    assert "path_not_found" in md
    assert "app_layout_dependency_by_app" in md


def test_build_p05_stage_report_flattens_metrics_and_flags_sparse_canvas():
    app_runs = [
        {
            "app_id": "qq",
            "software_name": "QQ",
            "discovery_status": "found",
            "launch_bind": {"status": "bound"},
            "launch_status": "bound",
            "bind_status": "bound",
            "capture_status": "captured",
            "observe_status": "canvas_created",
            "quality_metrics": {
                "canvas_created": True,
                "element_count": 1,
                "ocr_block_count": 0,
                "uia_candidate_count": 1,
                "fallback_vision_candidate_count": 0,
                "query_count": 4,
                "query_matched_count": 4,
                "query_correct_count": 0,
                "false_positive_count": 4,
                "risky_candidate_count": 0,
                "candidates_outside_capture_bounds_count": 0,
            },
        }
    ]

    report = p1_quality_report.build_p05_stage_report([], app_runs, min_apps=1, min_correct_rate=0.7)

    row = report["apps"][0]
    assert report["schema_version"] == "deskcanvas_p05_stage_report.v1"
    assert row["discovered"] is True
    assert row["launchable"] is False
    assert row["bound"] is True
    assert row["captured"] is True
    assert row["observe_ok"] is True
    assert row["canvas_created"] is True
    assert row["candidate_count"] == 1
    assert row["failed_stage"] == "canvas_sparse"
    assert row["next_fix_task"] == "Improve OCR/fallback candidate generation so this real app has enough queryable controls."
    assert report["acceptance"]["status"] == "not_ready"


def test_qq_query_matrix_uses_semantic_chat_item_for_sparse_layout():
    # QQ: index 1 = 聊天列表 (chat_item OK), index 4 = 联系人入口 (chat_item forbidden)
    chat_list = p1_quality_report.P1_QUERY_MATRIX["qq"][1]
    contact_entry = p1_quality_report.P1_QUERY_MATRIX["qq"][4]
    assert chat_list["label"] == "聊天列表"
    assert "chat_item" in chat_list["expected"]["semantic_roles"]
    assert contact_entry["label"] == "联系人入口"
    assert "chat_item" in contact_entry["expected"].get("forbidden_roles", [])


def test_flclash_query_matrix_uses_proxy_and_main_content_terms():
    queries = p1_quality_report.P1_QUERY_MATRIX["flclash"]

    assert queries[0]["label"] == "代理入口"
    assert queries[0]["target"] == {"natural_language": "代理"}
    assert queries[1]["label"] == "主内容区域"
    assert queries[1]["target"] == {"natural_language": "找主内容区域"}


def test_voicemeeter_query_matrix_targets_a1_by_label_not_generic_danger():
    query = p1_quality_report.P1_QUERY_MATRIX["voicemeeter"][0]

    assert query["label"] == "A1 输出"
    assert query["target"] == {"natural_language": "找 A1 输出按钮"}
    assert query["expected"]["text_any"] == ["A1"]


def test_priority_query_matrix_uses_natural_language_targets():
    for app_id in ["chrome", "vscode", "flclash", "qq", "voicemeeter"]:
        queries = p1_quality_report.P1_QUERY_MATRIX[app_id]

        assert len(queries) >= 3
        assert all("natural_language" in item["target"] for item in queries)


def test_vscode_query_matrix_avoids_sidebar_state_dependency():
    queries = p1_quality_report.P1_QUERY_MATRIX["vscode"]

    # VS Code matrix: index 1 = 文件资源管理器 (sidebar/tree_item)
    assert queries[1]["label"] == "文件资源管理器"
    assert queries[1]["target"] == {"natural_language": "找文件资源管理器"}
    assert "sidebar" in queries[1]["expected"]["semantic_roles"]


def test_p05_stage_report_explains_minimized_capture_failure():
    app_runs = [
        {
            "app_id": "wechat",
            "software_name": "WeChat",
            "discovery_status": "found",
            "bind_status": "bound",
            "capture_status": "failed",
            "observe_status": "failed",
            "launch_bind": {
                "capture_diagnostics": {"is_minimized": True},
            },
            "quality_metrics": {},
        }
    ]

    report = p1_quality_report.build_p05_stage_report([], app_runs, min_apps=1)

    assert report["apps"][0]["failed_stage"] == "capture"
    assert report["apps"][0]["next_fix_task"] == "Restore the minimized window or bind a non-minimized top-level window before capture."


class TestStrictEvaluation:
    def test_verdict_tp_for_correct_match(self):
        from scripts.p1_quality_report import diagnose_query_match
        spec = {"label": "搜索框", "target": {"natural_language": "找搜索框"}, "expected": {"semantic_roles": ["search_input"], "text_any": ["搜索"]}}
        cand = type("C", (), {
            "element_id": "e1", "text": "搜索", "name": "搜索", "semantic_role": type("R", (), {"value": "search_input"})(),
            "control_type": "EditControl", "bounding_rect": (0,0,100,30), "click_point": None,
            "confidence": 0.9, "confidence_level": type("L", (), {"value": "high"})(),
            "risk_tags": [], "risk_level": type("L", (), {"value": "L0"})(),
            "provider_sources": ["uia"], "interactable": True, "from_memory": False,
            "suggest_confirm": False, "region_id": "", "locator_ids": [], "attributes": {},
            "visual_type": "", "semantic_tags": [], "role_label": None, "role_confidence": 0.0,
            "role_source": "", "role_evidence": [], "refine_status": "unreviewed", "stable_key_id": None,
        })()
        result = diagnose_query_match(app_id="test", query_spec=spec, candidate=cand)
        assert result["verdict"] == "TP"

    def test_verdict_fp_for_forbidden_role(self):
        from scripts.p1_quality_report import diagnose_query_match
        spec = {"label": "联系人入口", "target": {"natural_language": "找联系人"}, "expected": {"semantic_roles": ["nav_item"], "forbidden_roles": ["chat_item"]}}
        cand = type("C", (), {
            "element_id": "e1", "text": "群助手", "name": "群助手", "semantic_role": type("R", (), {"value": "chat_item"})(),
            "control_type": "ListItemControl", "bounding_rect": (0,0,100,30), "click_point": None,
            "confidence": 1.0, "confidence_level": type("L", (), {"value": "high"})(),
            "risk_tags": [], "risk_level": type("L", (), {"value": "L0"})(),
            "provider_sources": ["ocr"], "interactable": True, "from_memory": False,
            "suggest_confirm": False, "region_id": "", "locator_ids": [], "attributes": {},
            "visual_type": "", "semantic_tags": [], "role_label": None, "role_confidence": 0.0,
            "role_source": "", "role_evidence": [], "refine_status": "unreviewed", "stable_key_id": None,
        })()
        result = diagnose_query_match(app_id="test", query_spec=spec, candidate=cand)
        assert result["verdict"] == "FP"
        assert any("forbidden" in r for r in result["reasons"])

    def test_verdict_fp_for_negative_text(self):
        from scripts.p1_quality_report import diagnose_query_match
        spec = {"label": "后退按钮", "target": {"natural_language": "找后退按钮"}, "expected": {"semantic_roles": ["button"], "negative_text": ["http", ".com"]}}
        cand = type("C", (), {
            "element_id": "e1", "text": "https://chatgpt.com", "name": "", "semantic_role": type("R", (), {"value": "button"})(),
            "control_type": "ButtonControl", "bounding_rect": (0,0,100,30), "click_point": None,
            "confidence": 0.9, "confidence_level": type("L", (), {"value": "high"})(),
            "risk_tags": [], "risk_level": type("L", (), {"value": "L0"})(),
            "provider_sources": ["uia"], "interactable": True, "from_memory": False,
            "suggest_confirm": False, "region_id": "", "locator_ids": [], "attributes": {},
            "visual_type": "", "semantic_tags": [], "role_label": None, "role_confidence": 0.0,
            "role_source": "", "role_evidence": [], "refine_status": "unreviewed", "stable_key_id": None,
        })()
        result = diagnose_query_match(app_id="test", query_spec=spec, candidate=cand)
        assert result["verdict"] == "FP"

    def test_verdict_miss_when_no_candidate(self):
        from scripts.p1_quality_report import diagnose_query_match
        spec = {"label": "播放按钮", "target": {"natural_language": "找播放按钮"}, "expected": {"semantic_roles": ["button"]}}
        result = diagnose_query_match(app_id="test", query_spec=spec, candidate=None)
        assert result["verdict"] == "MISS"