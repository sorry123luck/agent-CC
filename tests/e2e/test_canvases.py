"""E2E tests for canvas-related endpoints.

Tests:
- GET /api/v1/canvases (list cached canvases)
- GET /api/v1/canvases/{canvas_id} (canvas detail)
- GET /api/v1/canvases/{canvas_id}/screenshot (screenshot PNG)
"""

from __future__ import annotations

import io
import uuid

import pytest
import sqlalchemy
from PIL import Image

from src.perception.page_compiler_models import Candidate, Region, ScrollContext, SemanticRole
from tests.e2e.conftest import _make_test_canvas


class TestListCanvases:
    """GET /api/v1/canvases"""

    def test_list_canvases_empty(self, client):
        """Empty cache returns empty list."""
        resp = client.get("/api/v1/canvases")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_canvases_after_seed(self, client, seed_canvas):
        """Seeded canvas appears in list."""
        canvas_id, canvas = seed_canvas
        resp = client.get("/api/v1/canvases")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["canvas_id"] == canvas_id
        assert data[0]["app_id"] == "test_app"
        assert data[0]["element_count"] == 3
        assert data[0]["region_count"] == 1
        assert "uia" in data[0]["providers_used"]
        assert data[0]["stable"] is True

    def test_list_canvases_multiple(self, client, seed_canvas_varying):
        """Multiple canvases all appear."""
        resp = client.get("/api/v1/canvases")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_canvases_with_screenshot_flag(self, client, seed_canvas_with_screenshot):
        """has_screenshot is True when screenshot exists."""
        resp = client.get("/api/v1/canvases")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["has_screenshot"] is True

    def test_list_canvases_skips_mismatched_alias_entries(self, client, seed_canvas_varying):
        """Corrupted cache aliases from enhancement must not show duplicate canvas ids."""
        from src.canvas.canvas_cache import get_canvas_cache

        first_id, first, second_id, _second = seed_canvas_varying
        cache = get_canvas_cache()
        # Simulate the previous enhancement bug: cache key B points to object A.
        cache._cache[second_id] = first  # noqa: SLF001 - targeted regression for cache corruption

        resp = client.get("/api/v1/canvases")

        assert resp.status_code == 200
        ids = [item["canvas_id"] for item in resp.json()]
        assert ids == [first_id]


class TestGetCanvasDetail:
    """GET /api/v1/canvases/{canvas_id}"""

    def test_get_canvas_detail(self, client, seed_canvas):
        """Returns full canvas detail with elements, regions, provider_trace."""
        canvas_id, canvas = seed_canvas
        canvas.artifacts["observe_timing"] = {
            "stages": {"1_perception_analyze": 1.0},
            "total_seconds": 1.25,
            "slowest_stage": {"name": "1_perception_analyze", "seconds": 1.0},
        }
        resp = client.get(f"/api/v1/canvases/{canvas_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["canvas_id"] == canvas_id
        assert data["app_id"] == "test_app"
        assert data["window_title"] == "test_app Window"
        assert data["surface_type"] == "native_uia"
        assert data["page_class"] == "test_app/main/default"
        assert len(data["elements"]) == 3
        assert len(data["regions"]) == 1
        assert data["provider_trace"] is not None
        assert data["provider_trace"]["uia_used"] is True
        assert "uia" in data["providers_used"]
        assert data["observe_timing"]["total_seconds"] == 1.25

    def test_get_canvas_detail_element_fields(self, client, seed_canvas):
        """Elements include control_type, locator_ids, attributes."""
        canvas_id, _ = seed_canvas
        resp = client.get(f"/api/v1/canvases/{canvas_id}")
        data = resp.json()
        elem = data["elements"][0]
        assert "element_id" in elem
        assert "semantic_role" in elem
        assert "text" in elem
        assert "confidence" in elem
        assert "control_type" in elem
        assert "locator_ids" in elem
        assert "attributes" in elem
        assert "provider_sources" in elem
        assert "risk_tags" in elem
        assert "risk_level" in elem

    def test_get_canvas_detail_not_found(self, client):
        """Nonexistent canvas returns 404."""
        resp = client.get("/api/v1/canvases/nonexistent-id")
        assert resp.status_code == 404

    def test_get_canvas_detail_region_fields(self, client, seed_canvas):
        """Region includes region_id, role, bounds, element_count."""
        canvas_id, _ = seed_canvas
        resp = client.get(f"/api/v1/canvases/{canvas_id}")
        data = resp.json()
        region = data["regions"][0]
        assert "region_id" in region
        assert "role" in region
        assert "bounds" in region
        assert "element_count" in region

    def test_get_canvas_detail_window_dimensions(self, client, seed_canvas):
        """window_width and window_height are present."""
        canvas_id, _ = seed_canvas
        resp = client.get(f"/api/v1/canvases/{canvas_id}")
        data = resp.json()
        assert "window_width" in data
        assert "window_height" in data

    def test_get_canvas_detail_includes_fusion_diagnostics(self, client, seed_canvas):
        """Canvas detail returns semantic fusion diagnostics for console inspection."""
        canvas_id, canvas = seed_canvas
        canvas.artifacts["fusion_diagnostics"] = {
            "geometric_region_count": 2,
            "labeled_count": 1,
            "unknown_count": 1,
            "regions": [
                {
                    "region_id": "R0",
                    "semantic_label": "content",
                    "confidence": 0.5,
                    "reason": "large center pane",
                }
            ],
        }

        resp = client.get(f"/api/v1/canvases/{canvas_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["fusion_diagnostics"]["geometric_region_count"] == 2
        assert data["fusion_diagnostics"]["regions"][0]["semantic_label"] == "content"

    def test_get_canvas_detail_includes_local_evidence_artifacts(self, client, seed_canvas):
        """Canvas detail exposes OCR and vision candidates used by local evidence audits."""
        canvas_id, canvas = seed_canvas
        canvas.artifacts["ocr_blocks"] = [
            {"bbox": [10, 20, 60, 42], "text": "发送", "confidence": 0.91}
        ]
        canvas.artifacts["vision_candidates"] = [
            {"candidate_id": "vision_send", "bbox": [70, 20, 100, 42], "kind": "icon", "confidence": 0.72}
        ]

        resp = client.get(f"/api/v1/canvases/{canvas_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ocr_blocks"] == [{"bbox": [10, 20, 60, 42], "text": "发送", "confidence": 0.91}]
        assert data["vision_candidates"] == [
            {"candidate_id": "vision_send", "bbox": [70, 20, 100, 42], "kind": "icon", "confidence": 0.72}
        ]


class TestGetCanvasOperability:
    """GET /api/v1/canvases/{canvas_id}/operability"""

    def test_get_canvas_operability_exposes_agent_contract(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.app.process_name = "weixin.exe"
        canvas.page.page_class = "chat_workspace"
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(300, 80, 980, 600)),
            Region(region_id="composer", role="composer", bounds=(300, 600, 980, 720)),
        ]
        canvas.scroll_contexts = [
            ScrollContext(
                scroll_context_id="scroll_messages",
                region_id="messages",
                scroll_type="vertical",
                viewport_height=520,
                viewport_width=680,
                is_virtual=True,
            )
        ]
        canvas.elements = [
            Candidate(
                element_id="input_1",
                region_id="composer",
                semantic_role=SemanticRole.MESSAGE_INPUT,
                bounds=(320, 620, 880, 700),
                click_point=(600, 660),
                attributes={"actionability": "review", "safe_to_type": False},
                provider_sources=["geometry"],
            ),
            Candidate(
                element_id="send_1",
                region_id="composer",
                semantic_role=SemanticRole.SEND_BUTTON,
                bounds=(900, 670, 960, 710),
                click_point=(930, 690),
                risk_tags=["send"],
                attributes={"actionability": "controlled_probe"},
                provider_sources=["roi_vlm"],
            ),
        ]
        canvas.artifacts["roi_selection_plan"] = {
            "mode": "chat_workspace",
            "rois": [
                {"purpose": "message_stream", "bounds": [300, 80, 980, 600]},
                {"purpose": "composer", "bounds": [300, 600, 980, 720]},
            ],
        }

        resp = client.get(f"/api/v1/canvases/{canvas_id}/operability")

        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_status"] == "usable"
        row = data["rows"][0]
        assert row["page_class"] == "chat_workspace"
        assert row["agent_contract"]["schema_version"] == "2026-05-28.page-operability.v1"
        assert row["agent_contract"]["read_regions"] == ["message_stream"]
        assert row["agent_contract"]["input_regions"] == ["composer"]
        assert row["agent_contract"]["read_plan"] == [
            {
                "region": "message_stream",
                "method": "chat_crop_ocr_readback",
                "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
                "request": {
                    "region_role": "message_stream",
                    "include_elements": True,
                    "include_ocr": True,
                    "allow_crop_ocr": True,
                },
                "read_scope": "current_viewport",
                "scroll_context": {
                    "scroll_context_id": "scroll_messages",
                    "region_id": "messages",
                    "scroll_type": "vertical",
                    "is_virtual": True,
                },
                "long_content_strategy": {
                    "status": "current_viewport_only",
                    "reason": "controlled_scroll_available",
                },
                "next_scroll_probe": {
                    "endpoint": "POST /api/v1/canvases/{canvas_id}/scroll-region",
                    "request": {
                        "region_role": "message_stream",
                        "direction": "down",
                        "amount": "page",
                        "dry_run": True,
                        "execute_confirmed": False,
                        "read_after": True,
                    },
                },
                "scroll_supported": True,
            }
        ]
        assert row["agent_contract"]["safe_action_targets"] == []
        assert row["agent_contract"]["review_required_targets"][0]["role"] == "message_input"
        assert row["agent_contract"]["default_agent_type"] is False

    def test_get_canvas_operability_not_found(self, client):
        resp = client.get("/api/v1/canvases/missing-canvas/operability")

        assert resp.status_code == 404


class TestReadCanvasRegion:
    """POST /api/v1/canvases/{canvas_id}/read-region"""

    def test_read_region_harvests_candidate_and_ocr_text_inside_region(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.regions = [
            Region(region_id="status", role="status_panel", bounds=(0, 0, 400, 200)),
            Region(region_id="outside", role="content_area", bounds=(420, 0, 800, 200)),
        ]
        canvas.elements = [
            Candidate(
                element_id="status_text",
                region_id="status",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 20, 220, 50),
                text="网络检测 203.175.14.44",
                provider_sources=["uia"],
            ),
            Candidate(
                element_id="outside_text",
                region_id="outside",
                semantic_role=SemanticRole.TEXT,
                bounds=(430, 20, 700, 50),
                text="outside text",
                provider_sources=["uia"],
            ),
        ]
        canvas.artifacts["ocr_blocks"] = [
            {"bbox": [30, 80, 160, 110], "text": "内网 IP", "confidence": 0.92},
            {"bbox": [430, 80, 540, 110], "text": "不应读取", "confidence": 0.92},
        ]

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/read-region",
            json={"region_id": "status"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["canvas_id"] == canvas_id
        assert data["region_id"] == "status"
        assert data["method"] == "region_text_harvest"
        assert data["status"] == "pass"
        texts = [block["text"] for block in data["text_blocks"]]
        assert "网络检测 203.175.14.44" in texts
        assert "内网 IP" in texts
        assert "outside text" not in texts
        assert "不应读取" not in texts

    def test_read_region_supports_region_role_lookup(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        canvas.scroll_contexts = [
            ScrollContext(
                scroll_context_id="scroll_messages",
                region_id="messages",
                scroll_type="vertical",
                viewport_height=400,
                viewport_width=500,
                is_virtual=True,
            )
        ]
        canvas.elements = [
            Candidate(
                element_id="msg",
                region_id="messages",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 20, 220, 50),
                text="回复测试1",
                provider_sources=["uia"],
            ),
        ]

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/read-region",
            json={"region_role": "message_stream"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["region_id"] == "messages"
        assert data["method"] == "chat_crop_ocr_readback"
        assert data["text"] == "回复测试1"
        assert data["read_scope"] == "current_viewport"
        assert data["scroll_context"] == {
            "scroll_context_id": "scroll_messages",
            "region_id": "messages",
            "scroll_type": "vertical",
            "is_virtual": True,
        }
        assert data["long_content_strategy"] == {
            "status": "current_viewport_only",
            "reason": "controlled_scroll_available",
        }
        assert data["suggested_next_actions"] == [
            {
                "action": "scroll_region_dry_run",
                "endpoint": "POST /api/v1/canvases/{canvas_id}/scroll-region",
                "request": {
                    "region_role": "message_stream",
                    "direction": "down",
                    "amount": "page",
                    "dry_run": True,
                    "execute_confirmed": False,
                    "read_after": True,
                },
            }
        ]

    def test_read_region_can_run_crop_ocr_when_requested(
        self,
        client,
        seed_canvas_with_screenshot,
        monkeypatch,
    ):
        canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(100, 100, 300, 300)),
        ]
        canvas.elements = []
        canvas.artifacts["ocr_blocks"] = []

        class FakeOcrService:
            def extract_with_metadata(self, image, **kwargs):
                from src.perception.ocr_service import OCRExtractionResult, OCRTextBlock

                assert kwargs["region"] == (100, 100, 300, 300)
                return OCRExtractionResult(
                    blocks=[
                        OCRTextBlock(
                            text="截图OCR内容",
                            bbox=(110, 112, 220, 142),
                            confidence=0.93,
                        )
                    ],
                    success=True,
                    used_region=kwargs["region"],
                )

        monkeypatch.setattr(
            "src.perception.ocr_service.get_ocr_service",
            lambda: FakeOcrService(),
        )

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/read-region",
            json={
                "region_id": "messages",
                "include_elements": False,
                "include_ocr": False,
                "allow_crop_ocr": True,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pass"
        assert data["method"] == "chat_crop_ocr_readback"
        assert data["crop_ocr_used"] is True
        assert data["text"] == "截图OCR内容"
        assert data["text_blocks"] == [
            {
                "text": "截图OCR内容",
                "bounds": [110, 112, 220, 142],
                "confidence": 0.93,
                "source": "crop_ocr",
                "element_id": None,
            }
        ]

    def test_read_region_crop_ocr_missing_screenshot_warns(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(100, 100, 300, 300)),
        ]
        canvas.elements = []
        canvas.artifacts["ocr_blocks"] = []

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/read-region",
            json={
                "region_id": "messages",
                "include_elements": False,
                "include_ocr": False,
                "allow_crop_ocr": True,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "warn"
        assert data["crop_ocr_used"] is False
        assert "missing_screenshot_for_crop_ocr" in data["warnings"]

    def test_read_region_missing_region_returns_404(self, client, seed_canvas):
        canvas_id, _ = seed_canvas

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/read-region",
            json={"region_id": "missing"},
        )

        assert resp.status_code == 404


class TestScrollCanvasRegion:
    """POST /api/v1/canvases/{canvas_id}/scroll-region"""

    def test_scroll_region_dry_run_returns_plan_without_executing(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        canvas.scroll_contexts = [
            ScrollContext(
                scroll_context_id="scroll_messages",
                region_id="messages",
                scroll_type="vertical",
                viewport_height=400,
                viewport_width=500,
                is_virtual=True,
            )
        ]

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/scroll-region",
            json={"region_role": "message_stream", "direction": "down", "amount": "page"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "planned"
        assert data["dry_run"] is True
        assert data["can_execute"] is False
        assert data["region_id"] == "messages"
        assert data["scroll_context"]["scroll_context_id"] == "scroll_messages"
        assert data["action_plan"] == {
            "action": "scroll",
            "direction": "down",
            "amount": "page",
            "region_bounds": [0, 0, 500, 400],
            "scroll_context_id": "scroll_messages",
            "follow_up": "observe_then_read_region",
        }
        assert "scroll_region_dry_run_only" in data["warnings"]

    def test_scroll_region_without_scroll_context_blocks(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.regions = [
            Region(region_id="status", role="status_panel", bounds=(0, 0, 500, 400)),
        ]
        canvas.scroll_contexts = []

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/scroll-region",
            json={"region_role": "status_panel"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "blocked"
        assert data["can_execute"] is False
        assert data["action_plan"] == {}
        assert "no_scroll_context" in data["warnings"]

    def test_scroll_region_execute_without_confirmation_blocks(self, client, seed_canvas):
        canvas_id, canvas = seed_canvas
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        canvas.scroll_contexts = [
            ScrollContext(
                scroll_context_id="scroll_messages",
                region_id="messages",
                scroll_type="vertical",
                viewport_height=400,
                viewport_width=500,
            )
        ]

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/scroll-region",
            json={"region_role": "message_stream", "dry_run": False},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "blocked"
        assert data["dry_run"] is False
        assert data["can_execute"] is False
        assert data["action_plan"]["follow_up"] == "observe_then_read_region"
        assert "scroll_region_execute_requires_confirmation" in data["warnings"]

    def test_scroll_region_missing_region_returns_404(self, client, seed_canvas):
        canvas_id, _canvas = seed_canvas

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/scroll-region",
            json={"region_id": "missing"},
        )

        assert resp.status_code == 404

    def test_scroll_region_execute_observes_and_reads_after_scroll(
        self,
        client,
        seed_canvas,
        monkeypatch,
    ):
        from src.canvas.canvas_cache import get_canvas_cache
        from src.execution.action_executor import ActionResult

        canvas_id, canvas = seed_canvas
        canvas.window.hwnd = 24680
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        canvas.scroll_contexts = [
            ScrollContext(scroll_context_id="scroll_messages", region_id="messages"),
        ]
        canvas.elements = [
            Candidate(
                element_id="before_msg",
                region_id="messages",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 20, 220, 50),
                text="旧消息",
                provider_sources=["uia"],
            ),
        ]

        after = _make_test_canvas(hwnd=24680)
        after.regions = [
            Region(region_id="messages_after", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        after.scroll_contexts = [
            ScrollContext(scroll_context_id="scroll_messages_after", region_id="messages_after"),
        ]
        after.elements = [
            Candidate(
                element_id="after_msg",
                region_id="messages_after",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 20, 220, 50),
                text="新消息",
                provider_sources=["uia"],
            ),
        ]

        calls = []

        def fake_scroll_at(self, hwnd, client_x, client_y, delta):
            calls.append((hwnd, client_x, client_y, delta))
            return ActionResult(True, "scroll", "scrolled")

        def fake_observe(hwnd, **kwargs):
            get_canvas_cache().put(after)
            return after, "new_state", {"used": False, "status": "skipped", "provider": ""}

        monkeypatch.setattr("src.execution.action_executor.ActionExecutor.scroll_at", fake_scroll_at)
        monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/scroll-region",
            json={
                "region_role": "message_stream",
                "direction": "down",
                "amount": "page",
                "dry_run": False,
                "execute_confirmed": True,
                "read_after": True,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "executed"
        assert data["can_execute"] is True
        assert data["after_canvas_id"] == after.canvas_id
        assert data["execution_result"]["success"] is True
        assert data["before_read"]["text"] == "旧消息"
        assert data["after_read"]["text"] == "新消息"
        assert data["stitched_text"] == "旧消息\n新消息"
        assert data["stitch_report"] == {
            "before_block_count": 1,
            "after_block_count": 1,
            "stitched_block_count": 2,
            "deduped_count": 0,
        }
        assert calls == [(24680, 250, 200, -480)]

    def test_scroll_region_stitch_dedupes_repeated_text(
        self,
        client,
        seed_canvas,
        monkeypatch,
    ):
        from src.canvas.canvas_cache import get_canvas_cache
        from src.execution.action_executor import ActionResult

        canvas_id, canvas = seed_canvas
        canvas.window.hwnd = 24680
        canvas.regions = [
            Region(region_id="messages", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        canvas.scroll_contexts = [
            ScrollContext(scroll_context_id="scroll_messages", region_id="messages"),
        ]
        canvas.elements = [
            Candidate(
                element_id="before_msg",
                region_id="messages",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 20, 220, 50),
                text="重复消息",
                provider_sources=["uia"],
            ),
        ]

        after = _make_test_canvas(hwnd=24680)
        after.regions = [
            Region(region_id="messages_after", role="message_stream", bounds=(0, 0, 500, 400)),
        ]
        after.scroll_contexts = [
            ScrollContext(scroll_context_id="scroll_messages_after", region_id="messages_after"),
        ]
        after.elements = [
            Candidate(
                element_id="after_msg",
                region_id="messages_after",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 20, 220, 50),
                text="重复消息",
                provider_sources=["uia"],
            ),
            Candidate(
                element_id="after_new_msg",
                region_id="messages_after",
                semantic_role=SemanticRole.TEXT,
                bounds=(20, 80, 220, 110),
                text="新增消息",
                provider_sources=["uia"],
            ),
        ]

        monkeypatch.setattr(
            "src.execution.action_executor.ActionExecutor.scroll_at",
            lambda self, hwnd, client_x, client_y, delta: ActionResult(True, "scroll", "scrolled"),
        )

        def fake_observe(hwnd, **kwargs):
            get_canvas_cache().put(after)
            return after, "new_state", {"used": False, "status": "skipped", "provider": ""}

        monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)

        resp = client.post(
            f"/api/v1/canvases/{canvas_id}/scroll-region",
            json={
                "region_role": "message_stream",
                "dry_run": False,
                "execute_confirmed": True,
                "read_after": True,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["stitched_text"] == "重复消息\n新增消息"
        assert data["stitch_report"]["deduped_count"] == 1
        assert data["stitch_report"]["stitched_block_count"] == 2


class TestDeleteCanvasesAndModels:
    """DELETE endpoints used by the console cleanup buttons."""

    def test_delete_canvas_removes_cache_entry(self, client, seed_canvas):
        canvas_id, _ = seed_canvas

        resp = client.delete(f"/api/v1/canvases/{canvas_id}")

        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert client.get(f"/api/v1/canvases/{canvas_id}").status_code == 404

    def test_delete_canvas_prevents_background_resurrection(self, client, seed_canvas):
        from src.canvas.canvas_cache import get_canvas_cache

        canvas_id, canvas = seed_canvas
        resp = client.delete(f"/api/v1/canvases/{canvas_id}")

        assert resp.status_code == 200
        get_canvas_cache().put(canvas)
        assert client.get(f"/api/v1/canvases/{canvas_id}").status_code == 404

    def test_delete_canvas_removes_candidate_overrides(self, client, seed_canvas):
        from src.memory.candidate_override_store import CandidateOverrideStore
        from src.storage.db import Session

        canvas_id, _ = seed_canvas
        with Session() as session:
            CandidateOverrideStore().upsert(
                session,
                scope_key=f"canvas:{canvas_id}:elem_0",
                canvas_id=canvas_id,
                element_id="elem_0",
                semantic_role="button",
                source="manual",
            )
            session.commit()

        resp = client.delete(f"/api/v1/canvases/{canvas_id}")

        assert resp.status_code == 200
        assert resp.json()["records"]["candidate_overrides"] == 1
        with Session() as session:
            remaining = CandidateOverrideStore().list(session, canvas_id=canvas_id)
        assert remaining == []

    def test_delete_canvas_not_found(self, client):
        resp = client.delete("/api/v1/canvases/missing-canvas")

        assert resp.status_code == 404

    def test_delete_state_template_removes_snapshots_and_cached_canvas(self, client, seed_canvas):
        from src.storage.db import Session

        canvas_id, _ = seed_canvas
        pm_id = str(uuid.uuid4())
        st_id = str(uuid.uuid4())
        now = "2026-01-01T00:00:00"
        with Session() as session:
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO page_models
                       (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                        created_at, last_seen_at, observe_count, state_count)
                       VALUES (:pm, 'test_app', 'test_app/main', 'TestApp', 'native_uia',
                               :now, :now, 1, 1)"""
                ),
                {"pm": pm_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO state_templates
                       (state_template_id, page_model_id, app_id, page_class, state_label,
                        layout_fingerprint, state_signature, created_at, last_seen_at,
                        verify_count, snapshot_count, fixed_element_count, total_element_count)
                       VALUES (:st, :pm, 'test_app', 'test_app/main/default', 'main',
                               '{}', 'sig', :now, :now, 1, 1, 0, 0)"""
                ),
                {"st": st_id, "pm": pm_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO canvas_snapshots
                       (snapshot_id, canvas_id, page_model_id, state_template_id,
                        captured_at, element_count, has_screenshot)
                       VALUES (:snap, :canvas, :pm, :st, :now, 3, 0)"""
                ),
                {"snap": str(uuid.uuid4()), "canvas": canvas_id, "pm": pm_id, "st": st_id, "now": now},
            )
            session.commit()

        resp = client.delete(f"/api/v1/state-templates/{st_id}")

        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert client.get(f"/api/v1/canvases/{canvas_id}").status_code == 404

    def test_delete_last_canvas_keeps_state_template_and_state_vlm_responses(self, client, seed_canvas):
        from src.storage.db import Session

        canvas_id, _ = seed_canvas
        pm_id = str(uuid.uuid4())
        st_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        now = "2026-01-01T00:00:00"
        with Session() as session:
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO page_models
                       (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                        created_at, last_seen_at, observe_count, state_count)
                       VALUES (:pm, 'test_app', 'test_app/main', 'TestApp', 'native_uia',
                               :now, :now, 1, 1)"""
                ),
                {"pm": pm_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO state_templates
                       (state_template_id, page_model_id, app_id, page_class, state_label,
                        layout_fingerprint, state_signature, created_at, last_seen_at,
                        verify_count, snapshot_count, fixed_element_count, total_element_count)
                       VALUES (:st, :pm, 'test_app', 'test_app/main/default', 'main',
                               '{}', 'sig', :now, :now, 1, 1, 0, 0)"""
                ),
                {"st": st_id, "pm": pm_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO canvas_snapshots
                       (snapshot_id, canvas_id, page_model_id, state_template_id,
                        captured_at, element_count, has_screenshot)
                       VALUES (:snap, :canvas, :pm, :st, :now, 3, 0)"""
                ),
                {"snap": str(uuid.uuid4()), "canvas": canvas_id, "pm": pm_id, "st": st_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO vlm_responses
                       (response_id, cache_key, screenshot_hash, candidate_hash, prompt_version,
                        schema_version, provider_name, model_name, parsed_model, token_input,
                        token_output, cost_usd, latency_ms, status, page_model_id, state_template_id)
                       VALUES (:rid, 'cache', 'shot', 'cand', 'v1', 'v1', 'test', 'model',
                               '{}', 0, 0, 0, 1, 'success', :pm, :st)"""
                ),
                {"rid": response_id, "pm": pm_id, "st": st_id},
            )
            session.commit()

        resp = client.delete(f"/api/v1/canvases/{canvas_id}")

        assert resp.status_code == 200
        with Session() as session:
            remaining = session.execute(
                sqlalchemy.text("SELECT COUNT(*) FROM vlm_responses WHERE response_id = :rid"),
                {"rid": response_id},
            ).scalar()
            state_remaining = session.execute(
                sqlalchemy.text("SELECT COUNT(*) FROM state_templates WHERE state_template_id = :st"),
                {"st": st_id},
            ).scalar()
            page_remaining = session.execute(
                sqlalchemy.text("SELECT COUNT(*) FROM page_models WHERE page_model_id = :pm"),
                {"pm": pm_id},
            ).scalar()
        assert remaining == 1
        assert state_remaining == 1
        assert page_remaining == 1

    def test_delete_page_model_removes_model_tree_entry(self, client):
        from src.storage.db import Session

        pm_id = str(uuid.uuid4())
        now = "2026-01-01T00:00:00"
        with Session() as session:
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO page_models
                       (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                        created_at, last_seen_at, observe_count, state_count)
                       VALUES (:pm, 'test_app', 'test_app/main', 'TestApp', 'native_uia',
                               :now, :now, 1, 0)"""
                ),
                {"pm": pm_id, "now": now},
            )
            session.commit()

        resp = client.delete(f"/api/v1/page-models/{pm_id}")

        assert resp.status_code == 200
        tree = client.get("/api/v1/page-models/tree").json()
        assert all(pm["page_model_id"] != pm_id for pm in tree["page_models"])

    def test_page_model_tree_hides_orphaned_empty_models(self, client):
        from src.storage.db import Session

        orphan_id = str(uuid.uuid4())
        active_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        now = "2026-01-01T00:00:00"
        with Session() as session:
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO page_models
                       (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                        created_at, last_seen_at, observe_count, state_count)
                       VALUES (:pm, 'orphan_app', 'orphan/main', 'Orphan', 'native_uia',
                               :now, :now, 1, 0)"""
                ),
                {"pm": orphan_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO page_models
                       (page_model_id, app_id, page_class_prefix, display_name, surface_type,
                        created_at, last_seen_at, observe_count, state_count)
                       VALUES (:pm, 'active_app', 'active/main', 'Active', 'native_uia',
                               :now, :now, 1, 1)"""
                ),
                {"pm": active_id, "now": now},
            )
            session.execute(
                sqlalchemy.text(
                    """INSERT INTO state_templates
                       (state_template_id, page_model_id, app_id, page_class, state_label,
                        layout_fingerprint, state_signature, created_at, last_seen_at,
                        verify_count, snapshot_count, fixed_element_count, total_element_count)
                       VALUES (:st, :pm, 'active_app', 'active/main/default', 'main',
                               '{}', 'sig', :now, :now, 1, 0, 0, 0)"""
                ),
                {"st": state_id, "pm": active_id, "now": now},
            )
            session.commit()

        tree = client.get("/api/v1/page-models/tree").json()
        ids = {pm["page_model_id"] for pm in tree["page_models"]}
        assert active_id in ids
        assert orphan_id not in ids


class TestGetCanvasScreenshot:
    """GET /api/v1/canvases/{canvas_id}/screenshot"""

    def test_screenshot_raw(self, client, seed_canvas_with_screenshot):
        """overlay=false returns raw screenshot PNG."""
        canvas_id, _, _ = seed_canvas_with_screenshot
        resp = client.get(f"/api/v1/canvases/{canvas_id}/screenshot?overlay=false")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "PNG"
        assert img.size == (800, 600)

    def test_screenshot_overlay_true(self, client, seed_canvas_with_screenshot):
        """overlay=true returns overlay PNG (backend debug)."""
        canvas_id, _, _ = seed_canvas_with_screenshot
        resp = client.get(f"/api/v1/canvases/{canvas_id}/screenshot?overlay=true")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "PNG"

    def test_screenshot_with_layers(self, client, seed_canvas_with_screenshot):
        """layers parameter filters which layers to draw."""
        canvas_id, _, _ = seed_canvas_with_screenshot
        resp = client.get(
            f"/api/v1/canvases/{canvas_id}/screenshot?overlay=true&layers=elements,regions"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_screenshot_not_found(self, client, seed_canvas):
        """Canvas without screenshot returns 404."""
        canvas_id, _ = seed_canvas
        resp = client.get(f"/api/v1/canvases/{canvas_id}/screenshot")
        assert resp.status_code == 404

    def test_screenshot_nonexistent_canvas(self, client):
        """Nonexistent canvas returns 404."""
        resp = client.get("/api/v1/canvases/nonexistent/screenshot")
        assert resp.status_code == 404

    def test_screenshot_privacy_headers(self, client, seed_canvas_with_screenshot):
        """Screenshot response includes no-cache privacy headers."""
        canvas_id, _, _ = seed_canvas_with_screenshot
        resp = client.get(f"/api/v1/canvases/{canvas_id}/screenshot")
        assert resp.headers.get("cache-control") == "no-store"
        assert resp.headers.get("pragma") == "no-cache"
        assert resp.headers.get("expires") == "0"

    def test_screenshot_never_written_to_disk(self, tmp_path):
        """Screenshot is stored in memory only, not on disk.

        This test verifies the CanvasCache screenshot storage is memory-only.
        """
        import os

        from PIL import Image

        from src.canvas.canvas_cache import CanvasCache
        from src.perception.page_compiler_models import (
            AppInfo,
            Candidate,
            ConfidenceLevel,
            ElementState,
            InteractionCanvas,
            PageInfo,
            ProviderTrace,
            Region,
            RiskLevel,
            SemanticRole,
            SurfaceInfo,
            SurfaceType,
            WindowInfoSnapshot,
        )

        cache = CanvasCache(max_size=4)
        canvas = InteractionCanvas(
            canvas_id="test_mem_only",
            app=AppInfo(app_id="a", process_name="a.exe"),
            window=WindowInfoSnapshot(hwnd=1, title="t"),
            surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
            page=PageInfo(page_class="c", class_confidence=0.8),
            regions=[],
            elements=[],
            provider_trace=ProviderTrace(uia_used=True),
            providers_used=["uia"],
            canvas_schema_version="1.0",
        )
        screenshot = Image.new("RGB", (100, 100), color=(255, 0, 0))
        cache.put(canvas, screenshot=screenshot)

        # Verify screenshot is retrievable from memory
        retrieved = cache.get_screenshot("test_mem_only")
        assert retrieved is not None
        assert retrieved.size == (100, 100)

        # Verify no files were written to tmp_path (or anywhere)
        # The cache should not touch the filesystem
        cache.clear()
        assert cache.get_screenshot("test_mem_only") is None
