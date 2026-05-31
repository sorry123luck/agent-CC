"""Minimal P0 inspector/export service for snapshot inspection flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from src.execution.execution_record import build_execution_record
from src.perception.debug_tools import (
    build_action_target_debug,
    build_snapshot_evidence_trace,
    save_debug_bundle,
)
from src.perception.page_compiler_candidates import build_boundary_candidates
from src.perception.openclaw_protocol import (
    build_openclaw_payload,
    normalize_decision_record,
)

if TYPE_CHECKING:
    from src.perception.ocr_service import OCRService
    from src.perception.perception_service import PerceptionService
    from src.perception.uia_client import UIAClient as UIAClientType
    from src.windows.screenshot_service import ScreenshotService
    from src.windows.window_enum import WindowEnumService, WindowInfoExt


UIAClient = None
DEFAULT_UIA_TREE_DEPTH = 6


class InspectorService:
    """Exports a minimal inspection bundle for one window."""

    def __init__(
        self,
        window_enum_service: WindowEnumService | None = None,
        screenshot_service: ScreenshotService | None = None,
        perception_service: PerceptionService | None = None,
        ocr_service: OCRService | None = None,
    ) -> None:
        if window_enum_service is None:
            from src.windows.window_enum import WindowEnumService

            window_enum_service = WindowEnumService()
        if screenshot_service is None:
            from src.windows.screenshot_service import ScreenshotService

            screenshot_service = ScreenshotService()
        if perception_service is None:
            from src.perception.perception_service import PerceptionService

            perception_service = PerceptionService()
        if ocr_service is None:
            from src.perception.ocr_service import OCRService

            ocr_service = OCRService()

        self._window_enum_service = window_enum_service
        self._screenshot_service = screenshot_service
        self._perception_service = perception_service
        self._ocr_service = ocr_service

    def inspect_foreground_window(
        self,
        output_dir: str | Path = "data/inspect",
        include_ocr: bool = True,
    ) -> dict[str, Any]:
        """Inspect the current foreground window."""
        window_info = self._window_enum_service.get_foreground_window()
        if window_info is None:
            raise ValueError("No foreground window found")

        return self.inspect_window(
            hwnd=window_info.hwnd,
            output_dir=output_dir,
            include_ocr=include_ocr,
            window_info=window_info,
        )

    def inspect_window(
        self,
        hwnd: int,
        output_dir: str | Path = "data/inspect",
        include_ocr: bool = True,
        window_info: WindowInfoExt | None = None,
    ) -> dict[str, Any]:
        """Capture screenshot, UIA tree, OCR sidecars, and snapshot bundle."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        window_info = window_info or self._resolve_window_info(hwnd)
        bundle_name = f"inspect_{hwnd}"

        screenshot = self._screenshot_service.capture(mode="window", target=hwnd)
        preview = self._screenshot_service.create_preview(screenshot)
        preview_path = output_dir / f"{bundle_name}_preview.png"
        preview.save(preview_path)

        global UIAClient
        if UIAClient is None:
            from src.perception.uia_client import UIAClient as _UIAClient

            UIAClient = _UIAClient

        uia_client = UIAClient(hwnd)
        uia_tree = self._serialize_uia_tree(
            uia_client.get_element_tree(max_depth=DEFAULT_UIA_TREE_DEPTH)
        )
        uia_tree_path = output_dir / f"{bundle_name}_uia_tree.json"
        uia_tree_path.write_text(
            json.dumps(uia_tree, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ocr_blocks: list[dict[str, Any]] = []
        ocr_path: str | None = None
        if include_ocr:
            ocr_blocks = [
                {
                    "text": block.text,
                    "bbox": list(block.bbox),
                    "confidence": block.confidence,
                }
                for block in self._ocr_service.extract(screenshot)
            ]
            ocr_file = output_dir / f"{bundle_name}_ocr.json"
            ocr_file.write_text(
                json.dumps(ocr_blocks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            ocr_path = str(ocr_file)

        zone_page = self._perception_service.analyze(hwnd)
        snapshot = self._perception_service.create_page_snapshot(
            zone_page=zone_page,
            process_name=window_info.process_name,
        )
        bundle = save_debug_bundle(
            snapshot=snapshot,
            output_dir=output_dir,
            bundle_name=bundle_name,
            screenshot=zone_page.screenshot or screenshot,
        )

        window_path = output_dir / f"{bundle_name}_window.json"
        window_path.write_text(
            json.dumps(window_info.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        files = dict(bundle["files"])
        files["preview"] = str(preview_path)
        files["window"] = str(window_path)
        files["uia_tree"] = str(uia_tree_path)
        if ocr_path:
            files["ocr"] = ocr_path

        return {
            "hwnd": hwnd,
            "title": window_info.title,
            "output_dir": str(output_dir),
            "bundle_name": bundle["bundle_name"],
            "files": files,
            "summary": {
                **bundle["summary"],
                "uia_tree_node_count": len(uia_tree),
                "ocr_block_count": len(ocr_blocks),
                "vision_candidate_count": len(zone_page.vision_candidates),
                "window_rect": window_info.rect,
            },
        }

    def compare_snapshot_files(
        self,
        before_snapshot_path: str | Path,
        after_snapshot_path: str | Path,
        target_element_id: str | None = None,
    ) -> dict[str, Any]:
        """Provide a structured before/after diff summary for two snapshot JSON files."""
        before = self._load_snapshot_with_sidecars(before_snapshot_path)
        after = self._load_snapshot_with_sidecars(after_snapshot_path)
        before_regions = before.get("regions", [])
        after_regions = after.get("regions", [])
        before_elements = before.get("elements", [])
        after_elements = after.get("elements", [])
        before_region_roles = {region.get("role") for region in before_regions}
        after_region_roles = {region.get("role") for region in after_regions}
        before_scroll_contexts = before.get("scroll_contexts", [])
        after_scroll_contexts = after.get("scroll_contexts", [])
        before_anchors = before.get("anchors", [])
        after_anchors = after.get("anchors", [])
        before_provider = before.get("provider_trace", {}).get("provider_details", {})
        after_provider = after.get("provider_trace", {}).get("provider_details", {})
        before_artifacts = before.get("artifacts", {})
        after_artifacts = after.get("artifacts", {})

        result = {
            "before_surface_type": before.get("surface", {}).get("surface_type"),
            "after_surface_type": after.get("surface", {}).get("surface_type"),
            "before_page_class": before.get("page", {}).get("page_class"),
            "after_page_class": after.get("page", {}).get("page_class"),
            "before_region_count": len(before_regions),
            "after_region_count": len(after_regions),
            "before_element_count": len(before_elements),
            "after_element_count": len(after_elements),
            "region_count_delta": len(after_regions) - len(before_regions),
            "element_count_delta": len(after_elements) - len(before_elements),
            "before_anchor_count": len(before_anchors),
            "after_anchor_count": len(after_anchors),
            "before_scroll_context_count": len(before_scroll_contexts),
            "after_scroll_context_count": len(after_scroll_contexts),
            "added_region_roles": sorted(role for role in after_region_roles - before_region_roles if role),
            "removed_region_roles": sorted(role for role in before_region_roles - after_region_roles if role),
            "before_child_region_count": sum(len(region.get("child_region_ids", [])) for region in before_regions),
            "after_child_region_count": sum(len(region.get("child_region_ids", [])) for region in after_regions),
            "before_ocr_block_count": len(before_artifacts.get("ocr_blocks", [])),
            "after_ocr_block_count": len(after_artifacts.get("ocr_blocks", [])),
            "before_vision_candidate_count": len(before_artifacts.get("vision_candidates", [])),
            "after_vision_candidate_count": len(after_artifacts.get("vision_candidates", [])),
            "before_structure_evidence_score": before_artifacts.get("structure_evidence_score"),
            "after_structure_evidence_score": after_artifacts.get("structure_evidence_score"),
            "provider_diff": {
                "before_vision_layout_region_count": before_provider.get("vision", {}).get("layout_region_count", 0),
                "after_vision_layout_region_count": after_provider.get("vision", {}).get("layout_region_count", 0),
                "before_control_group_count": before_provider.get("vision", {}).get("control_group_count", 0),
                "after_control_group_count": after_provider.get("vision", {}).get("control_group_count", 0),
                "before_interaction_hint_count": before_provider.get("vision", {}).get("interaction_hint_count", 0),
                "after_interaction_hint_count": after_provider.get("vision", {}).get("interaction_hint_count", 0),
            },
            "before_trace": build_snapshot_evidence_trace(before),
            "after_trace": build_snapshot_evidence_trace(after),
        }
        if target_element_id:
            result["target_element_id"] = target_element_id
            result["target_debug"] = {
                "before": build_action_target_debug(before, target_element_id),
                "after": build_action_target_debug(after, target_element_id),
            }
        return result

    def trace_action_target(
        self,
        snapshot_path: str | Path,
        element_id: str,
    ) -> dict[str, Any]:
        """Load one snapshot JSON file and return target-level debug evidence."""
        snapshot = self._load_snapshot_with_sidecars(snapshot_path)
        return {
            "snapshot_path": str(snapshot_path),
            "element_id": element_id,
            "surface_type": snapshot.get("surface", {}).get("surface_type"),
            "page_class": snapshot.get("page", {}).get("page_class"),
            "evidence_trace": build_snapshot_evidence_trace(snapshot),
            "target_debug": build_action_target_debug(snapshot, element_id),
        }

    def trace_action_outcome(
        self,
        outcome_path: str | Path,
    ) -> dict[str, Any]:
        """Load one action outcome JSON file and return a compact execution trace."""
        outcome = json.loads(Path(outcome_path).read_text(encoding="utf-8"))
        return {
            "outcome_path": str(outcome_path),
            "action_trace": build_execution_record(outcome),
        }

    def build_openclaw_payload(
        self,
        snapshot_path: str | Path,
        task: str,
        max_candidates: int = 120,
    ) -> dict[str, Any]:
        """Build the weak-structure payload sent from local perception to DeskCanvas."""
        snapshot = self._load_snapshot_with_sidecars(snapshot_path)
        return {
            "snapshot_path": str(snapshot_path),
            "task": task,
            "openclaw_payload": build_openclaw_payload(
                snapshot,
                task=task,
                max_candidates=max_candidates,
            ),
        }

    def build_candidate_payload(
        self,
        snapshot_path: str | Path,
        max_candidates: int = 120,
    ) -> dict[str, Any]:
        """Expose normalized boundary candidates for inspection and DeskCanvas input checks."""
        snapshot = self._load_snapshot_with_sidecars(snapshot_path)
        return {
            "snapshot_path": str(snapshot_path),
            "candidates": build_boundary_candidates(
                snapshot,
                max_candidates=max_candidates,
            ),
        }

    def export_candidate_regression(
        self,
        snapshot_path: str | Path,
        task: str,
        image_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        max_candidates: int = 120,
        draw_labels: bool = False,
    ) -> dict[str, Any]:
        """Export a repeatable local regression bundle for one static snapshot/image pair."""
        snapshot_path = Path(snapshot_path)
        snapshot = self._load_snapshot_with_sidecars(snapshot_path)
        resolved_image_path = self._resolve_snapshot_image_path(snapshot_path, image_path)
        if resolved_image_path is not None:
            snapshot = self._refresh_snapshot_from_image(snapshot, resolved_image_path)

        candidates = build_boundary_candidates(snapshot, max_candidates=max_candidates)
        payload = build_openclaw_payload(snapshot, task=task, max_candidates=max_candidates)

        output_root = Path(output_dir) if output_dir is not None else snapshot_path.parent
        output_root.mkdir(parents=True, exist_ok=True)
        bundle_prefix = snapshot_path.stem.replace("_snapshot", "")

        candidates_path = output_root / f"{bundle_prefix}_candidate_regression.json"
        payload_path = output_root / f"{bundle_prefix}_openclaw_payload.json"
        candidates_path.write_text(
            json.dumps(candidates, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        payload_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        overlay_path: str | None = None
        if resolved_image_path is not None:
            overlay_file = output_root / f"{bundle_prefix}_candidate_overlay.png"
            overlay_path = str(
                self._save_candidate_overlay(
                    image_path=resolved_image_path,
                    candidates=candidates,
                    output_path=overlay_file,
                    draw_labels=draw_labels,
                )
            )

        return {
            "snapshot_path": str(snapshot_path),
            "image_path": str(resolved_image_path) if resolved_image_path is not None else None,
            "task": task,
            "candidate_count": len(candidates),
            "payload_candidate_count": len(payload.get("candidates") or []),
            "candidates_path": str(candidates_path),
            "payload_path": str(payload_path),
            "overlay_path": overlay_path,
            "provider_trace": dict(snapshot.get("provider_trace") or {}),
        }

    def export_real_app_regressions(
        self,
        baseline_dir: str | Path,
        task: str = "分析当前页面中的可交互候选",
        max_candidates: int = 160,
        draw_labels: bool = False,
    ) -> dict[str, Any]:
        """Export and summarize candidate regressions for every static app baseline snapshot."""
        baseline_dir = Path(baseline_dir)
        snapshots = sorted(baseline_dir.glob("*_snapshot.json"))
        summary_entries: list[dict[str, Any]] = []

        for snapshot_path in snapshots:
            result = self.export_candidate_regression(
                snapshot_path=snapshot_path,
                task=task,
                image_path=None,
                output_dir=baseline_dir,
                max_candidates=max_candidates,
                draw_labels=draw_labels,
            )
            payload = json.loads(Path(result["payload_path"]).read_text(encoding="utf-8"))
            snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
            provider_path = baseline_dir / snapshot_path.name.replace("_snapshot.json", "_provider.json")
            provider_report = {}
            if provider_path.exists():
                provider_report = json.loads(provider_path.read_text(encoding="utf-8"))
            summary_entries.append(
                self._build_real_app_regression_entry(
                    snapshot_path=snapshot_path,
                    regression_result=result,
                    snapshot=snapshot,
                    payload=payload,
                    provider_report=provider_report,
                )
            )

        summary_path = baseline_dir / "real_app_candidate_regression_summary.json"
        summary = {
            "baseline_dir": str(baseline_dir),
            "snapshot_count": len(summary_entries),
            "summary": summary_entries,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "baseline_dir": str(baseline_dir),
            "snapshot_count": len(summary_entries),
            "summary_path": str(summary_path),
            "summary": summary_entries,
        }

    def normalize_decision_record(
        self,
        decision_path: str | Path,
    ) -> dict[str, Any]:
        """Normalize one DeskCanvas decision JSON into the repo's canonical decision record."""
        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        return {
            "decision_path": str(decision_path),
            "decision_record": normalize_decision_record(decision),
        }

    def build_recrop_request(
        self,
        snapshot_path: str | Path,
        decision_path: str | Path,
        crop_scale: float = 1.5,
    ) -> dict[str, Any]:
        """Build a local recrop request when DeskCanvas returns a focus bbox."""
        snapshot = self._load_snapshot_with_sidecars(snapshot_path)
        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        decision_record = normalize_decision_record(decision)
        focus_bbox = decision_record.get("focus_bbox")
        if not focus_bbox:
            return {
                "snapshot_path": str(snapshot_path),
                "decision_path": str(decision_path),
                "error": "focus_bbox_missing",
            }

        return {
            "snapshot_path": str(snapshot_path),
            "decision_path": str(decision_path),
            "recrop_request": {
                "source_canvas_id": snapshot.get("canvas_id"),
                "focus_bbox": focus_bbox,
                "crop_scale": crop_scale,
                "page_state": decision_record.get("page_state"),
                "reason": decision_record.get("reason"),
                "next_action": decision_record.get("next_action"),
            },
        }

    def reanalyze_focus_region(
        self,
        snapshot_path: str | Path,
        decision_path: str | Path,
        task: str,
        image_path: str | Path | None = None,
        crop_scale: float = 1.5,
        max_candidates: int = 120,
        save_crop: bool = True,
    ) -> dict[str, Any]:
        """
        Crop one uncertain region from an existing screenshot, rerun local OCR/vision,
        and build a second-round DeskCanvas payload for that focused area.
        """
        snapshot_path = Path(snapshot_path)
        snapshot = self._load_snapshot_with_sidecars(snapshot_path)
        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        decision_record = normalize_decision_record(decision)
        focus_bbox = decision_record.get("focus_bbox")
        if not focus_bbox:
            return {
                "snapshot_path": str(snapshot_path),
                "decision_path": str(decision_path),
                "error": "focus_bbox_missing",
            }

        resolved_image_path = self._resolve_snapshot_image_path(snapshot_path, image_path)
        if resolved_image_path is None:
            return {
                "snapshot_path": str(snapshot_path),
                "decision_path": str(decision_path),
                "error": "source_image_missing",
            }

        source_image = Image.open(resolved_image_path).convert("RGB")
        crop_rect = self._expand_focus_bbox(
            tuple(int(value) for value in focus_bbox),
            source_image.size,
            crop_scale=crop_scale,
        )
        cropped = source_image.crop(crop_rect)

        crop_path: str | None = None
        if save_crop:
            crop_file = snapshot_path.parent / f"{snapshot_path.stem.replace('_snapshot', '')}_focus_crop.png"
            cropped.save(crop_file)
            crop_path = str(crop_file)

        ocr_result = self._ocr_service.extract_with_metadata(cropped)
        vision_candidates: list[dict[str, Any]] = []
        vision_provider_details: dict[str, Any] = {}
        try:
            vision_provider = getattr(self._perception_service, "_vision_provider", None)
            if vision_provider is not None:
                vision_result = vision_provider.parse_screenshot(cropped)
                vision_provider_details = {
                    "provider": vision_result.provider,
                    "success": vision_result.success,
                    "error": vision_result.error,
                    "candidate_count": len(vision_result.candidates),
                }
                vision_candidates = [
                    {
                        "candidate_id": candidate.element_id,
                        "bbox": list(candidate.bounding_box),
                        "kind": candidate.semantic_label,
                        "text": candidate.text,
                        "source": vision_result.provider,
                        "confidence": candidate.confidence,
                    }
                    for candidate in vision_result.candidates
                    if candidate.bounding_box
                ]
        except Exception as exc:
            vision_provider_details = {
                "provider": "omniparser",
                "success": False,
                "error": str(exc),
                "candidate_count": 0,
            }

        focus_snapshot = {
            "canvas_id": f"{snapshot.get('canvas_id', 'snapshot')}_focus",
            "app": dict(snapshot.get("app") or {}),
            "window": {"rect_client": [0, 0, cropped.size[0], cropped.size[1]]},
            "surface": dict(snapshot.get("surface") or {}),
            "page": {
                "page_class": snapshot.get("page", {}).get("page_class"),
            },
            "regions": [
                {
                    "region_id": "focus_region",
                    "role": "focus_region",
                    "subtype": "unknown",
                    "bounds": [0, 0, cropped.size[0], cropped.size[1]],
                }
            ],
            "elements": [],
            "artifacts": {
                "ocr_blocks": [
                    {
                        "bbox": list(block.bbox),
                        "text": block.text,
                        "confidence": block.confidence,
                    }
                    for block in ocr_result.blocks
                ],
                "vision_candidates": vision_candidates,
                "focus_source_bbox": list(crop_rect),
            },
        }

        return {
            "snapshot_path": str(snapshot_path),
            "decision_path": str(decision_path),
            "image_path": str(resolved_image_path),
            "crop_path": crop_path,
            "focus_bbox": list(focus_bbox),
            "crop_rect": list(crop_rect),
            "crop_size": list(cropped.size),
            "ocr_provider": {
                "provider": ocr_result.provider,
                "success": ocr_result.success,
                "error": ocr_result.error,
                "block_count": len(ocr_result.blocks),
            },
            "vision_provider": vision_provider_details,
            "focused_openclaw_payload": build_openclaw_payload(
                focus_snapshot,
                task=task,
                max_candidates=max_candidates,
            ),
        }

    def _resolve_window_info(self, hwnd: int) -> WindowInfoExt:
        windows = self._window_enum_service.enumerate_all(refresh=True)
        window_info = next((item for item in windows if item.hwnd == hwnd), None)
        if window_info is None:
            foreground = self._window_enum_service.get_foreground_window()
            if foreground and foreground.hwnd == hwnd:
                return foreground
            raise ValueError(f"Window with hwnd={hwnd} not found")
        return window_info

    def _serialize_uia_tree(
        self,
        uia_tree: list[tuple[int, Any]],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for depth, element in uia_tree:
            serialized.append(
                {
                    "depth": depth,
                    "element_id": element.element_id,
                    "name": element.name,
                    "automation_id": element.automation_id,
                    "control_type": element.control_type,
                    "bounding_rect": list(element.bounding_rect) if element.bounding_rect else None,
                    "is_enabled": element.is_enabled,
                }
            )
        return serialized

    def _resolve_snapshot_image_path(
        self,
        snapshot_path: Path,
        image_path: str | Path | None,
    ) -> Path | None:
        if image_path is not None:
            candidate = Path(image_path)
            return candidate if candidate.exists() else None

        bundle_prefix = snapshot_path.stem.replace("_snapshot", "")
        candidates = [
            snapshot_path.parent / f"{bundle_prefix}_raw.png",
            snapshot_path.parent / f"{bundle_prefix}_preview.png",
        ]
        generic_prefix = bundle_prefix.rsplit("_", 2)[0] if bundle_prefix.count("_") >= 2 else bundle_prefix
        candidates.extend(
            [
                snapshot_path.parent / f"{generic_prefix}_raw.png",
                snapshot_path.parent / f"{generic_prefix}_preview.png",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_snapshot_with_sidecars(
        self,
        snapshot_path: str | Path,
    ) -> dict[str, Any]:
        snapshot_path = Path(snapshot_path)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        artifacts = dict(snapshot.get("artifacts") or {})
        bundle_prefix = snapshot_path.stem.replace("_snapshot", "")

        ocr_path = snapshot_path.parent / f"{bundle_prefix}_ocr.json"
        if not artifacts.get("ocr_blocks") and ocr_path.exists():
            raw_ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            if isinstance(raw_ocr, list):
                artifacts["ocr_blocks"] = raw_ocr

        vision_path = snapshot_path.parent / f"{bundle_prefix}_vision.json"
        if vision_path.exists():
            raw_vision = json.loads(vision_path.read_text(encoding="utf-8"))
            vision_bundle = self._normalize_vision_sidecar(raw_vision)
            if not artifacts.get("vision_candidates") and vision_bundle["vision_candidates"]:
                artifacts["vision_candidates"] = vision_bundle["vision_candidates"]
            if not artifacts.get("vision_layout_regions") and vision_bundle["vision_layout_regions"]:
                artifacts["vision_layout_regions"] = vision_bundle["vision_layout_regions"]
            if not artifacts.get("vision_control_groups") and vision_bundle["vision_control_groups"]:
                artifacts["vision_control_groups"] = vision_bundle["vision_control_groups"]
            if (
                not artifacts.get("vision_interaction_hints")
                and vision_bundle["vision_interaction_hints"]
            ):
                artifacts["vision_interaction_hints"] = vision_bundle["vision_interaction_hints"]
            if vision_bundle["vision_provider"]:
                artifacts.setdefault("vision_provider", vision_bundle["vision_provider"])

        provider_path = snapshot_path.parent / f"{bundle_prefix}_provider.json"
        provider_trace = dict(snapshot.get("provider_trace") or {})
        if provider_path.exists():
            provider_report = json.loads(provider_path.read_text(encoding="utf-8"))
            provider_details = dict(provider_trace.get("provider_details") or {})
            for key, value in dict(provider_report.get("details") or {}).items():
                provider_details.setdefault(key, value)
            if provider_details:
                provider_trace["provider_details"] = provider_details
            for flag in ("uia_used", "ocr_used", "vision_used", "dom_used"):
                provider_trace.setdefault(flag, provider_report.get(flag))

        snapshot["artifacts"] = artifacts
        if provider_trace:
            snapshot["provider_trace"] = provider_trace
        return snapshot

    def _normalize_vision_sidecar(
        self,
        raw_vision: Any,
    ) -> dict[str, Any]:
        if isinstance(raw_vision, list):
            return {
                "vision_candidates": raw_vision,
                "vision_layout_regions": [],
                "vision_control_groups": [],
                "vision_interaction_hints": [],
                "vision_provider": {},
            }

        if not isinstance(raw_vision, dict):
            return {
                "vision_candidates": [],
                "vision_layout_regions": [],
                "vision_control_groups": [],
                "vision_interaction_hints": [],
                "vision_provider": {},
            }

        provider_name = raw_vision.get("vision_provider") or raw_vision.get("provider")
        return {
            "vision_candidates": list(raw_vision.get("vision_candidates") or []),
            "vision_layout_regions": list(raw_vision.get("vision_layout_regions") or []),
            "vision_control_groups": list(raw_vision.get("vision_control_groups") or []),
            "vision_interaction_hints": list(raw_vision.get("vision_interaction_hints") or []),
            "vision_provider": {
                "provider": provider_name,
                "candidate_count": int(raw_vision.get("vision_candidate_count") or 0),
                "layout_region_count": int(raw_vision.get("vision_layout_region_count") or 0),
                "control_group_count": int(raw_vision.get("vision_control_group_count") or 0),
            }
            if provider_name or raw_vision.get("vision_candidate_count") is not None
            else {},
        }

    def _refresh_snapshot_from_image(
        self,
        snapshot: dict[str, Any],
        image_path: Path,
    ) -> dict[str, Any]:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return snapshot

        artifacts = dict(snapshot.get("artifacts") or {})
        provider_trace = dict(snapshot.get("provider_trace") or {})
        provider_details = dict(provider_trace.get("provider_details") or {})

        try:
            ocr_result = self._ocr_service.extract_with_metadata(image)
            if isinstance(getattr(ocr_result, "success", None), bool):
                artifacts["ocr_blocks"] = [
                    {
                        "bbox": list(block.bbox),
                        "text": block.text,
                        "confidence": block.confidence,
                    }
                    for block in ocr_result.blocks
                ]
                ocr_provider = {
                    "provider": ocr_result.provider,
                    "success": ocr_result.success,
                    "error": ocr_result.error,
                    "elapsed_seconds": ocr_result.elapsed_seconds,
                    "used_region": list(ocr_result.used_region) if ocr_result.used_region else None,
                    "worker_command": list(ocr_result.worker_command or []),
                    "block_count": len(ocr_result.blocks),
                }
                artifacts["ocr_provider"] = ocr_provider
                provider_details["ocr_provider"] = ocr_provider
                provider_details["ocr"] = {
                    "used": bool(ocr_result.success and ocr_result.blocks),
                    "block_count": len(ocr_result.blocks),
                    "ocr_locator_count": len(ocr_result.blocks),
                }
                provider_trace["ocr_used"] = bool(ocr_result.success and ocr_result.blocks)
        except Exception as exc:
            ocr_provider = {
                "provider": "paddleocr_bridge",
                "success": False,
                "error": str(exc),
                "block_count": 0,
            }
            artifacts["ocr_blocks"] = []
            artifacts["ocr_provider"] = ocr_provider
            provider_details["ocr_provider"] = ocr_provider
            provider_details["ocr"] = {
                "used": False,
                "block_count": 0,
                "ocr_locator_count": 0,
            }
            provider_trace["ocr_used"] = False

        vision_provider = getattr(self._perception_service, "_vision_provider", None)
        if vision_provider is None:
            snapshot["artifacts"] = artifacts
            provider_trace["provider_details"] = provider_details
            snapshot["provider_trace"] = provider_trace
            return snapshot

        try:
            vision_result = vision_provider.parse_screenshot(image)
        except Exception as exc:
            vision_provider_details = {
                "provider": "omniparser",
                "success": False,
                "error": str(exc),
                "candidate_count": 0,
                "layout_region_count": 0,
                "control_group_count": 0,
            }
            artifacts["vision_candidates"] = []
            artifacts["vision_layout_regions"] = []
            artifacts["vision_control_groups"] = []
            artifacts["vision_interaction_hints"] = []
            artifacts["vision_provider"] = vision_provider_details
            provider_details["vision_provider"] = vision_provider_details
            provider_details["vision"] = {
                "used": False,
                "candidate_count": 0,
                "layout_region_count": 0,
                "control_group_count": 0,
                "interaction_hint_count": 0,
                "structure_evidence_score": None,
            }
            provider_details["vision_candidates"] = {"count": 0}
            provider_trace["provider_details"] = provider_details
            provider_trace["vision_used"] = False
            snapshot["artifacts"] = artifacts
            snapshot["provider_trace"] = provider_trace
            return snapshot
        if not isinstance(getattr(vision_result, "success", None), bool):
            snapshot["artifacts"] = artifacts
            provider_trace["provider_details"] = provider_details
            snapshot["provider_trace"] = provider_trace
            return snapshot

        artifacts["vision_candidates"] = [
            {
                "candidate_id": candidate.element_id,
                "bbox": list(candidate.bounding_box),
                "confidence": candidate.confidence,
                "kind": candidate.semantic_label,
                "text": candidate.text,
                "icon_type": candidate.icon_type,
                "region_role": candidate.region_role,
                "group_id": candidate.group_id,
                "interaction_hints": candidate.interaction_hints or {},
                "structure_evidence_score": candidate.structure_evidence_score,
                "attributes": candidate.attributes or {},
                "source": vision_result.provider,
            }
            for candidate in vision_result.candidates
            if candidate.bounding_box
        ]
        artifacts["vision_layout_regions"] = list(vision_result.layout_regions or [])
        artifacts["vision_control_groups"] = list(vision_result.control_groups or [])
        artifacts["vision_interaction_hints"] = list(vision_result.interaction_hints or [])
        if vision_result.structure_evidence_score is not None:
            artifacts["structure_evidence_score"] = vision_result.structure_evidence_score
        vision_provider_details = {
            "provider": vision_result.provider,
            "success": vision_result.success,
            "error": vision_result.error,
            "candidate_count": len(vision_result.candidates),
            "layout_region_count": len(vision_result.layout_regions or []),
            "control_group_count": len(vision_result.control_groups or []),
            "interaction_hint_count": len(vision_result.interaction_hints or []),
            "visual_score": vision_result.visual_score,
            "structure_evidence_score": vision_result.structure_evidence_score,
        }
        artifacts["vision_provider"] = vision_provider_details

        provider_details["vision_provider"] = vision_provider_details
        provider_details["vision"] = {
            "used": bool(
                vision_result.success
                and (
                    vision_result.candidates
                    or vision_result.layout_regions
                    or vision_result.control_groups
                )
            ),
            "candidate_count": len(vision_result.candidates),
            "layout_region_count": len(vision_result.layout_regions or []),
            "control_group_count": len(vision_result.control_groups or []),
            "interaction_hint_count": len(vision_result.interaction_hints or []),
            "structure_evidence_score": vision_result.structure_evidence_score,
        }
        provider_details["vision_candidates"] = {"count": len(artifacts["vision_candidates"])}
        if artifacts["vision_layout_regions"]:
            provider_details["vision_layout_regions"] = {
                "count": len(artifacts["vision_layout_regions"])
            }
        if artifacts["vision_control_groups"]:
            provider_details["vision_control_groups"] = {
                "count": len(artifacts["vision_control_groups"])
            }
        if artifacts["vision_interaction_hints"]:
            provider_details["vision_interaction_hints"] = {
                "count": len(artifacts["vision_interaction_hints"])
            }
        if vision_result.structure_evidence_score is not None:
            provider_details["structure_evidence_score"] = vision_result.structure_evidence_score

        provider_trace["provider_details"] = provider_details
        provider_trace["vision_used"] = bool(
            vision_result.success
            and (
                vision_result.candidates
                or vision_result.layout_regions
                or vision_result.control_groups
            )
        )
        snapshot["artifacts"] = artifacts
        snapshot["provider_trace"] = provider_trace
        return snapshot

    def _expand_focus_bbox(
        self,
        focus_bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        crop_scale: float,
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = focus_bbox
        width = max(1, right - left)
        height = max(1, bottom - top)
        center_x = left + width / 2
        center_y = top + height / 2
        expanded_width = max(width, int(width * crop_scale))
        expanded_height = max(height, int(height * crop_scale))
        crop_left = max(0, int(center_x - expanded_width / 2))
        crop_top = max(0, int(center_y - expanded_height / 2))
        crop_right = min(image_size[0], crop_left + expanded_width)
        crop_bottom = min(image_size[1], crop_top + expanded_height)
        return crop_left, crop_top, crop_right, crop_bottom

    def _save_candidate_overlay(
        self,
        image_path: Path,
        candidates: list[dict[str, Any]],
        output_path: Path,
        draw_labels: bool,
    ) -> Path:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        source_colors = {
            "ocr": (56, 189, 248),
            "vision": (250, 204, 21),
            "omniparser": (250, 204, 21),
            "uia": (74, 222, 128),
            "native": (74, 222, 128),
            "merged": (244, 114, 182),
            "unknown": (203, 213, 225),
        }
        for candidate in candidates:
            bbox = candidate.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            left, top, right, bottom = [int(value) for value in bbox]
            color = source_colors.get(str(candidate.get("source") or "unknown"), source_colors["unknown"])
            draw.rectangle([left, top, right, bottom], outline=color, width=2)
            if draw_labels:
                label = str(candidate.get("candidate_id") or "")
                if candidate.get("text"):
                    label = f"{label}:{candidate.get('text')}"
                draw.text((left + 2, max(0, top - 14)), label[:48], fill=color)
        image.save(output_path)
        return output_path

    def _build_real_app_regression_entry(
        self,
        snapshot_path: Path,
        regression_result: dict[str, Any],
        snapshot: dict[str, Any],
        payload: dict[str, Any],
        provider_report: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_summary = dict(payload.get("candidate_summary") or {})
        provider_health = dict(payload.get("provider_health") or {})
        vision_provider = dict(provider_health.get("vision_provider") or {})
        effective_provider_report = self._build_effective_provider_report(
            provider_report=provider_report,
            provider_health=provider_health,
            provider_details=((regression_result.get("provider_trace") or {}).get("provider_details") or {}),
            candidate_summary=candidate_summary,
        )
        issues: list[str] = []
        if vision_provider.get("success") is False:
            issues.append("omniparser_provider_unavailable")
        if not provider_health.get("vision_effective"):
            issues.append("vision_not_used")
        if int(provider_health.get("vision_sidecar_count") or 0) > 0 and not int(
            provider_health.get("omniparser_total_count") or 0
        ):
            issues.append("vision_sidecar_only")
        if not int(provider_health.get("omniparser_retained_count") or 0):
            issues.append("no_omniparser_candidates_retained")
        if int(regression_result.get("candidate_count") or 0) < 12:
            issues.append("candidate_count_sparse")
        if (
            int(len(snapshot.get("elements") or [])) <= 2
            and int(regression_result.get("candidate_count") or 0) < 12
            and not int(provider_health.get("omniparser_total_count") or 0)
        ):
            issues.append("element_count_sparse")

        return {
            "app": (((snapshot.get("app") or {}).get("process_name")) or "").replace(".exe", ""),
            "snapshot_path": str(snapshot_path),
            "image_path": regression_result.get("image_path"),
            "candidate_count": regression_result.get("candidate_count"),
            "payload_candidate_count": regression_result.get("payload_candidate_count"),
            "candidate_summary": candidate_summary,
            "provider_health": provider_health,
            "provider_report": effective_provider_report,
            "region_count": len(snapshot.get("regions") or []),
            "element_count": len(snapshot.get("elements") or []),
            "ocr_block_count": int(
                dict(provider_health.get("ocr_provider") or {}).get("block_count")
                or len((snapshot.get("artifacts") or {}).get("ocr_blocks") or [])
            ),
            "top_candidate_ids": list(candidate_summary.get("top_candidate_ids") or []),
            "issues": issues,
            "candidates_path": regression_result.get("candidates_path"),
            "payload_path": regression_result.get("payload_path"),
            "overlay_path": regression_result.get("overlay_path"),
        }

    def _build_effective_provider_report(
        self,
        provider_report: dict[str, Any],
        provider_health: dict[str, Any],
        provider_details: dict[str, Any],
        candidate_summary: dict[str, Any],
    ) -> dict[str, Any]:
        report = dict(provider_report or {})
        report.pop("details", None)
        report["uia_used"] = bool(provider_health.get("uia_used"))
        report["ocr_used"] = bool(provider_health.get("ocr_used"))
        report["vision_used"] = bool(provider_health.get("vision_used"))
        report["dom_used"] = bool(provider_health.get("dom_used"))

        vision_provider = dict(provider_health.get("vision_provider") or {})
        ocr_provider = dict(provider_health.get("ocr_provider") or {})
        live_vision = dict(provider_details.get("vision") or {})
        live_ocr = dict(provider_details.get("ocr") or {})

        report["vision"] = {
            **dict(report.get("vision") or {}),
            "used": bool(provider_health.get("vision_used")),
            "candidate_count": int(vision_provider.get("candidate_count") or 0),
            "layout_region_count": int(vision_provider.get("layout_region_count") or 0),
            "control_group_count": int(vision_provider.get("control_group_count") or 0),
            "interaction_hint_count": int(live_vision.get("interaction_hint_count") or 0),
            "structure_evidence_score": live_vision.get("structure_evidence_score"),
        }
        report["ocr"] = {
            **dict(report.get("ocr") or {}),
            "used": bool(provider_health.get("ocr_used")),
            "block_count": int(ocr_provider.get("block_count") or 0),
            "ocr_locator_count": int(live_ocr.get("ocr_locator_count") or 0),
        }
        report["vision_provider"] = vision_provider
        report["ocr_provider"] = ocr_provider
        report["vision_candidates"] = {
            "count": int(provider_details.get("vision_candidates", {}).get("count") or 0)
        }
        report["candidate_sources"] = dict(candidate_summary.get("by_source_total") or {})
        report["omniparser"] = {
            "available": bool(provider_health.get("omniparser_available")),
            "total_count": int(provider_health.get("omniparser_total_count") or 0),
            "retained_count": int(provider_health.get("omniparser_retained_count") or 0),
            "sidecar_count": int(provider_health.get("vision_sidecar_count") or 0),
        }
        return report
