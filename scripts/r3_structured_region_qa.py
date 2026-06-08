"""Phase R3.0 + U4: StructuredRegion Visual QA + Baseline Report.

Generates overlay images and quantitative metrics for structured_region_overlay
across multiple apps. Does NOT modify any production code.

Usage:
    python scripts/r3_structured_region_qa.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Enable U4 layout shadow and URE P0 shadow before importing PerceptionService
os.environ["OPENCLAW_U4_LAYOUT_SHADOW"] = "1"
os.environ["OPENCLAW_URE_P0_SHADOW"] = "1"
os.environ["OPENCLAW_VTL_0_SHADOW"] = "1"

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Output directory
REPORT_DIR = Path("reports/r3_structured_region_qa")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Colors for overlay
COLORS = {
    "top_bar": "#FFD700",         # Gold
    "toolbar": "#FF8C00",         # Dark Orange
    "side_rail": "#9370DB",       # Medium Purple
    "list_region": "#20B2AA",     # Light Sea Green
    "grid_region": "#4682B4",     # Steel Blue
    "canvas_region": "#3CB371",   # Medium Sea Green
    "document_region": "#87CEEB", # Sky Blue
    "media_control_bar": "#FF69B4", # Hot Pink
    "input_region": "#FFA07A",    # Light Salmon
    "status_region": "#D2B48C",   # Tan
    "floating_panel": "#C0C0C0",  # Silver
    "dialog": "#DDA0DD",          # Plum
    "unknown_structured": "#808080", # Gray
}
COLOR_EVIDENCE_OCR = "#FF4444"
COLOR_EVIDENCE_VISION = "#44FF44"


def get_font(size: int = 14):
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "arial.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_overlay(
    screenshot: Image.Image,
    overlay: dict,
    geometric_regions: list[dict] | None = None,
    ocr_blocks: list[dict] | None = None,
    vision_candidates: list[dict] | None = None,
    u4_override_regions: list[dict] | None = None,
) -> Image.Image:
    """Draw structured region overlay on screenshot.

    Args:
        u4_override_regions: If provided, draw these instead of the default
            U4 candidate_regions (used for merged overlay).
    """
    img = screenshot.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(12)
    font_small = get_font(10)

    # Draw geometric regions (light fill)
    if geometric_regions:
        for gr in geometric_regions:
            bounds = gr.get("bounds", [])
            if len(bounds) >= 4:
                l, t, r, b = bounds
                draw.rectangle([l, t, r, b], outline="#FFFFFF40", width=1)

    # Draw structured regions
    for reg in overlay.get("regions", []):
        bounds = reg.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        st = reg.get("structure_type", "unknown_structured")
        color = COLORS.get(st, "#808080")
        # Semi-transparent fill
        fill = color + "30"
        draw.rectangle([l, t, r, b], fill=fill, outline=color, width=2)
        # Label
        label = f"{reg.get('region_id', '?')} {st}"
        if reg.get("semantic_label"):
            label += f" ({reg['semantic_label']})"
        draw.text((l + 4, t + 4), label, fill=color, font=font)

    # Draw OCR blocks (red dots)
    if ocr_blocks:
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                bl, bt, br, bb = bbox
                cx, cy = (bl + br) // 2, (bt + bb) // 2
                draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=COLOR_EVIDENCE_OCR)

    # Draw vision candidates (green dots)
    if vision_candidates:
        for vc in vision_candidates:
            bbox = vc.get("bounding_rect") or vc.get("bbox", [])
            if len(bbox) >= 4:
                bl, bt, br, bb = bbox
                cx, cy = (bl + br) // 2, (bt + bb) // 2
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=COLOR_EVIDENCE_VISION)

    # Draw band proposals (R3.3)
    BAND_COLORS = {
        "TP": "#00FF00",   # Green
        "AMB": "#FFD700",  # Yellow
        "FP": "#FF4444",   # Red
    }
    band_audit = overlay.get("band_proposal_audit", {})
    audit_details = {d["band_id"]: d for d in band_audit.get("details", [])}
    for bp in overlay.get("band_proposals", []):
        bounds = bp.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        bid = bp.get("band_id", "?")
        audit = audit_details.get(bid, {})
        verdict = audit.get("verdict", "?")
        color = BAND_COLORS.get(verdict, "#808080")
        h_px = b - t
        # Semi-transparent fill
        fill = color + "20"
        draw.rectangle([l, t, r, b], fill=fill, outline=color, width=2)
        # Label at right edge
        label = f"{bid} {verdict} {h_px}px"
        draw.text((r - len(label) * 7 - 4, t + 2), label, fill=color, font=font_small)

    # Draw U4 layout candidate regions (cyan outlines)
    u4_data = overlay.get("ocr_visual_comparison", {}).get("u4_layout", {})
    if u4_data:
        COLOR_U4 = "#00CED1"  # Dark Turquoise
        COLOR_U4_SEP = "#00FFFF"  # Cyan for separators
        # Draw candidate regions (use override if provided)
        u4_regions_to_draw = u4_override_regions if u4_override_regions is not None else u4_data.get("candidate_regions", [])
        for cr in u4_regions_to_draw:
            bounds = cr.get("bounds", [])
            if len(bounds) >= 4:
                l, t, r, b = bounds
                fill = COLOR_U4 + "18"
                draw.rectangle([l, t, r, b], fill=fill, outline=COLOR_U4, width=2)
                label = f"U4:{cr.get('region_id','?')} ocr={cr.get('ocr_count',0)} vis={cr.get('vision_count',0)}"
                draw.text((l + 4, t + 4), label, fill=COLOR_U4, font=font_small)
        # Draw horizontal separators
        for sep in u4_data.get("horizontal_separators", []):
            y = sep.get("position", 0)
            draw.line([(0, y), (screenshot.width, y)], fill=COLOR_U4_SEP, width=1)
        # Draw vertical separators
        for sep in u4_data.get("vertical_separators", []):
            x = sep.get("position", 0)
            draw.line([(x, 0), (x, screenshot.height)], fill=COLOR_U4_SEP, width=1)

    return img


# Substructure type colors
SUB_COLORS = {
    "text_row": "#4488FF",          # Blue
    "code_line": "#8844FF",         # Purple
    "message_like": "#44CC88",      # Green
    "list_item_like": "#FF8844",    # Orange
    "table_row_like": "#CC44CC",    # Magenta
    "icon_candidate": "#FFCC00",    # Yellow
    "control_candidate": "#FF4444", # Red
    "whitespace_block": "#CCCCCC",  # Gray
    "unknown_evidence": "#888888",  # Dark gray
}


def draw_substructure_overlay(
    screenshot: Image.Image,
    overlay: dict,
) -> Image.Image:
    """Draw substructure/evidence overlay on screenshot."""
    img = screenshot.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(10)

    shadows = overlay.get("ocr_visual_comparison", {}).get(
        "u4_layout", {},
    ).get("page_region_model_shadow", [])

    for shadow in shadows:
        # Draw region boundary (thin white)
        rb = shadow.get("bounds", [])
        if len(rb) >= 4:
            draw.rectangle(rb, outline="#FFFFFF40", width=1)

        # Draw substructures
        for sub in shadow.get("substructures", []):
            bounds = sub.get("bounds", [])
            if len(bounds) < 4:
                continue
            l, t, r, b = bounds
            sub_type = sub.get("sub_type", "unknown_evidence")
            color = SUB_COLORS.get(sub_type, "#888888")
            fill = color + "30"
            draw.rectangle([l, t, r, b], fill=fill, outline=color, width=1)
            label = f"{sub_type[:8]} {sub.get('confidence', 0):.1f}"
            draw.text((l + 2, t + 2), label, fill=color, font=font)

    return img


# URE P0 overlay colors
URE_COLORS = {
    "geometric_region": "#4488FF",       # Blue
    "structured_group": "#8844FF",       # Purple
    "band_proposal": "#44CC88",          # Green
    "u4_candidate": "#FF8844",           # Orange
    "full_window_fallback": "#CCCCCC",   # Gray
}


def draw_ure_overlay(
    screenshot: Image.Image,
    overlay: dict,
) -> Image.Image:
    """Draw URE P0 PageRegion overlay on screenshot."""
    img = screenshot.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(11)

    ure_data = overlay.get("ocr_visual_comparison", {}).get(
        "universal_region_shadow", {},
    )
    if not ure_data:
        return img

    for pr in ure_data.get("page_regions", []):
        bounds = pr.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        skel = pr.get("skeleton_source", "unknown")
        color = URE_COLORS.get(skel, "#888888")
        fill = color + "25"
        draw.rectangle([l, t, r, b], fill=fill, outline=color, width=2)

        # Label
        sub_count = len(pr.get("substructures", []))
        ev = pr.get("evidence_sources", [])
        label = f"{pr.get('region_id', '?')} {skel[:8]} sub={sub_count} ev={'+'.join(ev) if ev else 'none'}"
        draw.text((l + 4, t + 4), label, fill=color, font=font)

    return img


# VTL-0 overlay colors
VTL_COLORS = {
    "page_container": "#4488FF",        # Blue
    "actionable_object": "#44CC88",     # Green
    "content_object": "#FF8844",        # Orange
    "text_substructure": "#88888880",   # Gray (semi-transparent)
}


def draw_vtl_overlay(
    screenshot: Image.Image,
    overlay: dict,
) -> Image.Image:
    """Draw VTL-0 VisualTraceObject overlay on screenshot.

    Color scheme:
    - page_containers: blue solid thick border
    - container_candidates: light blue thin border (not filled)
    - actionable_object: green
    - content_object: orange
    - text_substructure: gray dashed
    - rejected_text_like: gray dotted
    """
    img = screenshot.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(10)

    vtl_data = overlay.get("ocr_visual_comparison", {}).get(
        "visual_trace_shadow", {},
    )
    if not vtl_data:
        return img

    # Draw container_candidates (light blue thin, no fill) — only those NOT in page_containers
    container_ids = {pc.get("trace_id") for pc in vtl_data.get("page_containers", [])}
    for cc in vtl_data.get("container_candidates", []):
        if cc.get("trace_id") in container_ids:
            continue  # Skip, will be drawn as page_container
        bounds = cc.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        draw.rectangle([l, t, r, b], outline="#88CCFF40", width=1)

    # Draw page_containers (blue solid, thick)
    for pc in vtl_data.get("page_containers", []):
        bounds = pc.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        draw.rectangle([l, t, r, b], fill="#4488FF15", outline="#4488FF", width=3)
        label = f"PC:{pc.get('trace_id', '?')}"
        draw.text((l + 4, t + 4), label, fill="#4488FF", font=font)

    # Draw local_objects (green for actionable, orange for content)
    for obj in vtl_data.get("local_objects", []):
        bounds = obj.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        obj_class = obj.get("object_class", "content_object")
        color = VTL_COLORS.get(obj_class, "#888888")
        fill = color[:7] + "20"
        draw.rectangle([l, t, r, b], fill=fill, outline=color[:7], width=2)
        label = f"{obj_class[:6]} {obj.get('confidence', 0):.1f}"
        draw.text((l + 2, t + 2), label, fill=color[:7], font=font)

    # Draw text_substructures (gray dashed)
    for ts in vtl_data.get("text_substructures", []):
        bounds = ts.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        draw.rectangle([l, t, r, b], outline="#88888830", width=1)

    # Draw rejected_text_like (gray dotted, very faint)
    for rt in vtl_data.get("rejected_text_like", []):
        bounds = rt.get("bounds", [])
        if len(bounds) < 4:
            continue
        l, t, r, b = bounds
        draw.rectangle([l, t, r, b], outline="#88888820", width=1)

    return img


def collect_metrics(overlay: dict) -> dict:
    """Collect quantitative metrics from overlay."""
    regions = overlay.get("regions", [])
    by_type: dict[str, int] = {}
    for reg in regions:
        st = reg.get("structure_type", "unknown_structured")
        by_type[st] = by_type.get(st, 0) + 1

    # Group metrics
    groups = overlay.get("groups", [])
    by_group_type: dict[str, int] = {}
    for g in groups:
        st = g.get("structure_type", "unknown_structured")
        by_group_type[st] = by_group_type.get(st, 0) + 1

    total_ocr = 0
    total_vision = 0
    for reg in regions:
        for ev in reg.get("evidence", []):
            src = ev.get("source", "")
            if src == "ocr":
                detail = ev.get("detail", "")
                if "count=" in detail:
                    try:
                        count = int(detail.split("count=")[1].split(",")[0])
                        total_ocr += count
                    except ValueError:
                        pass
            elif src == "vision":
                detail = ev.get("detail", "")
                if "count=" in detail:
                    try:
                        count = int(detail.split("count=")[1].split(",")[0])
                        total_vision += count
                    except ValueError:
                        pass

    # Band proposal metrics
    band_proposals = overlay.get("band_proposals", [])
    band_audit = overlay.get("band_proposal_audit", {})

    # U4 layout metrics
    u4_data = overlay.get("ocr_visual_comparison", {}).get("u4_layout", {})
    u4_regions = u4_data.get("candidate_regions", [])
    u4_rejected = u4_data.get("rejected_reasons", {})

    return {
        "total_regions": overlay.get("total_regions", 0),
        "unknown_count": overlay.get("unknown_count", 0),
        "with_semantic_label": overlay.get("with_semantic_label", 0),
        "roi_eligible_count": overlay.get("roi_eligible_count", 0),
        "by_type": by_type,
        "total_ocr_evidence": total_ocr,
        "total_vision_evidence": total_vision,
        "total_groups": overlay.get("total_groups", 0),
        "group_unknown_count": overlay.get("group_unknown_count", 0),
        "group_roi_eligible_count": overlay.get("group_roi_eligible_count", 0),
        "by_group_type": by_group_type,
        "total_band_proposals": len(band_proposals),
        "band_tp": band_audit.get("tp", 0),
        "band_fp": band_audit.get("fp", 0),
        "band_amb": band_audit.get("amb", 0),
        "band_over_fragmented": band_audit.get("over_fragmented", 0),
        "band_proposal_audit": band_audit,
        "shadow_comparison": overlay.get("shadow_comparison", {}),
        "original_region_count": overlay.get("shadow_comparison", {}).get("original_region_count", 0),
        "original_unknown_count": overlay.get("shadow_comparison", {}).get("original_unknown_count", 0),
        "combined_region_count": overlay.get("shadow_comparison", {}).get("combined_region_count", 0),
        "combined_unknown_count": overlay.get("shadow_comparison", {}).get("combined_unknown_count", 0),
        "combined_roi_eligible_count": overlay.get("shadow_comparison", {}).get("combined_roi_eligible_count", 0),
        "injected_band_count": overlay.get("shadow_comparison", {}).get("injected_band_count", 0),
        "useful_injected_band_count": overlay.get("shadow_comparison", {}).get("useful_injected_band_count", 0),
        "noisy_injected_band_count": overlay.get("shadow_comparison", {}).get("noisy_injected_band_count", 0),
        # U4 layout metrics
        "u4_available": bool(u4_data),
        "u4_region_count": len(u4_regions),
        "u4_h_sep_count": len(u4_data.get("horizontal_separators", [])),
        "u4_v_sep_count": len(u4_data.get("vertical_separators", [])),
        "u4_density_transitions": len(u4_data.get("density_transitions", [])),
        "u4_proposals_count": u4_data.get("proposals_count", 0),
        "u4_useful_proposals": u4_data.get("useful_proposals", 0),
        "u4_neutral_proposals": u4_data.get("neutral_proposals", 0),
        "u4_noisy_proposals": u4_data.get("noisy_proposals", 0),
        "u4_rejected_reasons": u4_rejected,
        # U4.1 Phase 1: boundary signal diagnostics
        "u4_raw_separator_count": u4_data.get("raw_separator_count", 0),
        "u4_boundary_decisions": u4_data.get("boundary_decisions", []),
        "u4_signal_collection_errors": u4_data.get("signal_collection_errors", 0),
        # U4.2 Phase 1: region container signals
        "u4_region_signals": u4_data.get("region_signals", []),
        # U4.2 Phase B.1: pairwise flow signals
        "u4_pairwise_flow_signals": u4_data.get("pairwise_flow_signals", []),
    }


def run_app(name: str, hwnd: int, process: str) -> dict:
    """Run observe on one app and generate overlay."""
    from src.perception.perception_service import PerceptionService

    svc = PerceptionService()
    print(f"\n{'='*60}")
    print(f"  {name} (hwnd={hwnd}, process={process})")
    print(f"{'='*60}")

    t0 = time.time()
    zone_page = svc.analyze(hwnd=hwnd, allow_vlm=False)
    snapshot = svc.create_page_snapshot(zone_page, process_name=process)
    elapsed = time.time() - t0

    overlay = snapshot.artifacts.get("structured_region_overlay", {})
    geometric = snapshot.artifacts.get("geometric_regions", [])
    fusion = snapshot.artifacts.get("fusion_diagnostics", {})
    ocr_blocks = snapshot.artifacts.get("ocr_blocks", [])
    vision_candidates = snapshot.artifacts.get("vision_candidates", [])

    # Mark as excluded if no geometric regions (window not available / minimized)
    excluded = len(geometric) == 0

    metrics = collect_metrics(overlay)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["app"] = name
    metrics["hwnd"] = hwnd
    metrics["excluded"] = excluded

    if excluded:
        print(f"  EXCLUDED: no geometric regions (window not available or minimized)")
    else:
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Regions: {metrics['total_regions']} total, {metrics['unknown_count']} unknown, {metrics['with_semantic_label']} labeled")
        print(f"  By type: {json.dumps(metrics['by_type'], indent=2)}")
        print(f"  Groups: {metrics['total_groups']} total, {metrics['group_unknown_count']} unknown, {metrics['group_roi_eligible_count']} roi_eligible")
        print(f"  By group type: {json.dumps(metrics['by_group_type'], indent=2)}")
        print(f"  Band proposals: {metrics['total_band_proposals']} (TP={metrics['band_tp']}, FP={metrics['band_fp']}, AMB={metrics['band_amb']})")
        sc = overlay.get("shadow_comparison", {})
        details = sc.get("injected_band_details", [])
        if details:
            print(f"  Injected bands ({len(details)}):")
            for d in details:
                roi = "ROI" if d.get("roi_eligible") else "   "
                print(f"    {d.get('band_id','?'):5s} {d.get('band_type','?'):12s} h={d.get('height_px',0):3d}px → {d.get('resulting_structure_type','?'):20s} {roi} conf={d.get('confidence',0):.2f}")
    print(f"  OCR evidence: {metrics['total_ocr_evidence']} blocks in regions")
    print(f"  Vision evidence: {metrics['total_vision_evidence']} candidates in regions")
    print(f"  Geometric regions: {len(geometric)}")
    print(f"  Fusion labeled: {fusion.get('labeled_count', 0)}")
    # U4 layout metrics
    if metrics.get("u4_available"):
        print(f"  U4 layout: {metrics['u4_region_count']} regions, {metrics['u4_h_sep_count']}h/{metrics['u4_v_sep_count']}v seps, {metrics['u4_density_transitions']} density transitions")
        # U4.1 Phase 1: boundary signal report
        decisions = metrics.get("u4_boundary_decisions", [])
        if decisions:
            print(f"  U4.1 signals: {len(decisions)} gaps analyzed, {metrics.get('u4_signal_collection_errors', 0)} errors")
            for d in decisions[:8]:  # Show first 8
                sig = d.get("signals", {})
                ev = d.get("evidence_counts", {})
                has_ocr = d.get("has_ocr_evidence", False)
                has_vis = d.get("has_vision_evidence", False)
                has_uia = d.get("has_uia_evidence", False)
                evidence_tag = ""
                if has_ocr:
                    evidence_tag += "OCR"
                if has_vis:
                    evidence_tag += "+VIS"
                if has_uia:
                    evidence_tag += "+UIA"
                if not evidence_tag:
                    evidence_tag = "none"
                print(f"    {d['gap_id']:5s} pos={d['position']:4d} w={d['width']:3d} "
                      f"ocr_sim={sig.get('ocr_similarity',0):.2f} "
                      f"vis_sim={sig.get('vision_similarity',0):.2f} "
                      f"uia_same={sig.get('uia_same_parent',0):.2f} "
                      f"pixel_sim={sig.get('pixel_pattern_similarity',0):.2f} "
                      f"rhythm={sig.get('gap_rhythm_regularity',0):.2f} "
                      f"[{evidence_tag}] "
                      f"ocr:{ev.get('ocr_above',0)}/{ev.get('ocr_below',0)} "
                      f"vis:{ev.get('vision_above',0)}/{ev.get('vision_below',0)}")
            if len(decisions) > 8:
                print(f"    ... and {len(decisions) - 8} more")
    else:
        u4_rej = metrics.get("u4_rejected_reasons", {})
        print(f"  U4 layout: NOT AVAILABLE ({u4_rej})")

    # U4.1 Phase 2: merge details
    u4_data = overlay.get("ocr_visual_comparison", {}).get("u4_layout", {})
    merge_details = u4_data.get("merge_details", [])
    gap_classes = u4_data.get("gap_classifications", {})
    merged_regions = u4_data.get("merged_regions", [])
    if merge_details:
        print(f"  U4.1 merge: {len(merge_details)} merges, {len(merged_regions)} merged regions")
        for md in merge_details:
            print(f"    {md['merged_region_id']}: {md['original_count']} regions → {md['flow_type']} bounds={md['merged_bounds']}")
            print(f"      from: {', '.join(md['original_regions'])}")
    if gap_classes:
        print(f"  U4.1 gap classes: {gap_classes}")

    # U4.2 Phase 1: region signal report
    region_signals = metrics.get("u4_region_signals", [])
    if region_signals:
        print(f"  U4.2 signals: {len(region_signals)} regions")
        print(f"    {'RID':5s} {'area':>6s} {'ocr':>3s} {'vis':>3s} {'txt_dens':>8s} {'ctl_dens':>8s} {'h_ratio':>7s} {'w_ratio':>7s} {'ocr_rows':>8s} {'row_cv':>6s} {'x_align':>7s} {'pix_std':>7s}")
        for rs in region_signals:
            sig = rs.get("signals", {})
            print(f"    {rs['region_id']:5s} {rs['area_ratio']:6.3f} {rs['ocr_count']:3d} {rs['vision_count']:3d} "
                  f"{sig.get('text_density',0):8.5f} {sig.get('control_density',0):8.5f} "
                  f"{sig.get('height_ratio',0):7.3f} {sig.get('width_ratio',0):7.3f} "
                  f"{sig.get('ocr_row_count',0):8d} {sig.get('row_height_cv',0):6.3f} "
                  f"{sig.get('x_alignment',0):7.3f} {sig.get('pixel_uniformity',0):7.1f}")

    # U4.2 Phase A: substructure statistics
    u4_data = overlay.get("ocr_visual_comparison", {}).get("u4_layout", {})
    shadows = u4_data.get("page_region_model_shadow", [])
    if shadows:
        # Count substructure types
        type_counts: dict[str, int] = {}
        total_subs = 0
        for shadow in shadows:
            for sub in shadow.get("substructures", []):
                st = sub.get("sub_type", "unknown")
                type_counts[st] = type_counts.get(st, 0) + 1
                total_subs += 1
        print(f"  U4.2 substructures: {total_subs} total across {len(shadows)} regions")
        for st, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {st:20s}: {count}")

    # U4.2 Phase B.1: pairwise flow signals
    pairwise = metrics.get("u4_pairwise_flow_signals", [])
    if pairwise:
        def _fmt(v, width=5, prec=2):
            """Format value, handling None."""
            if v is None:
                return "NULL".rjust(width)
            return f"{v:{width}.{prec}f}"

        print(f"  U4.2 pairwise: {len(pairwise)} pairs")
        print(f"    {'Pair':12s} {'dir':5s} {'v_gap':>5s} {'h_ovlp':>6s} {'h_sim':>5s} {'sub_sim':>7s} {'txt_sim':>7s} {'ico_sim':>7s} {'x_aln':>5s} {'col_aln':>7s} {'edge':>4s} {'ico_r':>5s} {'inp':>3s} {'rel':5s} {'dom_a':16s} {'dom_b':16s}")
        for p in pairwise[:12]:
            s = p.get("signals", {})
            rel = p.get("signal_reliability", "?")
            print(f"    {p['pair_id']:12s} {p['direction']:5s} "
                  f"{s.get('vertical_gap_px',0) or 0:5d} "
                  f"{s.get('horizontal_overlap_ratio',0) or 0:6.2f} "
                  f"{_fmt(s.get('row_height_similarity'), 5)} "
                  f"{_fmt(s.get('dominant_subtype_similarity'), 7)} "
                  f"{_fmt(s.get('text_density_similarity'), 7)} "
                  f"{_fmt(s.get('icon_density_similarity'), 7)} "
                  f"{_fmt(s.get('x_alignment_similarity'), 5)} "
                  f"{_fmt(s.get('column_alignment_similarity'), 7)} "
                  f"{'Y' if s.get('edge_region_flag') else 'N':>4s} "
                  f"{s.get('icon_sub_ratio',0) or 0:5.2f} "
                  f"{'Y' if s.get('input_like_flag') else 'N':>3s} "
                  f"{rel:5s} "
                  f"{(p.get('region_a_dominant_sub') or 'None'):16s} "
                  f"{(p.get('region_b_dominant_sub') or 'None'):16s}")
        if len(pairwise) > 12:
            print(f"    ... and {len(pairwise) - 12} more")

    # Save overlay image
    if zone_page.screenshot:
        img_dir = REPORT_DIR / "images"
        img_dir.mkdir(exist_ok=True)

        # Raw screenshot
        zone_page.screenshot.save(img_dir / f"{name}_raw.png")

        # Original overlay image
        overlay_img = draw_overlay(
            zone_page.screenshot,
            overlay,
            geometric_regions=geometric,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
        )
        overlay_img.save(img_dir / f"{name}_overlay.png")
        print(f"  Saved: {img_dir / f'{name}_overlay.png'}")

        # Merged overlay image (U4.1 Phase 2)
        if merged_regions:
            merged_img = draw_overlay(
                zone_page.screenshot,
                overlay,
                geometric_regions=geometric,
                ocr_blocks=ocr_blocks,
                vision_candidates=vision_candidates,
                u4_override_regions=merged_regions,
            )
            merged_img.save(img_dir / f"{name}_merged_overlay.png")
            print(f"  Saved: {img_dir / f'{name}_merged_overlay.png'}")

        # Substructure overlay image (U4.2 Phase A)
        substructure_img = draw_substructure_overlay(
            zone_page.screenshot, overlay,
        )
        substructure_img.save(img_dir / f"{name}_substructure_overlay.png")
        print(f"  Saved: {img_dir / f'{name}_substructure_overlay.png'}")

        # URE P0 overlay
        ure_img = draw_ure_overlay(zone_page.screenshot, overlay)
        ure_img.save(img_dir / f"{name}_ure_overlay.png")
        print(f"  Saved: {img_dir / f'{name}_ure_overlay.png'}")

        # VTL-0 overlay
        vtl_img = draw_vtl_overlay(zone_page.screenshot, overlay)
        vtl_img.save(img_dir / f"{name}_vtl_overlay.png")
        print(f"  Saved: {img_dir / f'{name}_vtl_overlay.png'}")

    # URE P0 statistics
    ure_data = overlay.get("ocr_visual_comparison", {}).get("universal_region_shadow", {})
    if ure_data:
        prs = ure_data.get("page_regions", [])
        excluded = ure_data.get("excluded_reason", None)
        if excluded:
            print(f"  URE P0: EXCLUDED ({excluded})")
        else:
            # Count visible regions (area_ratio > 0.001, i.e., not zero-size)
            visible_prs = [pr for pr in prs if pr.get("area_ratio", 0) > 0.001]
            print(f"  URE P0: {len(prs)} page_regions ({len(visible_prs)} visible in overlay)")
            for pr in prs[:5]:
                subs = pr.get("substructures", [])
                ev = pr.get("evidence_sources", [])
                skel = pr.get("skeleton_source", "?")
                print(f"    {pr['region_id']}: {skel} bounds={pr['bounds']} "
                      f"sub={len(subs)} ev={'+'.join(ev) if ev else 'none'} "
                      f"area={pr.get('area_ratio', 0):.3f}")
            if len(prs) > 5:
                print(f"    ... and {len(prs) - 5} more")
    else:
        print(f"  URE P0: not available (OPENCLAW_URE_P0_SHADOW not set)")

    # VTL-0 statistics
    vtl_data = overlay.get("ocr_visual_comparison", {}).get("visual_trace_shadow", {})
    if vtl_data:
        containers = vtl_data.get("page_containers", [])
        container_cands = vtl_data.get("container_candidates", [])
        objects = vtl_data.get("local_objects", [])
        text_subs = vtl_data.get("text_substructures", [])
        rejected = vtl_data.get("rejected_text_like", [])
        excluded = vtl_data.get("excluded_reason", None)
        if excluded:
            print(f"  VTL-0: EXCLUDED ({excluded})")
        else:
            # Count by object_class
            class_counts = {}
            for obj in objects:
                oc = obj.get("object_class", "?")
                class_counts[oc] = class_counts.get(oc, 0) + 1
            print(f"  VTL-0: {len(containers)} containers ({len(container_cands)} cands), "
                  f"{len(objects)} objects, {len(text_subs)} text_subs, {len(rejected)} rejected_text")
            if class_counts:
                print(f"    object_class: {class_counts}")
            # Show first few containers
            for pc in containers[:3]:
                print(f"    {pc['trace_id']}: {pc['detection_method']} bounds={pc['bounds']} conf={pc['confidence']:.2f}")
            if len(containers) > 3:
                print(f"    ... and {len(containers) - 3} more")
    else:
        print(f"  VTL-0: not available (OPENCLAW_VTL_0_SHADOW not set)")

    # Save overlay JSON
    json_dir = REPORT_DIR / "json"
    json_dir.mkdir(exist_ok=True)
    with open(json_dir / f"{name}_overlay.json", "w", encoding="utf-8") as f:
        json.dump(overlay, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    """Run QA on all available apps."""
    from src.windows.window_enum import WindowEnumService

    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)

    # Target apps
    targets = [
        ("Paint", "mspaint.exe"),
        ("Notepad", "notepad.exe"),
        ("Chrome", "chrome.exe"),
        ("QQ", "qq.exe"),
        ("WeChat", "weixin.exe"),
        ("NetEase", "cloudmusic.exe"),
        ("VSCode", "code.exe"),
        ("FlClash", "flclash.exe"),
    ]

    results = []
    for name, process in targets:
        win = next(
            (w for w in windows if process.lower() in (w.process_name or "").lower()),
            None,
        )
        if not win:
            print(f"\n{name}: SKIP (window not found)")
            continue
        if hasattr(win, "is_minimized") and win.is_minimized:
            print(f"\n{name}: SKIP (minimized)")
            continue

        try:
            metrics = run_app(name, win.hwnd, process)
            results.append(metrics)
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results.append({"app": name, "error": str(e)})

    # Summary report
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    excluded_apps = [r["app"] for r in results if r.get("excluded")]
    active_results = [r for r in results if not r.get("excluded")]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "phase": "U1-shadow",
        "apps": results,
        "excluded_apps": excluded_apps,
    }

    total_regions = sum(r.get("total_regions", 0) for r in active_results)
    total_unknown = sum(r.get("unknown_count", 0) for r in active_results)
    total_groups = sum(r.get("total_groups", 0) for r in active_results)
    total_group_unknown = sum(r.get("group_unknown_count", 0) for r in active_results)
    total_group_roi = sum(r.get("group_roi_eligible_count", 0) for r in active_results)
    total_band_proposals = sum(r.get("total_band_proposals", 0) for r in active_results)
    total_band_tp = sum(r.get("band_tp", 0) for r in active_results)
    total_band_fp = sum(r.get("band_fp", 0) for r in active_results)
    total_band_amb = sum(r.get("band_amb", 0) for r in active_results)

    summary["total_regions"] = total_regions
    summary["unknown_count"] = total_unknown
    summary["total_groups"] = total_groups
    summary["group_unknown_count"] = total_group_unknown
    summary["group_roi_eligible_count"] = total_group_roi
    summary["total_band_proposals"] = total_band_proposals
    summary["band_tp"] = total_band_tp
    summary["band_fp"] = total_band_fp
    summary["useful_injected_band_count"] = sum(r.get("useful_injected_band_count", 0) for r in active_results)
    summary["noisy_injected_band_count"] = sum(r.get("noisy_injected_band_count", 0) for r in active_results)
    summary["band_amb"] = total_band_amb

    # Shadow comparison aggregates
    total_injected = sum(r.get("injected_band_count", 0) for r in active_results)
    total_combined_regions = sum(r.get("combined_region_count", 0) for r in active_results)
    total_combined_unknown = sum(r.get("combined_unknown_count", 0) for r in active_results)
    total_combined_roi = sum(r.get("combined_roi_eligible_count", 0) for r in active_results)

    summary["injected_band_count"] = total_injected
    summary["combined_region_count"] = total_combined_regions
    summary["combined_unknown_count"] = total_combined_unknown
    summary["combined_roi_eligible_count"] = total_combined_roi

    # U4 layout aggregates
    u4_available_count = sum(1 for r in active_results if r.get("u4_available"))
    total_u4_regions = sum(r.get("u4_region_count", 0) for r in active_results)
    total_u4_h_seps = sum(r.get("u4_h_sep_count", 0) for r in active_results)
    total_u4_v_seps = sum(r.get("u4_v_sep_count", 0) for r in active_results)
    summary["u4_available_count"] = u4_available_count
    summary["u4_total_regions"] = total_u4_regions
    summary["u4_total_h_separators"] = total_u4_h_seps
    summary["u4_total_v_separators"] = total_u4_v_seps
    summary["phase"] = "U4.2-B.1.2-signal-discrimination"

    print(f"  Apps tested: {len(results)} ({len(active_results)} active, {len(excluded_apps)} excluded)")
    if excluded_apps:
        print(f"  Excluded: {', '.join(excluded_apps)}")
    print(f"  Original: {total_regions} regions ({total_unknown} unknown, {total_group_roi} roi)")
    print(f"  Combined: {total_combined_regions} regions ({total_combined_unknown} unknown, {total_combined_roi} roi) [+{total_injected} bands injected]")
    total_useful = sum(r.get("useful_injected_band_count", 0) for r in active_results)
    total_noisy = sum(r.get("noisy_injected_band_count", 0) for r in active_results)
    print(f"  Bands:    {total_band_proposals} (TP={total_band_tp}, useful={total_useful}, noisy={total_noisy})")

    print(f"\n  {'App':12s} {'Status':>8s} {'Orig':>5s} {'Comb':>5s} {'Inj':>3s} {'Useful':>6s} {'Noisy':>5s}  Grouped types")
    print(f"  {'-'*80}")
    for r in results:
        app = r.get("app", "?")
        excluded = r.get("excluded", False)
        status = "EXCLUDED" if excluded else "active"
        total = r.get("total_regions", 0)
        combined = r.get("combined_region_count", 0)
        injected = r.get("injected_band_count", 0)
        useful = r.get("useful_injected_band_count", 0)
        noisy = r.get("noisy_injected_band_count", 0)
        grp_types = r.get("by_group_type", {})
        type_str = ", ".join(f"{k}:{v}" for k, v in sorted(grp_types.items(), key=lambda x: -x[1])[:3])
        print(f"  {app:12s} {status:>8s} {total:5d} {combined:5d} {injected:3d} {useful:6d} {noisy:5d}  {type_str}")

    # Save summary
    with open(REPORT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    main()
