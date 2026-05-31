from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX_TSX = ROOT / "src/ui/console/src/pages/Index.tsx"
VIRTUAL_MODEL_RENDERER = ROOT / "src/ui/console/src/components/VirtualModelRenderer.tsx"
PAGE_MODEL_BROWSER = ROOT / "src/ui/console/src/components/PageModelBrowser.tsx"
TOP_BAR = ROOT / "src/ui/console/src/components/TopBar.tsx"
MODEL_CANDIDATE_PANEL = ROOT / "src/ui/console/src/components/ModelCandidatePanel.tsx"


def test_model_view_restores_source_canvas_navigation_contract() -> None:
    source = INDEX_TSX.read_text(encoding="utf-8")

    assert "latestModelSnapshotId" in source
    assert "modelSourceCanvasId" in source
    assert "handleBackToScreenshot" in source
    assert "返回截图" in source
    assert "onBack={selectedStateTemplateId ? handleBackHome : handleBack}" in source
    assert "onClick={handleBackToScreenshot}" in source
    assert "<span>返回截图</span>" in source


def test_model_view_passes_source_canvas_to_candidate_list_and_renderer() -> None:
    source = INDEX_TSX.read_text(encoding="utf-8")

    assert "sourceCanvasId={modelSourceCanvasId}" in source
    assert "getModelAspectRatio(effectiveCanvasDetail ?? modelSourceCanvasDetail ?? undefined)" in source


def test_virtual_model_does_not_show_bottom_material_area() -> None:
    source = INDEX_TSX.read_text(encoding="utf-8")

    assert "showMaterialArea={false}" in source


def test_virtual_model_renderer_still_maps_stable_keys_to_canvas_crops() -> None:
    source = VIRTUAL_MODEL_RENDERER.read_text(encoding="utf-8")

    assert "stable_key_id" in source
    assert "getCandidateCropUrl(activeCanvasId, elementId)" in source
    assert "getCanvasScreenshotUrl(sourceCanvasId, false)" in source
    assert "ScreenshotBackedRegion" in source
    assert "fixedCandidates.map" in source
    assert "<CandidateMaterialArea" in source


def test_model_candidate_panel_has_fixed_control_mapping_action() -> None:
    source = MODEL_CANDIDATE_PANEL.read_text(encoding="utf-8")

    assert "固定控件并显示截图映射" in source
    assert "kind: 'fixed'" in source
    assert "visualType: safeCandidate.visual_type || 'button'" in source
    assert "semanticRole: safeCandidate.canonical_role || 'button'" in source
    assert "固定为消息输入框" not in source


def test_home_workbench_keeps_resizable_split_and_collapsed_model_tree() -> None:
    index_source = INDEX_TSX.read_text(encoding="utf-8")
    browser_source = PAGE_MODEL_BROWSER.read_text(encoding="utf-8")

    assert "leftTopRatio" in index_source
    assert "handleLeftSplitMouseDown" in index_source
    assert "cursor-row-resize" in index_source
    assert "useState<Set<string>>(new Set())" in browser_source
    assert "Trash2" in browser_source


def test_canvas_delete_notifies_home_selection_state() -> None:
    index_source = INDEX_TSX.read_text(encoding="utf-8")
    browser_source = (ROOT / "src/ui/console/src/components/CanvasBrowser.tsx").read_text(encoding="utf-8")

    assert "handleDeletedCanvas" in index_source
    assert "onDeletedCanvas={handleDeletedCanvas}" in index_source
    assert "onDeletedCanvas?: (canvasId: string) => void" in browser_source
    assert "onDeletedCanvas?.(c.canvas_id)" in browser_source


def test_model_tree_delete_notifies_home_selection_state() -> None:
    index_source = INDEX_TSX.read_text(encoding="utf-8")
    browser_source = PAGE_MODEL_BROWSER.read_text(encoding="utf-8")

    assert "handleDeletedStateTemplate" in index_source
    assert "handleDeletedPageModel" in index_source
    assert "onDeletedStateTemplate={handleDeletedStateTemplate}" in index_source
    assert "onDeletedPageModel={handleDeletedPageModel}" in index_source
    assert "onDeletedStateTemplate?: (stateTemplateId: string) => void" in browser_source
    assert "onDeletedPageModel?: (pageModelId: string) => void" in browser_source
    assert "onDeletedStateTemplate?.(st.state_template_id)" in browser_source
    assert "onDeletedPageModel?.(pm.page_model_id)" in browser_source


def test_top_bar_uses_explicit_virtual_model_back_labels() -> None:
    source = TOP_BAR.read_text(encoding="utf-8")

    assert "返回主页" in source
    assert "hasSourceCanvas" not in source
    assert "isVirtualModelView ? '返回主页' : '返回'" in source


def test_roi_vlm_panel_surfaces_response_latency_and_status_counts() -> None:
    index_source = INDEX_TSX.read_text(encoding="utf-8")
    types_source = (ROOT / "src/ui/console/src/api/types.ts").read_text(encoding="utf-8")

    assert "elapsed_ms: number" in types_source
    assert "result_status_counts: Record<string, number>" in types_source
    assert "耗时：{lastRoiVlmResult.elapsed_ms}ms" in index_source
    assert "返回状态：{formatStatusCounts(lastRoiVlmResult.result_status_counts)}" in index_source


def test_roi_vlm_raw_supplements_do_not_override_backend_projected_labels() -> None:
    index_source = INDEX_TSX.read_text(encoding="utf-8")
    apply_block = index_source.split("function applyRoiVlmSupplementsToElements", 1)[1].split("export default function Index", 1)[0]

    assert "role_label: element.role_label" in apply_block
    assert "role_source: element.role_source" in apply_block
    assert "annotation.label || element.role_label" not in apply_block
    assert "role_source: 'roi_vlm'" not in apply_block


def test_observe_response_type_includes_elapsed_ms() -> None:
    types_source = (ROOT / "src/ui/console/src/api/types.ts").read_text(encoding="utf-8")

    observe_block = types_source.split("export interface ObserveResponse", 1)[1].split("export interface QueryTargetModel", 1)[0]
    assert "elapsed_ms: number" in observe_block


def test_canvas_detail_type_includes_local_evidence_artifacts() -> None:
    types_source = (ROOT / "src/ui/console/src/api/types.ts").read_text(encoding="utf-8")

    canvas_detail_block = types_source.split("export interface CanvasDetail", 1)[1].split("export interface RoiVlmCandidateAnnotation", 1)[0]
    assert "ocr_blocks: Record<string, unknown>[]" in canvas_detail_block
    assert "vision_candidates: Record<string, unknown>[]" in canvas_detail_block
