from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW_CANVAS = ROOT / "src" / "ui" / "console" / "src" / "components" / "ReviewCanvas.tsx"
INDEX_PAGE = ROOT / "src" / "ui" / "console" / "src" / "pages" / "Index.tsx"
TYPES = ROOT / "src" / "ui" / "console" / "src" / "api" / "types.ts"


def test_review_canvas_separates_roi_vlm_from_omni_candidates():
    source = REVIEW_CANVAS.read_text(encoding="utf-8")

    assert "ROI裁剪" in source
    assert "roi_selection_plan" in source
    assert "roi_vlm_semantic_supplements" in source
    assert "Omni候选" in source
    assert "全图 VLM" in source


def test_roi_layer_is_available_in_default_layers_and_types():
    index_source = INDEX_PAGE.read_text(encoding="utf-8")
    types_source = TYPES.read_text(encoding="utf-8")

    assert "roi: false" in index_source
    assert "vlm_full: false" in index_source
    assert "roi_selection_plan?: Record<string, unknown>" in types_source
    assert "roi_vlm_semantic_supplements: RoiVlmSupplement[]" in types_source
