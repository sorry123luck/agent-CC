from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE_MODEL_BROWSER = ROOT / "src/ui/console/src/components/PageModelBrowser.tsx"


def test_model_tree_labels_unknown_app_group_as_unidentified_diagnostic() -> None:
    source = PAGE_MODEL_BROWSER.read_text(encoding="utf-8")

    assert "未识别应用" in source
    assert "app_id=unknown" in source
    assert "formatAppGroupLabel" in source
    assert "formatAppGroupTitle" in source
