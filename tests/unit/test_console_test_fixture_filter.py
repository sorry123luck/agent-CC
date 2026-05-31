from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANVAS_BROWSER = ROOT / "src" / "ui" / "console" / "src" / "components" / "CanvasBrowser.tsx"


def test_canvas_browser_filters_test_fixture_canvases_from_workbench():
    source = CANVAS_BROWSER.read_text(encoding="utf-8")

    assert "isTestFixtureCanvas" in source
    assert "test_app" in source
    assert "test_app.exe" in source
    assert "test_app Window" in source
    assert ".filter((c) => !isTestFixtureCanvas(c))" in source
