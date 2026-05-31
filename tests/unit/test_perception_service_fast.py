"""Fast-path perception snapshot tests."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.perception.perception_service import PerceptionService, ZonePageStructure
from src.windows.window_enum import WindowInfoExt


def _structured_fast_image() -> Image.Image:
    arr = np.full((600, 800, 3), (245, 245, 245), dtype=np.uint8)
    arr[:, :90, :] = (218, 218, 218)
    arr[:, 90:93, :] = (100, 100, 100)
    arr[:, 250:253, :] = (115, 115, 115)
    arr[72:75, :, :] = (125, 125, 125)
    arr[500:503, 250:, :] = (130, 130, 130)
    return Image.fromarray(arr)


def test_fast_snapshot_runs_lightweight_geometric_diagnostics():
    zone_page = ZonePageStructure(
        window_info=WindowInfoExt(
            hwnd=123,
            title="Fast Test",
            process_name="chatgpt.exe",
            rect=(0, 0, 800, 600),
        ),
        screenshot=_structured_fast_image(),
        screenshot_size=(800, 600),
        fast_mode=True,
    )

    snapshot = PerceptionService().create_page_snapshot(zone_page, process_name="chatgpt.exe")

    assert snapshot.artifacts["visual_pattern"]["mode"] == "chat_document"
    assert snapshot.artifacts["roi_selection_plan"]["rois"]
    assert len(snapshot.artifacts["geometric_regions"]) >= 3
    assert snapshot.artifacts["partition_diagnostics"]["fast_mode"] is True
    assert snapshot.provider_trace.provider_details["geometric_partitioner"]["mode"] == "fast_light"


def test_full_snapshot_bounds_large_geometric_diagnostics():
    zone_page = ZonePageStructure(
        window_info=WindowInfoExt(
            hwnd=124,
            title="Dense Console",
            process_name="voicemeeter8x64.exe",
            rect=(0, 0, 1600, 900),
        ),
        screenshot=_structured_fast_image().resize((1600, 900)),
        screenshot_size=(1600, 900),
        fast_mode=False,
    )

    snapshot = PerceptionService().create_page_snapshot(
        zone_page,
        process_name="voicemeeter8x64.exe",
    )

    geo = snapshot.provider_trace.provider_details["geometric_partitioner"]
    assert geo["mode"] == "bounded_full"
    assert snapshot.artifacts["partition_diagnostics"]["bounded_full"] is True
    assert snapshot.artifacts["partition_diagnostics"]["scale"] < 1.0
    assert snapshot.artifacts["geometric_regions"]
