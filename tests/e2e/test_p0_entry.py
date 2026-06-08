from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_launch_bind_endpoint_returns_window_and_capture_diagnostics(client, tmp_path):
    exe = tmp_path / "app.exe"
    exe.write_bytes(b"fake exe")
    service_result = {
        "status": "bound",
        "stage": "bind",
        "failure_reason": None,
        "app_id": "test_app",
        "exe_path": str(exe),
        "launch": {"attempted": True, "pid": 123, "message": "started"},
        "bound_window": {
            "hwnd": 456,
            "title": "Test App",
            "process_name": "app.exe",
            "process_id": 123,
            "source": "current_session",
            "confidence": 0.9,
        },
        "candidate_windows": [],
        "window_rect": [10, 20, 810, 620],
        "dpi_scale": 1.25,
        "capture_diagnostics": {
            "hwnd": 456,
            "window_bounds": [10, 20, 810, 620],
            "capture_bounds": [0, 0, 800, 600],
            "screenshot_size": [800, 600],
            "dpi_scale": 1.25,
            "monitor": {"index": 0},
            "is_minimized": False,
            "is_visible": True,
            "is_occluded": False,
            "valid": True,
        },
        "screenshot_evidence": {"captured": True, "size": [800, 600]},
    }

    with patch("src.integration.api_server.LaunchBindService") as service_cls:
        service_cls.return_value.launch_and_bind.return_value = service_result
        resp = client.post(
            "/api/v1/apps/launch-bind",
            json={"app_id": "test_app", "exe_path": str(exe), "timeout_seconds": 0.1},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "bound"
    assert data["bound_window"]["hwnd"] == 456
    assert data["capture_diagnostics"]["screenshot_size"] == [800, 600]


def test_observe_response_includes_capture_diagnostics(client, mock_observe):
    _mock, latest = mock_observe
    canvas = latest()

    with patch("src.integration.api_server._canvas_capture_diagnostics") as diagnostics:
        diagnostics.return_value = {
            "hwnd": 12345,
            "window_bounds": [10, 20, 810, 620],
            "capture_bounds": [0, 0, 800, 600],
            "screenshot_size": [800, 600],
            "dpi_scale": 1.0,
            "monitor": {"index": 0},
            "is_minimized": False,
            "is_visible": True,
            "is_occluded": False,
            "valid": True,
        }
        resp = client.post("/api/v1/observe", json={"hwnd": 12345, "async_enhance": False})

    assert resp.status_code == 200
    data = resp.json()
    assert data["capture_diagnostics"]["hwnd"] == 12345
    assert data["capture_diagnostics"]["valid"] is True
