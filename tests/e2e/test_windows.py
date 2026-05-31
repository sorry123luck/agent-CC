"""E2E tests for GET /api/v1/windows.

Tests the window enumeration endpoint.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock


class TestListWindows:
    """GET /api/v1/windows"""

    def test_list_windows(self, client):
        """Returns window list from WindowEnumService."""
        from src.windows.window_enum import WindowState

        mock_windows = [
            MagicMock(
                hwnd=12345,
                title="Test Window",
                class_name="TestClass",
                process_name="test.exe",
                process_id=100,
                state=WindowState.NORMAL,
            ),
            MagicMock(
                hwnd=67890,
                title="Another Window",
                class_name="AnotherClass",
                process_name="another.exe",
                process_id=200,
                state=WindowState.NORMAL,
            ),
        ]

        mock_fg = MagicMock(hwnd=12345)

        with patch("src.windows.window_enum.WindowEnumService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.enumerate_all.return_value = mock_windows
            mock_instance.get_foreground_window.return_value = mock_fg

            resp = client.get("/api/v1/windows")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["hwnd"] == 12345
            assert data[0]["title"] == "Test Window"
            assert data[0]["is_foreground"] is True
            assert data[1]["hwnd"] == 67890
            assert data[1]["title"] == "Another Window"
            assert data[1]["is_foreground"] is False

    def test_list_windows_empty(self, client):
        """No windows returns empty list."""
        with patch("src.windows.window_enum.WindowEnumService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.enumerate_all.return_value = []
            mock_instance.get_foreground_window.return_value = None

            resp = client.get("/api/v1/windows")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_list_windows_filters_empty_titles(self, client):
        """Windows with empty titles are filtered out."""
        from src.windows.window_enum import WindowState

        mock_windows = [
            MagicMock(
                hwnd=12345,
                title="Valid Window",
                class_name="C",
                process_name="v.exe",
                process_id=1,
                state=WindowState.NORMAL,
            ),
            MagicMock(
                hwnd=67890,
                title="",
                class_name="C2",
                process_name="e.exe",
                process_id=2,
                state=WindowState.NORMAL,
            ),
        ]

        mock_fg = MagicMock(hwnd=12345)

        with patch("src.windows.window_enum.WindowEnumService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.enumerate_all.return_value = mock_windows
            mock_instance.get_foreground_window.return_value = mock_fg

            resp = client.get("/api/v1/windows")
            assert resp.status_code == 200
            data = resp.json()
            # Only the window with a non-empty title should be returned
            assert len(data) == 1
            assert data[0]["title"] == "Valid Window"
