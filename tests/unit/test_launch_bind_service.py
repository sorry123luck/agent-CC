from __future__ import annotations

from unittest.mock import MagicMock

from src.windows.launch_bind_service import LaunchBindService
from src.windows.window_resolver import LaunchTarget, ResolveResult, WindowCandidate, WindowSource


class _FakeResolver:
    def __init__(self, results: list[ResolveResult]) -> None:
        self.results = list(results)
        self.last = self.results[-1]
        self.registered: list[int] = []

    def resolve(self, app_name: str, *, prefer_session: bool = True, refresh: bool = True):
        if self.results:
            self.last = self.results.pop(0)
        return self.last

    def register_session_pid(self, pid: int) -> None:
        self.registered.append(pid)


def _candidate(
    *,
    source: WindowSource = WindowSource.CURRENT_SESSION,
    process_id: int = 777,
    is_foreground: bool = True,
) -> WindowCandidate:
    return WindowCandidate(
        hwnd=12345,
        title="Chrome",
        class_name="Chrome_WidgetWin_1",
        process_name="chrome.exe",
        process_id=process_id,
        rect=(10, 20, 810, 620),
        source=source,
        is_foreground=is_foreground,
        confidence=0.9,
    )


def test_launch_bind_launches_and_binds_session_window(tmp_path):
    exe = tmp_path / "chrome.exe"
    exe.write_bytes(b"fake exe")
    resolver = _FakeResolver(
        [
            ResolveResult(
                app_name="chrome",
                found_window=False,
                launch_targets=[
                    LaunchTarget(app_name="chrome", executable=str(exe), args=["--new-window"]),
                ],
            ),
            ResolveResult(app_name="chrome", found_window=True, candidates=[_candidate()]),
        ]
    )
    popen = MagicMock(return_value=MagicMock(pid=777))
    diagnostics = MagicMock(return_value={"hwnd": 12345, "screenshot_size": [800, 600], "valid": True})

    result = LaunchBindService(
        resolver=resolver,
        popen=popen,
        sleep=lambda _seconds: None,
        capture_diagnostics=diagnostics,
    ).launch_and_bind(app_id="chrome", timeout_seconds=0.01)

    assert result["status"] == "bound"
    assert result["stage"] == "bind"
    assert result["failure_reason"] is None
    assert result["launch"]["pid"] == 777
    assert result["bound_window"]["hwnd"] == 12345
    assert result["window_rect"] == [10, 20, 810, 620]
    assert result["capture_diagnostics"]["valid"] is True
    assert resolver.registered == [777]


def test_launch_bind_reports_missing_path(tmp_path):
    missing = tmp_path / "missing.exe"

    result = LaunchBindService().launch_and_bind(
        app_id="missing",
        exe_path=str(missing),
        timeout_seconds=0.01,
    )

    assert result["status"] == "failed"
    assert result["stage"] == "discovery"
    assert result["failure_reason"] == "path_not_found"


def test_launch_bind_reports_multiple_candidates_without_session_match(tmp_path):
    exe = tmp_path / "app.exe"
    exe.write_bytes(b"fake exe")
    old_window = _candidate(source=WindowSource.EXISTING, process_id=42, is_foreground=False)
    resolver = _FakeResolver(
        [
            ResolveResult(app_name="app", found_window=False, launch_targets=[]),
            ResolveResult(app_name="app", found_window=True, candidates=[old_window, old_window]),
        ]
    )
    popen = MagicMock(return_value=MagicMock(pid=777))

    result = LaunchBindService(
        resolver=resolver,
        popen=popen,
        sleep=lambda _seconds: None,
    ).launch_and_bind(app_id="app", exe_path=str(exe), timeout_seconds=0.01)

    assert result["status"] == "failed"
    assert result["stage"] == "bind"
    assert result["failure_reason"] == "multiple_candidate_windows"
    assert len(result["candidate_windows"]) == 2


def test_launch_bind_binds_single_instance_foreground_window(tmp_path):
    exe = tmp_path / "chrome.exe"
    exe.write_bytes(b"fake exe")
    old_background = _candidate(source=WindowSource.EXISTING, process_id=42, is_foreground=False)
    foreground = _candidate(source=WindowSource.EXISTING, process_id=43, is_foreground=True)
    resolver = _FakeResolver(
        [
            ResolveResult(app_name="chrome", found_window=False, launch_targets=[]),
            ResolveResult(app_name="chrome", found_window=True, candidates=[foreground, old_background]),
        ]
    )
    popen = MagicMock(return_value=MagicMock(pid=777))
    diagnostics = MagicMock(return_value={"hwnd": 12345, "screenshot_size": [800, 600], "valid": True})

    result = LaunchBindService(
        resolver=resolver,
        popen=popen,
        sleep=lambda _seconds: None,
        capture_diagnostics=diagnostics,
    ).launch_and_bind(app_id="chrome", exe_path=str(exe), timeout_seconds=0.01)

    assert result["status"] == "bound"
    assert result["bound_window"]["source"] == "existing"
    assert result["launch"]["message"] == "started_single_instance_foreground_bound"
