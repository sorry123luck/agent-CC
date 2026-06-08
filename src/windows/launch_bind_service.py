"""Launch an app and bind the resulting top-level window."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from src.windows.capture_diagnostics import window_capture_diagnostics
from src.windows.window_resolver import WindowCandidate, WindowSource, get_window_resolver


class LaunchBindService:
    """Product-level entry helper: discovery/launch/window bind/capture evidence."""

    def __init__(
        self,
        *,
        resolver=None,
        popen: Callable[..., subprocess.Popen] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        capture_diagnostics: Callable[..., dict[str, Any]] = window_capture_diagnostics,
    ) -> None:
        self._resolver = resolver or get_window_resolver()
        self._popen = popen or subprocess.Popen
        self._sleep = sleep
        self._capture_diagnostics = capture_diagnostics

    def launch_and_bind(
        self,
        *,
        app_id: str | None = None,
        exe_path: str | None = None,
        args: list[str] | None = None,
        timeout_seconds: float = 10.0,
        allow_existing: bool = False,
    ) -> dict[str, Any]:
        app_id = str(app_id or Path(exe_path or "").stem or "").strip()
        args = list(args or [])

        if allow_existing and app_id:
            existing = self._resolver.resolve(app_id, prefer_session=True, refresh=True)
            if existing.candidates:
                return self._bound_response(
                    app_id=app_id,
                    exe_path=exe_path,
                    launch={"attempted": False, "pid": None, "message": "bound_existing_window"},
                    candidate=existing.candidates[0],
                    candidate_windows=existing.candidates,
                )

        resolved_exe = exe_path
        launch_args = args
        if not resolved_exe and app_id:
            resolved = self._resolver.resolve(app_id, prefer_session=True, refresh=True)
            if resolved.launch_targets:
                target = resolved.launch_targets[0]
                resolved_exe = target.executable
                launch_args = target.args or launch_args

        if not resolved_exe:
            return self._failed(
                app_id=app_id,
                exe_path=exe_path,
                stage="discovery",
                failure_reason="path_not_found",
                message="No executable path or launch target was found.",
            )

        launchable = Path(resolved_exe).is_file() or self._which(resolved_exe) is not None
        if not launchable:
            return self._failed(
                app_id=app_id,
                exe_path=resolved_exe,
                stage="discovery",
                failure_reason="path_not_found",
                message=f"Executable does not exist: {resolved_exe}",
            )

        try:
            command = [resolved_exe, *launch_args]
            proc = self._popen(command)
            pid = int(getattr(proc, "pid", 0) or 0)
            if pid:
                self._resolver.register_session_pid(pid)
        except Exception as exc:
            return self._failed(
                app_id=app_id,
                exe_path=resolved_exe,
                stage="launch",
                failure_reason="launch_failed",
                message=str(exc),
            )

        deadline = time.monotonic() + max(0.01, float(timeout_seconds))
        last_candidates: list[WindowCandidate] = []
        while time.monotonic() <= deadline:
            resolved = self._resolver.resolve(app_id, prefer_session=True, refresh=True)
            last_candidates = list(resolved.candidates)
            session_candidate = self._pick_session_candidate(last_candidates, pid)
            if session_candidate is not None:
                return self._bound_response(
                    app_id=app_id,
                    exe_path=resolved_exe,
                    launch={"attempted": True, "pid": pid, "message": "started"},
                    candidate=session_candidate,
                    candidate_windows=last_candidates,
                )
            foreground_candidate = self._pick_single_foreground_candidate(last_candidates)
            if foreground_candidate is not None:
                return self._bound_response(
                    app_id=app_id,
                    exe_path=resolved_exe,
                    launch={"attempted": True, "pid": pid, "message": "started_single_instance_foreground_bound"},
                    candidate=foreground_candidate,
                    candidate_windows=last_candidates,
                )
            self._sleep(0.25)

        if len(last_candidates) > 1:
            reason = "multiple_candidate_windows"
        elif len(last_candidates) == 1:
            reason = "binding_old_window"
        else:
            reason = "window_not_appeared"

        return self._failed(
            app_id=app_id,
            exe_path=resolved_exe,
            stage="bind",
            failure_reason=reason,
            message="Launched process did not produce a uniquely bindable session window.",
            launch={"attempted": True, "pid": pid, "message": "started"},
            candidate_windows=last_candidates,
        )

    def _pick_session_candidate(self, candidates: list[WindowCandidate], pid: int) -> WindowCandidate | None:
        for candidate in candidates:
            if candidate.source == WindowSource.CURRENT_SESSION or candidate.process_id == pid:
                return candidate
        return None

    def _pick_single_foreground_candidate(self, candidates: list[WindowCandidate]) -> WindowCandidate | None:
        foreground = [candidate for candidate in candidates if candidate.is_foreground]
        if len(foreground) == 1:
            return foreground[0]
        return None

    def _bound_response(
        self,
        *,
        app_id: str,
        exe_path: str | None,
        launch: dict[str, Any],
        candidate: WindowCandidate,
        candidate_windows: list[WindowCandidate],
    ) -> dict[str, Any]:
        diagnostics = self._capture_diagnostics(candidate.hwnd)
        screenshot_size = diagnostics.get("screenshot_size")
        return {
            "status": "bound",
            "stage": "bind",
            "failure_reason": None,
            "app_id": app_id,
            "exe_path": exe_path,
            "launch": launch,
            "bound_window": self._candidate_dict(candidate),
            "candidate_windows": [self._candidate_dict(item) for item in candidate_windows],
            "window_rect": list(candidate.rect) if candidate.rect else diagnostics.get("window_bounds"),
            "dpi_scale": diagnostics.get("dpi_scale", 1.0),
            "capture_diagnostics": diagnostics,
            "screenshot_evidence": {
                "captured": bool(screenshot_size),
                "size": screenshot_size,
            },
        }

    def _failed(
        self,
        *,
        app_id: str,
        exe_path: str | None,
        stage: str,
        failure_reason: str,
        message: str,
        launch: dict[str, Any] | None = None,
        candidate_windows: list[WindowCandidate] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "stage": stage,
            "failure_reason": failure_reason,
            "app_id": app_id,
            "exe_path": exe_path,
            "message": message,
            "launch": launch or {"attempted": False, "pid": None, "message": ""},
            "bound_window": None,
            "candidate_windows": [self._candidate_dict(item) for item in candidate_windows or []],
            "window_rect": None,
            "dpi_scale": None,
            "capture_diagnostics": {},
            "screenshot_evidence": {"captured": False, "size": None},
        }

    def _candidate_dict(self, candidate: WindowCandidate) -> dict[str, Any]:
        return {
            "hwnd": candidate.hwnd,
            "title": candidate.title,
            "class_name": candidate.class_name,
            "process_name": candidate.process_name,
            "process_id": candidate.process_id,
            "rect": list(candidate.rect) if candidate.rect else None,
            "is_visible": candidate.is_visible,
            "is_enabled": candidate.is_enabled,
            "state": candidate.state,
            "source": candidate.source.value if hasattr(candidate.source, "value") else str(candidate.source),
            "is_foreground": candidate.is_foreground,
            "confidence": candidate.confidence,
        }

    def _which(self, exe_path: str) -> str | None:
        if os.path.dirname(exe_path):
            return None
        import shutil

        return shutil.which(exe_path)
