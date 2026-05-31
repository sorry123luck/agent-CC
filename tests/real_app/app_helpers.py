"""App helpers for real-app tests.

Separation of concerns:
- resolve_window: find existing windows (never launches)
- launch_app: explicitly launch an app (Agent/test decision)
- close_app_safely: close apps we launched

WeChat is NEVER auto-launched. It must already be running.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from src.windows.window_resolver import (
    LaunchTarget,
    ResolveResult,
    WindowCandidate,
    get_window_resolver,
)


def find_chrome() -> str | None:
    """Find chrome.exe on Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return shutil.which("chrome")


def is_chrome_available() -> bool:
    """Check if Chrome is installed."""
    return find_chrome() is not None


# ===== Session tracking =====

_LAUNCHED_PROCS: dict[str, subprocess.Popen] = {}
_LAUNCHED_TEMP_DIRS: dict[str, str] = {}
_LOCK = Lock()


def register_launch(app_name: str, proc: subprocess.Popen) -> None:
    """Register a launched process with the WindowResolver."""
    resolver = get_window_resolver()
    resolver.register_session_pid(proc.pid)
    with _LOCK:
        _LAUNCHED_PROCS[app_name] = proc


# ===== Resolve (never launches) =====

def resolve_window(app_name: str, *, prefer_session: bool = True) -> ResolveResult:
    """Find existing windows for an app. NEVER launches.

    Returns ResolveResult with:
    - found_window: True if any window found
    - candidates: ranked list of WindowCandidate
    - launch_targets: available launch paths (if no window found)
    - suggested_next_step: what Agent should do next
    """
    resolver = get_window_resolver()
    return resolver.resolve(app_name, prefer_session=prefer_session)


def get_hwnd(app_name: str) -> int | None:
    """Get best hwnd for an app. Returns None if no window found."""
    result = resolve_window(app_name)
    return result.candidates[0].hwnd if result.candidates else None


# ===== Launch (explicit Agent/test decision) =====

def launch_app(app_name: str) -> subprocess.Popen | None:
    """Explicitly launch an app. Returns Popen or None if launch failed.

    IMPORTANT: This should only be called when Agent/test has decided to launch.
    WeChat is NEVER launched by this function — it must already be running.
    """
    if app_name == "wechat":
        # WeChat must already be running — never auto-launch
        return None

    proc = _do_launch(app_name)
    if proc is not None:
        register_launch(app_name, proc)
        wait_time = 3 if app_name in ("chrome", "vscode") else 1
        time.sleep(wait_time)
    return proc


def _do_launch(app_name: str) -> subprocess.Popen | None:
    """Internal: actually launch the app process."""
    if app_name == "notepad":
        return subprocess.Popen(["notepad.exe"])

    if app_name == "chrome":
        chrome_path = find_chrome()
        if chrome_path is None:
            return None
        temp_dir = tempfile.mkdtemp(prefix="oc_chrome_test_")
        proc = subprocess.Popen([
            chrome_path,
            f"--user-data-dir={temp_dir}",
            "--no-first-run",
            "--disable-default-apps",
            "about:blank",
        ])
        with _LOCK:
            _LAUNCHED_TEMP_DIRS[app_name] = temp_dir
        return proc

    if app_name == "vscode":
        temp_dir = tempfile.mkdtemp(prefix="oc_vscode_test_")
        proc = subprocess.Popen(["code.exe", f"--user-data-dir={temp_dir}", temp_dir])
        with _LOCK:
            _LAUNCHED_TEMP_DIRS[app_name] = temp_dir
        return proc

    raise ValueError(f"Unknown app: {app_name}")


# ===== Close =====

def close_app_safely(app_name: str) -> None:
    """Close an app that was launched by launch_app.

    Only closes processes we launched. Never closes WeChat.
    """
    with _LOCK:
        proc = _LAUNCHED_PROCS.pop(app_name, None)

    if proc is None:
        return

    if app_name == "wechat":
        return  # Never close WeChat

    resolver = get_window_resolver()

    try:
        proc.terminate()
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    resolver.unregister_session_pid(proc.pid)

    # Clean up temp directory
    with _LOCK:
        temp_dir = _LAUNCHED_TEMP_DIRS.pop(app_name, None)
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
