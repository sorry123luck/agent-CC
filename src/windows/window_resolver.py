"""WindowResolver: find existing windows for an app, return candidates + launch info.

Core principle: WindowResolver NEVER launches apps. It only:
1. Finds existing windows
2. Returns ranked candidates with confidence
3. If no window found, returns launch_targets so Agent can decide

Agent flow:
  1. Agent calls resolve(app_name)
  2. If candidates found → Agent picks one → observe(hwnd)
  3. If no candidates → Agent sees launch_targets → Agent decides to launch
  4. After Agent launches → Agent calls resolve again → observe(hwnd)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from src.windows.window_enum import WindowEnumService, WindowInfoExt


class WindowSource(str, Enum):
    """Where this window came from."""
    CURRENT_SESSION = "current_session"  # launched by us in this process
    EXISTING = "existing"  # was already open


class WindowCandidate(BaseModel):
    """A candidate window with resolution metadata."""

    # From WindowInfoExt
    hwnd: int
    title: str
    class_name: str | None = None
    process_name: str | None = None
    process_id: int | None = None
    rect: tuple[int, int, int, int] | None = None
    is_visible: bool = True
    is_enabled: bool = True
    state: str = "normal"

    # Resolution metadata
    source: WindowSource = WindowSource.EXISTING
    is_foreground: bool = False
    confidence: float = 0.0  # 0.0-1.0, how likely this is the target

    model_config = {"from_attributes": True}


class LaunchTarget(BaseModel):
    """A possible launch path for an app. Agent decides whether to use it."""

    app_name: str
    executable: str  # e.g. "notepad.exe" or full path
    args: list[str] = []
    description: str = ""  # human-readable description


class ResolveResult(BaseModel):
    """Result of resolving an app name to windows.

    If candidates is non-empty: Agent picks one and observe(hwnd).
    If candidates is empty: Agent sees launch_targets and decides next step.
    """

    app_name: str
    found_window: bool
    candidates: list[WindowCandidate] = []
    launch_targets: list[LaunchTarget] = []
    suggested_next_step: str = ""

    model_config = {"from_attributes": True}


# App search definitions: title patterns, process names, launch info
_APP_SEARCH_DEFS: dict[str, dict] = {
    "notepad": {
        "title_patterns": ["*Notepad*"],
        "process_name": "notepad.exe",
        "launch_executable": "notepad.exe",
        "launch_args": [],
    },
    "chrome": {
        "title_patterns": ["*Chrome*"],
        "process_name": "chrome.exe",
        "launch_executable": None,  # need to find path
        "launch_args": ["--no-first-run", "--disable-default-apps"],
    },
    "wechat": {
        "title_patterns": ["*微信*", "*WeChat*"],
        "process_name": "Weixin.exe",
        "launch_executable": None,  # need to find path
        "launch_args": [],
    },
    "weixin": {
        "title_patterns": ["*微信*", "*WeChat*"],
        "process_name": "Weixin.exe",
        "launch_executable": None,
        "launch_args": [],
    },
    "qq": {
        "title_patterns": ["*QQ*"],
        "process_name": "QQ.exe",
        "launch_executable": None,
        "launch_args": [],
    },
    "netease_cloud_music": {
        "title_patterns": ["*网易云*", "*CloudMusic*", "*Netease*", "*音乐*"],
        "process_name": "cloudmusic.exe",
        "launch_executable": None,
        "launch_args": [],
    },
    "cloudmusic": {
        "title_patterns": ["*网易云*", "*CloudMusic*", "*Netease*", "*音乐*"],
        "process_name": "cloudmusic.exe",
        "launch_executable": None,
        "launch_args": [],
    },
    "voicemeeter": {
        "title_patterns": ["*VoiceMeeter*", "*Voicemeeter*"],
        "process_name": "voicemeeter8x64.exe",
        "launch_executable": None,
        "launch_args": [],
    },
    "vscode": {
        "title_patterns": ["*Visual Studio Code*"],
        "process_name": "Code.exe",
        "launch_executable": "code.exe",
        "launch_args": [],
    },
}

# Common install paths for apps where executable isn't in PATH
_KNOWN_INSTALL_PATHS: dict[str, list[str]] = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ],
    "wechat": [
        r"C:\Program Files\Tencent\Weixin\Weixin.exe",
        r"C:\Program Files\Tencent\WeChat\WeChat.exe",
        r"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe",
    ],
    "weixin": [
        r"C:\Program Files\Tencent\Weixin\Weixin.exe",
        r"C:\Program Files\Tencent\WeChat\WeChat.exe",
        r"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe",
    ],
    "qq": [
        r"C:\Program Files\Tencent\QQNT\QQ.exe",
        r"C:\Program Files (x86)\Tencent\QQ\Bin\QQ.exe",
    ],
    "voicemeeter": [
        r"C:\Program Files (x86)\VB\Voicemeeter\voicemeeter8x64.exe",
        r"C:\Program Files (x86)\VB\Voicemeeter\voicemeeter8.exe",
        r"C:\Program Files (x86)\VB\Voicemeeter\voicemeeterpro.exe",
        r"C:\Program Files (x86)\VB\Voicemeeter\voicemeeter.exe",
    ],
    "vscode": [
        r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
        r"C:\Program Files\Microsoft VS Code\Code.exe",
        r"C:\Program Files (x86)\Microsoft VS Code\Code.exe",
    ],
}


def _find_executable(app_name: str) -> str | None:
    """Find executable for an app. Returns None if not found."""
    search_def = _APP_SEARCH_DEFS.get(app_name)
    if search_def is None:
        return shutil.which(f"{app_name}.exe")

    # Try explicit launch_executable first
    launch_exe = search_def.get("launch_executable")
    if launch_exe:
        if Path(launch_exe).exists() or shutil.which(launch_exe):
            return launch_exe

    # Try known install paths
    for path in _KNOWN_INSTALL_PATHS.get(app_name, []):
        expanded = os.path.expandvars(path)
        if Path(expanded).exists():
            return expanded

    # Try process_name in PATH
    process_name = search_def.get("process_name")
    if process_name:
        return shutil.which(process_name)

    return None


class WindowResolver:
    """Find existing windows for an app. NEVER launches apps.

    Maintains a set of PIDs from the current session so tests
    can prefer their own windows.
    """

    def __init__(self) -> None:
        self._enum = WindowEnumService()
        self._session_pids: set[int] = set()
        self._lock = Lock()

    def register_session_pid(self, pid: int) -> None:
        """Register a PID as launched in this session."""
        with self._lock:
            self._session_pids.add(pid)

    def unregister_session_pid(self, pid: int) -> None:
        """Unregister a session PID."""
        with self._lock:
            self._session_pids.discard(pid)

    def resolve(
        self,
        app_name: str,
        *,
        prefer_session: bool = True,
        refresh: bool = True,
    ) -> ResolveResult:
        """Resolve an app name to windows + launch info.

        NEVER launches apps. If no window found, returns launch_targets
        so Agent can decide whether to launch.

        Args:
            app_name: app identifier (e.g. "notepad", "chrome")
            prefer_session: if True, windows from current session rank higher
            refresh: force refresh of window list

        Returns:
            ResolveResult with candidates (if any) and launch_targets (if no window).
        """
        if refresh:
            self._enum._refresh()

        search_def = _APP_SEARCH_DEFS.get(app_name)
        if search_def is None:
            search_def = {
                "title_patterns": [f"*{app_name}*"],
                "process_name": f"{app_name}.exe",
            }

        # Collect matching windows by title and process
        seen_hwnds: set[int] = set()
        raw_windows: list[WindowInfoExt] = []

        for pattern in search_def.get("title_patterns", []):
            for w in self._enum.find_by_title(pattern):
                if w.hwnd not in seen_hwnds:
                    seen_hwnds.add(w.hwnd)
                    raw_windows.append(w)

        process_name = search_def.get("process_name")
        if process_name:
            for w in self._enum.find_by_process(process_name):
                if w.hwnd not in seen_hwnds:
                    seen_hwnds.add(w.hwnd)
                    raw_windows.append(w)

        # Get foreground hwnd
        fg = self._enum.get_foreground_window()
        fg_hwnd = fg.hwnd if fg else 0

        # Build candidates with confidence scores
        candidates: list[WindowCandidate] = []
        with self._lock:
            session_pids = set(self._session_pids)

        for w in raw_windows:
            is_session = w.process_id in session_pids
            is_fg = w.hwnd == fg_hwnd
            confidence = self._compute_confidence(
                w, is_session, is_fg, prefer_session
            )

            candidates.append(WindowCandidate(
                hwnd=w.hwnd,
                title=w.title,
                class_name=w.class_name,
                process_name=w.process_name,
                process_id=w.process_id,
                rect=w.rect,
                is_visible=w.is_visible,
                is_enabled=w.is_enabled,
                state=w.state.value if hasattr(w.state, "value") else str(w.state),
                source=WindowSource.CURRENT_SESSION if is_session else WindowSource.EXISTING,
                is_foreground=is_fg,
                confidence=confidence,
            ))

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        if candidates:
            return ResolveResult(
                app_name=app_name,
                found_window=True,
                candidates=candidates,
            )

        # No window found — return launch_targets
        launch_targets = self._build_launch_targets(app_name)
        return ResolveResult(
            app_name=app_name,
            found_window=False,
            launch_targets=launch_targets,
            suggested_next_step="agent_should_launch_app_then_observe_again",
        )

    def resolve_one(
        self,
        app_name: str,
        *,
        prefer_session: bool = True,
        refresh: bool = True,
    ) -> WindowCandidate | None:
        """Resolve to the single best candidate, or None if no match."""
        result = self.resolve(app_name, prefer_session=prefer_session, refresh=refresh)
        return result.candidates[0] if result.candidates else None

    def _build_launch_targets(self, app_name: str) -> list[LaunchTarget]:
        """Build launch targets for an app. Only returns info, never launches."""
        exe = _find_executable(app_name)
        if exe is None:
            return []

        search_def = _APP_SEARCH_DEFS.get(app_name, {})
        args = search_def.get("launch_args", [])

        return [LaunchTarget(
            app_name=app_name,
            executable=exe,
            args=args,
            description=f"Launch {app_name} from {exe}",
        )]

    def _compute_confidence(
        self,
        w: WindowInfoExt,
        is_session: bool,
        is_foreground: bool,
        prefer_session: bool,
    ) -> float:
        """Compute confidence score for a window candidate.

        Scoring:
        - Base: 0.5 for being a match
        - Session window: +0.3 if prefer_session, else +0.1
        - Foreground: +0.1
        - Visible + enabled: +0.1
        """
        score = 0.5
        if is_session and prefer_session:
            score += 0.3
        elif is_session:
            score += 0.1
        if is_foreground:
            score += 0.1
        if w.is_visible and w.is_enabled:
            score += 0.1
        return min(score, 1.0)


# Global singleton for test/product use
_resolver: WindowResolver | None = None
_resolver_lock = Lock()


def get_window_resolver() -> WindowResolver:
    """Get the global WindowResolver singleton."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = WindowResolver()
    return _resolver
