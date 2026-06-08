from pathlib import Path

from src.windows import window_resolver


def test_find_executable_uses_vscode_localappdata_install_path(monkeypatch):
    local_appdata = r"C:\Users\Tester\AppData\Local"
    expected = local_appdata + r"\Programs\Microsoft VS Code\Code.exe"

    monkeypatch.setenv("LOCALAPPDATA", local_appdata)
    monkeypatch.setattr(window_resolver.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == expected)

    assert window_resolver._find_executable("vscode") == expected
