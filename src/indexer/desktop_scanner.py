"""
桌面扫描器
扫描桌面上的快捷方式和 exe
"""

import os
from typing import Generator
from pathlib import Path

from src.common.logger import get_logger
from src.common.utils import expand_path
from src.indexer.shortcut_resolver import ShortcutResolver

logger = get_logger(__name__)


class DesktopScanner:
    """桌面软件扫描器"""

    DESKTOP_PATHS = [
        r"C:\Users\$USER\Desktop",
        r"C:\Users\Public\Desktop",
    ]

    def scan(self) -> Generator[dict, None, None]:
        """
        扫描桌面上的软件

        Yields:
            软件信息字典
        """
        shortcut_resolver = ShortcutResolver()

        for desktop_path in self.DESKTOP_PATHS:
            desktop_path = expand_path(desktop_path)
            if not os.path.isdir(desktop_path):
                continue

            yield from self._scan_directory(desktop_path, shortcut_resolver)

    def _scan_directory(self, directory: str, shortcut_resolver: ShortcutResolver) -> Generator[dict, None, None]:
        """扫描目录"""
        try:
            for entry in os.scandir(directory):
                if entry.is_file() and entry.name.lower().endswith((".lnk", ".exe")):
                    yield from self._process_entry(entry.path, shortcut_resolver)
        except Exception as e:
            logger.warning(f"无法扫描目录 {directory}: {e}")

    def _process_entry(self, path: str, shortcut_resolver: ShortcutResolver) -> Generator[dict, None, None]:
        """处理单个快捷方式或 exe"""
        try:
            if path.lower().endswith(".lnk"):
                exe_path = shortcut_resolver.resolve_lnk(path)
                if not exe_path:
                    return
                display_name = Path(path).stem
            else:
                exe_path = path
                display_name = Path(path).stem

            if self._is_likely_uninstaller(display_name):
                return

            info = {
                "display_name": display_name,
                "canonical_name": self._to_canonical_name(display_name),
                "launch_path": exe_path,
                "_source": "desktop",
            }

            yield info

        except Exception as e:
            logger.debug(f"无法处理 {path}: {e}")

    def _is_likely_uninstaller(self, name: str) -> bool:
        """判断是否可能是卸载程序"""
        name_lower = name.lower()
        uninstaller_keywords = ["uninstall", "update", "setup", "installer", "download", "helper"]
        return any(kw in name_lower for kw in uninstaller_keywords)

    def _to_canonical_name(self, display_name: str) -> str:
        """将显示名转换为规范名"""
        name = "".join(c if c.isalnum() else " " for c in display_name)
        return name.replace(" ", "").lower()
