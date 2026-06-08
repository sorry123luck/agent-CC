"""
文件系统扫描器
扫描常见安装目录中的软件
"""

import os
from typing import Generator
from pathlib import Path

from src.common.logger import get_logger
from src.common.utils import expand_path

logger = get_logger(__name__)


class FilesystemScanner:
    """文件系统软件扫描器"""

    INSTALL_DIRS = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"C:\Users\$USER\AppData\Local\Programs",
        r"C:\Users\$USER\AppData\Local",
    ]

    # 排除的目录
    EXCLUDE_DIRS = {
        "Windows",
        "Windows Defender",
        "Windows Security",
        "Microsoft",
        "Intel",
        "NVIDIA",
        "AMD",
        "Common Files",
        "DotNet",
        "WindowsPowerShell",
    }

    def scan(self) -> Generator[dict, None, None]:
        """
        扫描常见安装目录中的软件

        Yields:
            软件信息字典
        """
        for install_dir in self.INSTALL_DIRS:
            install_dir = expand_path(install_dir)
            if not os.path.isdir(install_dir):
                continue

            yield from self._scan_directory(install_dir)

    def _scan_directory(self, directory: str) -> Generator[dict, None, None]:
        """递归扫描目录"""
        try:
            for entry in os.scandir(directory):
                if not entry.is_dir():
                    continue

                entry_name = entry.name

                # 跳过排除目录
                if entry_name in self.EXCLUDE_DIRS or entry_name.startswith("."):
                    continue

                # 检查是否是软件主目录（通常有 exe）
                exe_path = self._find_main_exe(entry.path)
                if exe_path:
                    info = {
                        "display_name": entry_name,
                        "canonical_name": self._to_canonical_name(entry_name),
                        "install_location": entry.path,
                        "launch_path": exe_path,
                        "_source": "filesystem",
                    }
                    yield info
                else:
                    # 递归扫描子目录
                    yield from self._scan_directory(entry.path)

        except Exception as e:
            logger.debug(f"无法扫描目录 {directory}: {e}")

    def _find_main_exe(self, directory: str) -> str | None:
        """在目录中查找主程序 exe"""
        try:
            exe_files = []
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.lower().endswith(".exe") and not self._is_excluded_exe(f):
                        exe_files.append(os.path.join(root, f))

            if not exe_files:
                return None

            # 优先选择与目录名匹配的 exe
            dir_name = Path(directory).name.lower().replace(" ", "")
            for exe in exe_files:
                exe_name = Path(exe).stem.lower().replace(" ", "")
                if dir_name in exe_name or exe_name in dir_name:
                    return exe

            # 返回最短路径的 exe
            return min(exe_files, key=len)

        except Exception as e:
            logger.debug(f"无法查找 exe 于 {directory}: {e}")
            return None

    def _is_excluded_exe(self, exe_name: str) -> bool:
        """判断是否应排除的 exe"""
        name_lower = exe_name.lower()
        excluded = [
            "uninstall", "update", "setup", "installer", "download",
            "helper", "service", "crash", "report", "config",
            "engine", "launcher", "bootstrap",
        ]
        return any(kw in name_lower for kw in excluded)

    def _to_canonical_name(self, display_name: str) -> str:
        """将显示名转换为规范名"""
        name = "".join(c if c.isalnum() else " " for c in display_name)
        return name.replace(" ", "").lower()
