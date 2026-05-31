"""
注册表扫描器
扫描 Windows 注册表中的软件安装记录
"""

import os
from typing import Generator
from pathlib import Path
import winreg

from src.common.logger import get_logger
from src.common.utils import expand_path

logger = get_logger(__name__)


class RegistryScanner:
    """注册表软件扫描器"""

    # 注册表扫描路径
    REGISTRY_PATHS = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]

    def scan(self) -> Generator[dict, None, None]:
        """
        扫描注册表中的软件

        Yields:
            软件信息字典
        """
        for hkey, subkey in self.REGISTRY_PATHS:
            yield from self._scan_key(hkey, subkey)

    def _scan_key(self, hkey: int, subkey: str) -> Generator[dict, None, None]:
        """扫描单个注册表键"""
        try:
            with winreg.OpenKey(hkey, subkey, 0, winreg.KEY_READ) as key:
                i = 0
                while True:
                    try:
                        app_key_name = winreg.EnumKey(key, i)
                        i += 1
                        yield from self._read_app_key(hkey, f"{subkey}\\{app_key_name}")
                    except OSError:
                        break
        except Exception as e:
            logger.warning(f"无法打开注册表键 {hkey}\\{subkey}: {e}")

    def _read_app_key(self, hkey: int, app_key_path: str) -> Generator[dict, None, None]:
        """读取单个软件的注册表信息"""
        try:
            with winreg.OpenKey(hkey, app_key_path, 0, winreg.KEY_READ) as app_key:
                info = {}

                # 读取关键字段
                display_name = self._read_value(app_key, "DisplayName")
                if not display_name:
                    return  # 没有显示名称，跳过

                info["display_name"] = display_name
                info["canonical_name"] = self._to_canonical_name(display_name)
                info["publisher"] = self._read_value(app_key, "Publisher")
                info["version"] = self._read_value(app_key, "DisplayVersion")
                info["install_location"] = self._read_value(app_key, "InstallLocation")
                info["launch_path"] = self._resolve_exe_path(self._read_value(app_key, "InstallLocation"), display_name)
                info["launch_args"] = self._read_value(app_key, "QuietUninstallString") or ""

                # 判断是否为 UWP
                info["is_uwp"] = "ms-resource" in (self._read_value(app_key, "UrlUpdateInfo") or "")

                info["_source"] = "registry"

                yield info

        except Exception as e:
            logger.debug(f"无法读取注册表项 {app_key_path}: {e}")

    def _read_value(self, key, value_name: str) -> str | None:
        """读取注册表值"""
        try:
            value, _ = winreg.QueryValueEx(key, value_name)
            return str(value).strip() if value else None
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _to_canonical_name(self, display_name: str) -> str:
        """将显示名转换为规范名"""
        # 移除非字母数字字符，转为 PascalCase
        name = "".join(c if c.isalnum() else " " for c in display_name)
        # 简单处理：移除空格，转小写
        return name.replace(" ", "").lower()

    def _resolve_exe_path(self, install_location: str | None, display_name: str) -> str | None:
        """尝试解析 exe 路径"""
        if not install_location:
            return None

        install_location = expand_path(install_location)
        if not os.path.isdir(install_location):
            return None

        # 在安装目录中查找 exe
        try:
            exe_files = []
            for root, dirs, files in os.walk(install_location):
                for f in files:
                    if f.lower().endswith(".exe"):
                        exe_files.append(os.path.join(root, f))

            if not exe_files:
                return None

            # 优先选择与软件名匹配的 exe
            name_lower = display_name.lower().replace(" ", "")
            for exe in exe_files:
                exe_name = os.path.basename(exe).lower().replace(".exe", "").replace(" ", "")
                if name_lower in exe_name or exe_name in name_lower:
                    return exe

            # 否则返回最短路径的 exe（通常是主程序）
            return min(exe_files, key=len)

        except Exception as e:
            logger.debug(f"无法遍历目录 {install_location}: {e}")
            return None
