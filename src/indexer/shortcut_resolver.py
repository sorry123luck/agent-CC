"""
快捷方式解析器
解析 .lnk 快捷方式文件，获取其指向的目标路径
"""

import os
from pathlib import Path
import struct

from src.common.logger import get_logger

logger = get_logger(__name__)


class ShortcutResolver:
    """快捷方式解析器"""

    def resolve_lnk(self, lnk_path: str) -> str | None:
        """
        解析 .lnk 文件，获取其指向的目标路径

        Args:
            lnk_path: 快捷方式文件路径

        Returns:
            目标 exe 路径，如果解析失败返回 None
        """
        try:
            # 方法1：尝试用 COM 接口解析（最准确）
            exe_path = self._resolve_via_com(lnk_path)
            if exe_path:
                return exe_path

            # 方法2：手动解析 lnk 文件格式（备用）
            exe_path = self._resolve_via_binary(lnk_path)
            if exe_path:
                return exe_path

            return None

        except Exception as e:
            logger.debug(f"无法解析快捷方式 {lnk_path}: {e}")
            return None

    def _resolve_via_com(self, lnk_path: str) -> str | None:
        """通过 COM 接口解析快捷方式"""
        try:
            import pythoncom
            from win32com.shell.shell import ShellLink
            from win32com.shell.shell import Shell

            pythoncom.CoInitialize()

            try:
                link = ShellLink()
                link.SetPath(lnk_path)
                path = link.GetPath(0)[0]
                if path and path.lower().endswith(".exe"):
                    return path
            finally:
                pythoncom.CoUninitialize()

        except ImportError:
            logger.debug("pythoncom 不可用，跳过 COM 解析")
        except Exception as e:
            logger.debug(f"COM 解析失败: {e}")

        return None

    def _resolve_via_binary(self, lnk_path: str) -> str | None:
        """手动解析 lnk 二进制格式获取目标路径"""
        try:
            with open(lnk_path, "rb") as f:
                data = f.read()

            # 解析 Shell Item ID List
            # 简化实现：查找 .exe 路径字符串
            # 完整实现需要解析 lnk 文件格式的各个部分

            # 尝试在文件中查找可执行文件路径
            # 这是一个简化实现，复杂情况可能不准确
            for i in range(len(data) - 4):
                # 查找 .exe 字符串
                if data[i:i+4].lower() == b".exe":
                    # 向前查找路径开始
                    start = i
                    while start > 0 and data[start-1] not in (0, 0x20, 0x00):
                        start -= 1
                    path_bytes = data[start:i+4]
                    try:
                        path = path_bytes.decode("utf-8", errors="ignore")
                        if os.path.isfile(path):
                            return path
                    except:
                        pass

            return None

        except Exception as e:
            logger.debug(f"二进制解析失败: {e}")
            return None
