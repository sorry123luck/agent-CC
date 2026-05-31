"""
src.indexer package
软件索引与扫描模块
"""

from src.indexer.catalog_service import CatalogService
from src.indexer.registry_scanner import RegistryScanner
from src.indexer.startmenu_scanner import StartMenuScanner
from src.indexer.desktop_scanner import DesktopScanner
from src.indexer.filesystem_scanner import FilesystemScanner
from src.indexer.shortcut_resolver import ShortcutResolver
from src.indexer.launch_resolver import LaunchResolver
from src.indexer.alias_matcher import AliasMatcher

__all__ = [
    "CatalogService",
    "RegistryScanner",
    "StartMenuScanner",
    "DesktopScanner",
    "FilesystemScanner",
    "ShortcutResolver",
    "LaunchResolver",
    "AliasMatcher",
]
