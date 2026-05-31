"""
软件目录服务
统一管理软件扫描、搜索、启动的入口
"""

from typing import Iterator
from contextlib import contextmanager
from sqlalchemy.orm import Session

from src.common.logger import get_logger
from src.storage.db import Session as DB_Session
from src.storage.repositories import AppRepository, LaunchTargetRepository
from src.indexer.registry_scanner import RegistryScanner
from src.indexer.startmenu_scanner import StartMenuScanner
from src.indexer.desktop_scanner import DesktopScanner
from src.indexer.filesystem_scanner import FilesystemScanner
from src.indexer.launch_resolver import LaunchResolver
from src.indexer.alias_matcher import AliasMatcher

logger = get_logger(__name__)


class CatalogService:
    """
    软件目录服务
    统一入口，提供扫描、搜索、启动功能
    """

    def __init__(self):
        self._alias_matcher: AliasMatcher | None = None
        self._launch_resolver: LaunchResolver | None = None

    @property
    def alias_matcher(self) -> AliasMatcher:
        if self._alias_matcher is None:
            self._alias_matcher = AliasMatcher()
        return self._alias_matcher

    @property
    def launch_resolver(self) -> LaunchResolver:
        if self._launch_resolver is None:
            self._launch_resolver = LaunchResolver()
        return self._launch_resolver

    def scan_all(self) -> int:
        """
        执行全量扫描

        Returns:
            扫描到的软件数量
        """
        logger.info("开始全量软件扫描...")
        count = 0

        # 1. 注册表扫描
        logger.info("扫描注册表...")
        registry_count = 0
        for app_info in RegistryScanner().scan():
            self._save_app(app_info)
            registry_count += 1
        logger.info(f"注册表扫描完成，发现 {registry_count} 条记录")

        # 2. 开始菜单扫描
        logger.info("扫描开始菜单...")
        startmenu_count = 0
        for app_info in StartMenuScanner().scan():
            self._save_app(app_info)
            startmenu_count += 1
        logger.info(f"开始菜单扫描完成，发现 {startmenu_count} 条记录")

        # 3. 桌面扫描
        logger.info("扫描桌面...")
        desktop_count = 0
        for app_info in DesktopScanner().scan():
            self._save_app(app_info)
            desktop_count += 1
        logger.info(f"桌面扫描完成，发现 {desktop_count} 条记录")

        # 4. 文件系统扫描
        logger.info("扫描文件系统...")
        fs_count = 0
        for app_info in FilesystemScanner().scan():
            self._save_app(app_info)
            fs_count += 1
        logger.info(f"文件系统扫描完成，发现 {fs_count} 条记录")

        count = registry_count + startmenu_count + desktop_count + fs_count
        logger.info(f"全量扫描完成，共发现 {count} 条软件记录")
        return count

    def _save_app(self, app_info: dict) -> None:
        """保存软件到数据库"""
        with DB_Session() as sess:
            repo = AppRepository(sess)
            launch_repo = LaunchTargetRepository(sess)

            # 查找是否已存在（按规范名）
            existing = repo.get_by_canonical_name(app_info.get("canonical_name", ""))
            if existing:
                # 更新
                update_fields = {
                    "display_name": app_info.get("display_name"),
                    "publisher": app_info.get("publisher"),
                    "version": app_info.get("version"),
                    "install_location": app_info.get("install_location"),
                    "confidence": app_info.get("confidence", 0.5),
                }
                repo.update(existing.id, **update_fields)
                app = existing
            else:
                # 创建
                app = repo.create(
                    canonical_name=app_info.get("canonical_name", ""),
                    display_name=app_info.get("display_name", ""),
                    publisher=app_info.get("publisher"),
                    version=app_info.get("version"),
                    install_location=app_info.get("install_location"),
                    confidence=app_info.get("confidence", 0.5),
                )

            # 保存启动入口（去重：同一 app 同一路径只保留一个）
            if app_info.get("launch_path"):
                source = app_info.get("_source", "unknown")
                existing_targets = launch_repo.get_by_app_id(app.id)
                # 找同路径的已有记录，跳过重复
                existing = next(
                    (t for t in existing_targets if t.path == app_info["launch_path"]),
                    None,
                )
                if not existing:
                    launch_repo.create(
                        app_id=app.id,
                        path=app_info["launch_path"],
                        args=app_info.get("launch_args"),
                        target_type="exe",
                        score=80.0,
                        source=source,
                    )

    def search(self, query: str) -> list[dict]:
        """
        搜索软件

        Args:
            query: 搜索关键词

        Returns:
            匹配的软件列表
        """
        results = []

        with DB_Session() as sess:
            repo = AppRepository(sess)
            launch_repo = LaunchTargetRepository(sess)

            # 1. 别名匹配
            alias_result = self.alias_matcher.match(query)
            if alias_result:
                app = repo.get_by_canonical_name(alias_result.canonical_name)
                if app:
                    app_dict = self._app_to_dict(app, launch_repo)
                    if app_dict:
                        results.append(app_dict)

            # 2. 数据库模糊搜索
            for app in repo.list_all():
                if query.lower() in app.display_name.lower() or query.lower() in app.canonical_name.lower():
                    if not any(r["canonical_name"] == app.canonical_name for r in results):
                        app_dict = self._app_to_dict(app, launch_repo)
                        if app_dict:
                            results.append(app_dict)

        return results

    def launch(self, app_name: str) -> dict:
        """
        启动软件

        Args:
            app_name: 软件名称（显示名或别名）

        Returns:
            启动结果
        """
        from src.common.models import LaunchResult

        # 1. 查找软件
        search_results = self.search(app_name)
        if not search_results:
            return LaunchResult(success=False, app_name=app_name, message=f"未找到软件: {app_name}").model_dump()

        app_dict = search_results[0]

        # 2. 获取启动入口
        with DB_Session() as sess:
            repo = AppRepository(sess)
            app = repo.get_by_canonical_name(app_dict["canonical_name"])
            if not app:
                return LaunchResult(success=False, app_name=app_name, message="数据库记录不存在").model_dump()

            launch_repo = LaunchTargetRepository(sess)
            best_launch = launch_repo.get_best_for_app(app.id)
            if not best_launch:
                return LaunchResult(success=False, app_name=app_name, message="没有找到启动入口").model_dump()

            # 3. 启动
            return self.launch_resolver.launch(best_launch.path, best_launch.args)

    def _app_to_dict(self, app, launch_repo: LaunchTargetRepository) -> dict | None:
        """将 App 模型转换为字典"""
        # 获取启动路径
        launch_path = None
        best_launch = launch_repo.get_best_for_app(app.id)
        if best_launch:
            launch_path = best_launch.path

        return {
            "canonical_name": app.canonical_name,
            "display_name": app.display_name,
            "publisher": app.publisher,
            "install_location": app.install_location,
            "launch_path": launch_path,
            "version": app.version,
            "confidence": app.confidence,
        }
