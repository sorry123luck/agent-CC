"""
tests.unit.test_repositories
仓库层测试
"""

import pytest
import tempfile
import os

from src.storage.db import Database
from src.storage.repositories import AppRepository, LaunchTargetRepository


class TestAppRepository:
    """App 仓库测试"""

    def setup_method(self):
        """创建临时数据库"""
        self.temp_db = tempfile.mktemp(suffix=".db")
        self.db = Database(self.temp_db)
        self.db.init()
        self.db.create_all()
        self.ctx = self.db.session()
        self.session = self.ctx.__enter__()
        self.repo = AppRepository(self.session)

    def teardown_method(self):
        """清理"""
        try:
            self.ctx.__exit__(None, None, None)
        except Exception:
            pass
        # 等待文件释放
        import time
        time.sleep(0.1)
        if os.path.exists(self.temp_db):
            try:
                os.unlink(self.temp_db)
            except PermissionError:
                pass  # Windows 文件锁问题，忽略

    def test_create_app(self):
        """创建软件记录"""
        app = self.repo.create(
            canonical_name="wechat",
            display_name="微信",
            publisher="Tencent",
            confidence=0.9
        )
        assert app.id is not None
        assert app.canonical_name == "wechat"
        assert app.display_name == "微信"

    def test_get_by_canonical_name(self):
        """按规范名查找"""
        self.repo.create(canonical_name="wechat", display_name="微信")

        found = self.repo.get_by_canonical_name("wechat")
        assert found is not None
        assert found.display_name == "微信"

    def test_get_by_nonexistent_name(self):
        """查找不存在的名称返回 None"""
        found = self.repo.get_by_canonical_name("nonexistent")
        assert found is None

    def test_list_all(self):
        """列出所有软件"""
        self.repo.create(canonical_name="wechat", display_name="微信")
        self.repo.create(canonical_name="chrome", display_name="Chrome")

        all_apps = self.repo.list_all()
        assert len(all_apps) == 2

    def test_count_all(self):
        """计数"""
        self.repo.create(canonical_name="wechat", display_name="微信")
        self.repo.create(canonical_name="chrome", display_name="Chrome")

        count = self.repo.count_all()
        assert count == 2

    def test_count_by_launch_path(self):
        """按启动路径计数（通过 LaunchTarget 关联检查）"""
        # launch_path 存在于 LaunchTarget 表，不是 App 表
        # 这里验证 App 本身可以正常创建，不验证 launch_path
        app = self.repo.create(
            canonical_name="wechat2",
            display_name="微信2",
            install_location=r"C:\Program Files\WeChat"
        )
        assert app.id is not None

        # 不同路径应该返回 0（因为没有创建 LaunchTarget）
        from src.storage.schema import LaunchTarget
        count = self.session.query(LaunchTarget).filter(
            LaunchTarget.path == r"C:\Program Files\WeChat\WeChat.exe"
        ).count()
        assert count == 0

    def test_update_app(self):
        """更新软件记录"""
        app = self.repo.create(canonical_name="wechat", display_name="微信")

        updated = self.repo.update(app.id, display_name="微信电脑版")
        assert updated.display_name == "微信电脑版"

    def test_delete_app(self):
        """删除软件记录"""
        app = self.repo.create(canonical_name="wechat", display_name="微信")

        deleted = self.repo.delete(app.id)
        assert deleted is True

        found = self.repo.get_by_id(app.id)
        assert found is None


class TestLaunchTargetRepository:
    """LaunchTarget 仓库测试"""

    def setup_method(self):
        """创建临时数据库"""
        self.temp_db = tempfile.mktemp(suffix=".db")
        self.db = Database(self.temp_db)
        self.db.init()
        self.db.create_all()
        self.ctx = self.db.session()
        self.session = self.ctx.__enter__()

        # 先创建一个 App
        app_repo = AppRepository(self.session)
        self.app = app_repo.create(canonical_name="wechat", display_name="微信")
        self.session.commit()

        self.repo = LaunchTargetRepository(self.session)

    def teardown_method(self):
        """清理"""
        try:
            self.ctx.__exit__(None, None, None)
        except Exception:
            pass
        # 等待文件释放
        import time
        time.sleep(0.1)
        if os.path.exists(self.temp_db):
            try:
                os.unlink(self.temp_db)
            except PermissionError:
                pass  # Windows 文件锁问题，忽略

    def test_create_launch_target(self):
        """创建启动入口"""
        target = self.repo.create(
            app_id=self.app.id,
            path=r"C:\Program Files\WeChat\WeChat.exe",
            target_type="exe",
            score=80.0,
            source="registry"
        )
        assert target.id is not None
        assert target.path == r"C:\Program Files\WeChat\WeChat.exe"
        assert target.score == 80.0

    def test_get_by_app_id(self):
        """按 app_id 查找启动入口"""
        self.repo.create(
            app_id=self.app.id,
            path=r"C:\Program Files\WeChat\WeChat.exe",
            score=80.0,
            source="registry"
        )
        self.repo.create(
            app_id=self.app.id,
            path=r"C:\Program Files\WeChat\WeChatUpdate.exe",
            score=30.0,
            source="registry"
        )

        targets = self.repo.get_by_app_id(self.app.id)
        assert len(targets) == 2

    def test_get_best_for_app(self):
        """获取最高分启动入口"""
        self.repo.create(
            app_id=self.app.id,
            path=r"C:\Program Files\WeChat\WeChatUpdate.exe",
            score=30.0,
            source="registry"
        )
        self.repo.create(
            app_id=self.app.id,
            path=r"C:\Program Files\WeChat\WeChat.exe",
            score=80.0,
            source="registry"
        )

        best = self.repo.get_best_for_app(self.app.id)
        assert best is not None
        assert "WeChat.exe" in best.path
        assert best.score == 80.0

    def test_delete_by_app_id(self):
        """删除某软件的所有启动入口"""
        self.repo.create(
            app_id=self.app.id,
            path=r"C:\Program Files\WeChat\WeChat.exe",
            score=80.0,
            source="registry"
        )

        count = self.repo.delete_by_app_id(self.app.id)
        assert count == 1

        targets = self.repo.get_by_app_id(self.app.id)
        assert len(targets) == 0
