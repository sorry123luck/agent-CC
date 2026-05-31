"""
tests.unit.test_alias_matcher
别名匹配器测试
"""

import pytest
import tempfile
import os
import yaml

from src.indexer.alias_matcher import AliasMatcher


class TestAliasMatcher:
    """别名匹配器测试"""

    def setup_method(self):
        """创建临时别名文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.common_aliases_path = os.path.join(self.temp_dir, "common.yaml")
        self.user_aliases_path = os.path.join(self.temp_dir, "user.yaml")

        # 写入测试数据
        common_data = {
            "aliases": {
                "微信": "WeChat",
                "微信电脑版": "WeChat",
                "网易云": "NetEaseCloudMusic",
                "Chrome": "Chrome",
            }
        }
        with open(self.common_aliases_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(common_data, f)

        user_data = {"aliases": {}}
        with open(self.user_aliases_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(user_data, f)

        self.matcher = AliasMatcher(self.common_aliases_path, self.user_aliases_path)

    def teardown_method(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exact_match_common_alias(self):
        """精确匹配通用别名"""
        result = self.matcher.match("微信")
        assert result is not None
        assert result.canonical_name == "WeChat"
        assert result.score == 1.0

    def test_exact_match_case_insensitive(self):
        """别名匹配大小写不敏感"""
        result = self.matcher.match("chrome")
        assert result is not None
        assert result.canonical_name == "Chrome"

    def test_no_match(self):
        """无匹配时返回 None"""
        result = self.matcher.match("完全不存在的软件名称xyz")
        assert result is None

    def test_all_aliases_merged(self):
        """用户别名和通用别名应该合并"""
        all_aliases = self.matcher.get_all_aliases()
        assert "微信" in all_aliases
        assert "Chrome" in all_aliases

    def test_add_user_alias(self):
        """添加用户别名"""
        self.matcher.add_user_alias("我的微信", "WeChat")

        result = self.matcher.match("我的微信")
        assert result is not None
        assert result.canonical_name == "wechat"  # stored as lowercase
        assert result.score == 1.0

    def test_canonical_to_display(self):
        """规范名转显示名"""
        display = self.matcher._canonical_to_display("WeChat")
        assert "w" in display.lower()
        assert "e" in display.lower()
