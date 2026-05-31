"""
tests.unit.test_launch_resolver
启动入口解析器测试
"""

import pytest
import os
import tempfile

from src.indexer.launch_resolver import LaunchResolver


class TestLaunchResolver:
    """启动入口解析器测试"""

    def setup_method(self):
        self.resolver = LaunchResolver()

    def test_score_path_program_files(self):
        """Program Files 路径应该得高分"""
        score = self.resolver._score_path_location(
            r"C:\Program Files\Tencent\WeChat\WeChat.exe"
        )
        assert score >= 15

    def test_score_path_appdata(self):
        """AppData 路径应该有合理分数"""
        score = self.resolver._score_path_location(
            r"%LOCALAPPDATA%\Programs\WeChat\WeChat.exe"
        )
        assert score >= 10

    def test_score_filename_exact_match(self):
        """文件名完全匹配得最高分"""
        score = self.resolver._score_filename_match(
            r"C:\Program Files\WeChat\WeChat.exe",
            "WeChat"
        )
        assert score >= 25

    def test_score_suspicious_penalty(self):
        """可疑文件名应该扣分"""
        score = self.resolver._score_suspicious_penalty(
            r"C:\Program Files\WeChat\WeChatUpdate.exe"
        )
        assert score >= 20  # 至少扣 20 分

    def test_resolve_empty_candidates(self):
        """空候选列表返回 None"""
        result = self.resolver.resolve("WeChat", candidates=[])
        assert result is None

    def test_resolve_single_candidate(self):
        """单个候选直接返回"""
        candidates = [
            {
                "path": r"C:\Program Files\WeChat\WeChat.exe",
                "args": None,
                "source": "registry"
            }
        ]
        result = self.resolver.resolve("WeChat", candidates=candidates)
        assert result is not None
        assert "score" in result
        assert result["score"] >= 0

    def test_resolve_picks_best(self):
        """多个候选时应该选择分数最高的"""
        candidates = [
            {
                "path": r"C:\Program Files\WeChat\WeChatUpdate.exe",
                "args": None,
                "source": "registry"
            },
            {
                "path": r"C:\Program Files\WeChat\WeChat.exe",
                "args": None,
                "source": "registry"
            },
        ]
        result = self.resolver.resolve("WeChat", candidates=candidates)
        assert result is not None
        assert "WeChat.exe" in result["path"]
