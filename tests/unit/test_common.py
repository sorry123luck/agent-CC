"""
tests.unit.test_common
公共模块测试
"""

import pytest
from src.common.utils import fuzzy_match, safe_filename, expand_path, normalize_path
from src.common.models import LaunchResult, AliasMatchResult
from src.common.enums import AppSource, RiskLevel


class TestFuzzyMatch:
    """模糊匹配测试"""

    def test_exact_match(self):
        assert fuzzy_match("微信", "微信") == 1.0

    def test_partial_match(self):
        # 参数顺序：text=较长字符串, pattern=要匹配的短字符串
        score = fuzzy_match("微信电脑版", "微信")
        assert score >= 0.5

    def test_no_match(self):
        assert fuzzy_match("abc", "xyz") == 0

    def test_case_insensitive(self):
        assert fuzzy_match("WECHAT", "WeChat") == 1.0


class TestSafeFilename:
    """安全文件名测试"""

    def test_normal_name(self):
        assert safe_filename("微信") == "微信"

    def test_illegal_chars_removed(self):
        # 输入: file<>:"|?*.txt  ->  非法字符: < > : " | ? * 共7个被替换，. 保留
        assert safe_filename('file<>:"|?*.txt') == "file_______.txt"

    def test_empty_name(self):
        assert safe_filename("   ") == "unnamed"


class TestModels:
    """数据模型测试"""

    def test_launch_result_success(self):
        result = LaunchResult(
            success=True,
            app_name="WeChat",
            message="启动成功"
        )
        assert result.success is True
        assert result.app_name == "WeChat"

    def test_alias_match_result(self):
        result = AliasMatchResult(
            canonical_name="wechat",
            display_name="微信",
            score=0.95,
            matched_alias="微信"
        )
        assert result.canonical_name == "wechat"
        assert result.score == 0.95


class TestEnums:
    """枚举类型测试"""

    def test_appsource_values(self):
        assert AppSource.REGISTRY == "registry"
        assert AppSource.STARTMENU == "startmenu"
        assert AppSource.DESKTOP == "desktop"

    def test_risklevel_values(self):
        assert RiskLevel.LOW == "low"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
