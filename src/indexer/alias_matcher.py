"""
软件别名匹配器
支持通用别名和用户自定义别名，提供模糊匹配
"""

import os
from typing import Any

from src.common.logger import get_logger
from src.common.utils import read_yaml, fuzzy_match

logger = get_logger(__name__)


class AliasMatcher:
    """软件别名匹配器"""

    def __init__(self, common_aliases_path: str = "data/aliases/common_aliases.yaml",
                 user_aliases_path: str = "data/aliases/user_aliases.yaml"):
        self.common_aliases_path = common_aliases_path
        self.user_aliases_path = user_aliases_path
        self._common_aliases: dict[str, str] = {}
        self._user_aliases: dict[str, str] = {}
        self._load_aliases()

    def _load_aliases(self) -> None:
        """加载别名文件"""
        # 加载通用别名
        if os.path.isfile(self.common_aliases_path):
            data = read_yaml(self.common_aliases_path)
            self._common_aliases = data.get("aliases", {})
        else:
            logger.warning(f"通用别名文件不存在: {self.common_aliases_path}")

        # 加载用户别名
        if os.path.isfile(self.user_aliases_path):
            data = read_yaml(self.user_aliases_path)
            self._user_aliases = data.get("aliases", {})
        else:
            logger.warning(f"用户别名文件不存在: {self.user_aliases_path}")

    def match(self, query: str) -> Any:
        """
        匹配别名

        Args:
            query: 查询字符串（用户输入）

        Returns:
            AliasMatchResult 或 None
        """
        query_lower = query.lower().strip()

        # 1. 精确匹配（优先用户别名）
        if query_lower in self._user_aliases:
            canonical = self._user_aliases[query_lower]
            from src.common.models import AliasMatchResult
            return AliasMatchResult(
                canonical_name=canonical,
                display_name=self._canonical_to_display(canonical),
                score=1.0,
                matched_alias=query_lower
            )

        if query_lower in self._common_aliases:
            canonical = self._common_aliases[query_lower]
            from src.common.models import AliasMatchResult
            return AliasMatchResult(
                canonical_name=canonical,
                display_name=self._canonical_to_display(canonical),
                score=1.0,
                matched_alias=query_lower
            )

        # 2. 模糊匹配
        best_match = None
        best_score = 0.0

        # 检查所有别名
        all_aliases = {**self._user_aliases, **self._common_aliases}
        for alias, canonical in all_aliases.items():
            score = fuzzy_match(query_lower, alias, threshold=0.3)
            if score > best_score:
                best_score = score
                best_match = (alias, canonical)

        if best_match and best_score >= 0.5:
            from src.common.models import AliasMatchResult
            return AliasMatchResult(
                canonical_name=best_match[1],
                display_name=self._canonical_to_display(best_match[1]),
                score=best_score,
                matched_alias=best_match[0]
            )

        return None

    def _canonical_to_display(self, canonical: str) -> str:
        """将规范名转换为显示名（首字母大写）"""
        # 简单实现：按大写字母分隔
        result = []
        for i, c in enumerate(canonical):
            if i == 0 or (c.isupper() and i > 0):
                if result:
                    result.append(" ")
                result.append(c)
            else:
                result.append(c)
        return "".join(result)

    def add_user_alias(self, alias: str, canonical: str) -> None:
        """
        添加用户别名

        Args:
            alias: 别名
            canonical: 规范名
        """
        self._user_aliases[alias.lower().strip()] = canonical.lower().strip()

        # 保存到文件
        data = {"aliases": self._user_aliases}
        import yaml
        with open(self.user_aliases_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)

    def get_all_aliases(self) -> dict[str, str]:
        """获取所有别名（合并）"""
        return {**self._common_aliases, **self._user_aliases}
