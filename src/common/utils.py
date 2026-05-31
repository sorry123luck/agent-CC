"""
通用工具函数
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Any


def expand_path(path: str) -> str:
    """
    展开路径中的环境变量和特殊字符

    Args:
        path: 原始路径

    Returns:
        展开后的路径
    """
    # 展开 USER 环境变量
    path = path.replace("$USER", os.environ.get("USERNAME", "Default"))
    # 展开其他常见环境变量
    path = os.path.expandvars(path)
    # 展开 ~
    path = os.path.expanduser(path)
    return path


def normalize_path(path: str) -> str:
    """
    规范化路径（统一分隔符、转小写）

    Args:
        path: 原始路径

    Returns:
        规范化后的路径
    """
    return os.path.normpath(path).lower().replace("\\", "/")


def compute_file_hash(path: str, algorithm: str = "md5") -> str:
    """
    计算文件哈希值

    Args:
        path: 文件路径
        algorithm: 哈希算法（md5/sha1/sha256）

    Returns:
        十六进制哈希字符串
    """
    hash_func = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def compute_string_hash(s: str, algorithm: str = "md5") -> str:
    """
    计算字符串哈希值

    Args:
        s: 字符串
        algorithm: 哈希算法

    Returns:
        十六进制哈希字符串
    """
    return hashlib.new(algorithm, s.encode("utf-8")).hexdigest()


def fuzzy_match(text: str, pattern: str, threshold: float = 0.6) -> float:
    """
    模糊匹配（简单的包含+长度比例匹配）

    Args:
        text: 待匹配文本
        pattern: 模式
        threshold: 阈值（0-1）

    Returns:
        匹配分数，低于阈值返回 0
    """
    text_lower = text.lower()
    pattern_lower = pattern.lower()

    # 完全包含
    if pattern_lower in text_lower:
        return 1.0

    # 关键词匹配（按空格分割）
    pattern_words = pattern_lower.split()
    matched_words = sum(1 for w in pattern_words if w in text_lower)
    if matched_words > 0:
        score = matched_words / len(pattern_words)
        return score if score >= threshold else 0.0

    return 0.0


def safe_filename(name: str) -> str:
    """
    将字符串转换为安全的文件名

    Args:
        name: 原始名称

    Returns:
        安全的文件名
    """
    # 替换非法字符
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    # 去除首尾空格和点
    name = name.strip(". ")
    # 限制长度
    return name[:200] if name else "unnamed"


def ensure_dir(path: str) -> Path:
    """
    确保目录存在

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str) -> dict[str, Any]:
    """
    读取 YAML 文件

    Args:
        path: 文件路径

    Returns:
        解析后的字典
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str, data: dict[str, Any]) -> None:
    """
    写入 YAML 文件

    Args:
        path: 文件路径
        data: 要写入的数据
    """
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
