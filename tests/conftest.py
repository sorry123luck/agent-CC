"""
tests/conftest.py

pytest 配置文件
在测试收集阶段将仓库根目录注入到 sys.path，
确保 src 模块可以被正确导入。
"""

import sys
from pathlib import Path

# tests/conftest.py 位于 tests/ 目录，
# 仓库根目录是 tests/ 的父目录
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
