#!/usr/bin/env python
"""
初始化数据库脚本
运行此脚本创建所有表
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import get_db, init_db


def main():
    print("正在初始化数据库...")

    # 确保 data 目录存在
    Path("data").mkdir(exist_ok=True)

    # 初始化数据库
    db = init_db("data/openclaw.db")

    # 创建所有表
    db.create_all()

    print("数据库初始化完成！")
    print(f"数据库路径: {Path('data/openclaw.db').absolute()}")

    # 显示创建的所有表
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"已创建的表: {', '.join(tables)}")


if __name__ == "__main__":
    main()
