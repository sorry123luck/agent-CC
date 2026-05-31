#!/usr/bin/env python
"""
软件扫描脚本
扫描本机安装的软件并保存到数据库
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer.catalog_service import CatalogService
from src.common.logger import logger


def main():
    print("=" * 60)
    print("OpenClaw 软件扫描器")
    print("=" * 60)

    # 确保数据目录存在
    Path("data").mkdir(exist_ok=True)

    # 初始化数据库
    from src.storage.db import init_db
    db = init_db("data/openclaw.db")
    db.create_all()

    # 创建目录服务
    service = CatalogService()

    # 执行扫描
    print("\n开始扫描...")
    try:
        count = service.scan_all()
        print(f"\n扫描完成！共发现 {count} 个软件\n")

        # 显示部分结果
        print("部分软件列表：")
        print("-" * 60)
        results = []
        from src.storage.db import Session
        from src.storage.repositories import AppRepository
        with Session() as sess:
            repo = AppRepository(sess)
            all_apps = repo.list_all()
            for app in all_apps[:10]:
                print(f"  {app.display_name:<30}  {app.canonical_name}")
                results.append(app)

        if len(all_apps) > 10:
            print(f"  ... 还有 {len(all_apps) - 10} 个软件")

        print("-" * 60)

    except KeyboardInterrupt:
        print("\n\n扫描已取消")
    except Exception as e:
        logger.error(f"扫描失败: {e}")
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
