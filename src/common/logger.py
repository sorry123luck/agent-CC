"""
日志模块
使用 loguru，提供简洁的 API
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(data_dir: str = "data") -> None:
    """
    初始化日志配置

    Args:
        data_dir: 数据目录，用于存放日志文件
    """
    # 移除默认 handler
    logger.remove()

    # 创建 logs 目录
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 控制台输出
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
        colorize=True,
    )

    # 文件输出（按日期分割）
    logger.add(
        log_dir / "openclaw_{time}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
    )

    # 错误日志单独记录
    logger.add(
        log_dir / "error_{time}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def get_logger(name: str):
    """
    获取指定名称的 logger

    Args:
        name: 模块名称，通常用 __name__

    Returns:
        logger 实例
    """
    return logger.bind(name=name)


# 默认初始化
setup_logger()
