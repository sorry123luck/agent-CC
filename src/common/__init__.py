"""
src.common package
公共模块：日志、模型定义、错误类、枚举、工具函数
"""

from src.common.logger import get_logger, setup_logger
from src.common.models import *
from src.common.errors import *
from src.common.enums import *
from src.common.utils import *

__all__ = [
    "get_logger", "setup_logger",
    "App", "AppAlias", "LaunchTarget", "AliasMatchResult", "LaunchResult",
    "WindowInfo", "Screenshot", "Rect", "Candidate", "PageModel",
    "ActionStep", "ActionResult", "TaskRequest", "TaskResponse",
    "OperationPackManifest",
    "DeskCanvasError", "AppNotFoundError", "AppLaunchError", "WindowNotFoundError",
    "ElementNotFoundError", "PageRecognitionError", "ActionExecutionError",
    "VerificationError", "DriftDetectedError", "ConfigError", "DatabaseError",
    "MemoryError", "SafetyError",
    "AppSource", "LaunchTargetType", "PageType", "ElementRole", "ActionType",
    "RiskLevel", "TaskStatus", "VerifyResult",
    "expand_path", "normalize_path", "compute_file_hash", "compute_string_hash",
    "fuzzy_match", "safe_filename", "ensure_dir", "read_yaml", "write_yaml",
]
