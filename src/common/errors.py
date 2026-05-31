"""
自定义异常类
"""


class DeskCanvasError(Exception):
    """基础异常类"""
    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AppNotFoundError(DeskCanvasError):
    """软件未找到"""
    pass


class AppLaunchError(DeskCanvasError):
    """软件启动失败"""
    pass


class WindowNotFoundError(DeskCanvasError):
    """窗口未找到"""
    pass


class ElementNotFoundError(DeskCanvasError):
    """界面元素未找到"""
    pass


class PageRecognitionError(DeskCanvasError):
    """页面识别失败"""
    pass


class ActionExecutionError(DeskCanvasError):
    """动作执行失败"""
    pass


class VerificationError(DeskCanvasError):
    """验证失败"""
    pass


class DriftDetectedError(DeskCanvasError):
    """页面漂移检测到"""
    pass


class ConfigError(DeskCanvasError):
    """配置错误"""
    pass


class DatabaseError(DeskCanvasError):
    """数据库错误"""
    pass


class MemoryError(DeskCanvasError):
    """记忆系统错误"""
    pass


class SafetyError(DeskCanvasError):
    """安全拦截错误"""
    pass
