"""
自定义异常类
"""


class OpenClawError(Exception):
    """基础异常类"""
    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AppNotFoundError(OpenClawError):
    """软件未找到"""
    pass


class AppLaunchError(OpenClawError):
    """软件启动失败"""
    pass


class WindowNotFoundError(OpenClawError):
    """窗口未找到"""
    pass


class ElementNotFoundError(OpenClawError):
    """界面元素未找到"""
    pass


class PageRecognitionError(OpenClawError):
    """页面识别失败"""
    pass


class ActionExecutionError(OpenClawError):
    """动作执行失败"""
    pass


class VerificationError(OpenClawError):
    """验证失败"""
    pass


class DriftDetectedError(OpenClawError):
    """页面漂移检测到"""
    pass


class ConfigError(OpenClawError):
    """配置错误"""
    pass


class DatabaseError(OpenClawError):
    """数据库错误"""
    pass


class MemoryError(OpenClawError):
    """记忆系统错误"""
    pass


class SafetyError(OpenClawError):
    """安全拦截错误"""
    pass
