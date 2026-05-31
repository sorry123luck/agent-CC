"""
src.memory package
记忆模块：页面记忆、漂移检测、重学

主要类：
- MemoryService: 统一记忆服务入口
- PageTemplateManager: 页面模板管理
- DriftDetector: 漂移检测
- ActionRecipeManager: 动作配方管理
"""
from src.memory.memory_service import MemoryService
from src.memory.page_template_manager import PageTemplateManager
from src.memory.drift_detector import DriftDetector, DriftReport
from src.memory.action_recipe_manager import ActionRecipeManager, ActionRecipeData, ActionStep

__all__ = [
    "MemoryService",
    "PageTemplateManager",
    "DriftDetector",
    "DriftReport",
    "ActionRecipeManager",
    "ActionRecipeData",
    "ActionStep",
]
