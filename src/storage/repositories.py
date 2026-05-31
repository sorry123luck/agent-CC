"""
数据仓库层
提供数据库表操作的高级封装
"""

from typing import Any
from sqlalchemy.orm import Session

from src.storage.schema import App, AppAlias, LaunchTarget, PageTemplate, Candidate, ActionRecipe, ExecutionLog, DriftEvent


class AppRepository:
    """软件仓库"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **kwargs) -> App:
        app = App(**kwargs)
        self.session.add(app)
        self.session.flush()
        return app

    def get_by_id(self, app_id: int) -> App | None:
        return self.session.query(App).filter(App.id == app_id).first()

    def get_by_canonical_name(self, canonical_name: str) -> App | None:
        return self.session.query(App).filter(App.canonical_name == canonical_name).first()

    def list_all(self) -> list[App]:
        return self.session.query(App).all()

    def count_all(self) -> int:
        return self.session.query(App).count()

    def count_by_launch_path(self, launch_path: str) -> int:
        """统计具有相同启动路径的软件数量（用于去重检查）"""
        from src.storage.schema import LaunchTarget
        return self.session.query(LaunchTarget).filter(LaunchTarget.path == launch_path).count()

    def update(self, app_id: int, **kwargs) -> App | None:
        app = self.get_by_id(app_id)
        if app:
            for key, value in kwargs.items():
                setattr(app, key, value)
            self.session.flush()
        return app

    def delete(self, app_id: int) -> bool:
        app = self.get_by_id(app_id)
        if app:
            self.session.delete(app)
            return True
        return False


class AppAliasRepository:
    """软件别名仓库"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **kwargs) -> AppAlias:
        alias = AppAlias(**kwargs)
        self.session.add(alias)
        self.session.flush()
        return alias

    def get_by_alias(self, alias: str) -> list[AppAlias]:
        return self.session.query(AppAlias).filter(AppAlias.alias == alias).all()

    def get_by_app_id(self, app_id: int) -> list[AppAlias]:
        return self.session.query(AppAlias).filter(AppAlias.app_id == app_id).all()

    def bulk_create(self, app_id: int, aliases: list[str], source: str = "common") -> list[AppAlias]:
        results = []
        for alias_str in aliases:
            alias = AppAlias(app_id=app_id, alias=alias_str, source=source)
            self.session.add(alias)
            results.append(alias)
        self.session.flush()
        return results


class LaunchTargetRepository:
    """启动入口仓库"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **kwargs) -> LaunchTarget:
        target = LaunchTarget(**kwargs)
        self.session.add(target)
        self.session.flush()
        return target

    def get_by_id(self, target_id: int) -> LaunchTarget | None:
        return self.session.query(LaunchTarget).filter(LaunchTarget.id == target_id).first()

    def get_by_app_id(self, app_id: int) -> list[LaunchTarget]:
        return self.session.query(LaunchTarget).filter(LaunchTarget.app_id == app_id).order_by(LaunchTarget.score.desc()).all()

    def get_best_for_app(self, app_id: int) -> LaunchTarget | None:
        return (
            self.session.query(LaunchTarget)
            .filter(LaunchTarget.app_id == app_id)
            .order_by(LaunchTarget.score.desc())
            .first()
        )

    def delete_by_app_id(self, app_id: int) -> int:
        count = self.session.query(LaunchTarget).filter(LaunchTarget.app_id == app_id).delete()
        self.session.flush()
        return count


class PageTemplateRepository:
    """页面模板仓库"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **kwargs) -> PageTemplate:
        template = PageTemplate(**kwargs)
        self.session.add(template)
        self.session.flush()
        return template

    def get_by_id(self, template_id: int) -> PageTemplate | None:
        return self.session.query(PageTemplate).filter(PageTemplate.id == template_id).first()

    def get_by_app_and_type(self, app_id: str, page_type: str) -> PageTemplate | None:
        return (
            self.session.query(PageTemplate)
            .filter(PageTemplate.app_id == app_id, PageTemplate.page_type == page_type)
            .first()
        )

    def list_by_app(self, app_id: str) -> list[PageTemplate]:
        return self.session.query(PageTemplate).filter(PageTemplate.app_id == app_id).all()


class ExecutionLogRepository:
    """执行日志仓库"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **kwargs) -> ExecutionLog:
        log = ExecutionLog(**kwargs)
        self.session.add(log)
        self.session.flush()
        return log

    def get_by_task_id(self, task_id: str) -> list[ExecutionLog]:
        return self.session.query(ExecutionLog).filter(ExecutionLog.task_id == task_id).order_by(ExecutionLog.created_at).all()

    def get_recent(self, limit: int = 100) -> list[ExecutionLog]:
        return self.session.query(ExecutionLog).order_by(ExecutionLog.created_at.desc()).limit(limit).all()
