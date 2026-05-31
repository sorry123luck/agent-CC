"""
数据库模块
使用 SQLAlchemy ORM + SQLite
"""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """ORM 基类"""
    pass


class Database:
    """数据库管理器"""

    _instance = None

    def __init__(self, db_path: str = "data/openclaw.db"):
        self.db_path = db_path
        self._engine = None
        self._session_factory = None

    def init(self) -> None:
        """初始化数据库连接"""
        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建引擎。File-backed SQLite must not use StaticPool because the
        # background enhancement workers perform concurrent DB work; sharing one
        # raw sqlite connection across threads can trigger sqlite3.InterfaceError.
        if self.db_path == ":memory:":
            db_url = "sqlite:///:memory:"
            self._engine = create_engine(
                db_url,
                echo=False,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            db_url = f"sqlite:///{self.db_path}"
            self._engine = create_engine(
                db_url,
                echo=False,
                connect_args={"check_same_thread": False},
            )

        # 设置 SQLAlchemy 2.0 模式
        event.listen(self._engine, "connect", lambda *args, **kwargs: None)

        # 创建会话工厂
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    @property
    def engine(self):
        if self._engine is None:
            self.init()
        return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """获取数据库会话的上下文管理器"""
        if self._session_factory is None:
            self.init()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        """创建所有表"""
        from src.storage.schema import Base
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """删除所有表（谨慎使用）"""
        from src.storage.schema import Base
        Base.metadata.drop_all(self.engine)


# 全局数据库实例
_db: Database | None = None


def init_db(db_path: str = "data/openclaw.db") -> Database:
    """初始化全局数据库实例"""
    global _db
    _db = Database(db_path)
    _db.init()
    return _db


def get_db() -> Database:
    """获取全局数据库实例"""
    global _db
    if _db is None:
        return init_db()
    return _db


@contextmanager
def Session() -> Iterator[Session]:
    """获取数据库会话的便捷函数（上下文管理器）"""
    db = get_db()
    with db.session() as sess:
        yield sess
