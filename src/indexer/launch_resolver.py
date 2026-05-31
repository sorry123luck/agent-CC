"""
启动入口解析器
对同一软件的多个启动候选进行打分，选择最优启动入口
"""

import os
from pathlib import Path

from src.common.logger import get_logger
from src.common.models import LaunchResult
from src.common.utils import expand_path

logger = get_logger(__name__)


class LaunchResolver:
    """启动入口解析器"""

    # 路径权重
    PATH_WEIGHTS = {
        "program_files": 20,
        "user_local": 15,
        "appdata_roaming": 15,
        "appdata_local": 10,
        "other": 0,
    }

    # 文件名权重指标
    MAIN_INDICATORS = ["app", "application", ""]
    SUSPICIOUS_PATTERNS = [
        "uninstall", "update", "setup", "installer", "download",
        "helper", "service", "crash", "report", "config",
        "engine", "launcher", "bootstrap", "redist"
    ]

    def resolve(self, app_name: str, candidates: list[dict] | None = None) -> dict | None:
        """
        解析最优启动入口

        Args:
            app_name: 软件名称
            candidates: 候选启动入口列表，如果为 None 则从数据库加载

        Returns:
            最优启动入口信息
        """
        if candidates is None:
            candidates = self._load_candidates_from_db(app_name)
            if not candidates:
                return None

        # 对每个候选打分
        scored = []
        for candidate in candidates:
            score = self._calculate_score(candidate, app_name)
            candidate["score"] = score
            scored.append(candidate)

        # 返回分数最高的
        return max(scored, key=lambda x: x["score"]) if scored else None

    def _calculate_score(self, candidate: dict, app_name: str) -> float:
        """计算启动入口分数"""
        score = 0.0
        path = candidate.get("path", "")

        if not path:
            return 0.0

        # 1. 路径正规性加分
        score += self._score_path_location(path)

        # 2. 文件名匹配度
        score += self._score_filename_match(path, app_name)

        # 3. 文件大小合理性
        score += self._score_filesize(path)

        # 4. 惩罚可疑文件名
        score -= self._score_suspicious_penalty(path)

        return max(0.0, min(100.0, score))

    def _score_path_location(self, path: str) -> float:
        """根据路径位置打分"""
        path_lower = path.lower()

        if "program files" in path_lower:
            return self.PATH_WEIGHTS["program_files"]
        elif "users" in path_lower and "local" in path_lower:
            return self.PATH_WEIGHTS["user_local"]
        elif "appdata" in path_lower and "roaming" in path_lower:
            return self.PATH_WEIGHTS["appdata_roaming"]
        elif "appdata" in path_lower and "local" in path_lower:
            return self.PATH_WEIGHTS["appdata_local"]
        else:
            return self.PATH_WEIGHTS["other"]

    def _score_filename_match(self, path: str, app_name: str) -> float:
        """根据文件名与软件名匹配度打分"""
        filename = Path(path).stem.lower()
        app_name_lower = app_name.lower().replace(" ", "")

        # 完全匹配
        if filename == app_name_lower:
            return 30.0

        # 包含关系
        if app_name_lower in filename or filename in app_name_lower:
            return 20.0

        # 部分词匹配
        app_words = set(app_name_lower)
        filename_words = set(filename)
        overlap = len(app_words & filename_words)
        if overlap > 0:
            return 10.0 * (overlap / max(len(app_words), 1))

        return 0.0

    def _score_filesize(self, path: str) -> float:
        """根据文件大小打分"""
        try:
            size = os.path.getsize(path)

            # 理想范围：100KB - 50MB
            IDEAL_MIN = 102400  # 100KB
            IDEAL_MAX = 52428800  # 50MB

            if IDEAL_MIN <= size <= IDEAL_MAX:
                return 30.0
            elif size < IDEAL_MIN:
                # 太小，可能是快捷方式或辅助程序
                return max(0, 15.0 * size / IDEAL_MIN)
            else:
                # 太大，可能是安装包
                return max(0, 15.0 - 5.0 * (size - IDEAL_MAX) / IDEAL_MAX)

        except Exception:
            return 0.0

    def _score_suspicious_penalty(self, path: str) -> float:
        """惩罚可疑的文件名"""
        filename_lower = Path(path).stem.lower()

        penalty = 0.0
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in filename_lower:
                penalty += 20.0

        return penalty

    def _load_candidates_from_db(self, app_name: str) -> list[dict]:
        """从数据库加载软件的启动候选"""
        from src.indexer.alias_matcher import AliasMatcher
        from src.storage.db import Session
        from src.storage.repositories import AppRepository, LaunchTargetRepository

        # 1. 别名匹配
        matcher = AliasMatcher()
        alias_result = matcher.match(app_name)

        with Session() as sess:
            repo = AppRepository(sess)
            lt_repo = LaunchTargetRepository(sess)

            # 查找 App
            app = None
            if alias_result:
                app = repo.get_by_canonical_name(alias_result.canonical_name)

            if not app:
                # 模糊搜索
                for a in repo.list_all():
                    if app_name.lower() in a.display_name.lower() or app_name.lower() in a.canonical_name.lower():
                        app = a
                        break

            if not app:
                return []

            # 加载所有启动入口
            targets = lt_repo.get_by_app_id(app.id)
            return [
                {
                    "path": t.path,
                    "args": t.args,
                    "source": t.source,
                    "target_type": t.target_type,
                }
                for t in targets
            ]

    def launch(self, exe_path: str, args: str | None = None) -> dict:
        """
        启动可执行文件

        Args:
            exe_path: exe 路径
            args: 启动参数

        Returns:
            LaunchResult
        """
        from src.common.models import LaunchResult

        try:
            exe_path = expand_path(exe_path)
            if not os.path.isfile(exe_path):
                return LaunchResult(
                    success=False,
                    app_name=Path(exe_path).stem,
                    message=f"文件不存在: {exe_path}"
                ).model_dump()

            # 使用 Windows API 启动
            import subprocess
            full_cmd = f'"{exe_path}"' + (f" {args}" if args else "")
            subprocess.Popen(full_cmd, shell=True)

            return LaunchResult(
                success=True,
                app_name=Path(exe_path).stem,
                message=f"成功启动: {exe_path}",
                hwnd=None  # TODO: 获取窗口句柄
            ).model_dump()

        except Exception as e:
            logger.error(f"启动失败 {exe_path}: {e}")
            return LaunchResult(
                success=False,
                app_name=Path(exe_path).stem if exe_path else "unknown",
                message=f"启动异常: {str(e)}"
            ).model_dump()
