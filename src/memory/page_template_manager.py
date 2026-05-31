"""
页面模板管理器

负责保存、加载和匹配页面模板。
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.storage.db import Session as DBSession
from src.storage.schema import PageTemplate, Candidate


class PageTemplateManager:
    """
    管理页面模板的保存、加载和匹配

    核心流程：
    1. capture() — 从当前感知结果捕获页面结构
    2. save() — 保存模板到数据库
    3. match() — 用当前感知结果匹配已有模板
    4. update() — 更新已有模板
    """

    def capture(
        self,
        app_id: str,
        page_type: str,
        zone_structure: Any,
        screenshot_path: str | None = None,
    ) -> dict[str, Any]:
        """
        从感知结果捕获页面模板

        Args:
            app_id: 应用标识
            page_type: 页面类型（如 "editor", "browser"）
            zone_structure: PerceptionService.analyze() 返回的 ZonePageStructure
            screenshot_path: 截图文件路径

        Returns:
            模板数据字典（含 fingerprint_hash）
        """
        # 生成结构指纹
        elements_data = []
        for zone in zone_structure.zones:
            for elem in zone.elements:
                elements_data.append({
                    "zone": zone.zone_type.value,
                    "type": elem.control_type,
                    "name": elem.name,
                    "rect": elem.bounding_rect,
                })

        structure_json = json.dumps(elements_data, sort_keys=True)
        fingerprint = hashlib.sha256(structure_json.encode()).hexdigest()[:16]

        return {
            "app_id": app_id,
            "page_type": page_type,
            "structure_json": structure_json,
            "fingerprint_hash": fingerprint,
            "screenshot_path": screenshot_path,
            "confidence": 0.5,
            "elements_data": elements_data,
        }

    def save(
        self,
        app_id: str,
        page_type: str,
        zone_structure: Any,
        screenshot_path: str | None = None,
        overwrite: bool = False,
    ) -> PageTemplate:
        """
        保存页面模板

        Args:
            app_id: 应用标识
            page_type: 页面类型
            zone_structure: 感知结果
            screenshot_path: 截图路径
            overwrite: 是否覆盖已有模板

        Returns:
            PageTemplate 实例
        """
        captured = self.capture(app_id, page_type, zone_structure, screenshot_path)

        with DBSession() as s:
            # 检查是否已存在
            existing = s.query(PageTemplate).filter(
                PageTemplate.app_id == app_id,
                PageTemplate.page_type == page_type,
            ).first()

            if existing:
                if overwrite:
                    existing.structure_json = captured["structure_json"]
                    existing.fingerprint_hash = captured["fingerprint_hash"]
                    existing.screenshot_path = screenshot_path
                    existing.confidence = 0.5
                    existing.updated_at = datetime.now(timezone.utc)
                    s.commit()
                    s.refresh(existing)
                    return existing
                return existing

            template = PageTemplate(
                app_id=app_id,
                page_type=page_type,
                structure_json=captured["structure_json"],
                fingerprint_hash=captured["fingerprint_hash"],
                screenshot_path=screenshot_path,
                confidence=0.5,
            )
            s.add(template)
            s.commit()
            s.refresh(template)
            return template

    def match(
        self,
        app_id: str,
        zone_structure: Any,
    ) -> tuple[PageTemplate | None, float]:
        """
        用当前感知结果匹配已有模板

        Args:
            app_id: 应用标识
            zone_structure: 当前感知结果

        Returns:
            (匹配的模板, 置信度) 或 (None, 0.0)
        """
        captured = self.capture(app_id, "", zone_structure)
        current_fingerprint = captured["fingerprint_hash"]

        with DBSession() as s:
            templates = s.query(PageTemplate).filter(
                PageTemplate.app_id == app_id
            ).all()

            if not templates:
                return None, 0.0

            # 精确匹配
            for t in templates:
                if t.fingerprint_hash == current_fingerprint:
                    return t, 1.0

            # 模糊匹配：计算元素重合度
            current_elements = set(
                f"{e['zone']}:{e['type']}:{e['name']}"
                for e in captured["elements_data"]
            )

            best_match = None
            best_score = 0.0

            for t in templates:
                if not t.structure_json:
                    continue
                stored_elements = json.loads(t.structure_json)
                stored_set = set(
                    f"{e['zone']}:{e['type']}:{e.get('name', '')}"
                    for e in stored_elements
                )
                if not stored_set:
                    continue

                # Jaccard 相似度
                intersection = len(current_elements & stored_set)
                union = len(current_elements | stored_set)
                score = intersection / union if union > 0 else 0.0

                if score > best_score:
                    best_score = score
                    best_match = t

            if best_match and best_score >= 0.5:
                # 更新置信度
                best_match.confidence = best_score
                s.commit()
                return best_match, best_score

            return None, 0.0

    def load(self, app_id: str, page_type: str) -> PageTemplate | None:
        """加载指定模板"""
        with DBSession() as s:
            return s.query(PageTemplate).filter(
                PageTemplate.app_id == app_id,
                PageTemplate.page_type == page_type,
            ).first()

    def list_templates(self, app_id: str) -> list[PageTemplate]:
        """列出应用的所有模板"""
        with DBSession() as s:
            return s.query(PageTemplate).filter(
                PageTemplate.app_id == app_id
            ).all()
