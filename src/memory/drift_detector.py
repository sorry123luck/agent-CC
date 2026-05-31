"""
漂移检测器

检测当前页面结构与已存储模板的偏差。
"""
import json
from dataclasses import dataclass
from typing import Any

from src.storage.db import Session as DBSession
from src.storage.schema import PageTemplate


@dataclass
class DriftReport:
    """漂移报告"""
    drifted: bool
    confidence_drop: float  # 置信度下降
    changed_zones: list[str]  # 发生变化的区域
    new_elements: list[str]  # 新增元素
    missing_elements: list[str]  # 消失元素
    suggestion: str  # 建议（relearn / ignore）


class DriftDetector:
    """
    漂移检测器

    通过比较当前感知结果与已存储模板，检测页面是否发生漂移。
    触发条件：
    - 模板匹配置信度 < 0.5
    - 新增元素比例 > 30%
    - 消失元素比例 > 30%
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        template: PageTemplate,
        current_structure: Any,
    ) -> DriftReport:
        """
        检测漂移

        Args:
            template: 已存储的页面模板
            current_structure: 当前感知结果

        Returns:
            DriftReport 漂移报告
        """
        if not template or not template.structure_json:
            return DriftReport(
                drifted=True,
                confidence_drop=1.0,
                changed_zones=[],
                new_elements=[],
                missing_elements=[],
                suggestion="relearn",
            )

        stored_elements = json.loads(template.structure_json)
        current_elements = self._extract_elements(current_structure)

        stored_keys = set(self._element_key(e) for e in stored_elements)
        current_keys = set(self._element_key(e) for e in current_elements)

        new_keys = current_keys - stored_keys
        missing_keys = stored_keys - current_keys

        # 计算变化比例
        total = max(len(stored_keys), len(current_keys), 1)
        new_ratio = len(new_keys) / total
        missing_ratio = len(missing_keys) / total

        drifted = (
            new_ratio > 0.3 or
            missing_ratio > 0.3
        )

        # 归类变化区域
        stored_by_zone = self._group_by_zone(stored_elements)
        current_by_zone = self._group_by_zone(current_elements)

        changed_zones = []
        for zone in set(stored_by_zone.keys()) | set(current_by_zone.keys()):
            stored_set = stored_by_zone.get(zone, set())
            current_set = current_by_zone.get(zone, set())
            if stored_set != current_set:
                changed_zones.append(zone)

        confidence_drop = template.confidence if template.confidence else 0.0

        if drifted:
            suggestion = "relearn"
        else:
            suggestion = "ignore"

        return DriftReport(
            drifted=drifted,
            confidence_drop=1.0 - confidence_drop,
            changed_zones=changed_zones,
            new_elements=sorted(new_keys),
            missing_elements=sorted(missing_keys),
            suggestion=suggestion,
        )

    def _extract_elements(self, structure: Any) -> list[dict[str, Any]]:
        """从感知结果提取元素列表"""
        elements = []
        for zone in structure.zones:
            for elem in zone.elements:
                elements.append({
                    "zone": zone.zone_type.value,
                    "type": elem.control_type,
                    "name": elem.name or "",
                    "rect": elem.bounding_rect,
                })
        return elements

    def _element_key(self, elem: dict[str, Any]) -> str:
        """生成元素唯一键"""
        return f"{elem['zone']}:{elem['type']}:{elem.get('name', '')}"

    def _group_by_zone(self, elements: list[dict[str, Any]]) -> dict[str, set[str]]:
        """按区域分组元素"""
        groups: dict[str, set[str]] = {}
        for e in elements:
            zone = e["zone"]
            key = self._element_key(e)
            groups.setdefault(zone, set()).add(key)
        return groups
