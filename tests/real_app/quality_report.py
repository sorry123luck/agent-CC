"""Quality report generator for real-app observe/query validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_TEXT_PREVIEW_MAX = 50


def _sanitize_text(text: str) -> str:
    """Truncate and sanitize text for quality report privacy.

    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single space
    - Truncate to _TEXT_PREVIEW_MAX chars
    - Replace potentially sensitive patterns (URLs, paths, long numbers)
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Mask URLs
    text = re.sub(r"https?://\S+", "[URL]", text)
    # Mask file paths
    text = re.sub(r"[A-Za-z]:\\[\w\\.\\]+", "[PATH]", text)
    if len(text) > _TEXT_PREVIEW_MAX:
        text = text[:_TEXT_PREVIEW_MAX] + "..."
    return text


@dataclass
class ObserveResult:
    status: str = "not_run"  # success / failed / skipped
    surface_type: str = ""
    page_class: str = ""
    element_count: int = 0
    region_count: int = 0
    providers_used: list[str] = field(default_factory=list)
    canvas_id: str = ""
    error: str | None = None


@dataclass
class QueryResult:
    found: bool = False
    count: int = 0
    text_preview: str = ""
    top_role: str = ""
    error: str | None = None


@dataclass
class AppQuality:
    name: str
    observe: ObserveResult = field(default_factory=ObserveResult)
    queries: dict[str, QueryResult] = field(default_factory=dict)
    found_roles: list[str] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    quality: str = "unknown"  # strong / adequate / weak / failed
    issues: list[str] = field(default_factory=list)


class QualityReport:
    """Collect and export real-app quality data."""

    def __init__(self) -> None:
        self.apps: dict[str, AppQuality] = {}
        self._timestamp = datetime.now(timezone.utc).isoformat()

    def record_observe(self, app_name: str, result: ObserveResult) -> None:
        if app_name not in self.apps:
            self.apps[app_name] = AppQuality(name=app_name)
        self.apps[app_name].observe = result

    def record_query(self, app_name: str, query_text: str, result: QueryResult) -> None:
        if app_name not in self.apps:
            self.apps[app_name] = AppQuality(name=app_name)
        self.apps[app_name].queries[query_text] = result

    def record_key_elements(self, app_name: str, found: set[str], required_roles: list[str]) -> None:
        if app_name not in self.apps:
            self.apps[app_name] = AppQuality(name=app_name)
        app = self.apps[app_name]
        for role in required_roles:
            if role in found:
                app.found_roles.append(role)
            else:
                app.missing_roles.append(role)
                app.issues.append(f"Required role '{role}' not found in observe output")

    def assess_quality(self, app_name: str) -> None:
        if app_name not in self.apps:
            return
        app = self.apps[app_name]

        if app.observe.status != "success":
            app.quality = "failed"
            return

        missing_required = len(app.missing_roles)
        found_queries = sum(1 for q in app.queries.values() if q.found)
        total_queries = len(app.queries)

        if missing_required == 0 and total_queries > 0 and found_queries >= total_queries * 0.5:
            app.quality = "strong"
        elif missing_required <= 1 and found_queries >= 1:
            app.quality = "adequate"
        elif app.observe.element_count > 0:
            app.quality = "weak"
        else:
            app.quality = "failed"

        # Add issues for weak/failed
        if app.quality == "weak":
            if missing_required > 0:
                app.issues.append(f"{missing_required} required roles missing")
            if found_queries == 0:
                app.issues.append("No queries returned candidates")

    def to_dict(self) -> dict:
        apps_dict = {}
        observe_success = 0
        query_success = 0
        weak_count = 0

        for name, app in self.apps.items():
            if app.observe.status == "success":
                observe_success += 1
            if any(q.found for q in app.queries.values()):
                query_success += 1
            if app.quality in ("weak", "failed"):
                weak_count += 1

            apps_dict[name] = {
                "observe": {
                    "status": app.observe.status,
                    "surface_type": app.observe.surface_type,
                    "page_class": app.observe.page_class,
                    "element_count": app.observe.element_count,
                    "region_count": app.observe.region_count,
                    "providers_used": app.observe.providers_used,
                    "error": app.observe.error,
                },
                "queries": {
                    qt: {
                        "found": q.found,
                        "count": q.count,
                        "text_preview": _sanitize_text(q.text_preview),
                        "top_role": q.top_role,
                        "error": q.error,
                    }
                    for qt, q in app.queries.items()
                },
                "key_elements": {
                    **{r: "found" for r in app.found_roles},
                    **{r: "missing" for r in app.missing_roles},
                },
                "quality": app.quality,
                "issues": app.issues,
            }

        return {
            "timestamp": self._timestamp,
            "apps": apps_dict,
            "summary": {
                "total": len(self.apps),
                "observe_success": observe_success,
                "query_success": query_success,
                "weak": weak_count,
            },
        }

    def write_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def print_summary(self) -> None:
        d = self.to_dict()
        s = d["summary"]
        print(f"\n{'='*60}")
        print(f"识别质量报告 — {d['timestamp']}")
        print(f"{'='*60}")
        print(f"总计: {s['total']} apps | observe 成功: {s['observe_success']} | query 成功: {s['query_success']} | 弱/失败: {s['weak']}")
        print(f"{'-'*60}")
        for name, app_data in d["apps"].items():
            obs = app_data["observe"]
            q_count = sum(1 for q in app_data["queries"].values() if q["found"])
            print(f"  {name}: observe={obs['status']}, surface={obs['surface_type']}, "
                  f"elements={obs['element_count']}, queries_found={q_count}/{len(app_data['queries'])}, "
                  f"quality={app_data['quality']}")
            if app_data["issues"]:
                for issue in app_data["issues"]:
                    print(f"    ⚠ {issue}")
        print(f"{'='*60}\n")
