"""
Page Model Store — SQLite 持久化 PageModel、StateTemplate、CanvasSnapshot。

E Phase 2 核心模块：管理页面级持久身份。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from src.memory.page_identity import (
    CanvasSnapshotRef,
    LayoutFingerprint,
    PageModelData,
    StateTemplateData,
    WORKFLOW_DISPLAY,
    compute_fingerprint_similarity,
    fingerprint_to_signature,
    get_app_display_name,
    infer_app_from_process,
)
from src.storage.schema import (
    CanvasSnapshotRecord,
    CandidateStateRecord,
    PageModelRecord,
    StateTemplateRecord,
)

# StateTemplate 匹配阈值
# similarity >= STATE_SIM_THRESHOLD → 复用（同一 StateTemplate）
# similarity < STATE_SIM_THRESHOLD → 创建新 StateTemplate
# 注：不再使用 merge 行为，因为用户要求状态变化（如输入框空/有内容、菜单开/关）
# 必须创建不同的 StateTemplate。
STATE_SIM_THRESHOLD = 0.80


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PageModelStore:
    """页面模型持久存储。"""

    # =========================================================================
    # PageModel
    # =========================================================================

    def find_or_create_page_model(
        self,
        session,
        app_id: str,
        page_class: str,
        process_name: str | None = None,
        surface_type: str | None = None,
        layout_signature: str | None = None,
        window_title: str | None = None,
        *,
        vlm_app_name: str | None = None,
        vlm_display_name: str | None = None,
    ) -> tuple[PageModelData, str]:
        """查找或创建 PageModel。

        匹配链：
        1. app_id + page_class_prefix 精确匹配
        2. app_id + surface_type + layout_signature 相似匹配（兜底 unknown）

        当 app_id 为 "unknown" 时，尝试从 process_name 或 vlm_app_name 推断真实 app_id。

        Returns:
            (PageModelData, "reused" | "new")
        """
        from src.memory.page_identity import extract_page_class_prefix

        # 当 app_id 为 unknown 时，尝试从 process_name 或 vlm_app_name 推断
        effective_app_id = app_id
        if app_id in ("unknown", "") and process_name:
            inferred = infer_app_from_process(process_name)
            if inferred:
                effective_app_id = inferred[0]
        # VLM app_name 兜底：process_name 无法推断时使用
        if effective_app_id in ("unknown", "") and vlm_app_name:
            effective_app_id = vlm_app_name.lower().replace(" ", "_")

        prefix = extract_page_class_prefix(page_class, process_name, surface_type)

        # 如果 prefix 以 unknown/ 开头且已推断出 app_id，重建 prefix
        if prefix.startswith("unknown/") and effective_app_id != "unknown":
            workflow_part = prefix.split("/")[1] if "/" in prefix else "gui"
            prefix = f"{effective_app_id}/{workflow_part}"

        # 1. 精确匹配：app_id + page_class_prefix
        row = session.query(PageModelRecord).filter(
            PageModelRecord.app_id == effective_app_id,
            PageModelRecord.page_class_prefix == prefix,
        ).first()

        if row:
            row.last_seen_at = _now_iso()
            row.observe_count = (row.observe_count or 0) + 1
            session.flush()
            return self._row_to_page_model(row), "reused"

        # 2. 兜底：app_id + surface_type + layout_signature（仅 unknown 时）
        if layout_signature and page_class.startswith("unknown"):
            fallback = session.query(PageModelRecord).filter(
                PageModelRecord.app_id == effective_app_id,
                PageModelRecord.surface_type == surface_type,
                PageModelRecord.layout_signature == layout_signature,
            ).first()

            if fallback:
                fallback.last_seen_at = _now_iso()
                fallback.observe_count = (fallback.observe_count or 0) + 1
                session.flush()
                return self._row_to_page_model(fallback), "reused"

        # 3. 创建新 PageModel
        now = _now_iso()
        model_id = str(uuid.uuid4())
        display_name = vlm_display_name or _guess_display_name(effective_app_id, prefix, process_name)

        new_row = PageModelRecord(
            page_model_id=model_id,
            app_id=effective_app_id,
            page_class_prefix=prefix,
            display_name=display_name,
            surface_type=surface_type,
            layout_signature=layout_signature,
            created_at=now,
            last_seen_at=now,
            observe_count=1,
            state_count=0,
        )
        session.add(new_row)
        session.flush()
        return self._row_to_page_model(new_row), "new"

    # =========================================================================
    # StateTemplate
    # =========================================================================

    def find_or_create_state_template(
        self,
        session,
        page_model_id: str,
        app_id: str,
        page_class: str,
        fingerprint: LayoutFingerprint,
        total_element_count: int,
        fixed_element_count: int,
        state_label: str | None = None,
        *,
        vlm_state_label: str | None = None,
        vlm_state_flags: list[str] | None = None,
    ) -> tuple[StateTemplateData, str]:
        """查找或创建 StateTemplate。

        匹配链：
        1. state_signature exact match → "reused"
        2. layout_fingerprint similarity >= 0.80 → "reused"
        3. similarity < 0.80 → "new"（创建新 StateTemplate）

        注：不再使用 merge 行为，因为用户要求状态变化（如输入框空/有内容、菜单开/关）
        必须创建不同的 StateTemplate。

        Returns:
            (StateTemplateData, "reused" | "new")
        """
        sig = fingerprint_to_signature(fingerprint)
        fp_dict = {
            "fixed_roles": list(fingerprint.fixed_roles),
            "region_structure": [list(r) for r in fingerprint.region_structure],
            "state_flags": [list(f) for f in fingerprint.state_flags],
            "element_count_bucket": fingerprint.element_count_bucket,
            "input_state": fingerprint.input_state,
            "has_dialog": fingerprint.has_dialog,
            "has_menu": fingerprint.has_menu,
        }
        if vlm_state_flags:
            fp_dict["vlm_state_flags"] = vlm_state_flags
        fp_json = json.dumps(fp_dict, sort_keys=True, ensure_ascii=False)

        # 1. exact match
        exact = session.query(StateTemplateRecord).filter(
            StateTemplateRecord.page_model_id == page_model_id,
            StateTemplateRecord.state_signature == sig,
        ).first()

        if exact:
            exact.last_seen_at = _now_iso()
            exact.verify_count = (exact.verify_count or 0) + 1
            exact.total_element_count = max(exact.total_element_count or 0, total_element_count)
            exact.fixed_element_count = max(exact.fixed_element_count or 0, fixed_element_count)
            if vlm_state_label and not exact.state_label:
                exact.state_label = vlm_state_label
            session.flush()
            return self._row_to_state_template(exact), "reused"

        # 2. similarity match
        candidates = session.query(StateTemplateRecord).filter(
            StateTemplateRecord.page_model_id == page_model_id,
        ).all()

        best_row = None
        best_sim = 0.0
        for row in candidates:
            existing_fp = self._parse_fingerprint(row.layout_fingerprint)
            if existing_fp:
                sim = compute_fingerprint_similarity(fingerprint, existing_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row

        if best_row and best_sim >= STATE_SIM_THRESHOLD:
            best_row.last_seen_at = _now_iso()
            best_row.verify_count = (best_row.verify_count or 0) + 1
            best_row.total_element_count = max(best_row.total_element_count or 0, total_element_count)
            best_row.fixed_element_count = max(best_row.fixed_element_count or 0, fixed_element_count)
            if vlm_state_label and not best_row.state_label:
                best_row.state_label = vlm_state_label
            session.flush()
            return self._row_to_state_template(best_row), "reused"

        # 3. 创建新 StateTemplate
        now = _now_iso()
        st_id = str(uuid.uuid4())
        effective_state_label = vlm_state_label or state_label

        new_row = StateTemplateRecord(
            state_template_id=st_id,
            page_model_id=page_model_id,
            app_id=app_id,
            page_class=page_class,
            state_signature=sig,
            layout_fingerprint=fp_json,
            state_label=effective_state_label,
            fixed_element_count=fixed_element_count,
            total_element_count=total_element_count,
            created_at=now,
            last_seen_at=now,
            verify_count=1,
            snapshot_count=0,
            max_snapshots=10,
        )
        session.add(new_row)
        session.flush()

        # 更新 PageModel 的 state_count
        pm = session.query(PageModelRecord).filter(
            PageModelRecord.page_model_id == page_model_id,
        ).first()
        if pm:
            pm.state_count = (pm.state_count or 0) + 1
            session.flush()

        return self._row_to_state_template(new_row), "new"

    # =========================================================================
    # CanvasSnapshot
    # =========================================================================

    def record_canvas_snapshot(
        self,
        session,
        canvas_id: str,
        page_model_id: str,
        state_template_id: str,
        element_count: int,
        has_screenshot: bool,
    ) -> CanvasSnapshotRef:
        """记录画布快照归属。

        自动执行保留策略：每个 StateTemplate 最多保留 max_snapshots 个。
        """
        now = _now_iso()
        existing = session.query(CanvasSnapshotRecord).filter(
            CanvasSnapshotRecord.canvas_id == canvas_id,
        ).first()
        if existing is not None:
            old_state_template_id = existing.state_template_id
            existing.page_model_id = page_model_id
            existing.state_template_id = state_template_id
            existing.captured_at = now
            existing.element_count = element_count
            existing.has_screenshot = 1 if has_screenshot else 0
            session.flush()

            for affected_state_id in {old_state_template_id, state_template_id}:
                if not affected_state_id:
                    continue
                st = session.query(StateTemplateRecord).filter(
                    StateTemplateRecord.state_template_id == affected_state_id,
                ).first()
                if st:
                    st.snapshot_count = session.query(CanvasSnapshotRecord).filter(
                        CanvasSnapshotRecord.state_template_id == affected_state_id,
                    ).count()
                    session.flush()

            return CanvasSnapshotRef(
                snapshot_id=existing.snapshot_id,
                canvas_id=canvas_id,
                page_model_id=page_model_id,
                state_template_id=state_template_id,
                captured_at=now,
                element_count=element_count,
                has_screenshot=has_screenshot,
            )

        snapshot_id = str(uuid.uuid4())
        new_row = CanvasSnapshotRecord(
            snapshot_id=snapshot_id,
            canvas_id=canvas_id,
            page_model_id=page_model_id,
            state_template_id=state_template_id,
            captured_at=now,
            element_count=element_count,
            has_screenshot=1 if has_screenshot else 0,
        )
        session.add(new_row)
        session.flush()  # check constraints early; commit happens when caller's Session() exits

        # 更新 StateTemplate 的 snapshot_count
        st = session.query(StateTemplateRecord).filter(
            StateTemplateRecord.state_template_id == state_template_id,
        ).first()
        if st:
            st.snapshot_count = (st.snapshot_count or 0) + 1
            session.flush()

            # 保留策略：删除超出 max_snapshots 的最旧记录
            max_n = st.max_snapshots or 10
            excess = session.query(CanvasSnapshotRecord).filter(
                CanvasSnapshotRecord.state_template_id == state_template_id,
            ).order_by(
                CanvasSnapshotRecord.captured_at.asc(),
            ).offset(max_n).all()

            for old in excess:
                session.delete(old)

        session.flush()

        return CanvasSnapshotRef(
            snapshot_id=snapshot_id,
            canvas_id=canvas_id,
            page_model_id=page_model_id,
            state_template_id=state_template_id,
            captured_at=now,
            element_count=element_count,
            has_screenshot=has_screenshot,
        )

    # =========================================================================
    # CandidateState
    # =========================================================================

    def record_candidate_state(
        self,
        session,
        key_id: str,
        state_template_id: str,
        page_model_id: str,
    ) -> None:
        """记录候选-状态关联（upsert）。"""
        now = _now_iso()

        existing = session.query(CandidateStateRecord).filter(
            CandidateStateRecord.key_id == key_id,
            CandidateStateRecord.state_template_id == state_template_id,
        ).first()

        if existing:
            existing.last_seen_at = now
            existing.seen_count = (existing.seen_count or 0) + 1
        else:
            session.add(CandidateStateRecord(
                key_id=key_id,
                state_template_id=state_template_id,
                page_model_id=page_model_id,
                first_seen_at=now,
                last_seen_at=now,
                seen_count=1,
            ))

        session.flush()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_page_models(
        self, session, app_id: str | None = None,
    ) -> list[PageModelData]:
        """获取所有 PageModel。"""
        q = session.query(PageModelRecord)
        if app_id:
            q = q.filter(PageModelRecord.app_id == app_id)
        rows = q.order_by(PageModelRecord.last_seen_at.desc()).all()
        return [self._row_to_page_model(r) for r in rows]

    def get_state_templates(
        self, session, page_model_id: str,
    ) -> list[StateTemplateData]:
        """获取某个 PageModel 的所有 StateTemplate。"""
        rows = session.query(StateTemplateRecord).filter(
            StateTemplateRecord.page_model_id == page_model_id,
        ).order_by(StateTemplateRecord.last_seen_at.desc()).all()
        return [self._row_to_state_template(r) for r in rows]

    def get_state_template_by_id(
        self, session, state_template_id: str,
    ) -> StateTemplateData | None:
        """通过 state_template_id 获取单个 StateTemplate。"""
        row = session.query(StateTemplateRecord).filter(
            StateTemplateRecord.state_template_id == state_template_id,
        ).first()
        return self._row_to_state_template(row) if row else None

    def get_canvas_snapshots(
        self,
        session,
        state_template_id: str | None = None,
        page_model_id: str | None = None,
    ) -> list[CanvasSnapshotRef]:
        """获取画布快照列表。"""
        q = session.query(CanvasSnapshotRecord)
        if state_template_id:
            q = q.filter(CanvasSnapshotRecord.state_template_id == state_template_id)
        if page_model_id:
            q = q.filter(CanvasSnapshotRecord.page_model_id == page_model_id)
        rows = q.order_by(CanvasSnapshotRecord.captured_at.desc()).all()
        return [self._row_to_snapshot(r) for r in rows]

    def get_candidate_states(
        self, session, key_id: str,
    ) -> list[str]:
        """获取某个候选出现过的所有 state_template_id。"""
        rows = session.query(CandidateStateRecord).filter(
            CandidateStateRecord.key_id == key_id,
        ).all()
        return [r.state_template_id for r in rows]

    # =========================================================================
    # Internal Converters
    # =========================================================================

    def _row_to_page_model(self, row: PageModelRecord) -> PageModelData:
        return PageModelData(
            page_model_id=row.page_model_id,
            app_id=row.app_id,
            page_class_prefix=row.page_class_prefix,
            display_name=row.display_name,
            surface_type=row.surface_type,
            layout_signature=row.layout_signature,
            created_at=row.created_at,
            last_seen_at=row.last_seen_at,
            observe_count=row.observe_count or 0,
            state_count=row.state_count or 0,
        )

    def _row_to_state_template(self, row: StateTemplateRecord) -> StateTemplateData:
        return StateTemplateData(
            state_template_id=row.state_template_id,
            page_model_id=row.page_model_id,
            app_id=row.app_id,
            page_class=row.page_class,
            state_signature=row.state_signature,
            layout_fingerprint=row.layout_fingerprint,
            state_label=row.state_label,
            fixed_element_count=row.fixed_element_count or 0,
            total_element_count=row.total_element_count or 0,
            created_at=row.created_at,
            last_seen_at=row.last_seen_at,
            verify_count=row.verify_count or 0,
            snapshot_count=row.snapshot_count or 0,
            max_snapshots=row.max_snapshots or 10,
        )

    def _row_to_snapshot(self, row: CanvasSnapshotRecord) -> CanvasSnapshotRef:
        return CanvasSnapshotRef(
            snapshot_id=row.snapshot_id,
            canvas_id=row.canvas_id,
            page_model_id=row.page_model_id or "",
            state_template_id=row.state_template_id or "",
            captured_at=row.captured_at,
            element_count=row.element_count or 0,
            has_screenshot=bool(row.has_screenshot),
        )

    def _parse_fingerprint(self, fp_json: str) -> LayoutFingerprint | None:
        """从 JSON 解析 LayoutFingerprint。"""
        try:
            data = json.loads(fp_json)
            return LayoutFingerprint(
                fixed_roles=tuple(data.get("fixed_roles", [])),
                region_structure=tuple(tuple(r) for r in data.get("region_structure", [])),
                state_flags=tuple(tuple(f) for f in data.get("state_flags", [])),
                element_count_bucket=data.get("element_count_bucket", "unknown"),
                input_state=data.get("input_state", "unknown"),
                has_dialog=data.get("has_dialog", False),
                has_menu=data.get("has_menu", False),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    # =========================================================================
    # Delete
    # =========================================================================

    def delete_page_model(self, session, page_model_id: str) -> bool:
        """删除 PageModel 及其所有关联数据（级联删除 StateTemplate、CanvasSnapshot、CandidateState）。

        Returns:
            True if deleted, False if not found.
        """
        pm = session.query(PageModelRecord).filter_by(page_model_id=page_model_id).first()
        if not pm:
            return False

        # 删除关联的 CandidateState
        st_ids = [
            st.state_template_id
            for st in session.query(StateTemplateRecord).filter_by(page_model_id=page_model_id).all()
        ]
        if st_ids:
            session.query(CandidateStateRecord).filter(
                CandidateStateRecord.state_template_id.in_(st_ids)
            ).delete(synchronize_session=False)

        # 删除关联的 CanvasSnapshot
        session.query(CanvasSnapshotRecord).filter_by(page_model_id=page_model_id).delete()

        # 删除关联的 StateTemplate
        session.query(StateTemplateRecord).filter_by(page_model_id=page_model_id).delete()

        # 删除 PageModel
        session.delete(pm)
        session.flush()
        return True

    def delete_state_template(self, session, state_template_id: str) -> tuple[bool, str | None, list[str]]:
        """删除单个 StateTemplate 及其 CanvasSnapshot/CandidateState 关联。"""
        st = session.query(StateTemplateRecord).filter_by(state_template_id=state_template_id).first()
        if not st:
            return False, None, []

        page_model_id = st.page_model_id
        canvas_ids = [
            row.canvas_id
            for row in session.query(CanvasSnapshotRecord.canvas_id).filter_by(
                state_template_id=state_template_id,
            ).all()
        ]

        session.query(CandidateStateRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)
        session.query(CanvasSnapshotRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)
        session.delete(st)
        session.flush()

        pm = session.query(PageModelRecord).filter_by(page_model_id=page_model_id).first()
        if pm:
            pm.state_count = session.query(StateTemplateRecord).filter_by(
                page_model_id=page_model_id,
            ).count()
            if pm.state_count == 0:
                session.delete(pm)
            session.flush()

        return True, page_model_id, canvas_ids


# =============================================================================
# Helpers
# =============================================================================


def _guess_display_name(
    app_id: str,
    prefix: str,
    process_name: str | None = None,
) -> str | None:
    """从 app_id、prefix 和 process_name 推断中文显示名。

    格式：{app_display_name}{workflow_display_name}
    例如：微信聊天页面、记事本文档编辑页、Chrome网页、codex主界面
    """
    workflow = prefix.split("/")[1] if "/" in prefix else ""
    workflow_name = WORKFLOW_DISPLAY.get(workflow, "")
    app_name = get_app_display_name(app_id, process_name)

    if app_name and workflow_name:
        return f"{app_name}{workflow_name}"
    elif app_name:
        return app_name
    elif workflow_name:
        return workflow_name
    # 最终兜底：用 app_id
    return app_id if app_id else None
