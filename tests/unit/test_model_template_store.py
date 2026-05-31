"""ModelTemplateStore tests."""

from __future__ import annotations

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.memory.model_template_store import ModelTemplateStore
from src.storage.schema import AppShellTemplateRecord, Base, ElementTemplateRecord, RegionTemplateRecord


def _session_factory():
    engine = sqlalchemy.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_upsert_workbench_summary_persists_templates():
    Session = _session_factory()
    store = ModelTemplateStore()
    workbench = {
        "app_shell": {"app_id": "wechat", "region_roles": ["navigation"]},
        "region_templates": [
            {"region_id": "left_nav", "role": "navigation", "purpose": "入口", "bounds": [0, 0, 80, 600], "candidate_count": 2},
        ],
        "element_templates": [
            {"key_id": "stable_1", "role": "search_input", "label": "搜索", "region_id": "left_nav", "kind": "fixed", "actionability": "safe", "confidence": 0.9},
        ],
    }

    with Session() as session:
        counts = store.upsert_workbench_summary(
            session,
            app_id="wechat",
            page_model_id="pm_1",
            state_template_id="st_1",
            display_name="微信",
            surface_type="native_uia",
            page_class="wechat/chat",
            state_label="聊天",
            workbench=workbench,
        )
        session.commit()
        assert counts == {"app_shell": 1, "regions": 1, "elements": 1}

        assert session.query(AppShellTemplateRecord).count() == 1
        assert session.query(RegionTemplateRecord).count() == 1
        assert session.query(ElementTemplateRecord).count() == 1

        regions = store.list_region_templates(session, "st_1")
        elements = store.list_element_templates(session, "st_1")
        assert regions[0]["region_id"] == "left_nav"
        assert elements[0]["key_id"] == "stable_1"


def test_upsert_workbench_summary_updates_existing_rows():
    Session = _session_factory()
    store = ModelTemplateStore()

    with Session() as session:
        for label in ["搜索", "搜索框"]:
            store.upsert_workbench_summary(
                session,
                app_id="wechat",
                page_model_id="pm_1",
                state_template_id="st_1",
                display_name="微信",
                surface_type="native_uia",
                page_class="wechat/chat",
                state_label="聊天",
                workbench={
                    "app_shell": {"app_id": "wechat"},
                    "region_templates": [{"region_id": "left_nav", "role": "navigation"}],
                    "element_templates": [{"key_id": "stable_1", "role": "search_input", "label": label}],
                },
            )
        session.commit()

        assert session.query(AppShellTemplateRecord).count() == 1
        assert session.query(RegionTemplateRecord).count() == 1
        assert session.query(ElementTemplateRecord).count() == 1
        assert store.list_element_templates(session, "st_1")[0]["label"] == "搜索框"
