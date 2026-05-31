"""Real-app test fixtures: TestClient, quality report."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.real_app.quality_report import QualityReport

_PROJECT_ROOT = Path(__file__).parents[2]

# Module-level quality report — shared across all tests in the process
_QUALITY_REPORT = QualityReport()

# Gate: real_app tests only run when OPENCLAW_RUN_REAL_APP_TESTS=1
_RUN_REAL = os.environ.get("OPENCLAW_RUN_REAL_APP_TESTS", "0") == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "real_app: mark test as requiring real app interaction")


def pytest_collection_modifyitems(config, items):
    if _RUN_REAL:
        return
    skip_marker = pytest.mark.skip(reason="real_app tests disabled (set OPENCLAW_RUN_REAL_APP_TESTS=1 to enable)")
    for item in items:
        if "real_app" in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture
def client():
    """FastAPI TestClient with in-memory DB (reuses e2e pattern)."""
    import sqlalchemy
    from sqlalchemy.orm import sessionmaker
    from unittest.mock import patch

    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    from src.storage.schema import Base
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    with patch("src.integration.api_server.init_db") as mock_init_db:
        class FakeDB:
            def create_all(self):
                Base.metadata.create_all(engine)
        mock_init_db.return_value = FakeDB()

        with patch("src.storage.db.Session", TestSession):
            from src.integration.api_server import app as fastapi_app
            with TestClient(fastapi_app) as c:
                yield c


@pytest.fixture
def quality_report():
    """Module-level quality report — shared across all tests."""
    yield _QUALITY_REPORT


def pytest_sessionfinish(session, exitstatus):
    """Write quality report after all tests complete."""
    if _QUALITY_REPORT.apps:
        # Assess quality for each app
        for app_name in list(_QUALITY_REPORT.apps.keys()):
            _QUALITY_REPORT.assess_quality(app_name)
        _QUALITY_REPORT.write_json(str(_PROJECT_ROOT / "tests" / "real_app" / "quality_report.json"))
        _QUALITY_REPORT.print_summary()
