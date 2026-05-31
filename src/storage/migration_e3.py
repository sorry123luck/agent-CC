"""E Phase 3-lite migration: candidate override overlay."""

from src.storage.db import get_db
from src.storage.schema import CandidateOverrideRecord


def run() -> None:
    """Create candidate override table if it does not already exist."""
    db = get_db()
    CandidateOverrideRecord.__table__.create(db.engine, checkfirst=True)


if __name__ == "__main__":
    run()
