"""
tests/integration/conftest.py

Shared fixtures for the integration tests: a fresh in-memory PaceDB per
test, and `insert`, which writes one object through a DAO inside its
own transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pace.db.pace_db import PaceDB
from pace.db.relational.versioned_dao import VersionedDAO


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


@pytest.fixture
def insert(pace_db: PaceDB) -> Callable[[VersionedDAO[Any, Any, Any], Any], None]:
    def _insert(dao: VersionedDAO[Any, Any, Any], model: Any) -> None:
        with pace_db.registry.transaction() as session:
            dao.insert(session, model)

    return _insert
