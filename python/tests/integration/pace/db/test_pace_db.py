"""
tests/integration/pace/db/test_pace_db.py

Schema and wiring sanity for PaceDB/RegistryDB. This is the test that
would have caught the "geometriess" table-name typo earlier — it only
ever surfaced via create_all() actually resolving the FK graph against
a real engine, which no unit test exercises.
"""

from __future__ import annotations

import pytest
from pace.db.pace_db import PaceDB


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


def test_create_all_succeeds(pace_db: PaceDB):
    """The actual regression guard: create_all() must resolve every
    FK target. A stale/mistyped __tablename__ or ForeignKey string
    fails here, not at some unrelated call site later."""
    # create_all() already ran in the fixture; reaching this line
    # without an exception is the test.
    assert pace_db.registry is not None


def test_registry_exposes_all_five_daos(pace_db: PaceDB):
    registry = pace_db.registry
    assert registry.geometries is not None
    assert registry.materials is not None
    assert registry.lcomponents is not None
    assert registry.ccomponents is not None
    assert registry.references is not None


def test_all_daos_share_one_sql_db(pace_db: PaceDB):
    """RegistryDB's DAOs must all be constructed from the same SqlDB
    instance PaceDB owns — not each opening an independent engine
    against the same URL, which would defeat SingletonThreadPool's
    :memory: persistence and silently give each DAO its own empty
    database."""
    registry = pace_db.registry
    engines = {
        registry.geometries._db,
        registry.materials._db,
        registry.lcomponents._db,
        registry.ccomponents._db,
        registry.references._db,
    }
    assert len(engines) == 1
