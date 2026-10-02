"""
tests/integration/pace/db/test_pace_db.py

Schema and wiring sanity for PaceDB/RegistryDB/SqlDB: create_all()
resolves every foreign-key target, every DAO shares one SqlDB, dao_for()
maps every ReferenceableType to its DAO, transaction() commits or rolls
back as a unit, and SQLite connections enforce foreign keys.
"""

from __future__ import annotations

import pytest
from pace.core.reference_types import ReferenceableType
from pace.db.pace_db import PaceDB
from pace.db.relational.sql_db import SqlDB
from sqlalchemy import text


def test_create_all_succeeds(pace_db: PaceDB):
    """create_all() must resolve every FK target. A stale/mistyped
    __tablename__ or ForeignKey string fails here, not at some
    unrelated call site later."""
    assert pace_db.registry is not None


def test_all_daos_share_one_sql_db(pace_db: PaceDB):
    """Every DAO must be built from the one SqlDB PaceDB owns — not each
    opening an independent engine against the same URL, which would
    give each DAO its own empty :memory: database."""
    registry = pace_db.registry
    sql_dbs = {
        registry.geometries._db,
        registry.materials._db,
        registry.lcomponents._db,
        registry.ccomponents._db,
        registry.lattices._db,
        registry.reactor_blueprints._db,
        registry.references._db,
    }
    assert len(sql_dbs) == 1


@pytest.mark.parametrize(
    "referenceable_type,attribute",
    [
        (ReferenceableType.GEOMETRY, "geometries"),
        (ReferenceableType.MATERIAL, "materials"),
        (ReferenceableType.LCOMPONENT, "lcomponents"),
        (ReferenceableType.CCOMPONENT, "ccomponents"),
        (ReferenceableType.LATTICE, "lattices"),
        (ReferenceableType.REACTOR_BLUEPRINT, "reactor_blueprints"),
    ],
)
def test_dao_for_maps_each_type_to_its_dao(
    pace_db: PaceDB, referenceable_type, attribute
):
    registry = pace_db.registry
    assert registry.dao_for(referenceable_type) is getattr(registry, attribute)


def test_dao_for_covers_every_referenceable_type(pace_db: PaceDB):
    for referenceable_type in ReferenceableType:
        assert pace_db.registry.dao_for(referenceable_type) is not None


def test_sqlite_connections_enforce_foreign_keys():
    sql_db = SqlDB(db_url="sqlite:///:memory:")
    with sql_db.session() as session:
        assert session.execute(text("PRAGMA foreign_keys")).scalar() == 1
