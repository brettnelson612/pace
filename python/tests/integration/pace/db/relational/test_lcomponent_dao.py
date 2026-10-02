"""
tests/integration/pace/db/relational/test_lcomponent_dao.py

LComponentDAO-specific behaviour against a real DB: the geometry and
material foreign-key columns are written on insert, reported by
describe(), and enforced by SQLite (PRAGMA foreign_keys=ON, set by
SqlDB) — a row can't point at a missing Geometry/Material, and a
referenced Geometry/Material can't be deleted. The generic DAO methods
are covered in test_versioned_dao.py.
"""

from __future__ import annotations

import pytest
from pace.core.component import LComponent
from pace.core.geometry import GCylinder
from pace.core.ids import GeometryID, MaterialID
from pace.core.material import (
    DensityUnit,
    MaterialComponentEntry,
    MIsotopic,
    PercentType,
)
from pace.db.pace_db import PaceDB
from sqlalchemy.exc import IntegrityError


def _geometry() -> GCylinder:
    return GCylinder.create(
        family_name="test_cyl", version_label="1", radius_m=0.005, height_m=0.01
    )


def _material() -> MIsotopic:
    return MIsotopic.create(
        family_name="test_uo2",
        version_label="1",
        components={"U235": MaterialComponentEntry(percent=100.0)},
        percent_type=PercentType.WO,
        density_value=10.3,
        density_unit=DensityUnit.G_PER_CM3,
    )


def _pellet(geometry_id: str = "test_cyl-1", material_id: str = "test_uo2-1"):
    return LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=GeometryID(geometry_id),
        material=MaterialID(material_id),
    )


@pytest.fixture
def registered_parts(pace_db: PaceDB, insert) -> None:
    insert(pace_db.registry.geometries, _geometry())
    insert(pace_db.registry.materials, _material())


def test_insert_then_get(pace_db: PaceDB, insert, registered_parts):
    pellet = _pellet()
    insert(pace_db.registry.lcomponents, pellet)
    fetched = pace_db.registry.lcomponents.get(pellet.id)
    assert fetched is not None
    assert fetched.to_dict() == pellet.to_dict()


def test_describe_includes_geometry_and_material_columns(
    pace_db: PaceDB, insert, registered_parts
):
    pellet = _pellet()
    insert(pace_db.registry.lcomponents, pellet)
    described = pace_db.registry.lcomponents.describe(pellet.id)
    assert described is not None
    assert described["geometry"] == "test_cyl-1"
    assert described["material"] == "test_uo2-1"


@pytest.mark.parametrize(
    "geometry_id,material_id",
    [("missing_cyl-1", "test_uo2-1"), ("test_cyl-1", "missing_uo2-1")],
)
def test_insert_pointing_at_a_missing_part_violates_foreign_key(
    pace_db: PaceDB, insert, registered_parts, geometry_id, material_id
):
    pellet = _pellet(geometry_id=geometry_id, material_id=material_id)
    with pytest.raises(IntegrityError):
        insert(pace_db.registry.lcomponents, pellet)
    assert not pace_db.registry.lcomponents.exists(pellet.id)


def test_deleting_a_referenced_geometry_violates_foreign_key(
    pace_db: PaceDB, insert, registered_parts
):
    insert(pace_db.registry.lcomponents, _pellet())
    with pytest.raises(IntegrityError):
        with pace_db.registry.transaction() as session:
            pace_db.registry.geometries.delete(session, GeometryID("test_cyl-1"))
    assert pace_db.registry.geometries.exists(GeometryID("test_cyl-1"))
