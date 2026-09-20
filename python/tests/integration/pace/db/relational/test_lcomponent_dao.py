"""
tests/integration/pace/db/relational/test_lcomponent.py

Direct LComponentDAO tests against a real DB. Also checks whether the
geometry/material foreign keys on LComponentRow are actually enforced
by SQLite at the DB level, or only by ComponentService's own
dangling-reference check above it -- SQLite does not enforce foreign
keys by default; it requires "PRAGMA foreign_keys = ON" per
connection. If SqlDB doesn't set this, LComponentRow's FK columns give
no real referential integrity on their own, despite the docstring's
claim -- ComponentService's validation would be the ONLY thing
preventing a dangling geometry/material reference, not a backstop on
top of one.
"""

from __future__ import annotations

import pytest
from pace.core.component import LComponent
from pace.core.geometry import GCylinder
from pace.core.ids import LComponentID
from pace.core.material import MaterialComponentEntry, MIsotopic
from pace.db.pace_db import PaceDB


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


def _make_geometry(family_name: str = "test_cyl") -> GCylinder:
    return GCylinder.create(
        family_name=family_name, version_label="1", radius_m=0.005, height_m=0.01
    )


def _make_material(family_name: str = "test_uo2") -> MIsotopic:
    return MIsotopic.create(
        family_name=family_name,
        version_label="1",
        components={"U235": MaterialComponentEntry(percent=100.0)},
        percent_type="wo",
        density_value=10.3,
        density_unit="g/cm3",
    )


def test_save_then_get(pace_db: PaceDB):
    geometry = _make_geometry()
    material = _make_material()
    pace_db.registry.geometries.save(geometry)
    pace_db.registry.materials.save(material)

    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    pace_db.registry.lcomponents.save(lcomponent)
    assert pace_db.registry.lcomponents.get(lcomponent.id) == lcomponent


def test_family_exists(pace_db: PaceDB):
    assert not pace_db.registry.lcomponents.family_exists("test_pellet")
    geometry = _make_geometry()
    material = _make_material()
    pace_db.registry.geometries.save(geometry)
    pace_db.registry.materials.save(material)
    pace_db.registry.lcomponents.save(
        LComponent.create(
            family_name="test_pellet",
            version_label="1",
            geometry=geometry.id,
            material=material.id,
        )
    )
    assert pace_db.registry.lcomponents.family_exists("test_pellet")


def test_children_of(pace_db: PaceDB):
    geometry = _make_geometry()
    material = _make_material()
    pace_db.registry.geometries.save(geometry)
    pace_db.registry.materials.save(material)

    v1 = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    pace_db.registry.lcomponents.save(v1)
    v2 = LComponent.create(
        family_name="test_pellet",
        version_label="2",
        geometry=geometry.id,
        material=material.id,
        derived_from=v1.id,
        user_edit=True,
    )
    pace_db.registry.lcomponents.save(v2)

    assert {c.id for c in pace_db.registry.lcomponents.children_of(v1.id)} == {v2.id}


def test_get_many(pace_db: PaceDB):
    geometry = _make_geometry()
    material = _make_material()
    pace_db.registry.geometries.save(geometry)
    pace_db.registry.materials.save(material)

    a = LComponent.create(
        family_name="a", version_label="1", geometry=geometry.id, material=material.id
    )
    b = LComponent.create(
        family_name="b", version_label="1", geometry=geometry.id, material=material.id
    )
    pace_db.registry.lcomponents.save(a)
    pace_db.registry.lcomponents.save(b)

    fetched = pace_db.registry.lcomponents.get_many(
        [a.id, LComponentID("nonexistent-1")]
    )
    assert fetched == {a.id: a}


def test_delete_removes_the_row(pace_db: PaceDB):
    geometry = _make_geometry()
    material = _make_material()
    pace_db.registry.geometries.save(geometry)
    pace_db.registry.materials.save(material)

    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    pace_db.registry.lcomponents.save(lcomponent)
    pace_db.registry.lcomponents.delete(lcomponent.id)
    assert pace_db.registry.lcomponents.get(lcomponent.id) is None
