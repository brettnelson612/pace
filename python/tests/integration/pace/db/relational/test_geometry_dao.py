"""
tests/integration/pace/db/relational/test_geometry.py

Direct GeometryDAO tests against a real DB — covers the methods
ComponentService doesn't call yet (versions_of_family,
latest_version_of_family, delete, children_of), so they're not
otherwise exercised anywhere.
"""

from __future__ import annotations

import pytest
from pace.core.geometry import GCylinder
from pace.core.ids import GeometryID
from pace.db.pace_db import PaceDB


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


def _make_cylinder(
    family_name: str = "test_cylinder", version_label: str = "1", **overrides
) -> GCylinder:
    kwargs = {"radius_m": 0.005, "height_m": 0.01}
    kwargs.update(overrides)
    return GCylinder.create(
        family_name=family_name, version_label=version_label, **kwargs
    )


def test_save_then_get(pace_db: PaceDB):
    geometry = _make_cylinder()
    pace_db.registry.geometries.save(geometry)
    assert pace_db.registry.geometries.get(geometry.id) == geometry


def test_get_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.geometries.get(GeometryID("nonexistent-1")) is None


def test_exists(pace_db: PaceDB):
    geometry = _make_cylinder()
    assert not pace_db.registry.geometries.exists(geometry.id)
    pace_db.registry.geometries.save(geometry)
    assert pace_db.registry.geometries.exists(geometry.id)


def test_family_exists(pace_db: PaceDB):
    assert not pace_db.registry.geometries.family_exists("test_cylinder")
    pace_db.registry.geometries.save(_make_cylinder())
    assert pace_db.registry.geometries.family_exists("test_cylinder")


def test_versions_of_family_returns_every_version(pace_db: PaceDB):
    v1 = _make_cylinder(version_label="1")
    v2 = _make_cylinder(
        version_label="2", derived_from=v1.id, user_edit=True, radius_m=0.006
    )
    pace_db.registry.geometries.save(v1)
    pace_db.registry.geometries.save(v2)

    versions = pace_db.registry.geometries.versions_of_family("test_cylinder")
    assert {v.id for v in versions} == {v1.id, v2.id}


def test_versions_of_family_empty_when_no_such_family(pace_db: PaceDB):
    assert pace_db.registry.geometries.versions_of_family("nonexistent_family") == []


def test_latest_version_of_family_returns_most_recently_created(pace_db: PaceDB):
    v1 = _make_cylinder(version_label="1")
    pace_db.registry.geometries.save(v1)
    v2 = _make_cylinder(
        version_label="2", derived_from=v1.id, user_edit=True, radius_m=0.006
    )
    pace_db.registry.geometries.save(v2)

    latest = pace_db.registry.geometries.latest_version_of_family("test_cylinder")
    assert latest is not None
    assert latest.id == v2.id


def test_children_of_returns_direct_derivations_only(pace_db: PaceDB):
    v1 = _make_cylinder(version_label="1")
    v2 = _make_cylinder(
        version_label="2", derived_from=v1.id, user_edit=True, radius_m=0.006
    )
    v3 = _make_cylinder(
        version_label="3", derived_from=v2.id, user_edit=True, radius_m=0.007
    )
    pace_db.registry.geometries.save(v1)
    pace_db.registry.geometries.save(v2)
    pace_db.registry.geometries.save(v3)

    # v3 derives from v2, not v1 -- v1's children are just v2.
    children = pace_db.registry.geometries.children_of(v1.id)
    assert {c.id for c in children} == {v2.id}


def test_delete_removes_the_row(pace_db: PaceDB):
    geometry = _make_cylinder()
    pace_db.registry.geometries.save(geometry)
    pace_db.registry.geometries.delete(geometry.id)
    assert pace_db.registry.geometries.get(geometry.id) is None


def test_delete_nonexistent_id_is_a_no_op(pace_db: PaceDB):
    # Must not raise -- ComponentService is responsible for the
    # refcount check before ever calling this.
    pace_db.registry.geometries.delete(GeometryID("nonexistent-1"))


def test_get_many_returns_only_requested_and_existing_ids(pace_db: PaceDB):
    a = _make_cylinder(family_name="a")
    b = _make_cylinder(family_name="b")
    pace_db.registry.geometries.save(a)
    pace_db.registry.geometries.save(b)

    fetched = pace_db.registry.geometries.get_many([a.id, GeometryID("nonexistent-1")])
    assert fetched == {a.id: a}


def test_get_many_empty_list_returns_empty_dict(pace_db: PaceDB):
    assert pace_db.registry.geometries.get_many([]) == {}
