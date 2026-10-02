"""
tests/integration/pace/db/relational/test_versioned_dao.py

VersionedDAO behaviour against a real (in-memory) SQLite DB, exercised
through GeometryDAO: insert/get, insert never overwriting an existing
id, the self-referential derived_from foreign key, existing_ids,
describe, version queries, children_of, delete, and transaction
rollback. Ends with a smoke test that every concrete DAO round-trips
its own type.
"""

from __future__ import annotations

import pytest
from pace.core.boundary_conditions import NeutronBC
from pace.core.bounds import faces_for_geometry_type
from pace.core.component import (
    CComponent,
    ComponentPlacement,
    ComponentRef,
    ComponentType,
    LComponent,
)
from pace.core.geometry import GCylinder, GeometryType, GPose
from pace.core.ids import CComponentID, ComponentPlacementID, GeometryID
from pace.core.lattice import LatticeElement, RectLattice
from pace.core.material import (
    DensityUnit,
    MaterialComponentEntry,
    MIsotopic,
    PercentType,
)
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.db.pace_db import PaceDB
from sqlalchemy.exc import IntegrityError


def _cylinder(
    family_name: str = "test_cylinder", version_label: str = "1", **overrides
) -> GCylinder:
    kwargs = {"radius_m": 0.005, "height_m": 0.01}
    kwargs.update(overrides)
    return GCylinder.create(
        family_name=family_name, version_label=version_label, **kwargs
    )


def _v2(v1: GCylinder, version_label: str = "2", radius_m: float = 0.006):
    return _cylinder(
        version_label=version_label,
        derived_from=v1.id,
        user_edit=True,
        radius_m=radius_m,
    )


# =============================================================================
# insert / get
# =============================================================================


def test_insert_then_get(pace_db: PaceDB, insert):
    geometry = _cylinder()
    insert(pace_db.registry.geometries, geometry)
    fetched = pace_db.registry.geometries.get(geometry.id)
    assert fetched is not None
    assert fetched.to_dict() == geometry.to_dict()


def test_get_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.geometries.get(GeometryID("nonexistent-1")) is None


def test_insert_never_overwrites_an_existing_id(pace_db: PaceDB, insert):
    original = _cylinder(radius_m=0.005)
    insert(pace_db.registry.geometries, original)
    with pytest.raises(IntegrityError):
        insert(pace_db.registry.geometries, _cylinder(radius_m=0.009))
    fetched = pace_db.registry.geometries.get(original.id)
    assert fetched is not None
    assert fetched.radius_m == 0.005


def test_insert_with_missing_predecessor_violates_foreign_key(pace_db: PaceDB, insert):
    orphan = _cylinder(
        version_label="2", derived_from=GeometryID("missing-1"), user_edit=True
    )
    with pytest.raises(IntegrityError):
        insert(pace_db.registry.geometries, orphan)
    assert not pace_db.registry.geometries.exists(orphan.id)


def test_insert_revalidates_the_object(pace_db: PaceDB, insert):
    geometry = _cylinder()
    object.__setattr__(geometry, "radius_m", -1.0)
    with pytest.raises(ValueError):
        insert(pace_db.registry.geometries, geometry)


def test_failed_transaction_writes_nothing(pace_db: PaceDB):
    geometry = _cylinder()
    with pytest.raises(RuntimeError):
        with pace_db.registry.transaction() as session:
            pace_db.registry.geometries.insert(session, geometry)
            raise RuntimeError("abort")
    assert not pace_db.registry.geometries.exists(geometry.id)


# =============================================================================
# Lookups
# =============================================================================


def test_exists(pace_db: PaceDB, insert):
    geometry = _cylinder()
    assert not pace_db.registry.geometries.exists(geometry.id)
    insert(pace_db.registry.geometries, geometry)
    assert pace_db.registry.geometries.exists(geometry.id)


def test_existing_ids_returns_the_registered_subset(pace_db: PaceDB, insert):
    a, b = _cylinder(family_name="a"), _cylinder(family_name="b")
    insert(pace_db.registry.geometries, a)
    insert(pace_db.registry.geometries, b)
    found = pace_db.registry.geometries.existing_ids(
        [a.id, b.id, GeometryID("missing-1")]
    )
    assert found == {a.id, b.id}


def test_existing_ids_of_nothing_is_empty(pace_db: PaceDB):
    assert pace_db.registry.geometries.existing_ids([]) == set()


def test_family_exists(pace_db: PaceDB, insert):
    assert not pace_db.registry.geometries.family_exists("test_cylinder")
    insert(pace_db.registry.geometries, _cylinder())
    assert pace_db.registry.geometries.family_exists("test_cylinder")


def test_describe_returns_every_column_except_data(pace_db: PaceDB, insert):
    v1 = _cylinder()
    insert(pace_db.registry.geometries, v1)
    described = pace_db.registry.geometries.describe(v1.id)
    assert described is not None
    assert "data" not in described
    assert described["id"] == v1.id
    assert described["family_name"] == "test_cylinder"
    assert described["version_label"] == "1"
    assert described["type"] == "cylinder"
    assert described["derived_from"] is None
    assert described["gt_run_id"] is None
    assert described["user_edit"] is False
    assert described["created_at"] is not None


def test_describe_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.geometries.describe(GeometryID("nonexistent-1")) is None


def test_get_many_returns_only_requested_and_existing_ids(pace_db: PaceDB, insert):
    a, b = _cylinder(family_name="a"), _cylinder(family_name="b")
    insert(pace_db.registry.geometries, a)
    insert(pace_db.registry.geometries, b)
    fetched = pace_db.registry.geometries.get_many([a.id, GeometryID("nonexistent-1")])
    assert fetched.keys() == {a.id}
    assert fetched[a.id].to_dict() == a.to_dict()


def test_get_many_of_nothing_is_empty(pace_db: PaceDB):
    assert pace_db.registry.geometries.get_many([]) == {}


# =============================================================================
# Version queries
# =============================================================================


def test_versions_of_family_returns_every_version_oldest_first(pace_db: PaceDB, insert):
    v1 = _cylinder()
    v2 = _v2(v1)
    insert(pace_db.registry.geometries, v1)
    insert(pace_db.registry.geometries, v2)
    versions = pace_db.registry.geometries.versions_of_family("test_cylinder")
    assert [v.id for v in versions] == [v1.id, v2.id]


def test_versions_of_family_empty_when_no_such_family(pace_db: PaceDB):
    assert pace_db.registry.geometries.versions_of_family("nonexistent_family") == []


def test_latest_version_of_family_returns_most_recently_created(
    pace_db: PaceDB, insert
):
    v1 = _cylinder()
    insert(pace_db.registry.geometries, v1)
    v2 = _v2(v1)
    insert(pace_db.registry.geometries, v2)
    latest = pace_db.registry.geometries.latest_version_of_family("test_cylinder")
    assert latest is not None
    assert latest.id == v2.id


def test_latest_version_of_family_none_when_no_such_family(pace_db: PaceDB):
    assert pace_db.registry.geometries.latest_version_of_family("nope") is None


def test_children_of_returns_direct_derivations_only(pace_db: PaceDB, insert):
    v1 = _cylinder()
    v2 = _v2(v1)
    v3 = _cylinder(version_label="3", derived_from=v2.id, user_edit=True)
    for version in (v1, v2, v3):
        insert(pace_db.registry.geometries, version)
    children = pace_db.registry.geometries.children_of(v1.id)
    assert {c.id for c in children} == {v2.id}


# =============================================================================
# delete
# =============================================================================


def test_delete_removes_the_row(pace_db: PaceDB, insert):
    geometry = _cylinder()
    insert(pace_db.registry.geometries, geometry)
    with pace_db.registry.transaction() as session:
        pace_db.registry.geometries.delete(session, geometry.id)
    assert pace_db.registry.geometries.get(geometry.id) is None


def test_delete_nonexistent_id_is_a_no_op(pace_db: PaceDB):
    with pace_db.registry.transaction() as session:
        pace_db.registry.geometries.delete(session, GeometryID("nonexistent-1"))


def test_delete_of_a_predecessor_violates_foreign_key(pace_db: PaceDB, insert):
    v1 = _cylinder()
    insert(pace_db.registry.geometries, v1)
    insert(pace_db.registry.geometries, _v2(v1))
    with pytest.raises(IntegrityError):
        with pace_db.registry.transaction() as session:
            pace_db.registry.geometries.delete(session, v1.id)
    assert pace_db.registry.geometries.exists(v1.id)


# =============================================================================
# Every concrete DAO round-trips its own type
# =============================================================================


def test_every_dao_round_trips(pace_db: PaceDB, insert):
    registry = pace_db.registry
    cylinder = _cylinder()
    fuel = MIsotopic.create(
        family_name="uo2",
        version_label="1",
        components={"U235": MaterialComponentEntry(percent=100.0)},
        percent_type=PercentType.WO,
        density_value=10.257,
        density_unit=DensityUnit.G_PER_CM3,
    )
    pellet = LComponent.create(
        family_name="pellet",
        version_label="1",
        geometry=cylinder.id,
        material=fuel.id,
    )
    pin_cell = CComponent.create(
        family_name="pin_cell",
        version_label="1",
        bounds=cylinder.id,
        placements=[
            ComponentPlacement(
                id=ComponentPlacementID("pellet"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                ref=ComponentRef(type=ComponentType.LCOMPONENT, id=pellet.id),
            )
        ],
    )
    pin_ref = ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID(pin_cell.id))
    lattice = RectLattice.create(
        family_name="grid",
        version_label="1",
        pitch_m=0.0126,
        fill=fuel.id,
        shape=(1, 1),
        elements=[LatticeElement(ref=pin_ref, addresses=((0, 0),))],
    )
    blueprint = ReactorBlueprint.create(
        family_name="core",
        version_label="1",
        bounds=cylinder.id,
        root=ComponentPlacement(
            id=ComponentPlacementID("root"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            ref=pin_ref,
        ),
        neutron_bcs=dict.fromkeys(
            faces_for_geometry_type(GeometryType.CYLINDER), NeutronBC.VACUUM
        ),
    )
    pairs = [
        (registry.geometries, cylinder),
        (registry.materials, fuel),
        (registry.lcomponents, pellet),
        (registry.ccomponents, pin_cell),
        (registry.lattices, lattice),
        (registry.reactor_blueprints, blueprint),
    ]
    for dao, model in pairs:
        insert(dao, model)
    for dao, model in pairs:
        fetched = dao.get(model.id)
        assert type(fetched) is type(model)
        assert fetched.to_dict() == model.to_dict()
