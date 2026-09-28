"""
tests/integration/pace/db/relational/test_lattice_dao.py

Direct LatticeDAO tests against a real (in-memory) DB: save/get
round-trips for both concrete lattice types (polymorphic dispatch via
the `type` column), describe(), exists/family_exists, version queries,
delete, and get_many(). Mirrors test_geometry_dao.py.
"""

from __future__ import annotations

import pytest
from pace.core.component_ref import ComponentKind, ComponentRef
from pace.core.ids import CComponentID, LatticeID, MaterialID
from pace.core.lattice import (
    HexLattice,
    HexOrientation,
    Lattice,
    LatticePlacement,
    RectLattice,
)
from pace.db.pace_db import PaceDB

PIN = ComponentRef(kind=ComponentKind.CCOMPONENT, id=CComponentID("pin_cell-1"))
WATER = MaterialID("water-1")


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


def _rect(family_name: str = "rect_lattice", version_label: str = "1", **kwargs):
    return RectLattice.create(
        family_name=family_name,
        version_label=version_label,
        pitch_m=0.0126,
        fill=WATER,
        shape=(2, 2),
        placements=[LatticePlacement(ref=PIN, addresses=((0, 0), (1, 1)))],
        **kwargs,
    )


def _hex() -> HexLattice:
    return HexLattice.create(
        family_name="hex_lattice",
        version_label="1",
        pitch_m=0.009,
        fill=WATER,
        num_rings=2,
        orientation=HexOrientation.FLAT_TOP,
        placements=[LatticePlacement(ref=PIN, addresses=((0, 0), (1, 3)))],
    )


@pytest.mark.parametrize("build,cls", [(_rect, RectLattice), (_hex, HexLattice)])
def test_save_then_get_dispatches_to_concrete_type(pace_db: PaceDB, build, cls):
    lattice: Lattice = build()
    pace_db.registry.lattices.save(lattice)
    fetched = pace_db.registry.lattices.get(lattice.id)
    assert isinstance(fetched, cls)
    assert fetched.to_dict() == lattice.to_dict()


def test_describe_includes_type(pace_db: PaceDB):
    pace_db.registry.lattices.save(_hex())
    described = pace_db.registry.lattices.describe(LatticeID("hex_lattice-1"))
    assert described is not None
    assert described["type"] == "hex"


def test_get_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.lattices.get(LatticeID("nope-1")) is None


def test_exists_and_family_exists(pace_db: PaceDB):
    lattice = _rect()
    assert not pace_db.registry.lattices.exists(lattice.id)
    pace_db.registry.lattices.save(lattice)
    assert pace_db.registry.lattices.exists(lattice.id)
    assert pace_db.registry.lattices.family_exists("rect_lattice")
    assert not pace_db.registry.lattices.family_exists("other")


def test_versions_children_and_latest(pace_db: PaceDB):
    v1 = _rect()
    v2 = _rect(version_label="2", derived_from=v1.id, user_edit=True)
    pace_db.registry.lattices.save(v1)
    pace_db.registry.lattices.save(v2)

    versions = pace_db.registry.lattices.versions_of_family("rect_lattice")
    assert {v.id for v in versions} == {v1.id, v2.id}
    assert [c.id for c in pace_db.registry.lattices.children_of(v1.id)] == [v2.id]
    latest = pace_db.registry.lattices.latest_version_of_family("rect_lattice")
    assert latest is not None
    assert latest.id in {v1.id, v2.id}


def test_delete_removes_the_row(pace_db: PaceDB):
    lattice = _rect()
    pace_db.registry.lattices.save(lattice)
    pace_db.registry.lattices.delete(lattice.id)
    assert pace_db.registry.lattices.get(lattice.id) is None


def test_get_many_returns_only_requested_and_existing_ids(pace_db: PaceDB):
    rect, hexagonal = _rect(), _hex()
    pace_db.registry.lattices.save(rect)
    pace_db.registry.lattices.save(hexagonal)
    fetched = pace_db.registry.lattices.get_many([rect.id, LatticeID("nope-1")])
    assert fetched.keys() == {rect.id}
    assert pace_db.registry.lattices.get_many([]) == {}
