"""
tests/integration/pace/db/relational/test_lattice_dao.py

LatticeDAO tests against a real (in-memory) DB: insert/get round-trips
for both concrete lattice types (polymorphic dispatch via the `type`
column), describe(), exists/family_exists, version queries, delete, and
get_many(). The generic DAO behaviour is covered in
test_versioned_dao.py.
"""

from __future__ import annotations

import pytest
from pace.core.component import ComponentRef, ComponentType
from pace.core.ids import CComponentID, LatticeID, MaterialID
from pace.core.lattice import (
    HexLattice,
    HexOrientation,
    Lattice,
    LatticeElement,
    RectLattice,
)
from pace.db.pace_db import PaceDB
from sqlalchemy.exc import IntegrityError

PIN = ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("pin_cell-1"))
WATER = MaterialID("water-1")


def _rect(family_name: str = "rect_lattice", version_label: str = "1", **kwargs):
    return RectLattice.create(
        family_name=family_name,
        version_label=version_label,
        pitch_m=0.0126,
        fill=WATER,
        shape=(2, 2),
        elements=[LatticeElement(ref=PIN, addresses=((0, 0), (1, 1)))],
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
        elements=[LatticeElement(ref=PIN, addresses=((0, 0), (1, 3)))],
    )


@pytest.mark.parametrize("build,cls", [(_rect, RectLattice), (_hex, HexLattice)])
def test_insert_then_get_dispatches_to_concrete_type(
    pace_db: PaceDB, insert, build, cls
):
    lattice: Lattice = build()
    insert(pace_db.registry.lattices, lattice)
    fetched = pace_db.registry.lattices.get(lattice.id)
    assert isinstance(fetched, cls)
    assert fetched.to_dict() == lattice.to_dict()


def test_describe_includes_type(pace_db: PaceDB, insert):
    insert(pace_db.registry.lattices, _hex())
    described = pace_db.registry.lattices.describe(LatticeID("hex_lattice-1"))
    assert described is not None
    assert described["type"] == "hex"


def test_get_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.lattices.get(LatticeID("nope-1")) is None


def test_exists_and_family_exists(pace_db: PaceDB, insert):
    lattice = _rect()
    assert not pace_db.registry.lattices.exists(lattice.id)
    insert(pace_db.registry.lattices, lattice)
    assert pace_db.registry.lattices.exists(lattice.id)
    assert pace_db.registry.lattices.family_exists("rect_lattice")
    assert not pace_db.registry.lattices.family_exists("other")


def test_versions_children_and_latest(pace_db: PaceDB, insert):
    v1 = _rect()
    v2 = _rect(version_label="2", derived_from=v1.id, user_edit=True)
    insert(pace_db.registry.lattices, v1)
    insert(pace_db.registry.lattices, v2)

    versions = pace_db.registry.lattices.versions_of_family("rect_lattice")
    assert {v.id for v in versions} == {v1.id, v2.id}
    assert [c.id for c in pace_db.registry.lattices.children_of(v1.id)] == [v2.id]
    latest = pace_db.registry.lattices.latest_version_of_family("rect_lattice")
    assert latest is not None
    assert latest.id == v2.id


def test_delete_removes_the_row(pace_db: PaceDB, insert):
    lattice = _rect()
    insert(pace_db.registry.lattices, lattice)
    with pace_db.registry.transaction() as session:
        pace_db.registry.lattices.delete(session, lattice.id)
    assert pace_db.registry.lattices.get(lattice.id) is None


def test_get_many_returns_only_requested_and_existing_ids(pace_db: PaceDB, insert):
    rect, hexagonal = _rect(), _hex()
    insert(pace_db.registry.lattices, rect)
    insert(pace_db.registry.lattices, hexagonal)
    fetched = pace_db.registry.lattices.get_many([rect.id, LatticeID("nope-1")])
    assert fetched.keys() == {rect.id}
    assert pace_db.registry.lattices.get_many([]) == {}


def test_insert_never_overwrites_an_existing_id(pace_db: PaceDB, insert):
    insert(pace_db.registry.lattices, _rect())
    with pytest.raises(IntegrityError):
        insert(pace_db.registry.lattices, _rect())
