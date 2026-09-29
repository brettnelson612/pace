"""
tests/integration/pace/db/relational/test_reactor_dao.py

Direct ReactorDAO tests against a real (in-memory) DB: save/get
round-trip of every Reactor field (boundary conditions keyed by face,
flow conditions, operating state), exists/family_exists, version
queries, delete, and get_many(). Mirrors test_geometry_dao.py.
"""

from __future__ import annotations

import pytest
from pace.core.bounds import BoundsFace, faces_for_geometry_type
from pace.core.component import Placement
from pace.core.component_ref import ComponentType, ComponentRef
from pace.core.geometry import GeometryType, GPose
from pace.core.ids import CComponentID, GeometryID, MaterialID, PlacementID, ReactorID
from pace.core.reactor import (
    FlowInlet,
    FlowOutlet,
    NeutronBC,
    OperatingState,
    Reactor,
)
from pace.db.pace_db import PaceDB


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


def _reactor(version_label: str = "1", **kwargs) -> Reactor:
    bcs = dict.fromkeys(
        faces_for_geometry_type(GeometryType.RECT_PRISM), NeutronBC.REFLECTIVE
    )
    bcs[BoundsFace.Z_MIN] = NeutronBC.VACUUM
    bcs[BoundsFace.Z_MAX] = NeutronBC.VACUUM
    return Reactor.create(
        family_name="pin_3d",
        version_label=version_label,
        bounds=GeometryID("pin_box-1"),
        root=Placement(
            id=PlacementID("pin"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            ref=ComponentRef(
                type=ComponentType.CCOMPONENT, id=CComponentID("pin_cell-1")
            ),
        ),
        neutron_bcs=bcs,
        flow_bcs={
            BoundsFace.Z_MIN: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
            BoundsFace.Z_MAX: FlowOutlet(pressure_pa=15.5e6),
        },
        operating_state=OperatingState(
            power_w=6.5e4, initial_temperatures_k={MaterialID("uo2-1"): 900.0}
        ),
        **kwargs,
    )


def test_save_then_get_round_trips_every_field(pace_db: PaceDB):
    reactor = _reactor()
    pace_db.registry.reactors.save(reactor)
    fetched = pace_db.registry.reactors.get(reactor.id)
    assert fetched is not None
    assert fetched.to_dict() == reactor.to_dict()


def test_get_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.reactors.get(ReactorID("nope-1")) is None


def test_exists_and_family_exists(pace_db: PaceDB):
    reactor = _reactor()
    assert not pace_db.registry.reactors.exists(reactor.id)
    pace_db.registry.reactors.save(reactor)
    assert pace_db.registry.reactors.exists(reactor.id)
    assert pace_db.registry.reactors.family_exists("pin_3d")


def test_versions_and_children(pace_db: PaceDB):
    v1 = _reactor()
    v2 = _reactor(version_label="2", derived_from=v1.id, user_edit=True)
    pace_db.registry.reactors.save(v1)
    pace_db.registry.reactors.save(v2)
    versions = pace_db.registry.reactors.versions_of_family("pin_3d")
    assert {v.id for v in versions} == {v1.id, v2.id}
    assert [c.id for c in pace_db.registry.reactors.children_of(v1.id)] == [v2.id]


def test_delete_and_get_many(pace_db: PaceDB):
    reactor = _reactor()
    pace_db.registry.reactors.save(reactor)
    assert pace_db.registry.reactors.get_many([reactor.id]).keys() == {reactor.id}
    pace_db.registry.reactors.delete(reactor.id)
    assert pace_db.registry.reactors.get_many([reactor.id]) == {}
