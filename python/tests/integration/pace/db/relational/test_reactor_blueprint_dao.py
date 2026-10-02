"""
tests/integration/pace/db/relational/test_reactor_blueprint_dao.py

ReactorBlueprintDAO tests against a real (in-memory) DB: insert/get
round-trip of every ReactorBlueprint field (boundary conditions keyed
by face, flow faces), exists/family_exists, version queries, delete, and
get_many(). The generic DAO behaviour is covered in
test_versioned_dao.py.
"""

from __future__ import annotations

from pace.core.boundary_conditions import FlowBC, NeutronBC, ThermalBC
from pace.core.bounds import BoundsFace, faces_for_geometry_type
from pace.core.component import ComponentPlacement, ComponentRef, ComponentType
from pace.core.geometry import GeometryType, GPose
from pace.core.ids import (
    CComponentID,
    ComponentPlacementID,
    GeometryID,
    MaterialID,
    ReactorBlueprintID,
)
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.db.pace_db import PaceDB


def _blueprint(version_label: str = "1", **kwargs) -> ReactorBlueprint:
    bcs = dict.fromkeys(
        faces_for_geometry_type(GeometryType.RECT_PRISM), NeutronBC.REFLECTIVE
    )
    bcs[BoundsFace.Z_MIN] = NeutronBC.VACUUM
    bcs[BoundsFace.Z_MAX] = NeutronBC.VACUUM
    return ReactorBlueprint.create(
        family_name="pin_3d",
        version_label=version_label,
        description="3D pin cell",
        bounds=GeometryID("pin_box-1"),
        fill=MaterialID("water-1"),
        root=ComponentPlacement(
            id=ComponentPlacementID("pin"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            ref=ComponentRef(
                type=ComponentType.CCOMPONENT, id=CComponentID("pin_cell-1")
            ),
        ),
        neutron_bcs=bcs,
        thermal_bcs={BoundsFace.X_MIN: ThermalBC.ADIABATIC},
        flow_bcs={BoundsFace.Z_MIN: FlowBC.INLET, BoundsFace.Z_MAX: FlowBC.OUTLET},
        **kwargs,
    )


def test_insert_then_get_round_trips_every_field(pace_db: PaceDB, insert):
    blueprint = _blueprint()
    insert(pace_db.registry.reactor_blueprints, blueprint)
    fetched = pace_db.registry.reactor_blueprints.get(blueprint.id)
    assert fetched is not None
    assert fetched.to_dict() == blueprint.to_dict()
    assert fetched.flow_bcs == blueprint.flow_bcs


def test_get_returns_none_when_absent(pace_db: PaceDB):
    assert pace_db.registry.reactor_blueprints.get(ReactorBlueprintID("nope-1")) is None


def test_exists_and_family_exists(pace_db: PaceDB, insert):
    blueprint = _blueprint()
    assert not pace_db.registry.reactor_blueprints.exists(blueprint.id)
    insert(pace_db.registry.reactor_blueprints, blueprint)
    assert pace_db.registry.reactor_blueprints.exists(blueprint.id)
    assert pace_db.registry.reactor_blueprints.family_exists("pin_3d")


def test_versions_and_children(pace_db: PaceDB, insert):
    v1 = _blueprint()
    v2 = _blueprint(version_label="2", derived_from=v1.id, user_edit=True)
    insert(pace_db.registry.reactor_blueprints, v1)
    insert(pace_db.registry.reactor_blueprints, v2)
    dao = pace_db.registry.reactor_blueprints
    assert [v.id for v in dao.versions_of_family("pin_3d")] == [v1.id, v2.id]
    assert [c.id for c in dao.children_of(v1.id)] == [v2.id]


def test_delete_and_get_many(pace_db: PaceDB, insert):
    blueprint = _blueprint()
    insert(pace_db.registry.reactor_blueprints, blueprint)
    dao = pace_db.registry.reactor_blueprints
    assert dao.get_many([blueprint.id]).keys() == {blueprint.id}
    with pace_db.registry.transaction() as session:
        dao.delete(session, blueprint.id)
    assert dao.get_many([blueprint.id]) == {}
