"""
tests/unit/pace/core/test_reactor_blueprint.py

Covers: ReactorBlueprint construction and defaults, lineage,
boundary-condition rules (at least one neutron condition, periodic faces
paired, flow conditions absent or exactly one inlet and one outlet),
round-trip of every field, and id-based equality. Checks that need the
bounds geometry (faces match the shape) live in the RegistryService
integration tests.
"""

import pytest
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
    SimulationJobID,
)
from pace.core.reactor_blueprint import ReactorBlueprint

BOX_FACES = faces_for_geometry_type(GeometryType.RECT_PRISM)
WATER = MaterialID("borated_water-1")
ROOT = ComponentPlacement(
    id=ComponentPlacementID("pin_cell"),
    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
    ref=ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("pin_cell-1")),
)


def _blueprint(**overrides) -> ReactorBlueprint:
    kwargs = {
        "family_name": "vera_problem_1",
        "version_label": "1",
        "bounds": GeometryID("pin_cell_box-1"),
        "root": ROOT,
        "neutron_bcs": dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE),
    }
    kwargs.update(overrides)
    return ReactorBlueprint.create(**kwargs)


# =============================================================================
# Construction and lineage
# =============================================================================


def test_create_derives_id():
    assert _blueprint().id == "vera_problem_1-1"


def test_defaults():
    blueprint = _blueprint()
    assert blueprint.fill is None
    assert blueprint.description == ""
    assert blueprint.thermal_bcs == {}
    assert blueprint.flow_bcs == {}


def test_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        ReactorBlueprint(
            id=ReactorBlueprintID("wrong-1"),
            family_name="vera_problem_1",
            version_label="1",
            bounds=GeometryID("pin_cell_box-1"),
            root=ROOT,
            neutron_bcs=dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE),
        )


def test_lineage_rule_applies():
    with pytest.raises(ValueError):
        _blueprint(simulation_job_id=SimulationJobID("job-1"))


# =============================================================================
# Boundary-condition rules
# =============================================================================


def test_requires_a_neutron_condition():
    with pytest.raises(ValueError):
        _blueprint(neutron_bcs={})


def test_periodic_pair_is_accepted():
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    bcs[BoundsFace.X_MIN] = NeutronBC.PERIODIC
    bcs[BoundsFace.X_MAX] = NeutronBC.PERIODIC
    blueprint = _blueprint(neutron_bcs=bcs)
    assert blueprint.neutron_bcs[BoundsFace.X_MIN] == NeutronBC.PERIODIC


def test_unpaired_periodic_face_is_rejected():
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    bcs[BoundsFace.X_MIN] = NeutronBC.PERIODIC
    with pytest.raises(ValueError):
        _blueprint(neutron_bcs=bcs)


def test_periodic_radial_face_is_rejected():
    with pytest.raises(ValueError):
        _blueprint(neutron_bcs={BoundsFace.RADIAL: NeutronBC.PERIODIC})


def test_inlet_and_outlet_are_accepted():
    blueprint = _blueprint(
        flow_bcs={BoundsFace.Z_MIN: FlowBC.INLET, BoundsFace.Z_MAX: FlowBC.OUTLET}
    )
    assert len(blueprint.flow_bcs) == 2


@pytest.mark.parametrize(
    "flow_bcs",
    [
        {BoundsFace.Z_MIN: FlowBC.INLET},
        {BoundsFace.Z_MAX: FlowBC.OUTLET},
        {
            BoundsFace.Z_MIN: FlowBC.INLET,
            BoundsFace.X_MIN: FlowBC.INLET,
            BoundsFace.Z_MAX: FlowBC.OUTLET,
        },
        {
            BoundsFace.Z_MIN: FlowBC.INLET,
            BoundsFace.Z_MAX: FlowBC.OUTLET,
            BoundsFace.X_MAX: FlowBC.OUTLET,
        },
    ],
)
def test_flow_conditions_need_exactly_one_inlet_and_one_outlet(flow_bcs):
    with pytest.raises(ValueError):
        _blueprint(flow_bcs=flow_bcs)


# =============================================================================
# Round-trip and equality
# =============================================================================


def test_full_round_trip():
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    bcs[BoundsFace.Z_MIN] = NeutronBC.VACUUM
    bcs[BoundsFace.Z_MAX] = NeutronBC.VACUUM
    blueprint = _blueprint(
        description="3D pin cell with axial leakage",
        fill=WATER,
        neutron_bcs=bcs,
        thermal_bcs={BoundsFace.Z_MIN: ThermalBC.ADIABATIC},
        flow_bcs={BoundsFace.Z_MIN: FlowBC.INLET, BoundsFace.Z_MAX: FlowBC.OUTLET},
    )
    restored = ReactorBlueprint.from_dict(blueprint.to_dict())
    assert restored.to_dict() == blueprint.to_dict()
    assert restored.neutron_bcs == blueprint.neutron_bcs
    assert restored.thermal_bcs == blueprint.thermal_bcs
    assert restored.flow_bcs == blueprint.flow_bcs
    assert restored.root == blueprint.root


def test_to_dict_stores_faces_and_conditions_as_plain_strings():
    data = _blueprint(
        flow_bcs={BoundsFace.Z_MIN: FlowBC.INLET, BoundsFace.Z_MAX: FlowBC.OUTLET}
    ).to_dict()
    for key in ("neutron_bcs", "flow_bcs"):
        for face, condition in data[key].items():
            assert type(face) is str
            assert type(condition) is str


def test_equality_and_hash_are_id_based():
    a = _blueprint()
    b = _blueprint(description="different text, same id")
    assert a == b
    assert hash(a) == hash(b) == hash(a.id)
