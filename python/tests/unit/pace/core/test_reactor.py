"""
tests/pace/core/test_reactor.py

Covers: FlowInlet/FlowOutlet validation, round-trip and dispatch;
OperatingState validation and round-trip; Reactor construction,
lineage, boundary-condition rules (at least one neutron condition,
periodic faces paired, flow conditions all-or-one-inlet-one-outlet),
round-trip of every field, and id-based equality. Checks that need the
bounds geometry (faces match the shape) live in the ComponentService
integration tests.
"""

import pytest
from pace.core.bounds import BoundsFace, faces_for_geometry_type
from pace.core.component import ComponentPlacement
from pace.core.component_ref import ComponentType, ComponentRef
from pace.core.geometry import GeometryType, GPose
from pace.core.ids import (
    CComponentID,
    GeometryID,
    GTRunID,
    MaterialID,
    ComponentPlacementID,
)
from pace.core.reactor_blueprint import (
    FlowInlet,
    FlowOutlet,
    NeutronBC,
    OperatingState,
    Reactor,
    ThermalBC,
    flow_bc_from_dict,
)

BOX_FACES = faces_for_geometry_type(GeometryType.RECT_PRISM)
UO2 = MaterialID("uo2-1")
WATER = MaterialID("borated_water-1")
ROOT = ComponentPlacement(
    id=ComponentPlacementID("pin_cell"),
    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
    ref=ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("pin_cell-1")),
)


def _reactor(**overrides) -> Reactor:
    kwargs = {
        "family_name": "vera_problem_1",
        "version_label": "1",
        "bounds": GeometryID("pin_cell_box-1"),
        "root": ROOT,
        "neutron_bcs": dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE),
        "operating_state": OperatingState(
            initial_temperatures_k={UO2: 565.0, WATER: 565.0}
        ),
    }
    kwargs.update(overrides)
    return Reactor.create(**kwargs)


# =============================================================================
# Flow boundary conditions
# =============================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_flow_rate_kg_s": 0.0, "temperature_k": 565.0},
        {"mass_flow_rate_kg_s": 0.3, "temperature_k": 0.0},
    ],
)
def test_inlet_values_must_be_positive(kwargs):
    with pytest.raises(ValueError):
        FlowInlet(**kwargs)


def test_outlet_pressure_must_be_positive():
    with pytest.raises(ValueError):
        FlowOutlet(pressure_pa=0.0)


@pytest.mark.parametrize(
    "condition",
    [
        FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
        FlowOutlet(pressure_pa=15.5e6),
    ],
)
def test_flow_bc_round_trip_dispatches_on_kind(condition):
    assert flow_bc_from_dict(condition.to_dict()) == condition


def test_flow_bc_rejects_unknown_kind():
    with pytest.raises(ValueError):
        flow_bc_from_dict({"kind": "pump"})


# =============================================================================
# OperatingState
# =============================================================================


def test_operating_state_defaults_to_no_power_and_no_temperatures():
    state = OperatingState()
    assert state.power_w is None
    assert state.initial_temperatures_k == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"power_w": 0.0},
        {"power_w": -1.0},
        {"initial_temperatures_k": {UO2: 0.0}},
    ],
)
def test_operating_state_rejects_non_positive_values(kwargs):
    with pytest.raises(ValueError):
        OperatingState(**kwargs)


def test_operating_state_round_trip():
    state = OperatingState(power_w=6.5e4, initial_temperatures_k={UO2: 900.0})
    assert OperatingState.from_dict(state.to_dict()) == state


# =============================================================================
# Reactor — construction and lineage
# =============================================================================


def test_reactor_create_derives_id():
    assert _reactor().id == "vera_problem_1-1"


def test_reactor_defaults():
    reactor = _reactor()
    assert reactor.fill is None
    assert reactor.description == ""
    assert reactor.thermal_bcs == {}
    assert reactor.flow_bcs == {}


def test_reactor_lineage_rule_applies():
    with pytest.raises(ValueError):
        _reactor(gt_run_id=GTRunID("run-1"))


# =============================================================================
# Reactor — boundary-condition rules
# =============================================================================


def test_requires_a_neutron_condition():
    with pytest.raises(ValueError):
        _reactor(neutron_bcs={})


def test_periodic_pair_is_accepted():
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    bcs[BoundsFace.X_MIN] = NeutronBC.PERIODIC
    bcs[BoundsFace.X_MAX] = NeutronBC.PERIODIC
    assert _reactor(neutron_bcs=bcs).neutron_bcs[BoundsFace.X_MIN] == NeutronBC.PERIODIC


def test_unpaired_periodic_face_is_rejected():
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    bcs[BoundsFace.X_MIN] = NeutronBC.PERIODIC
    with pytest.raises(ValueError):
        _reactor(neutron_bcs=bcs)


def test_periodic_radial_face_is_rejected():
    with pytest.raises(ValueError):
        _reactor(neutron_bcs={BoundsFace.RADIAL: NeutronBC.PERIODIC})


def test_inlet_and_outlet_are_accepted():
    reactor = _reactor(
        flow_bcs={
            BoundsFace.Z_MIN: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
            BoundsFace.Z_MAX: FlowOutlet(pressure_pa=15.5e6),
        }
    )
    assert len(reactor.flow_bcs) == 2


@pytest.mark.parametrize(
    "flow_bcs",
    [
        {BoundsFace.Z_MIN: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0)},
        {BoundsFace.Z_MAX: FlowOutlet(pressure_pa=15.5e6)},
        {
            BoundsFace.Z_MIN: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
            BoundsFace.X_MIN: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
            BoundsFace.Z_MAX: FlowOutlet(pressure_pa=15.5e6),
        },
    ],
)
def test_flow_conditions_need_exactly_one_inlet_and_one_outlet(flow_bcs):
    with pytest.raises(ValueError):
        _reactor(flow_bcs=flow_bcs)


# =============================================================================
# Reactor — round-trip and equality
# =============================================================================


def test_full_reactor_round_trip():
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    bcs[BoundsFace.Z_MIN] = NeutronBC.VACUUM
    bcs[BoundsFace.Z_MAX] = NeutronBC.VACUUM
    reactor = _reactor(
        description="3D pin cell with axial leakage",
        fill=WATER,
        neutron_bcs=bcs,
        thermal_bcs={BoundsFace.Z_MIN: ThermalBC.ADIABATIC},
        flow_bcs={
            BoundsFace.Z_MIN: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
            BoundsFace.Z_MAX: FlowOutlet(pressure_pa=15.5e6),
        },
        operating_state=OperatingState(
            power_w=6.5e4, initial_temperatures_k={UO2: 900.0, WATER: 565.0}
        ),
    )
    restored = Reactor.from_dict(reactor.to_dict())
    assert restored.to_dict() == reactor.to_dict()
    assert restored.flow_bcs[BoundsFace.Z_MIN] == reactor.flow_bcs[BoundsFace.Z_MIN]


def test_equality_and_hash_are_id_based():
    a = _reactor()
    b = _reactor(description="different text, same id")
    assert a == b
    assert hash(a) == hash(b) == hash(a.id)
