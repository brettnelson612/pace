"""
tests/unit/pace/core/test_operating_conditions.py

Covers: CoolantInlet and OperatingConditions field constraints,
defaults, round-trips with and without a coolant inlet, and value-based
equality (neither is a versioned object).
"""

import pytest
from pace.core.ids import MaterialID
from pace.core.operating_conditions import CoolantInlet, OperatingConditions

UO2 = MaterialID("uo2-1")
WATER = MaterialID("borated_water-1")
PRESSURE_PA = 15.513e6


# =============================================================================
# CoolantInlet
# =============================================================================


def test_coolant_inlet_valid_construction():
    inlet = CoolantInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0)
    assert inlet.mass_flow_rate_kg_s == 0.3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_flow_rate_kg_s": 0.0, "temperature_k": 565.0},
        {"mass_flow_rate_kg_s": -0.3, "temperature_k": 565.0},
        {"mass_flow_rate_kg_s": 0.3, "temperature_k": 0.0},
    ],
)
def test_coolant_inlet_values_must_be_positive(kwargs):
    with pytest.raises(ValueError):
        CoolantInlet(**kwargs)


def test_coolant_inlet_round_trip():
    inlet = CoolantInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0)
    assert CoolantInlet.from_dict(inlet.to_dict()) == inlet


# =============================================================================
# OperatingConditions
# =============================================================================


def test_defaults_are_no_power_no_temperatures_no_inlet():
    conditions = OperatingConditions(system_pressure_pa=PRESSURE_PA)
    assert conditions.power_w is None
    assert conditions.initial_temperatures_k == {}
    assert conditions.coolant_inlet is None


def test_system_pressure_is_required():
    with pytest.raises(TypeError):
        OperatingConditions()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"system_pressure_pa": 0.0},
        {"system_pressure_pa": -1.0},
        {"system_pressure_pa": PRESSURE_PA, "power_w": 0.0},
        {"system_pressure_pa": PRESSURE_PA, "power_w": -1.0},
        {"system_pressure_pa": PRESSURE_PA, "initial_temperatures_k": {UO2: 0.0}},
    ],
)
def test_rejects_non_positive_values(kwargs):
    with pytest.raises(ValueError):
        OperatingConditions(**kwargs)


def test_round_trip_without_inlet():
    conditions = OperatingConditions(
        system_pressure_pa=PRESSURE_PA,
        initial_temperatures_k={UO2: 565.0, WATER: 565.0},
    )
    assert OperatingConditions.from_dict(conditions.to_dict()) == conditions


def test_round_trip_with_every_field():
    conditions = OperatingConditions(
        power_w=6.5e4,
        system_pressure_pa=PRESSURE_PA,
        initial_temperatures_k={UO2: 900.0, WATER: 565.0},
        coolant_inlet=CoolantInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0),
    )
    restored = OperatingConditions.from_dict(conditions.to_dict())
    assert restored == conditions
    assert restored.coolant_inlet == conditions.coolant_inlet


def test_equality_is_value_based():
    a = OperatingConditions(system_pressure_pa=PRESSURE_PA)
    b = OperatingConditions(system_pressure_pa=PRESSURE_PA, power_w=1.0)
    assert a != b
