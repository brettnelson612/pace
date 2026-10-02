"""
pace/core/operating_conditions.py

The plant conditions a simulation starts from: power, system pressure,
starting temperatures, and the coolant inlet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import MaterialID
from pace.core.pace_object import PaceObject


@dataclass(frozen=True, kw_only=True)
class CoolantInlet(PaceObject):
    """Coolant enters through the blueprint's inlet face at this total
    mass flow rate and temperature. The THM translator splits the total
    across channels (v1: by flow area, i.e. equal inlet velocity).

    Example: CoolantInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0)
    """

    mass_flow_rate_kg_s: float = field(metadata={"constraint": Constraint.POSITIVE})
    temperature_k: float = field(metadata={"constraint": Constraint.POSITIVE})

    def validate(self) -> None:
        validate_fields(self)

    def to_dict(self) -> dict:
        return {
            "mass_flow_rate_kg_s": self.mass_flow_rate_kg_s,
            "temperature_k": self.temperature_k,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CoolantInlet:
        return cls(
            mass_flow_rate_kg_s=data["mass_flow_rate_kg_s"],
            temperature_k=data["temperature_k"],
        )


@dataclass(frozen=True, kw_only=True)
class OperatingConditions(PaceObject):
    """The conditions every solver starts from.

    - power_w: total thermal power. None means no heat source (e.g. a
          hot-zero-power eigenvalue problem, where power only
          normalizes tallies).
    - system_pressure_pa: coolant system pressure. The THM translator
          applies it at the outlet face.
    - initial_temperatures_k: starting temperature per material. Fixed
          values in a standalone neutronics run; initial guesses in a
          coupled run.
    - coolant_inlet: inlet flow and temperature. None for models with
          no flow path.

    Example — VERA problem 1A (565 K, 2250 psia, no flow, zero power):
        OperatingConditions(
            system_pressure_pa=15.513e6,
            initial_temperatures_k={
                fuel.id: 565.0,
                clad.id: 565.0,
                borated_water.id: 565.0,
            },
        )
    """

    power_w: float | None = field(
        default=None, metadata={"constraint": Constraint.POSITIVE}
    )
    system_pressure_pa: float = field(metadata={"constraint": Constraint.POSITIVE})
    initial_temperatures_k: dict[MaterialID, float] = field(default_factory=dict)
    coolant_inlet: CoolantInlet | None = None

    def validate(self) -> None:
        validate_fields(self)
        for material_id, temperature_k in self.initial_temperatures_k.items():
            if temperature_k <= 0:
                raise ValueError(
                    f"initial temperature for {material_id!r} must be positive, "
                    f"got {temperature_k}"
                )

    def to_dict(self) -> dict:
        return {
            "power_w": self.power_w,
            "system_pressure_pa": self.system_pressure_pa,
            "initial_temperatures_k": dict(self.initial_temperatures_k),
            "coolant_inlet": (
                self.coolant_inlet.to_dict() if self.coolant_inlet is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> OperatingConditions:
        return cls(
            power_w=data["power_w"],
            system_pressure_pa=data["system_pressure_pa"],
            initial_temperatures_k={
                MaterialID(material_id): temperature_k
                for material_id, temperature_k in data["initial_temperatures_k"].items()
            },
            coolant_inlet=(
                CoolantInlet.from_dict(data["coolant_inlet"])
                if data["coolant_inlet"] is not None
                else None
            ),
        )
