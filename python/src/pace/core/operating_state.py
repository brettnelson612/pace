"""
pace/core/operating_state.py

The operating state the solvers start from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pace.core.ids import MaterialID
from pace.core.pace_object import PaceObject


@dataclass(frozen=True, kw_only=True)
class FlowInlet(PaceObject):
    """Coolant enters through this face at the given total mass flow
    rate and temperature. The THM translator splits the total across
    channels (v1: by flow area, i.e. equal inlet velocity).

    Example: FlowInlet(mass_flow_rate_kg_s=0.3, temperature_k=565.0)
    """

    mass_flow_rate_kg_s: float
    temperature_k: float

    def validate(self) -> None:
        if self.mass_flow_rate_kg_s <= 0:
            raise ValueError(
                f"mass_flow_rate_kg_s must be positive, got {self.mass_flow_rate_kg_s}"
            )
        if self.temperature_k <= 0:
            raise ValueError(
                f"temperature_k must be positive, got {self.temperature_k}"
            )

    def to_dict(self) -> dict:
        return {
            "mass_flow_rate_kg_s": self.mass_flow_rate_kg_s,
            "temperature_k": self.temperature_k,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FlowInlet:
        return cls(
            mass_flow_rate_kg_s=data["mass_flow_rate_kg_s"],
            temperature_k=data["temperature_k"],
        )


@dataclass(frozen=True, kw_only=True)
class FlowOutlet(PaceObject):
    """Coolant leaves through this face; the system pressure is set
    here.

    Example: FlowOutlet(pressure_pa=15.5e6)
    """

    pressure_pa: float

    def validate(self) -> None:
        if self.pressure_pa <= 0:
            raise ValueError(f"pressure_pa must be positive, got {self.pressure_pa}")

    def to_dict(self) -> dict:
        return {"pressure_pa": self.pressure_pa}

    @classmethod
    def from_dict(cls, data: dict) -> FlowOutlet:
        return cls(pressure_pa=data["pressure_pa"])


@dataclass(frozen=True, kw_only=True)
class OperatingState(PaceObject):
    """"""

    power_w: float | None = None
    initial_temperatures_k: dict[MaterialID, float] = field(default_factory=dict)
    flow_inlet: FlowInlet | None
    flow_outlet: FlowOutlet | None

    def validate(self) -> None:
        if self.power_w is not None and self.power_w <= 0:
            raise ValueError(f"power_w must be positive, got {self.power_w}")
        for material_id, temperature_k in self.initial_temperatures_k.items():
            if temperature_k <= 0:
                raise ValueError(
                    f"initial temperature for {material_id!r} must be positive, "
                    f"got {temperature_k}"
                )

    def to_dict(self) -> dict:
        return {
            "power_w": self.power_w,
            "initial_temperatures_k": dict(self.initial_temperatures_k),
            "flow_inlet": self.flow_inlet.to_dict() if self.flow_inlet else None,
            "flow_outlet": self.flow_outlet.to_dict() if self.flow_outlet else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OperatingState:
        return cls(
            power_w=data["power_w"],
            initial_temperatures_k={
                MaterialID(material_id): temperature_k
                for material_id, temperature_k in data["initial_temperatures_k"].items()
            },
            flow_inlet=(
                FlowInlet.from_dict(data["flow_inlet"])
                if data["flow_inlet"] is not None
                else None
            ),
            flow_outlet=(
                FlowOutlet.from_dict(data["flow_outlet"])
                if data["flow_outlet"] is not None
                else None
            ),
        )
