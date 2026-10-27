""" """

from __future__ import annotations

from dataclasses import dataclass

from pace.core.ids import SimulationCaseID
from pace.core.operating_conditions import OperatingConditions
from pace.core.pace_object import PaceObject
from pace.core.reactor_blueprint import ReactorBlueprintID


@dataclass(frozen=True, kw_only=True)
class OpenMCSpec(PaceObject):
    pass


@dataclass(frozen=True, kw_only=True)
class HeatTransferSpec(PaceObject):
    pass


@dataclass(frozen=True, kw_only=True)
class CardinalSpec(PaceObject):
    pass


@dataclass(frozen=True, kw_only=True)
class THMSpec(PaceObject):
    pass


@dataclass(frozen=True, kw_only=True)
class CouplingSpec(PaceObject):
    pass


@dataclass(frozen=True, kw_only=True)
class SimulationCase(PaceObject):
    id: SimulationCaseID
    reactor_blueprint: ReactorBlueprintID
    operating_conditions: OperatingConditions
    neutronics: OpenMCSpec | None
    heat_transfer: HeatTransferSpec | None
    thermal_hydraulics: THMSpec | None
    coupling: CouplingSpec | None
