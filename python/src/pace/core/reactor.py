"""
pace/core/reactor.py

Reactor — the top-level, simulatable model: the region of space being
modeled (`bounds`), what fills any space the root component leaves
(`fill`), the root component placed inside it, the boundary conditions
on each face of the bounds, and the operating state the solvers start
from.

Everything spatially varying (temperature and density fields, burned
compositions) is run state, not part of the Reactor.

Boundary conditions exist ONLY here, at the edge of the modeled system.
Interior surfaces (between a pin cell and its neighbour, between a
composite and its parent) are transmission surfaces in OpenMC and
internal interfaces in MOOSE; subsystems get bounds, never conditions.

Every kind of boundary condition is keyed by BoundsFace (see bounds.py),
a named face of the bounds shape. That includes coolant flow: the inlet
and outlet are faces too (coolant typically enters at z_min and leaves
at z_max).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pace.core.bounds import PERIODIC_PARTNER, BoundsFace
from pace.core.component import Placement
from pace.core.ids import GeometryID, MaterialID, ReactorID
from pace.core.pace_object import PaceObject
from pace.core.versioned import Versioned


class NeutronBC(str, Enum):
    """What happens to a neutron reaching a face of the Reactor's bounds.

    Values (OpenMC's boundary types, minus 'transmission', which is the
    interior-surface default and never valid on the outer boundary):
        - vacuum: the neutron leaves and never returns (leakage). Use
              where the real system ends, e.g. the top and bottom of a
              3D pin.
        - reflective: mirror reflection — "my neighbour is my mirror
              image". Reflective on all sides of a pin cell models an
              infinite array (VERA problems 1 and 2).
        - periodic: leaves one face, re-enters the paired opposite face
              — "my neighbour is an identical copy". Must be set on both
              faces of a pair (see PERIODIC_PARTNER).
        - white: returns with a random (cosine-distributed) direction;
              mainly for cylindricalized unit cells.
    """

    VACUUM = "vacuum"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"
    WHITE = "white"


class ThermalBC(str, Enum):
    """Heat-conduction condition on a face of the Reactor's bounds.

    Values:
        - adiabatic: no heat crosses the face. Also the symmetry-plane
              condition, and the default for any face not listed.

    FIXED_TEMPERATURE and CONVECTIVE carry values, so they arrive as
    dataclasses when a model first needs them.
    """

    ADIABATIC = "adiabatic"


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
            "kind": "inlet",
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
        return {"kind": "outlet", "pressure_pa": self.pressure_pa}

    @classmethod
    def from_dict(cls, data: dict) -> FlowOutlet:
        return cls(pressure_pa=data["pressure_pa"])


FlowBC = FlowInlet | FlowOutlet


def flow_bc_from_dict(data: dict) -> FlowBC:
    """Dispatch on the "kind" key written by FlowInlet/FlowOutlet."""
    if data["kind"] == "inlet":
        return FlowInlet.from_dict(data)
    if data["kind"] == "outlet":
        return FlowOutlet.from_dict(data)
    raise ValueError(f"unknown flow boundary condition kind: {data['kind']!r}")


@dataclass(frozen=True, kw_only=True)
class OperatingState(PaceObject):
    """The few scalars the solvers start from that aren't geometry,
    composition, or a boundary condition.

    - power_w: total thermal power. Only coupled runs need it (Cardinal
          normalizes OpenMC's heating tallies with it); None for
          standalone eigenvalue runs.
    - initial_temperatures_k: a uniform starting temperature per
          material. OpenMC picks cross sections by temperature, so even
          standalone runs need these (the VERA problem 1 variants differ
          only in fuel temperature). Any material the model uses but
          doesn't list here is an error at translation time.

    Deliberately absent: inlet temperature, flow and pressure (flow
    boundary conditions on the Reactor), resolution and depletion steps
    (run spec), and power shapes or temperature fields (run output).

    Example — VERA problem 1 at hot zero power:
        OperatingState(
            power_w=None,
            initial_temperatures_k={
                uo2.id: 565.0,
                helium.id: 565.0,
                zircaloy_4.id: 565.0,
                borated_water.id: 565.0,
            },
        )
    """

    power_w: float | None = None
    initial_temperatures_k: dict[MaterialID, float] = field(default_factory=dict)

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> OperatingState:
        return cls(
            power_w=data["power_w"],
            initial_temperatures_k={
                MaterialID(material_id): temperature_k
                for material_id, temperature_k in data["initial_temperatures_k"].items()
            },
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class Reactor(Versioned[ReactorID]):
    """The top-level, simulatable model.

    - bounds: the region being modeled (GRectanglePrism, GHexPrism or
          GCylinder — checked by ComponentService).
    - fill: the material occupying space inside the bounds that the root
          doesn't. Usually None (e.g. VERA problem 2, whose assembly
          bounds equal the Reactor's); needed for things like bypass
          flow around a core. It never reaches coolant inside the
          root's nested composites — each composite owns its own fill.
    - root: the placed root component (a Placement, so it carries a
          ComponentRef plus a pose).
    - neutron_bcs: one condition per face of the bounds. ComponentService
          checks the keys are exactly the faces of the bounds shape.
    - thermal_bcs: optional; any face not listed is adiabatic.
    - flow_bcs: optional; if given, exactly one FlowInlet and one
          FlowOutlet.
    - operating_state: starting values (see OperatingState).

    Example — VERA problem 1 (2D, reflective on every face):
        Reactor.create(
            family_name="vera_problem_1",
            version_label="1",
            bounds=pin_cell_box.id,
            root=Placement(
                id=PlacementID("pin_cell"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                ref=ComponentRef(
                    type=ComponentType.CCOMPONENT, id=fuel_pin_cell.id
                ),
            ),
            neutron_bcs=dict.fromkeys(
                faces_for_geometry_type(GeometryType.RECT_PRISM),
                NeutronBC.REFLECTIVE,
            ),
            operating_state=OperatingState(initial_temperatures_k={...}),
        )

    eq=False / id-based equality: several fields are dicts.
    """

    description: str = ""
    bounds: GeometryID
    fill: MaterialID | None = None
    root: Placement
    neutron_bcs: dict[BoundsFace, NeutronBC]
    thermal_bcs: dict[BoundsFace, ThermalBC] = field(default_factory=dict)
    flow_bcs: dict[BoundsFace, FlowBC] = field(default_factory=dict)
    operating_state: OperatingState

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Reactor) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self) -> None:
        super().validate()
        self._validate_boundary_conditions()

    def _validate_boundary_conditions(self) -> None:
        """Within-object checks. Whether the faces match the bounds
        shape needs the bounds geometry — ComponentService's job.

        Rules enforced:
            - at least one neutron condition.
            - a PERIODIC face's partner is also PERIODIC.
            - flow conditions are either absent, or exactly one inlet
                  and one outlet.
        """
        if not self.neutron_bcs:
            raise ValueError(f"Reactor {self.id!r} has no neutron boundary conditions")

        for face, condition in self.neutron_bcs.items():
            if condition != NeutronBC.PERIODIC:
                continue
            partner = PERIODIC_PARTNER.get(face)
            if partner is None:
                raise ValueError(f"face {face.value!r} has no periodic partner")
            if self.neutron_bcs.get(partner) != NeutronBC.PERIODIC:
                raise ValueError(
                    f"face {face.value!r} is periodic but its partner "
                    f"{partner.value!r} is not"
                )

        if self.flow_bcs:
            inlets = [c for c in self.flow_bcs.values() if isinstance(c, FlowInlet)]
            outlets = [c for c in self.flow_bcs.values() if isinstance(c, FlowOutlet)]
            if len(inlets) != 1 or len(outlets) != 1:
                raise ValueError(
                    "flow boundary conditions need exactly one inlet and one "
                    f"outlet, got {len(inlets)} inlet(s) and {len(outlets)} outlet(s)"
                )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "description": self.description,
            "bounds": self.bounds,
            "fill": self.fill,
            "root": self.root.to_dict(),
            "neutron_bcs": {f.value: c.value for f, c in self.neutron_bcs.items()},
            "thermal_bcs": {f.value: c.value for f, c in self.thermal_bcs.items()},
            "flow_bcs": {f.value: c.to_dict() for f, c in self.flow_bcs.items()},
            "operating_state": self.operating_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Reactor:
        return cls(
            **cls._base_fields_from_dict(data),
            description=data["description"],
            bounds=GeometryID(data["bounds"]),
            fill=MaterialID(data["fill"]) if data["fill"] is not None else None,
            root=Placement.from_dict(data["root"]),
            neutron_bcs={
                BoundsFace(f): NeutronBC(c) for f, c in data["neutron_bcs"].items()
            },
            thermal_bcs={
                BoundsFace(f): ThermalBC(c) for f, c in data["thermal_bcs"].items()
            },
            flow_bcs={
                BoundsFace(f): flow_bc_from_dict(c) for f, c in data["flow_bcs"].items()
            },
            operating_state=OperatingState.from_dict(data["operating_state"]),
        )
