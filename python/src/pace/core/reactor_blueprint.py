"""
pace/core/reactor_blueprint.py

Reactor — the top-level, simulatable model: the region of space being
modeled (`bounds`), what fills any space the root component leaves
(`fill`), the root component placed inside it, and the operating state
the solvers start from.

Everything spatially varying (temperature and density fields, burned
compositions) is run state, not part of the Reactor.

Every kind of boundary condition is keyed by BoundsFace (see bounds.py),
a named face of the bounds shape. That includes coolant flow: the inlet
and outlet are faces too (coolant typically enters at z_min and leaves
at z_max).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pace.core.bounds import PERIODIC_PARTNER, BoundsFace
from pace.core.component import ComponentPlacement
from pace.core.ids import GeometryID, MaterialID, ReactorBlueprintID
from pace.core.versioned import Versioned
from pace.core.boundary_conditions import NeutronBC, ThermalBC, FlowBC


@dataclass(frozen=True, kw_only=True, eq=False)
class ReactorBlueprint(Versioned[ReactorBlueprintID]):
    """The top-level, simulatable model.

    - bounds: the region being modeled (GRectanglePrism, GHexPrism or
          GCylinder — checked by ComponentService).
    - fill: the material occupying space inside the bounds that the root
          doesn't. Usually None (e.g. VERA problem 2, whose assembly
          bounds equal the Reactor's); needed for things like bypass
          flow around a core. It never reaches coolant inside the
          root's nested composites — each composite owns its own fill.
    - root: the placed root component (a ComponentPlacement, so it carries a
          ComponentRef plus a pose).
    - neutron_bcs: one condition per face of the bounds. ComponentService
          checks the keys are exactly the faces of the bounds shape.
    - thermal_bcs: optional; any face not listed is adiabatic.
    - flow_bcs: optional; if given, exactly one FlowInlet face and one
          FlowOutlet face.

    Example — VERA problem 1 (2D, reflective on every face):
        Reactor.create(
            family_name="vera_problem_1",
            version_label="1",
            bounds=pin_cell_box.id,
            root=ComponentPlacement(
                id=ComponentPlacementID("pin_cell"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                ref=ComponentRef(
                    type=ComponentType.CCOMPONENT, id=fuel_pin_cell.id
                ),
            ),
            neutron_bcs=dict.fromkeys(
                faces_for_geometry_type(GeometryType.RECT_PRISM),
                NeutronBC.REFLECTIVE,
            ),
        )

    eq=False / id-based equality: several fields are dicts.
    """

    description: str = ""
    bounds: GeometryID
    fill: MaterialID | None = None
    root: ComponentPlacement
    neutron_bcs: dict[BoundsFace, NeutronBC]
    thermal_bcs: dict[BoundsFace, ThermalBC] = field(default_factory=dict)
    flow_bcs: dict[BoundsFace, FlowBC] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReactorBlueprint) and self.id == other.id

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

        # ensure that exactly one inlet and one outlet provided, if any
        if self.flow_bcs:
            inlets = [
                condition
                for condition in self.flow_bcs.values()
                if condition == FlowBC.INLET
            ]
            outlets = [
                condition
                for condition in self.flow_bcs.values()
                if condition == FlowBC.OUTLET
            ]

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
            "neutron_bcs": {
                face.value: condition.value
                for face, condition in self.neutron_bcs.items()
            },
            "thermal_bcs": {
                face.value: condition.value
                for face, condition in self.thermal_bcs.items()
            },
            "flow_bcs": {
                face.value: condition.value for face, condition in self.flow_bcs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReactorBlueprint:
        return cls(
            **cls._base_fields_from_dict(data),
            description=data["description"],
            bounds=GeometryID(data["bounds"]),
            fill=MaterialID(data["fill"]) if data["fill"] is not None else None,
            root=ComponentPlacement.from_dict(data["root"]),
            neutron_bcs={
                BoundsFace(face): NeutronBC(condition)
                for face, condition in data["neutron_bcs"].items()
            },
            thermal_bcs={
                BoundsFace(face): ThermalBC(condition)
                for face, condition in data["thermal_bcs"].items()
            },
            flow_bcs={
                BoundsFace(face): FlowBC(condition)
                for face, condition in data["flow_bcs"].items()
            },
        )
