"""
pace/core/reactor_blueprint.py

ReactorBlueprint — the top-level model: a root Enclosure and the
boundary conditions on the faces of that Enclosure's bounds. The root's
bounds are the region being modeled; the blueprint owns no space of its
own. Operating conditions (power, pressure, temperatures, inlet flow)
are not part of the blueprint.

Everything spatially varying (temperature and density fields, burned
compositions) is run state, not part of the blueprint.

Every kind of boundary condition is keyed by BoundsFace (see
bounds.py), a named face of the root Enclosure's bounds shape. That
includes coolant flow: the inlet and outlet are faces too (coolant
typically enters at z_min and leaves at z_max).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pace.core.boundary_conditions import FlowBC, NeutronBC, ThermalBC
from pace.core.bounds import PERIODIC_PARTNER, BoundsFace
from pace.core.ids import EnclosureID, ReactorBlueprintID
from pace.core.versioned import Versioned


@dataclass(frozen=True, kw_only=True, eq=False)
class ReactorBlueprint(Versioned[ReactorBlueprintID]):
    """The top-level, simulatable model: what is modeled (the root
    Enclosure) and what happens at its outer boundary (the boundary
    conditions). Simulation cases reference a blueprint by id.

    - root: the Enclosure being modeled. Its bounds are the problem
          boundary and its fill covers whatever its members don't (e.g.
          the water gap around a fuel assembly). The root is always an
          Enclosure: a Part has no fill to surround it, and a Lattice
          must be placed inside an Enclosure. The root sits at the
          origin; it has no pose.
    - neutron_bcs: one condition per face of the root's bounds.
          RegistryService checks the keys are exactly the faces of the
          bounds shape.
    - thermal_bcs: optional; any face not listed is adiabatic.
    - flow_bcs: optional; if given, exactly one FlowBC.INLET face and
          one FlowBC.OUTLET face.

    Example — VERA problem 1 (2D, reflective on every face), whose root
    is the pin cell itself:
        ReactorBlueprint.create(
            family_name="vera_problem_1",
            version_label="1",
            root=fuel_pin_cell.id,
            neutron_bcs=dict.fromkeys(
                faces_for_geometry_type(GeometryType.RECT_PRISM),
                NeutronBC.REFLECTIVE,
            ),
        )
    """

    description: str = ""
    root: EnclosureID
    neutron_bcs: dict[BoundsFace, NeutronBC]
    thermal_bcs: dict[BoundsFace, ThermalBC] = field(default_factory=dict)
    flow_bcs: dict[BoundsFace, FlowBC] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if not self.root:
            raise ValueError(f"Reactor {self.id!r} root must be a non-empty id")
        self._validate_boundary_conditions()

    def _validate_boundary_conditions(self) -> None:
        """Within-object checks. Whether the faces match the root's
        bounds shape needs the root and its bounds geometry —
        RegistryService's job.

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
            "root": self.root,
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
            **cls._base_versioned_fields_from_dict(data),
            description=data["description"],
            root=EnclosureID(data["root"]),
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
