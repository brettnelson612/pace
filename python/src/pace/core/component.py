"""
pace/core/component.py

Reactor structure objects:
    - LComponent (geometry version + material version, no position) —
          a leaf: a specific shape paired with a specific composition,
          e.g. a fuel pellet (a GCylinder geometry version + a UO2
          MIsotopic material version).
    - PComponent (a position + one LComponent or CComponent; only
          ever a member of a CComponent) — a leaf or composite placed
          somewhere in space, e.g. that fuel pellet positioned at a
          specific height within a fuel pin's stack.
    - CComponent (a collection of >=2 PComponents, no position) — a
          composite assembled from multiple positioned components,
          e.g. a fuel pin (a stack of positioned pellets), or a full
          fuel assembly (a lattice of positioned fuel pins).

PComponent.component being LComponentID | CComponentID (rather than
embedding either directly) is what makes structures recursive: a
CComponent's members can themselves be positioned CComponents, so an
arbitrarily deep hierarchy — pellets -> pins -> assemblies -> a full
core — is built the same way at every level, just nesting one more
PComponent/CComponent layer each time.

Reference direction follows PACE's existing bare-IDs-always rule:
LComponent and CComponent each have their own registry ("LComponents",
"components"), so PComponent references them by ID. PComponent itself
has no registry (it is only ever a member of a CComponent — it has no
independent lifecycle), so CComponent embeds PComponent objects directly
rather than referencing them by ID, since there's nowhere for such an ID
to be looked up from.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.geometry import GPose
from pace.core.ids import (
    CComponentID,
    GeometryVersionID,
    LComponentID,
    MaterialVersionID,
    PComponentID,
)
from pace.core.pace_object import PaceObject


@dataclass(frozen=True, kw_only=True)
class LComponent(PaceObject):
    """A leaf component — one geometry version paired with one
    material version, no position of its own.

    Example — a UO2 fuel pellet, referencing an already-defined
    cylinder geometry version and enriched-UO2 material version:
        LComponent(
            id=LComponentID("pellet-1"),
            geometry_version=GeometryVersionID("gv-pellet-cyl-1"),
            material_version=MaterialVersionID("mv-uo2-3p2-1"),
        )
    """

    id: LComponentID
    geometry_version: GeometryVersionID
    material_version: MaterialVersionID

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "geometry_version": self.geometry_version,
            "material_version": self.material_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LComponent:
        return cls(
            id=data["id"],
            geometry_version=data["geometry_version"],
            material_version=data["material_version"],
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True, eq=False)
class CComponent(PaceObject):
    """A composite component — a collection of >=2 positioned
    (PComponent) members, no position of its own.

    Example — a simple 2-pellet fuel pin stack:
        CComponent(
            id=CComponentID("pin-1"),
            components=[
                PComponent(
                    id=PComponentID("pin-1-pos-1"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    component=LComponentID("pellet-1"),
                ),
                PComponent(
                    id=PComponentID("pin-1-pos-2"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
                    component=LComponentID("pellet-1"),
                ),
            ],
        )
    A CComponent's own members can themselves be positioned
    CComponents (via PComponent.component being a CComponentID) —
    e.g. an assembly built from positioned pins, each of which is
    itself a CComponent of positioned pellets.

    eq=False / id-based equality: `components` is a list of PComponent
    objects, not hashable via the frozen-dataclass default — same
    reasoning, and same pattern, as GAddition/GSubtraction and
    MIsotopic/MMixture.
    """

    id: CComponentID
    components: list[PComponent]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CComponent) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CComponent:
        return cls(
            id=data["id"],
            components=[
                PComponent.from_dict(component) for component in data["components"]
            ],
        )

    def validate(self):
        self._validate_composition()

    def _validate_composition(self):
        """A composite of fewer than 2 members isn't a composite."""
        if len(self.components) < 2:
            raise ValueError(
                "CComponent must be provided a list of at least two components."
            )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class PComponent(PaceObject):
    """A positioned component — a GPose plus a reference to the one
    LComponent or CComponent placed at that position. Only ever
    appears as a member of a CComponent's `components` list — has no
    independent registry/lifecycle of its own.
    """

    # TODO: decide how the PComponentID is determined...
    id: PComponentID
    pose: GPose
    component: LComponentID | CComponentID

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "component": self.component,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PComponent:
        return cls(
            id=data["id"],
            pose=GPose.from_dict(data["pose"]),
            component=data["component"],
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError
