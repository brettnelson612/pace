"""
pace/core/component_ref.py

ComponentKind and ComponentRef — a typed pointer to anything that can be
placed inside a composite: an LComponent, a CComponent, or a Lattice.

Why a pair and not a bare id: ids are opaque strings built from
family_name + version_label, and family names are only unique within
one kind's registry. An LComponent "pin-1" and a CComponent "pin-1" can
both exist, so a bare id can't say which registry to look in.
ComponentRef carries the kind alongside the id, and is the one shape
used everywhere a placement happens (PComponent, Lattice placements,
Reactor.root).

ComponentKind is deliberately narrower than ReferenceableType: it lists
only the kinds that can be *placed*. Geometries and materials are
referenced too, but never placed on their own, so a ComponentRef to a
geometry is unrepresentable rather than merely invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pace.core.ids import CComponentID, LatticeID, LComponentID
from pace.core.pace_object import PaceObject
from pace.core.reference_types import ReferenceableType


class ComponentKind(str, Enum):
    """The kinds of object a composite or lattice can place.

    Values match ReferenceableType's, so converting between the two is a
    lookup by value (see ComponentRef.referenceable_type).
    """

    LCOMPONENT = "lcomponent"
    CCOMPONENT = "ccomponent"
    LATTICE = "lattice"


PlaceableID = LComponentID | CComponentID | LatticeID


@dataclass(frozen=True, kw_only=True)
class ComponentRef(PaceObject):
    """A typed reference to one placeable object.

    Example — pointing at a registered pin-cell composite:
        ComponentRef(kind=ComponentKind.CCOMPONENT, id=pin_cell.id)
    """

    kind: ComponentKind
    id: PlaceableID

    @property
    def referenceable_type(self) -> ReferenceableType:
        """The reference-index type tag for this ref's target."""
        return ReferenceableType(self.kind.value)

    def validate(self) -> None:
        if not isinstance(self.kind, ComponentKind):
            raise TypeError(
                f"kind must be a ComponentKind, got {self.kind!r} "
                f"({type(self.kind).__name__})"
            )
        if not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> ComponentRef:
        kind = ComponentKind(data["kind"])
        raw_id = data["id"]
        typed_id: PlaceableID
        if kind == ComponentKind.LCOMPONENT:
            typed_id = LComponentID(raw_id)
        elif kind == ComponentKind.CCOMPONENT:
            typed_id = CComponentID(raw_id)
        else:
            typed_id = LatticeID(raw_id)
        return cls(kind=kind, id=typed_id)
