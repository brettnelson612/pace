"""
pace/core/placeable.py

What a Placement can point at. A placeable is a Part (part.py), an
Enclosure (enclosure.py) or a Lattice (lattice.py); each has its own
registry and is referenced by id.

    - PlaceableType — which of the three kinds a reference points at.
    - PlaceableRef — a kind plus an id. Bare ids are opaque strings with
          no runtime type information, so anything that may point at
          more than one kind of placeable (a Placement, the root of a
          ResolvedPlaceable) carries a PlaceableRef rather than a bare
          id.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pace.core.ids import EnclosureID, LatticeID, PartID, PlaceableID
from pace.core.pace_object import PaceObject
from pace.core.reference_types import ReferenceableType


class PlaceableType(StrEnum):
    """The kinds of object a Placement can point at.

    Values match ReferenceableType's, so converting between the two is a
    lookup by value (see PlaceableRef.referenceable_type).
    """

    PART = "part"
    ENCLOSURE = "enclosure"
    LATTICE = "lattice"


@dataclass(frozen=True, kw_only=True)
class PlaceableRef(PaceObject):
    """A typed reference to a placeable object.

    Example — pointing at a registered pin-cell Enclosure:
        PlaceableRef(type=PlaceableType.ENCLOSURE, id=pin_cell.id)
    """

    type: PlaceableType
    id: PlaceableID

    @property
    def referenceable_type(self) -> ReferenceableType:
        """The reference-index type tag for this ref's target."""
        return ReferenceableType(self.type.value)

    def validate(self) -> None:
        if not isinstance(self.type, PlaceableType):
            raise TypeError(
                f"type must be a PlaceableType, got {self.type!r} "
                f"({type(self.type).__name__})"
            )
        if not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> dict:
        return {"type": self.type.value, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> PlaceableRef:
        type = PlaceableType(data["type"])
        raw_id = data["id"]
        typed_id: PlaceableID
        if type == PlaceableType.PART:
            typed_id = PartID(raw_id)
        elif type == PlaceableType.ENCLOSURE:
            typed_id = EnclosureID(raw_id)
        else:
            typed_id = LatticeID(raw_id)
        return cls(type=type, id=typed_id)
