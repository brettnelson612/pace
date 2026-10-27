"""
pace/core/placement.py

Placement — a pose plus a PlaceableRef to one Part, Enclosure, or
Lattice. Only ever a member of an Enclosure's `placements` list; has no
independent registry, lifecycle, or lineage of its own. Any change to a
Placement (a moved pose, a repointed ref) is a change to its parent's
own serialized data, which is enough to mint a new parent version
through the existing mechanism.

Reference direction follows PACE's bare-ids-always rule: Part,
Enclosure and Lattice have their own registries, so a Placement
references them by id. Placement has no registry, so Enclosure embeds
Placement objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.geometry import GPose
from pace.core.ids import PlacementID
from pace.core.pace_object import PaceObject
from pace.core.placeable import PlaceableRef


@dataclass(frozen=True, kw_only=True)
class Placement(PaceObject):
    """A GPose plus a PlaceableRef to the Part, Enclosure, or Lattice
    placed at that position. Only ever appears as a member of an
    Enclosure's `placements` list — has no independent registry,
    lifecycle, or lineage.

    The pose positions the placed object's own origin (the center of
    its bounds) in the parent's coordinate frame.

    id must be unique WITHIN its own parent's `placements` list, not
    globally — the same underlying placeable legitimately gets reused
    across many placements, each with its own locally-unique
    Placement.id. This is what makes path-based region addressing
    (e.g. "/fuel_pin_cell-1/rod/") work: an id stays attached to its
    Placement regardless of list order, unlike an array index.

    Example — a fuel rod placed at its pin cell's center:
        Placement(
            id=PlacementID("rod"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            ref=PlaceableRef(type=PlaceableType.ENCLOSURE, id=fuel_rod.id),
        )
    """

    id: PlacementID
    pose: GPose
    ref: PlaceableRef

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Placement id must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "ref": self.ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Placement:
        return cls(
            id=PlacementID(data["id"]),
            pose=GPose.from_dict(data["pose"]),
            ref=PlaceableRef.from_dict(data["ref"]),
        )
