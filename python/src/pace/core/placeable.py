"""
pace/core/placeable.py

Reactor structure objects. A placeable is anything a Placement can put
somewhere: a Part, an Enclosure, or a Lattice (pace/core/lattice.py).

    - Part — a leaf: one geometry paired with one material, no
          position of its own. Versioned like Geometry/Material/
          Enclosure: family_name + version_label identity, and the
          same three-state derived_from/simulation_job_id/user_edit
          lineage rule.
    - Placement — a pose plus a PlaceableRef to one Part, Enclosure,
          or Lattice. Only ever a member of an Enclosure's `placements`
          list (or a Reactor's root); has no independent registry,
          lifecycle, or lineage of its own. Any change to a Placement (a
          moved pose, a repointed ref) is a change to its parent's own
          serialized data, which is enough to mint a new parent version
          through the existing mechanism.
    - Enclosure — a region of space it owns (`bounds`), the positioned
          members inside it, and optionally one material (`fill`) that
          occupies whatever part of the bounds no member occupies.
          Versioned the same way as Geometry/Material/Part.

Bounds and fill:
    - bounds is the region an Enclosure owns — never a physical wall.
          Walls, ducts and cladding are members. A bound sits either on
          a repetition line (a pin cell's pitch box, drawn through open
          water) or on a wall's inner face, when the coolant on each
          side must be tracked separately.
    - fill is always a single material. Nobody authors a fill
          geometry: the fill region is derived as bounds minus the
          union of members, holes included. That is what makes the
          fill impossible to overlap with, or leave gaps next to, its
          neighbours.
    - fill=None means the members are expected to occupy the bounds
          exactly (e.g. concentric layers that tile a cylinder).
          Leftover space with no fill is a translation-time error.
    - Placeables are placed, never used as a fill. A placed child's
          size comes from its own bounds; the parent only says where it
          goes.

Structures are recursive: a Placement's ref can point at another
Enclosure (or a Lattice), so pellets -> rods -> pin cells -> lattices
-> assemblies are all built the same way, one nesting level at a time.
Bare ids are opaque strings with no runtime type information, which is
why placements go through PlaceableRef (kind + id) rather than a bare id.

Reference direction follows PACE's bare-ids-always rule: Part,
Enclosure and Lattice have their own registries, so a placement
references them by id. Placement has no registry, so Enclosure embeds
Placement objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pace.core.geometry import GPose
from pace.core.ids import (
    EnclosureID,
    GeometryID,
    LatticeID,
    MaterialID,
    PartID,
    PlaceableID,
    PlacementID,
)
from pace.core.pace_object import PaceObject
from pace.core.reference_types import ReferenceableType
from pace.core.versioned import Versioned


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


@dataclass(frozen=True, kw_only=True)
class Placement(PaceObject):
    """A GPose plus a PlaceableRef to the Part, Enclosure, or Lattice
    placed at that position. Only ever appears as a member of an
    Enclosure's `placements` list, or as a Reactor's root — has no
    independent registry, lifecycle, or lineage.

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


@dataclass(frozen=True, kw_only=True, eq=False)
class Part(Versioned[PartID]):
    """A leaf — one geometry paired with one material, no position of
    its own.

    Identity/lineage (id/family_name/version_label/derived_from/
    simulation_job_id/user_edit, build_id(), create(), the three-state
    lineage rule) are inherited from Versioned — see that class's
    docstring.

    family_name has no shape/composition of its own to anchor it to
    the way it does for Geometry or Material — a Part IS the
    pairing, so family_name is the caller-chosen name for the
    conceptual SLOT this pairing fills (e.g. "fuel_pellet"), not a
    description of any content that survives a version bump.

    A version minted as part of a cascading edit (RegistryService
    propagating a change up through every reference path) carries the
    SAME cause as the root edit that triggered the cascade: if a human
    hand-edited the referenced geometry, the new Part version
    this produces is also user_edit=True; if a simulation job produced
    the root change, the new Part version carries that same
    simulation_job_id. The human or the simulation job is the root
    cause of the whole chain, not just the one object they directly
    touched.

    Example — a UO2 fuel pellet, referencing an already-defined
    cylinder geometry and enriched-UO2 material:
        Part.create(
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )
    """

    geometry: GeometryID
    material: MaterialID

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "geometry": self.geometry,
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Part:
        return cls(
            **cls._base_versioned_fields_from_dict(data),
            geometry=GeometryID(data["geometry"]),
            material=MaterialID(data["material"]),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class Enclosure(Versioned[EnclosureID]):
    """A region of space it owns (`bounds`), the positioned members
    inside it, and optionally the one material
    (`fill`) occupying whatever part of the bounds no member occupies.
    Has no position of its own; whoever places it supplies one.

    Identity and versioning mirror Geometry/Material exactly (see
    Versioned).

    bounds must reference a GRectanglePrism, GHexPrism or GCylinder —
    the shapes every translator can turn into a clipping region and the
    Reactor module can mesh. That check needs the referenced geometry,
    so it's enforced by RegistryService, not here. The same goes for
    "every member lies inside the bounds".

    Example — the VERA problem 1 pin cell: a fuel rod (itself an
    Enclosure bounded by its clad's outer cylinder) placed in a 1.26 cm
    pitch box, with borated water filling the rest:
        Enclosure.create(
            family_name="fuel_pin_cell",
            version_label="1",
            bounds=pin_cell_box.id,        # GRectanglePrism, 12.6 mm square
            fill=borated_water.id,
            placements=[
                Placement(
                    id=PlacementID("rod"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    ref=PlaceableRef(
                        type=PlaceableType.ENCLOSURE, id=fuel_rod.id
                    ),
                ),
            ],
        )
    """

    bounds: GeometryID
    fill: MaterialID | None = None
    placements: list[Placement]

    def validate(self) -> None:
        """Check identity/lineage (via Versioned), then the
        composition-level rules."""
        super().validate()
        self._validate_composition()

    def _validate_composition(self) -> None:
        """Enforce the within-object composition rules — no registry
        access needed (dangling references, bounds shape and
        containment are RegistryService's job).

        Rules enforced:
            - at least 1 member. An Enclosure with no members would be
                  only its fill; an empty lattice slot already covers
                  that case.
            - unique Placement ids within this Enclosure — ids are the
                  path segments region addresses are built from.
            - no duplicate (ref, pose) pairs — the same placeable
                  placed twice at the exact same pose contributes
                  nothing physically and indicates a construction
                  mistake. The ref's kind is part of the key, so a
                  Part and an Enclosure that happen to share an id
                  string are not mistaken for one another.
            - no direct self-reference (an ENCLOSURE ref to this
                  Enclosure's own id). Multi-hop cycles need registry
                  access and are RegistryService's job.
        """
        if len(self.placements) < 1:
            raise ValueError("Enclosure must have at least one placement.")

        seen_ids: set[str] = set()
        seen_placements: set[tuple] = set()
        for placement in self.placements:
            if placement.id in seen_ids:
                raise ValueError(
                    f"duplicate Placement id {placement.id!r} in {self.id!r}"
                )
            seen_ids.add(placement.id)

            ref = placement.ref
            if ref.type == PlaceableType.ENCLOSURE and ref.id == self.id:
                raise ValueError(
                    f"Enclosure {self.id!r} cannot reference itself directly"
                )

            pose = placement.pose
            key = (
                ref.type,
                ref.id,
                pose.x_m,
                pose.y_m,
                pose.z_m,
                pose.z_rotation_rad,
            )
            if key in seen_placements:
                raise ValueError(f"duplicate placeable+position: {ref.id} at {pose}")
            seen_placements.add(key)

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "bounds": self.bounds,
            "fill": self.fill,
            "placements": [placement.to_dict() for placement in self.placements],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Enclosure:
        return cls(
            **cls._base_versioned_fields_from_dict(data),
            bounds=GeometryID(data["bounds"]),
            fill=MaterialID(data["fill"]) if data["fill"] is not None else None,
            placements=[
                Placement.from_dict(placement) for placement in data["placements"]
            ],
        )
