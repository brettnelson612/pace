"""
pace/core/enclosure.py

Enclosure — a region of space it owns (`bounds`), the positioned
members inside it (Placements), and optionally one material (`fill`)
that occupies whatever part of the bounds no member occupies. Versioned
the same way as Geometry/Material/Part.

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
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.ids import EnclosureID, GeometryID, MaterialID
from pace.core.placeable import PlaceableType
from pace.core.placement import Placement
from pace.core.versioned import Versioned


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
            - a fill, at least one member, or both. An Enclosure
                  with a fill and no members is a region of a single
                  material, e.g. a lattice slot holding only coolant.
                  One with neither would be empty space.
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
        if self.fill is None and not self.placements:
            raise ValueError(
                f"Enclosure {self.id!r} needs a fill, at least one placement, or both"
            )

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
