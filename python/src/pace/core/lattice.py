"""
pace/core/lattice.py

Lattice — a grid of slots, where a slot's position is implied by
its address and the pitch rather than stored per member. Exists for
repeated structures (a fuel assembly's 17x17 pin array) where an
Enclosure's explicit Placement-per-member approach doesn't scale.

Two concrete shapes:
    - RectLattice — addresses are (row, col); row 0 is the TOP row and
          col 0 the leftmost column, the order an assembly map is drawn
          in. Matches both OpenMC's RectLattice.universes and the MOOSE
          Reactor module's `pattern`, so neither translator reorders.
    - HexLattice — addresses are (ring, index); ring 0 is the single
          center slot, ring k has 6k slots. Within a ring, indices
          proceed clockwise from a starting slot that depends on the
          orientation: the top slot (+y) for FLAT_TOP, the right-hand
          slot (+x) for POINT_TOP. This is OpenMC's own within-ring
          convention for its 'y' and 'x' orientations, so the OpenMC
          translator only has to reverse the ring order (OpenMC lists
          rings outermost first).

Tiling. Every slot holds exactly one occupant, and every occupant is an
Enclosure whose bounds are exactly the slot's shape: a square prism of
side pitch_m for a RectLattice, a hexagonal prism of flat-to-flat width
pitch_m for a HexLattice. All occupants share one height. The lattice
therefore has no fill of its own: each slot's coolant is its occupant's
fill, which gives every slot its own coolant region (its own cell
instance, mesh region and THM channel), and the MOOSE Reactor module's
patterned mesh generators, which tile unit cells of the pitch size, can
mesh it directly. A slot holding only coolant is an Enclosure with a
fill and no other members. Whether each occupant's bounds match the
slot needs the referenced geometries, so RegistryService checks it.

A POINT_TOP lattice places each occupant rotated by pi/6 about z, so an
occupant's flat-top GHexPrism bounds (see geometry.py) line up with its
point-top slot.

Placement. A lattice is always placed inside an Enclosure, never used
as a ReactorBlueprint's root. Its placement pose locates the lattice's
center: the center slot of a HexLattice, the middle of the grid of a
RectLattice. Lattices are 2D: their height is their occupants' shared
height, centered on the pose like every other shape, and axial
variation comes from stacking in the parent.

Versioned like Geometry/Material/Part/Enclosure.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Self

from pace.core.constraints import Constraint, validate_fields
from pace.core.geometry import Geometry, GHexPrism, GRectanglePrism
from pace.core.ids import EnclosureID, LatticeID
from pace.core.pace_object import PaceObject
from pace.core.versioned import Versioned

LatticeAddress = tuple[int, int]


class LatticeType(StrEnum):
    """Discriminator tag for Lattice subclasses."""

    RECT = "rect"
    HEX = "hex"


class HexOrientation(StrEnum):
    """Which way a hexagonal lattice's slots point.

    Values:
        - flat_top: each slot has two faces perpendicular to the y-axis,
              so neighboring slots sit directly above and below each
              other (OpenMC orientation 'y', its default).
        - point_top: each slot has two faces perpendicular to the
              x-axis, so neighboring slots sit directly left and right
              of each other (OpenMC orientation 'x').
    """

    FLAT_TOP = "flat_top"
    POINT_TOP = "point_top"


# lattice types registered here upon definition — see
# Lattice.__init_subclass__
LATTICE_TYPE_TO_CLASS: dict[str, type[Lattice]] = {}


@dataclass(frozen=True, kw_only=True)
class LatticeOccupant(PaceObject):
    """One Enclosure in a lattice and every address it occupies.

    Grouping addresses per Enclosure (rather than storing one entry per
    slot) keeps a lattice's serialized form small when one pin cell
    fills hundreds of slots, and gives each referenced Enclosure exactly
    one entry.

    Example — the same pin cell in three slots of a rect lattice's top
    row:
        LatticeOccupant(
            enclosure=pin_cell.id,
            addresses=((0, 0), (0, 1), (0, 2)),
        )
    """

    enclosure: EnclosureID
    addresses: tuple[LatticeAddress, ...]

    def validate(self) -> None:
        if not self.enclosure:
            raise ValueError("occupant enclosure must be a non-empty id")
        if len(self.addresses) < 1:
            raise ValueError(f"occupant {self.enclosure!r} lists no addresses")
        if len(set(self.addresses)) != len(self.addresses):
            raise ValueError(f"occupant {self.enclosure!r} repeats an address")

    def to_dict(self) -> dict:
        return {
            "enclosure": self.enclosure,
            "addresses": [list(address) for address in self.addresses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> LatticeOccupant:
        return cls(
            enclosure=EnclosureID(data["enclosure"]),
            addresses=tuple((a, b) for a, b in data["addresses"]),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class Lattice(Versioned[LatticeID]):
    """Shared base for RectLattice and HexLattice.

    pitch_m is the center-to-center distance between adjacent slots
    (for hex: across flats), which is also the width of each slot.
    occupants say which Enclosure fills each slot; every slot is filled
    (see the module docstring for the tiling rule).

    Identity, lineage and equality are inherited from Versioned.
    """

    lattice_type: ClassVar[LatticeType]

    pitch_m: float = field(metadata={"constraint": Constraint.POSITIVE})
    occupants: list[LatticeOccupant]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "lattice_type" in cls.__dict__:
            LATTICE_TYPE_TO_CLASS[cls.lattice_type.value] = cls

    @abstractmethod
    def addresses(self) -> list[LatticeAddress]:
        """Every valid address in this lattice, in a stable order."""

    @abstractmethod
    def slot_center(self, address: LatticeAddress) -> tuple[float, float]:
        """The (x, y) center of the slot at `address`, in meters,
        relative to the lattice's center. The viewer and every solver
        translator place slots with this one function, so they cannot
        disagree about where a slot is."""

    @abstractmethod
    def matches_slot(self, bounds: Geometry) -> bool:
        """Whether `bounds` is exactly this lattice's slot shape, as an
        occupant's bounds must be."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Self: ...

    @abstractmethod
    def _validate_shape(self) -> None:
        """Subclass-specific checks on the lattice's own dimensions."""

    def validate(self) -> None:
        """Field constraints, then identity/lineage (via Versioned),
        then shape, then occupants."""
        validate_fields(self)
        super().validate()
        self._validate_shape()
        self._validate_occupants()

    def _validate_occupants(self) -> None:
        """Rules enforced:
        - each Enclosure appears in at most one occupant — one entry
              per Enclosure, holding all of its addresses.
        - every address is valid for this lattice's shape.
        - no address is occupied twice.
        - every address is occupied.
        """
        valid = set(self.addresses())
        seen_enclosures: set[EnclosureID] = set()
        occupied: set[LatticeAddress] = set()

        for occupant in self.occupants:
            if occupant.enclosure in seen_enclosures:
                raise ValueError(
                    f"{occupant.enclosure!r} appears in more than one occupant "
                    f"of {self.id!r}"
                )
            seen_enclosures.add(occupant.enclosure)

            for address in occupant.addresses:
                if address not in valid:
                    raise ValueError(
                        f"address {address} is outside lattice {self.id!r}"
                    )
                if address in occupied:
                    raise ValueError(
                        f"address {address} is occupied twice in {self.id!r}"
                    )
                occupied.add(address)

        empty = [address for address in self.addresses() if address not in occupied]
        if empty:
            raise ValueError(
                f"every slot of lattice {self.id!r} must be occupied; "
                f"empty: {empty}"
            )

    def occupant_at(self, address: LatticeAddress) -> EnclosureID:
        """The Enclosure filling the slot at `address`."""
        for occupant in self.occupants:
            if address in occupant.addresses:
                return occupant.enclosure
        raise ValueError(f"address {address} is outside lattice {self.id!r}")

    def to_dict(self) -> dict:
        return {
            "type": self.lattice_type.value,
            **super().to_dict(),
            "pitch_m": self.pitch_m,
            "occupants": [occupant.to_dict() for occupant in self.occupants],
        }

    @classmethod
    def _lattice_fields_from_dict(cls, data: dict) -> dict:
        """Shared parsing for every field Lattice adds on top of
        Versioned."""
        return {
            **cls._base_versioned_fields_from_dict(data),
            "pitch_m": data["pitch_m"],
            "occupants": [LatticeOccupant.from_dict(p) for p in data["occupants"]],
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class RectLattice(Lattice):
    """A rectangular grid with square pitch.

    shape is (n_rows, n_cols). Addresses are (row, col) with row 0 at
    the top and col 0 at the left. Each slot is a square of side
    pitch_m.

    Example — a 3x3 of fuel pin cells around a central guide-tube cell,
    authored as a grid:
        RectLattice.from_grid(
            family_name="mini_lattice_3x3",
            version_label="1",
            pitch_m=0.0126,
            grid=[
                [pin_cell.id, pin_cell.id, pin_cell.id],
                [pin_cell.id, guide_cell.id, pin_cell.id],
                [pin_cell.id, pin_cell.id, pin_cell.id],
            ],
        )
    """

    lattice_type: ClassVar[LatticeType] = LatticeType.RECT
    shape: tuple[int, int]

    def _validate_shape(self) -> None:
        n_rows, n_cols = self.shape
        if n_rows < 1 or n_cols < 1:
            raise ValueError(f"shape must be positive on both axes, got {self.shape}")

    def addresses(self) -> list[LatticeAddress]:
        n_rows, n_cols = self.shape
        return [(row, col) for row in range(n_rows) for col in range(n_cols)]

    def slot_center(self, address: LatticeAddress) -> tuple[float, float]:
        """Columns increase toward +x and rows increase toward -y (row 0
        is the top row), with the grid centered on the origin."""
        if address not in set(self.addresses()):
            raise ValueError(f"address {address} is outside lattice {self.id!r}")
        row, col = address
        n_rows, n_cols = self.shape
        x = (col - (n_cols - 1) / 2) * self.pitch_m
        y = ((n_rows - 1) / 2 - row) * self.pitch_m
        return (x, y)

    def matches_slot(self, bounds: Geometry) -> bool:
        """A square prism of side pitch_m."""
        return (
            isinstance(bounds, GRectanglePrism)
            and math.isclose(bounds.length_m, self.pitch_m)
            and math.isclose(bounds.width_m, self.pitch_m)
        )

    @classmethod
    def from_grid(
        cls,
        *,
        family_name: str,
        version_label: str,
        pitch_m: float,
        grid: list[list[EnclosureID]],
        **kwargs,
    ) -> RectLattice:
        """Build a RectLattice from a top-row-first grid of Enclosure
        ids. Occupants are ordered by each Enclosure's first appearance,
        reading row by row."""

        # verify at least one row/col present
        if not grid or not grid[0]:
            raise ValueError("grid must have at least one row and one column")
        n_cols = len(grid[0])

        # verify all columns are the same length
        if any(len(row) != n_cols for row in grid):
            raise ValueError("every grid row must have the same length")

        addresses_by_enclosure: dict[EnclosureID, list[LatticeAddress]] = {}
        for row_index, row in enumerate(grid):
            for col_index, enclosure in enumerate(row):
                addresses_by_enclosure.setdefault(enclosure, []).append(
                    (row_index, col_index)
                )

        return cls.create(
            family_name=family_name,
            version_label=version_label,
            pitch_m=pitch_m,
            shape=(len(grid), n_cols),
            occupants=[
                LatticeOccupant(enclosure=enclosure, addresses=tuple(addresses))
                for enclosure, addresses in addresses_by_enclosure.items()
            ],
            **kwargs,
        )

    def to_grid(self) -> list[list[EnclosureID]]:
        """The top-row-first grid form."""
        n_rows, n_cols = self.shape
        return [
            [self.occupant_at((row, col)) for col in range(n_cols)]
            for row in range(n_rows)
        ]

    def to_dict(self) -> dict:
        return {**super().to_dict(), "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, data: dict) -> RectLattice:
        n_rows, n_cols = data["shape"]
        return cls(**cls._lattice_fields_from_dict(data), shape=(n_rows, n_cols))


@dataclass(frozen=True, kw_only=True, eq=False)
class HexLattice(Lattice):
    """A hexagonal grid of concentric rings.

    num_rings counts the center slot as ring 0, so num_rings=3 has
    1 + 6 + 12 = 19 slots. Addresses are (ring, index); ring k has 6k
    slots, numbered clockwise from the top slot (FLAT_TOP) or the
    right-hand slot (POINT_TOP). Each slot is a regular hexagon of
    flat-to-flat width pitch_m.

    Example — a 2-ring (7-slot) bundle: one center pin plus six around
    it, all the same pin cell:
        HexLattice.create(
            family_name="seven_pin_bundle",
            version_label="1",
            num_rings=2,
            orientation=HexOrientation.FLAT_TOP,
            pitch_m=0.009,
            occupants=[
                LatticeOccupant(
                    enclosure=pin_cell.id,
                    addresses=((0, 0), *((1, i) for i in range(6))),
                ),
            ],
        )
    """

    lattice_type: ClassVar[LatticeType] = LatticeType.HEX
    num_rings: int
    orientation: HexOrientation

    @staticmethod
    def ring_size(ring: int) -> int:
        """Number of slots in `ring` (1 for the center, 6 * ring otherwise)."""
        return 1 if ring == 0 else 6 * ring

    def _validate_shape(self) -> None:
        if self.num_rings < 1:
            raise ValueError(f"num_rings must be at least 1, got {self.num_rings}")
        if not isinstance(self.orientation, HexOrientation):
            raise TypeError(
                f"orientation must be a HexOrientation, got {self.orientation!r}"
            )

    def addresses(self) -> list[LatticeAddress]:
        """Every (ring, index) address, center ring first, then each ring
        outward, clockwise from its starting slot."""
        return [
            (ring, index)
            for ring in range(self.num_rings)
            for index in range(self.ring_size(ring))
        ]

    def slot_center(self, address: LatticeAddress) -> tuple[float, float]:
        """Ring k is a hexagon of slots whose six corner slots sit at
        distance k * pitch_m from the center, in the directions of the
        center slot's six neighbors. Index 0 is the first corner (at
        +y for FLAT_TOP, +x for POINT_TOP); indices then walk clockwise
        along the ring's edges, k slots per edge."""
        if address not in set(self.addresses()):
            raise ValueError(f"address {address} is outside lattice {self.id!r}")
        ring, index = address
        if ring == 0:
            return (0.0, 0.0)

        first_corner_angle = (
            math.pi / 2 if self.orientation == HexOrientation.FLAT_TOP else 0.0
        )
        edge, step = divmod(index, ring)

        def corner(corner_index: int) -> tuple[float, float]:
            angle = first_corner_angle - corner_index * math.pi / 3
            radius = ring * self.pitch_m
            return (radius * math.cos(angle), radius * math.sin(angle))

        start_x, start_y = corner(edge)
        end_x, end_y = corner(edge + 1)
        fraction = step / ring
        return (
            start_x + fraction * (end_x - start_x),
            start_y + fraction * (end_y - start_y),
        )

    def matches_slot(self, bounds: Geometry) -> bool:
        """A hexagonal prism whose flat-to-flat width, sqrt(3) times its
        circumradius, equals pitch_m."""
        return isinstance(bounds, GHexPrism) and math.isclose(
            math.sqrt(3) * bounds.circumradius_m, self.pitch_m
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "num_rings": self.num_rings,
            "orientation": self.orientation.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HexLattice:
        return cls(
            **cls._lattice_fields_from_dict(data),
            num_rings=data["num_rings"],
            orientation=HexOrientation(data["orientation"]),
        )


def lattice_from_dict(data: dict) -> Lattice:
    """Reconstruct the correct concrete Lattice subclass from a dict
    produced by any subclass's to_dict(), dispatching on "type"."""
    cls = LATTICE_TYPE_TO_CLASS.get(data["type"])
    if cls is None:
        raise ValueError(f"unknown lattice type: {data['type']!r}")
    return cls.from_dict(data)
