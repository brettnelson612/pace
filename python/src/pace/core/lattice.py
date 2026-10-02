"""
pace/core/lattice.py

Lattice — a grid of slots, where a slot's position is implied by
its address and the pitch rather than stored per member. Exists for
repeated structures (a fuel assembly's 17x17 pin array) where a
CComponent's explicit ComponentPlacement-per-member approach doesn't scale.

Two concrete shapes:
    - RectLattice — addresses are (row, col); row 0 is the TOP row and
          col 0 the leftmost column, the order an assembly map is drawn
          in. Matches both OpenMC's RectLattice.universes and the MOOSE
          Reactor module's `pattern`, so neither translator reorders.
    - HexLattice — addresses are (ring, index); ring 0 is the single
          center slot, ring k has 6k slots. Within a ring, index 0 is
          the "top" slot and indices proceed clockwise — OpenMC's own
          within-ring convention, so the OpenMC translator only has to
          reverse the ring order (OpenMC lists rings outermost first).

fill is the single material around every occupant and in every empty
slot. Translators always split it PER SLOT (each slot's fill becomes its
own cell instance / mesh region / THM channel) — never into one region
spanning the whole lattice, which would give every pin's coolant one
shared temperature and density.

An occupant's bounds must fit inside its slot (checked by
ComponentService, since it needs the referenced geometries); the
lattice fill covers the rest of the slot. Lattices are 2D: axial
variation comes from stacking in the parent, and every occupant shares
one height.

Versioned like Geometry/Material/LComponent/CComponent.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Self

from pace.core.component import ComponentType, ComponentRef
from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import LatticeID, MaterialID
from pace.core.pace_object import PaceObject
from pace.core.versioned import Versioned

LatticeAddress = tuple[int, int]


class LatticeType(str, Enum):
    """Discriminator tag for Lattice subclasses."""

    RECT = "rect"
    HEX = "hex"


class HexOrientation(str, Enum):
    """Which way a hexagonal lattice's slots point.

    Values:
        - flat_top: each slot has two faces perpendicular to the y-axis
              (OpenMC orientation 'y', its default).
        - point_top: each slot has two faces perpendicular to the
              x-axis (OpenMC orientation 'x').
    """

    FLAT_TOP = "flat_top"
    POINT_TOP = "point_top"


# lattice types registered here upon definition — see
# Lattice.__init_subclass__
LATTICE_TYPE_TO_CLASS: dict[str, type[Lattice]] = {}


@dataclass(frozen=True, kw_only=True)
class LatticeElement(PaceObject):
    """A lattice element is just a single component and a list of every
    lattice address it occupies.

    Example — the same pin cell in three slots of a rect lattice's top
    row:
        LatticeElement(
            ref=ComponentRef(type=ComponentType.CCOMPONENT, id=pin_cell.id),
            addresses=((0, 0), (0, 1), (0, 2)),
        )
    """

    ref: ComponentRef
    addresses: tuple[LatticeAddress, ...]

    def validate(self) -> None:
        if len(self.addresses) < 1:
            raise ValueError(f"element of {self.ref.id!r} lists no addresses")
        if len(set(self.addresses)) != len(self.addresses):
            raise ValueError(f"element of {self.ref.id!r} repeats an address")

    def to_dict(self) -> dict:
        return {
            "ref": self.ref.to_dict(),
            "addresses": [list(address) for address in self.addresses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> LatticeElement:
        return cls(
            ref=ComponentRef.from_dict(data["ref"]),
            addresses=tuple((a, b) for a, b in data["addresses"]),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class Lattice(Versioned[LatticeID]):
    """Shared base for RectLattice and HexLattice.

    pitch_m is the center-to-center distance between adjacent slots
    (for hex: across flats). fill is the material in every empty slot
    and around every occupant.

    Identity/lineage are inherited from Versioned. eq=False / id-based
    equality: `elements` is a list — same pattern as CComponent.
    """

    lattice_type: ClassVar[LatticeType]

    pitch_m: float = field(metadata={"constraint": Constraint.POSITIVE})
    fill: MaterialID
    elements: list[LatticeElement]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "lattice_type" in cls.__dict__:
            LATTICE_TYPE_TO_CLASS[cls.lattice_type.value] = cls

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Lattice) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @abstractmethod
    def addresses(self) -> list[LatticeAddress]:
        """Every valid address in this lattice, in a stable order."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Self: ...

    @abstractmethod
    def _validate_shape(self) -> None:
        """Subclass-specific checks on the lattice's own dimensions."""

    def validate(self) -> None:
        """Field constraints, then identity/lineage (via Versioned),
        then shape, then elements."""
        validate_fields(self)
        super().validate()
        self._validate_shape()
        self._validate_elements()

    def _validate_elements(self) -> None:
        """Rules enforced:
        - at least one element — a lattice of nothing but fill is
              just a fill.
        - each ref appears in at most one element — one entry per
              placed component, holding all of its addresses.
        - every address is valid for this lattice's shape.
        - no address is occupied twice.
        - no direct self-reference.
        """
        if len(self.elements) < 1:
            raise ValueError(f"Lattice {self.id!r} has no elements")

        valid = set(self.addresses())
        seen_refs: set[ComponentRef] = set()
        occupied: set[LatticeAddress] = set()

        for element in self.elements:
            ref = element.ref

            # check that ref isn't listed more than once
            if ref in seen_refs:
                raise ValueError(
                    f"{ref.id!r} appears in more than one element of {self.id!r}"
                )
            seen_refs.add(ref)

            # shallow check that there is no circular dependency
            if ref.type == ComponentType.LATTICE and ref.id == self.id:
                raise ValueError(f"Lattice {self.id!r} cannot reference itself")

            # check that all addresses are valid for given shape
            for address in element.addresses:
                if address not in valid:
                    raise ValueError(
                        f"address {address} is outside lattice {self.id!r}"
                    )
                if address in occupied:
                    raise ValueError(
                        f"address {address} is occupied twice in {self.id!r}"
                    )
                occupied.add(address)

    def occupant_at(self, address: LatticeAddress) -> ComponentRef | None:
        """The component placed at `address`, or None for an empty slot
        (fill only)."""
        if address not in set(self.addresses()):
            raise ValueError(f"address {address} is outside lattice {self.id!r}")
        for element in self.elements:
            if address in element.addresses:
                return element.ref
        return None

    def empty_addresses(self) -> list[LatticeAddress]:
        """Addresses holding only the lattice fill, in address order."""
        occupied = {a for p in self.elements for a in p.addresses}
        return [a for a in self.addresses() if a not in occupied]

    def to_dict(self) -> dict:
        return {
            "type": self.lattice_type.value,
            **super().to_dict(),
            "pitch_m": self.pitch_m,
            "fill": self.fill,
            "elements": [element.to_dict() for element in self.elements],
        }

    @classmethod
    def _lattice_fields_from_dict(cls, data: dict) -> dict:
        """Shared parsing for every field Lattice adds on top of
        Versioned."""
        return {
            **cls._base_fields_from_dict(data),
            "pitch_m": data["pitch_m"],
            "fill": MaterialID(data["fill"]),
            "elements": [LatticeElement.from_dict(p) for p in data["elements"]],
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class RectLattice(Lattice):
    """A rectangular grid with square pitch.

    shape is (n_rows, n_cols). Addresses are (row, col) with row 0 at
    the top.

    Example — a 3x3 of fuel pin cells around a central guide-tube cell,
    authored as a grid:
        fuel = ComponentRef(type=ComponentType.CCOMPONENT, id=pin_cell.id)
        guide = ComponentRef(type=ComponentType.CCOMPONENT, id=guide_cell.id)
        RectLattice.from_grid(
            family_name="mini_lattice_3x3",
            version_label="1",
            pitch_m=0.0126,
            fill=borated_water.id,
            grid=[
                [fuel, fuel, fuel],
                [fuel, guide, fuel],
                [fuel, fuel, fuel],
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

    @classmethod
    def from_grid(
        cls,
        *,
        family_name: str,
        version_label: str,
        pitch_m: float,
        fill: MaterialID,
        grid: list[list[ComponentRef | None]],
        **kwargs,
    ) -> RectLattice:
        """Build a RectLattice from a top-row-first grid of refs.
        ComponentPlacements are ordered by each ref's first appearance,
        reading row by row. None = empty slot."""

        # verify at least one row/col present
        if not grid or not grid[0]:
            raise ValueError("grid must have at least one row and one column")
        n_cols = len(grid[0])

        # verify all columns are the same length
        if any(len(row) != n_cols for row in grid):
            raise ValueError("every grid row must have the same length")

        addresses_by_ref: dict[ComponentRef, list[LatticeAddress]] = {}
        for row_index, row in enumerate(grid):
            for col_index, ref in enumerate(row):
                if ref is not None:
                    addresses_by_ref.setdefault(ref, []).append((row_index, col_index))

        return cls.create(
            family_name=family_name,
            version_label=version_label,
            pitch_m=pitch_m,
            fill=fill,
            shape=(len(grid), n_cols),
            elements=[
                LatticeElement(ref=ref, addresses=tuple(addresses))
                for ref, addresses in addresses_by_ref.items()
            ],
            **kwargs,
        )

    def to_grid(self) -> list[list[ComponentRef | None]]:
        """The top-row-first grid form (None = empty slot)."""
        n_rows, n_cols = self.shape
        grid: list[list[ComponentRef | None]] = [[None] * n_cols for _ in range(n_rows)]
        for element in self.elements:
            for row, col in element.addresses:
                grid[row][col] = element.ref
        return grid

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
    1 + 6 + 12 = 19 slots. Addresses are (ring, index): index 0 is the
    top slot of the ring, proceeding clockwise; ring k has 6k slots.

    Example — a 2-ring (7-slot) bundle: one center pin plus six around
    it, all the same pin:
        HexLattice.create(
            family_name="seven_pin_bundle",
            version_label="1",
            num_rings=2,
            orientation=HexOrientation.FLAT_TOP,
            pitch_m=0.009,
            fill=sodium.id,
            elements=[
                LatticeElement(
                    ref=ComponentRef(type=ComponentType.CCOMPONENT, id=pin.id),
                    addresses=((0, 0),) + tuple((1, i) for i in range(6)),
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
        outward from its top slot clockwise."""
        return [
            (ring, index)
            for ring in range(self.num_rings)
            for index in range(self.ring_size(ring))
        ]

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
