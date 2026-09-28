"""
pace/core/lattice.py

Lattice — a regular rectangular grid of positions, where position is
implicit from grid index * pitch rather than an explicit PComponent
list. Exists for repeated structures (a fuel assembly's pin array)
where CComponent's explicit-position-per-member approach doesn't
scale to hundreds of members; mirrors OpenMC's RectLattice.

2D rectangular only — 3D and hexagonal lattices are a direct
extension of the same shape, deliberately deferred until needed.

Versioned the same way as Geometry/Material/LComponent/CComponent:
family_name + version_label identity, the same three-state
derived_from/gt_run_id/user_edit lineage rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.ids import CComponentID, LatticeID, LComponentID
from pace.core.pace_object import PaceObject
from pace.core.reference_types import ReferenceableType
from pace.core.versioned import Versioned


@dataclass(frozen=True, kw_only=True)
class LatticeCell(PaceObject):
    """One lattice position's contents — an LComponent, CComponent, or
    nested Lattice, with an explicit type discriminator for the same
    reason PComponent.component_type exists: the bare id string alone
    can't say which registry it belongs to."""

    component: LComponentID | CComponentID | LatticeID
    component_type: ReferenceableType

    def validate(self) -> None:
        if self.component_type not in (
            ReferenceableType.LCOMPONENT,
            ReferenceableType.CCOMPONENT,
            ReferenceableType.LATTICE,
        ):
            raise ValueError(
                "component_type must be LCOMPONENT, CCOMPONENT, or LATTICE, "
                f"got {self.component_type!r}"
            )

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "component_type": self.component_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LatticeCell:
        return cls(
            component=data["component"],
            component_type=ReferenceableType(data["component_type"]),
        )


@dataclass(kw_only=True, frozen=True, eq=False)
class Lattice(Versioned[LatticeID]):
    """A regular 2D rectangular grid — dimensions is (nx, ny),
    pitch_m is center-to-center spacing per axis, lower_left_m is the
    position of the (0,0) cell's own center. universes is flat,
    row-major (index j*nx + i for grid cell (i, j)), length exactly
    nx*ny. outer, if set, fills space outside the lattice's own
    defined extent (e.g. surrounding coolant); if unset, a
    to_open_mc() translation is expected to bound the lattice exactly
    to its extent with no space left over.

    Identity/lineage (id/family_name/version_label/derived_from/
    gt_run_id/user_edit, build_id(), create(), the three-state
    lineage rule) are inherited from Versioned — see that class's
    docstring.

    Example — a 2x1 row of the same pin repeated twice:
        Lattice.create(
            family_name="two_pin_row",
            version_label="1",
            dimensions=(2, 1),
            pitch_m=(0.0126, 0.0126),
            lower_left_m=(-0.0063, -0.0063),
            universes=[
                LatticeCell(
                    component=pin.id,
                    component_type=ReferenceableType.CCOMPONENT,
                ),
                LatticeCell(
                    component=pin.id,
                    component_type=ReferenceableType.CCOMPONENT,
                ),
            ],
        )

    eq=False / id-based equality: `universes` is a list, not hashable
    via the frozen-dataclass default — same pattern as CComponent/
    GAddition/MIsotopic.
    """

    dimensions: tuple[int, int]
    pitch_m: tuple[float, float]
    lower_left_m: tuple[float, float]
    universes: list[LatticeCell]
    outer: LatticeCell | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Lattice) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self) -> None:
        """Check identity/lineage (via Versioned), then the
        lattice-specific grid rules."""
        super().validate()
        self._validate_grid()

    def _validate_grid(self) -> None:
        """Enforce the lattice-specific structural rules:
        - dimensions and pitch_m strictly positive on each axis.
        - universes length exactly matches dimensions[0]*dimensions[1]
              — a lattice whose cell count doesn't match its own
              declared shape is malformed.
        - no direct self-reference — a cell (or outer) whose
              component_type is LATTICE and whose component id
              equals this Lattice's own id. Needs no registry
              access, same reasoning as CComponent's own
              self-reference guard.
        """
        nx, ny = self.dimensions
        if nx < 1 or ny < 1:
            raise ValueError(f"dimensions must be positive, got {self.dimensions}")

        if self.pitch_m[0] <= 0 or self.pitch_m[1] <= 0:
            raise ValueError(f"pitch_m must be positive, got {self.pitch_m}")

        expected_count = nx * ny
        if len(self.universes) != expected_count:
            raise ValueError(
                f"universes has {len(self.universes)} entries, expected "
                f"{expected_count} for dimensions {self.dimensions}"
            )

        cells_to_check = list(self.universes)
        if self.outer is not None:
            cells_to_check.append(self.outer)
        for cell in cells_to_check:
            if (
                cell.component_type == ReferenceableType.LATTICE
                and cell.component == self.id
            ):
                raise ValueError(
                    f"Lattice {self.id!r} cannot reference itself directly"
                )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "dimensions": list(self.dimensions),
            "pitch_m": list(self.pitch_m),
            "lower_left_m": list(self.lower_left_m),
            "universes": [cell.to_dict() for cell in self.universes],
            "outer": self.outer.to_dict() if self.outer is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Lattice:
        return cls(
            **cls._base_fields_from_dict(data),
            dimensions=tuple(data["dimensions"]),
            pitch_m=tuple(data["pitch_m"]),
            lower_left_m=tuple(data["lower_left_m"]),
            universes=[LatticeCell.from_dict(c) for c in data["universes"]],
            outer=(
                LatticeCell.from_dict(data["outer"])
                if data["outer"] is not None
                else None
            ),
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError
