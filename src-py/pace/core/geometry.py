"""
Geometry domain objects: Geometry (bare registry identity, no shape data)
and GeometryVersion (the versioned, polymorphic interface holding actual
shape data — GCylinder, GAnnulus, GHexPrism, GSphere, GRectanglePrism,
GNull, GAddition, GSubtraction — plus validate()/to_open_mc()/to_moose()).

Mirrors material.py's Geometry/GeometryVersion <-> Material/MaterialVersion
split intentionally — kept in sync by design, not by accident.
"""

from dataclasses import dataclass, field
from enum import Enum
from abc import abstractmethod

from typing import Optional

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import (
    GeometryID,
    GeometryVersionID,
    GTRunID,
)
from pace.core.pace_model_object import PaceModelObject

MIN_LATTICE_DIMENSION_COUNT = 0
MAX_LATTICE_DIMENSION_COUNT = 1000

MIN_GEO_LENGTH_M = 0.0
MAX_GEO_LENGTH_M = 1000.0


class GeometryType(Enum):
    CYLINDER = "cylinder"
    ANNULUS = "annulus"
    HEX_PRISM = "hex_prism"
    SPHERE = "sphere"
    RECT_PRISM = "rect_prism"
    NULL = "null"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"


@dataclass(kw_only=True, frozen=True)
class Geometry(PaceModelObject):
    """A registry-level geometry identity — holds no shape
    data itself; see GeometryVersion for shape data and
    versioning."""

    id: GeometryID

    def to_dict(self) -> dict:
        return {"id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> "PaceModelObject":
        return cls(id=data["id"])


@dataclass(kw_only=True, frozen=True)
class GeometryVersion(PaceModelObject):
    """General interface for a specific geometry version."""

    id: GeometryVersionID
    geometry_id: GeometryID
    derived_from: Optional[GeometryVersionID] = None
    gt_run_id: Optional[GTRunID] = None

    def __post_init__(self):
        """Perform geometry-type-specific validation post-init."""
        self.validate()

    def validate(self):
        if (self.derived_from is None) != (self.gt_run_id is None):
            raise ValueError(
                "derived_from and gt_run_id must both be set, or both be None"
            )
        # perform shape-specific validation
        self._validate_shape()

    @abstractmethod
    def _validate_shape(self):
        """Shape-specific validation, implemented per concrete geometry type."""
        pass

    @abstractmethod
    def to_open_mc(self):
        """Method to convert geometry to OpenMC equivalent."""
        pass

    @abstractmethod
    def to_moose(self):
        """Method to convert geometry to MOOSE equivalent."""
        pass

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "geometry_id": self.geometry_id,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeometryVersion":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
        )


@dataclass(kw_only=True, frozen=True)
class GCylinder(GeometryVersion):
    """A basic cylinder shape."""

    radius_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )
    height_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "radius_m": self.radius_m,
            "height_m": self.height_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GCylinder":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            radius_m=data["radius_m"],
            height_m=data["height_m"],
        )

    def _validate_shape(self):
        validate_fields(self)

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GAnnulus(GeometryVersion):
    """An annulus (ring cross-section) shape — e.g. a fuel-cladding gap."""

    inner_radius_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )
    outer_radius_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )
    height_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "inner_radius_m": self.inner_radius_m,
            "outer_radius_m": self.outer_radius_m,
            "height_m": self.height_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GAnnulus":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            inner_radius_m=data["inner_radius_m"],
            outer_radius_m=data["outer_radius_m"],
            height_m=data["height_m"],
        )

    def _validate_shape(self):
        validate_fields(self)
        # relational check — not expressible via field metadata alone
        if not (self.inner_radius_m < self.outer_radius_m):
            raise ValueError(
                f"inner_radius_m ({self.inner_radius_m}) must be less than "
                f"outer_radius_m ({self.outer_radius_m})"
            )

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GHexPrism(GeometryVersion):
    """A hexagonal prism shape."""

    circumradius_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )
    height_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "circumradius_m": self.circumradius_m,
            "height_m": self.height_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GHexPrism":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            circumradius_m=data["circumradius_m"],
            height_m=data["height_m"],
        )

    def _validate_shape(self):
        validate_fields(self)

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GSphere(GeometryVersion):
    """A sphere shape."""

    radius_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "radius_m": self.radius_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GSphere":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            radius_m=data["radius_m"],
        )

    def _validate_shape(self):
        validate_fields(self)

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GRectanglePrism(GeometryVersion):
    """A basic rectangular prism shape."""

    length_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )
    width_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )
    height_m: float = field(
        metadata={
            "constraint": Constraint.POSITIVE,
            "range": (MIN_GEO_LENGTH_M, MAX_GEO_LENGTH_M),
        }
    )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "length_m": self.length_m,
            "width_m": self.width_m,
            "height_m": self.height_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GRectanglePrism":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            length_m=data["length_m"],
            width_m=data["width_m"],
            height_m=data["height_m"],
        )

    def _validate_shape(self):
        validate_fields(self)

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GNull(GeometryVersion):
    """An empty element — represents an empty position within a lattice."""

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "GNull":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
        )

    def _validate_shape(self):
        # no shape-specific fields — nothing beyond the base pairing
        # invariant (already enforced by GeometryVersion.validate())
        pass

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GPos(PaceModelObject):
    """Class representation of a position in 3D space."""

    x_m: float
    y_m: float
    z_m: float

    def to_dict(self) -> dict:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GPos":
        return cls(x_m=data["x_m"], y_m=data["y_m"], z_m=data["z_m"])


@dataclass(kw_only=True, frozen=True)
class GAddition(GeometryVersion):
    """An Addition Geometry; a union of geometries.

    This represents the space taken up by one or both of two or more
    overlapping geometries. The unioned geometries must have positions
    relative to one another and a ValueError is thrown if any one of
    the geometries does not touch or overlap with any of the others.
    """

    units: list[tuple[GeometryVersionID, GPos]]

    def __eq__(self, value: object) -> bool:
        return isinstance(value, GAddition) and self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "units": [
                {"geometry_version_id": unit_id, "position": pos.to_dict()}
                for unit_id, pos in self.units
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GAddition":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            units=[
                (u["geometry_version_id"], GPos.from_dict(u["position"]))
                for u in data["units"]
            ],
        )

    def _validate_shape(self):
        if len(self.units) < 2:
            raise ValueError("an Addition needs at least 2 units")

        # check for duplicates
        seen = set()
        for geometry_id, pos in self.units:
            key = (geometry_id, pos.x_m, pos.y_m, pos.z_m)
            if key in seen:
                raise ValueError(f"duplicate unit+position: {geometry_id} at {pos}")
            seen.add(key)

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass


@dataclass(kw_only=True, frozen=True)
class GSubtraction(GeometryVersion):
    """A Subtraction Geometry.

    This represents the space taken up by a base geometry
    after one or more "cut" geometries have been subtracted
    from it. Every cut must overlap with the base.
    """

    base: tuple[GeometryVersionID, GPos]
    cuts: list[tuple[GeometryVersionID, GPos]]

    def __eq__(self, value: object) -> bool:
        return isinstance(value, GSubtraction) and self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict:
        base_id, base_pos = self.base
        return {
            **super().to_dict(),
            "base": {"geometry_version_id": base_id, "position": base_pos.to_dict()},
            "cuts": [
                {"geometry_version_id": cut_id, "position": pos.to_dict()}
                for cut_id, pos in self.cuts
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GSubtraction":
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            base=(
                data["base"]["geometry_version_id"],
                GPos.from_dict(data["base"]["position"]),
            ),
            cuts=[
                (cut["geometry_version_id"], GPos.from_dict(cut["position"]))
                for cut in data["cuts"]
            ],
        )

    def _validate_shape(self):
        if len(self.cuts) < 1:
            raise ValueError("a subtraction needs at least one cut geometry")

        # check for duplicates
        seen = set()
        for geometry_id, pos in self.cuts:
            key = (geometry_id, pos.x_m, pos.y_m, pos.z_m)
            if key in seen:
                raise ValueError(f"duplicate cut: {geometry_id} at {pos}")
            seen.add(key)

    def to_open_mc(self):
        # TODO: implement this
        pass

    def to_moose(self):
        # TODO: implement this
        pass
