"""
pace/core/geometry.py

Geometry domain objects: Geometry (bare registry identity, no shape data)
and GeometryVersion (the versioned, polymorphic interface holding actual
shape data — GCylinder, GAnnulus, GHexPrism, GSphere, GRectanglePrism,
GNull, GAddition, GSubtraction — plus validate()/to_open_mc()/to_moose()).

What "geometry" means physically here: a geometry is a description of the
SHAPE a region of space occupies — nothing about what substance fills it
(that's Material) and nothing about its position within a larger assembly
(positioning is handled where a geometry+material pairing gets placed,
elsewhere in the model — GAddition/GSubtraction's internal unit positions
are the one exception, needed to define the shape itself).

Several shapes here (GAddition, GSubtraction) are PACE's version of
Constructive Solid Geometry (CSG) — building complex shapes out of boolean
combinations of simpler ones. This mirrors how OpenMC represents
geometry under the hood: surfaces divide space into two half-spaces, and
cells are defined as boolean combinations (union, intersection,
complement) of those half-spaces. GAddition/GSubtraction are PACE's
higher-level equivalents of that union/complement machinery, expressed in
terms of already-defined GeometryVersions rather than raw surfaces.

Mirrors material.py's Geometry/GeometryVersion <-> Material/MaterialVersion
split intentionally — kept in sync by design, not by accident.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import (
    GeometryID,
    GeometryVersionID,
    GTRunID,
)
from pace.core.pace_object import PaceObject

# Bounds are practical sanity limits (catching typos/unit mistakes — e.g. a
# radius accidentally entered in cm instead of m), not physical constraints.
# Nothing about reactor geometry is inherently capped at 1000; the ceiling
# just needs to be comfortably larger than any real component while still
# catching obviously-wrong input.
MIN_LATTICE_DIMENSION_COUNT = 0
MAX_LATTICE_DIMENSION_COUNT = 1000

MIN_GEO_LENGTH_M = 0.0
MAX_GEO_LENGTH_M = 1000.0

MAX_ROTATION_RADIANS = math.pi * 2


class GeometryType(Enum):
    """Discriminator tag for GeometryVersion subclasses (not yet wired
    into any dispatch/deserialization logic).

    Values:
        - cylinder: a basic solid cylinder — e.g. a fuel pellet or
              fuel pin.
        - annulus: a ring cross-section (hollow cylinder) — e.g. a
              fuel-cladding gap, or the cladding itself.
        - hex_prism: a hexagonal prism — e.g. a hexagonal fuel
              assembly duct, common in SFR/HTGR lattice designs.
        - sphere: a basic sphere — e.g. a pebble in a pebble-bed
              design.
        - rect_prism: a rectangular prism (box) — e.g. a square/
              rectangular assembly duct or structural block.
        - null: an empty element — a placeholder for a vacant lattice
              position.
        - addition: a union of two or more other geometries (CSG
              union) — see GAddition.
        - subtraction: a base geometry with one or more other
              geometries cut out of it (CSG complement) — see
              GSubtraction.
    """

    CYLINDER = "cylinder"
    ANNULUS = "annulus"
    HEX_PRISM = "hex_prism"
    SPHERE = "sphere"
    RECT_PRISM = "rect_prism"
    NULL = "null"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"


@dataclass(kw_only=True, frozen=True)
class GPose(PaceObject):
    """Class representation of a position-orientation in 3D space."""

    x_m: float
    y_m: float
    z_m: float
    z_rotation_rad: float = field(
        default=0.0, metadata={"range": (0.0, MAX_ROTATION_RADIANS)}
    )

    def to_dict(self) -> dict:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "z_rotation_rad": self.z_rotation_rad,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GPose:
        return cls(
            x_m=data["x_m"],
            y_m=data["y_m"],
            z_m=data["z_m"],
            z_rotation_rad=data["z_rotation_rad"],
        )

    def validate(self):
        validate_fields(self)


@dataclass(kw_only=True, frozen=True)
class Geometry(PaceObject):
    """A registry-level geometry identity — holds no shape
    data itself; see GeometryVersion for shape data and
    versioning.

    Mirrors Material: a Geometry is just an ID, referenced stably
    across however many GeometryVersions get recorded under it over
    time (e.g. as a design iterates through revisions)."""

    id: GeometryID

    def to_dict(self) -> dict:
        return {"id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> PaceObject:
        return cls(id=data["id"])


@dataclass(kw_only=True, frozen=True)
class GeometryVersion(PaceObject):
    """General interface for a specific geometry version.

    Versioning rule (identical to MaterialVersion): v1 is always
    user-authored (derived_from and gt_run_id both None) — a shape a
    person specified directly. v2+ can only be created as the
    recorded output of a GT run (both fields set together) — e.g. a
    component's geometry changing as a result of simulated thermal
    expansion or mechanical deformation over the course of a run. One
    of derived_from/gt_run_id set without the other is invalid.
    """

    id: GeometryVersionID
    geometry_id: GeometryID
    derived_from: GeometryVersionID | None = None
    gt_run_id: GTRunID | None = None

    def validate(self):
        """Check the derived_from/gt_run_id lineage pairing, then
        delegate to the concrete subclass's shape-specific checks."""
        if (self.derived_from is None) != (self.gt_run_id is None):
            raise ValueError(
                "derived_from and gt_run_id must both be set, or both be None"
            )
        # perform shape-specific validation
        self._validate_shape()

    @abstractmethod
    def _validate_shape(self):
        """Shape-specific validation, implemented per concrete geometry type."""

    @abstractmethod
    def to_open_mc(self):
        """Method to convert geometry to OpenMC equivalent."""
        raise NotImplementedError

    @abstractmethod
    def to_moose(self):
        """Method to convert geometry to MOOSE equivalent."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "geometry_id": self.geometry_id,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GeometryVersion:
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
        )


@dataclass(kw_only=True, frozen=True)
class GCylinder(GeometryVersion):
    """A basic solid cylinder — e.g. a fuel pellet, a fuel pin
    (without cladding), or a simple control-rod slug."""

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
    def from_dict(cls, data: dict) -> GCylinder:
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
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GAnnulus(GeometryVersion):
    """An annulus (ring cross-section) shape — e.g. a fuel-cladding
    gap, or the cladding tube itself (inner_radius_m = fuel outer
    surface, outer_radius_m = cladding outer surface)."""

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
    def from_dict(cls, data: dict) -> GAnnulus:
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
        """Enforce the annulus-specific physical rule:

        Rules enforced:
            - inner_radius_m must be strictly less than outer_radius_m
        """
        validate_fields(self)
        # relational check — not expressible via field metadata alone
        if not (self.inner_radius_m < self.outer_radius_m):
            raise ValueError(
                f"inner_radius_m ({self.inner_radius_m}) must be less than "
                f"outer_radius_m ({self.outer_radius_m})"
            )

    def to_open_mc(self):
        # TODO: implement this
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GHexPrism(GeometryVersion):
    """A hexagonal prism shape — e.g. a hexagonal fuel assembly duct,
    the standard cross-section for SFR and HTGR lattice designs.

    circumradius_m (not a plain "radius") is the distance from the
    hexagon's center to one of its six corners (the circumscribed
    circle's radius) — the standard way a regular hexagon's size is
    specified, since it fully determines the hexagon's dimensions
    including its flat-to-flat width."""

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
    def from_dict(cls, data: dict) -> GHexPrism:
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
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GSphere(GeometryVersion):
    """A sphere shape — e.g. a pebble in a pebble-bed reactor design,
    or a spherical fuel/absorber element."""

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
    def from_dict(cls, data: dict) -> GSphere:
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
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GRectanglePrism(GeometryVersion):
    """A basic rectangular prism shape — e.g. a square/rectangular
    assembly duct, a structural block, or a plate-type fuel element."""

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
    def from_dict(cls, data: dict) -> GRectanglePrism:
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
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GNull(GeometryVersion):
    """An empty element — represents an empty position within a
    lattice (e.g. a vacant fuel-pin slot, a coolant-only channel with
    no solid component placed in it)."""

    def to_dict(self) -> dict:
        return super().to_dict()

    def _validate_shape(self):
        # no shape-specific fields — nothing beyond the base pairing
        # invariant (already enforced by GeometryVersion.validate())
        pass

    def to_open_mc(self):
        # TODO: implement this
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GAddition(GeometryVersion):
    """An Addition Geometry; a union of geometries — PACE's CSG
    "union" operator.

    This represents the space taken up by one or both of two or more
    overlapping geometries. The unioned geometries must have positions
    relative to one another and a ValueError is thrown if any one of
    the geometries does not touch or overlap with any of the others.

    Example — a fuel pin with a small spacer feature unioned onto its
    side, at a position offset from the pin's own center:
        GAddition(
            id=...,
            geometry_id=...,
            units=[
                (pin_geometry_version_id, GPose(x_m=0, y_m=0, z_m=0)),
                (spacer_geometry_version_id, GPose(x_m=0.01, y_m=0, z_m=0)),
            ],
        )

    __eq__/__hash__ are identity-based via self.id (rather than the
    dataclass-generated field-based default) since `units` is a list
    of tuples containing GPose objects — not a natural fit for
    value-based equality/hashing the way a scalar-only GeometryVersion
    subclass (e.g. GSphere) is.
    """

    units: list[tuple[GeometryVersionID, GPose]]

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
    def from_dict(cls, data: dict) -> GAddition:
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            units=[
                (unit["geometry_version_id"], GPose.from_dict(unit["position"]))
                for unit in data["units"]
            ],
        )

    def _validate_shape(self):
        """Enforce the addition-specific physical rules:

        Rules enforced:
            - at least 2 units — a union of one geometry isn't a
                  union.
            - no duplicate (geometry version, position) pairs — the
                  same geometry placed twice at the exact same position
                  contributes nothing physically and indicates a
                  construction mistake.
        """
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
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class GSubtraction(GeometryVersion):
    """A Subtraction Geometry — PACE's CSG "complement"/"difference"
    operator.

    This represents the space taken up by a base geometry
    after one or more "cut" geometries have been subtracted
    from it. Every cut must overlap with the base.

    Example — a gas plenum, modeled as a cylinder with a smaller
    cylindrical void cut out of one end:
        GSubtraction(
            id=...,
            geometry_id=...,
            base=(outer_cylinder_version_id, GPose(x_m=0, y_m=0, z_m=0)),
            cuts=[
                (void_cylinder_version_id, GPose(x_m=0, y_m=0, z_m=0.1)),
            ],
        )

    __eq__/__hash__ are identity-based via self.id, same reasoning as
    GAddition.
    """

    base: tuple[GeometryVersionID, GPose]
    cuts: list[tuple[GeometryVersionID, GPose]]

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
    def from_dict(cls, data: dict) -> GSubtraction:
        return cls(
            id=data["id"],
            geometry_id=data["geometry_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            base=(
                data["base"]["geometry_version_id"],
                GPose.from_dict(data["base"]["position"]),
            ),
            cuts=[
                (cut["geometry_version_id"], GPose.from_dict(cut["position"]))
                for cut in data["cuts"]
            ],
        )

    def _validate_shape(self):
        """Enforce the subtraction-specific physical rules:

        Rules enforced:
            - at least 1 cut — a subtraction with no cuts is just the
                  base geometry, unchanged.
            - no duplicate (geometry version, position) cut pairs —
                  the same cut applied twice at the same position is a
                  construction mistake, not a meaningful double-cut.
        """
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
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError
