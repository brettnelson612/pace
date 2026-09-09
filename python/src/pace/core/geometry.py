"""
core/geometry.py

Geometry domain objects: GeometryVersion (the versioned, polymorphic
class holding actual shape data — GCylinder, GAnnulus, GHexPrism,
GSphere, GRectanglePrism, GNull, GAddition, GSubtraction — plus
validate()/to_open_mc()/to_moose()).

No separate bare "Geometry" identity class/table — a version's family
is just a `family_name` string carried on the version itself, not a
separate row. A version's id is one opaque string built from
family_name + version_label (e.g. "uranium3.2-1",
"uranium3.2-6month_depletion" — see GeometryVersion.build_id()), which
is what every reference elsewhere in the system (LComponent,
GAddition.units, etc.) actually points at.

family_name is NOT unique per row — every version in a family shares
it. Rejecting a duplicate family_name is enforced only at "create a
brand-new family" (v1, derived_from=None) time, and that's application
logic (ComponentService), not a database constraint, since the column
legitimately repeats across many rows.

What "geometry" means physically here: a geometry is a description of
the SHAPE a region of space occupies — nothing about what substance
fills it (that's Material) and nothing about its position within a
larger assembly (positioning is handled where a geometry+material
pairing gets placed, elsewhere in the model — GAddition/GSubtraction's
internal unit positions are the one exception, needed to define the
shape itself).

Several shapes here (GAddition, GSubtraction) are PACE's version of
Constructive Solid Geometry (CSG) — building complex shapes out of
boolean combinations of simpler ones. This mirrors how OpenMC itself
represents geometry under the hood: surfaces divide space into two
half-spaces, and cells are defined as boolean combinations (union,
intersection, complement) of those half-spaces. GAddition/GSubtraction
are PACE's higher-level equivalents of that union/complement machinery,
expressed in terms of already-defined GeometryVersions rather than raw
surfaces.

Mirrors material.py's structure intentionally — kept in sync by
design, not by accident.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import GeometryVersionID, GTRunID
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


class GeometryType(Enum):
    """Discriminator tag for GeometryVersion subclasses.

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
class GeometryVersion(PaceObject):
    """General interface for a specific geometry version.

    Identity: id is a single opaque string built from family_name +
    version_label via build_id() — e.g. family_name="fuel_pellet",
    version_label="1" -> id="fuel_pellet-1". _validate_id() confirms
    the two stay consistent, catching an accidentally mismatched/
    copy-pasted id at construction time.

    Versioning rule: a version is either v1 (user-authored from
    scratch) or a derived version, and if derived, it has exactly one
    recorded cause:
        - derived_from is None: version 1. gt_run_id must be None and
              user_edit must be False — a v1 has no derivation cause
              because it has no predecessor.
        - derived_from is set: a derived version. Exactly one of
              gt_run_id (this version is the recorded output of a GT
              run — e.g. thermal expansion or mechanical deformation
              simulated over the course of a run) or user_edit (a
              person directly edited a predecessor version, e.g. via
              the workshop) must also be set — never neither, and
              never both, since a single version bump can't
              simultaneously be GT-run output and a manual edit.
    """

    id: GeometryVersionID
    family_name: str
    version_label: str
    derived_from: GeometryVersionID | None = None
    gt_run_id: GTRunID | None = None
    user_edit: bool = False

    @staticmethod
    def build_id(family_name: str, version_label: str) -> GeometryVersionID:
        """The one canonical way an id is constructed from a
        family_name + version_label pair. Used both when constructing
        a new version and by _validate_id() to confirm an existing
        id actually matches its own family_name/version_label."""
        return GeometryVersionID(f"{family_name}-{version_label}")

    def validate(self):
        """Check id/family_name/version_label consistency, then the
        derived_from/gt_run_id/user_edit lineage rule, then delegate
        to the concrete subclass's shape-specific checks."""
        self._validate_id()
        self._validate_lineage()
        self._validate_shape()

    def _validate_id(self) -> None:
        expected = self.build_id(self.family_name, self.version_label)
        if self.id != expected:
            raise ValueError(
                f"id {self.id!r} does not match family_name/version_label "
                f"(expected {expected!r})"
            )

    def _validate_lineage(self) -> None:
        has_predecessor = self.derived_from is not None
        has_gt_run = self.gt_run_id is not None

        if not has_predecessor:
            if has_gt_run or self.user_edit:
                raise ValueError(
                    "gt_run_id must be unset and user_edit must be False "
                    "when derived_from is unset (version 1 has no "
                    "derivation cause)"
                )
        elif has_gt_run == self.user_edit:
            # both set, or both unset — either way, invalid
            raise ValueError(
                "when derived_from is set, exactly one of gt_run_id or "
                "user_edit must also be set (got gt_run_id="
                f"{self.gt_run_id!r}, user_edit={self.user_edit!r})"
            )

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
        type_value = CLASS_TO_GEOMETRY_TYPE.get(type(self))
        if type_value is None:
            raise ValueError(
                f"{type(self).__name__} is not registered in "
                "GEOMETRY_TYPE_TO_CLASS — add it there before calling "
                "to_dict() on this class"
            )
        return {
            "type": type_value,
            "id": self.id,
            "family_name": self.family_name,
            "version_label": self.version_label,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "user_edit": self.user_edit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GeometryVersion:
        return cls(
            id=data["id"],
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
            inner_radius_m=data["inner_radius_m"],
            outer_radius_m=data["outer_radius_m"],
            height_m=data["height_m"],
        )

    def _validate_shape(self):
        """Enforce the annulus-specific physical rule:

        Rules enforced:
            - inner_radius_m must be strictly less than outer_radius_m
                  — an annulus with inner >= outer describes a solid
                  cylinder or a degenerate/inverted shape, not a ring.
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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
class GPose(PaceObject):
    """A position (and z-axis orientation) in 3D space, used to place
    one GeometryVersion relative to another (e.g. within a GAddition's
    units or a GSubtraction's base/cuts).

    Deliberately a plain value object, not a PaceModelObject-style
    registered entity — a GPose has no independent identity or
    lifecycle of its own; it only exists as an attribute of whatever
    placement it describes.

    z_rotation_rad defaults to 0.0 (no rotation) so existing call
    sites that don't need rotation are unaffected. Only z-axis
    rotation is modeled — sufficient for placing a non-rotationally-
    symmetric geometry (e.g. an off-axis GHexPrism within a lattice)
    without needing full 3D orientation machinery not yet needed
    anywhere in the current design.
    """

    x_m: float
    y_m: float
    z_m: float
    z_rotation_rad: float = 0.0

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
            z_rotation_rad=data.get("z_rotation_rad", 0.0),
        )


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
            id=GeometryVersion.build_id("pin_with_spacer", "1"),
            family_name="pin_with_spacer",
            version_label="1",
            units=[
                (pin_geometry_version_id, GPose(x_m=0, y_m=0, z_m=0)),
                (spacer_geometry_version_id, GPose(x_m=0.01, y_m=0, z_m=0)),
            ],
        )
    Note: the ids inside `units` reference OTHER, already-existing
    GeometryVersions — unaffected by this class's own identity scheme.

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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
            key = (geometry_id, pos.x_m, pos.y_m, pos.z_m, pos.z_rotation_rad)
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
            id=GeometryVersion.build_id("gas_plenum", "1"),
            family_name="gas_plenum",
            version_label="1",
            base=(outer_cylinder_version_id, GPose(x_m=0, y_m=0, z_m=0)),
            cuts=[
                (void_cylinder_version_id, GPose(x_m=0, y_m=0, z_m=0.1)),
            ],
        )
    Note: the ids inside `base`/`cuts` reference OTHER, already-existing
    GeometryVersions — unaffected by this class's own identity scheme.

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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
            key = (geometry_id, pos.x_m, pos.y_m, pos.z_m, pos.z_rotation_rad)
            if key in seen:
                raise ValueError(f"duplicate cut: {geometry_id} at {pos}")
            seen.add(key)

    def to_open_mc(self):
        # TODO: implement this
        raise NotImplementedError

    def to_moose(self):
        # TODO: implement this
        raise NotImplementedError


# =============================================================================
# Type dispatch — maps GeometryType values to their concrete class, and
# back. Lives here, not in the persistence layer, because it's a fact about
# this module's own class hierarchy, not about how anything gets stored.
#
# GeometryVersion.to_dict() (defined earlier in this file) references
# CLASS_TO_GEOMETRY_TYPE before it's defined here — fine, since Python
# resolves names inside a method body at call time, not at
# class-definition time, and by the time to_dict() is ever actually called
# the whole module has finished loading.
#
# Also used by anything doing polymorphic reconstruction from a stored
# `type` string (e.g. GeometryVersionRepository).
#
# Must be kept in sync by hand whenever a new GeometryVersion subclass is
# added — nothing enforces that automatically.
# =============================================================================
GEOMETRY_TYPE_TO_CLASS: dict[str, type[GeometryVersion]] = {
    GeometryType.CYLINDER.value: GCylinder,
    GeometryType.ANNULUS.value: GAnnulus,
    GeometryType.HEX_PRISM.value: GHexPrism,
    GeometryType.SPHERE.value: GSphere,
    GeometryType.RECT_PRISM.value: GRectanglePrism,
    GeometryType.NULL.value: GNull,
    GeometryType.ADDITION.value: GAddition,
    GeometryType.SUBTRACTION.value: GSubtraction,
}

# Inverse lookup, keyed on exact type (not isinstance) — avoids any
# ambiguity if the class hierarchy ever grows a subclass of a subclass.
CLASS_TO_GEOMETRY_TYPE: dict[type[GeometryVersion], str] = {
    cls: type_value for type_value, cls in GEOMETRY_TYPE_TO_CLASS.items()
}
