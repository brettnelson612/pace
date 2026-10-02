"""
pace/core/component.py

Reactor structure objects:
    - LComponent — a leaf: one geometry paired with one material, no
          position of its own. Versioned like Geometry/Material/
          CComponent: family_name + version_label identity, and the
          same three-state derived_from/gt_run_id/user_edit lineage
          rule.
    - ComponentPlacement — a component placement: a pose plus a ComponentRef to one
          LComponent, CComponent, or Lattice. Only ever a member of a
          CComponent's `placements` list (or a Reactor's root); has no
          independent registry, lifecycle, or lineage of its own. Any
          change to a ComponentPlacement (a moved pose, a repointed ref) is a
          change to its parent's own serialized data, which is enough
          to mint a new parent version through the existing mechanism.
    - CComponent — a composite: a region of space it owns (`bounds`),
          the positioned members inside it, and optionally one material
          (`fill`) that occupies whatever part of the bounds no member
          occupies. Versioned the same way as Geometry/Material/
          LComponent.

Bounds and fill:
    - bounds is the region a composite owns — never a physical wall.
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
    - Components are placed, never used as a fill. A placed child's
          size comes from its own bounds; the parent only says where it
          goes.

Structures are recursive: a ComponentPlacement's ref can point at another
CComponent (or a Lattice), so pellets -> rods -> pin cells -> lattices
-> assemblies are all built the same way, one nesting level at a time.
Bare ids are opaque strings with no runtime type information, which is
why placements go through ComponentRef (kind + id) rather than a bare id.

Reference direction follows PACE's bare-ids-always rule: LComponent,
CComponent and Lattice have their own registries, so a placement
references them by id. ComponentPlacement has no registry, so CComponent embeds
ComponentPlacement objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.reference_types import ReferenceableType
from pace.core.geometry import GPose
from pace.core.ids import (
    CComponentID,
    GeometryID,
    LComponentID,
    MaterialID,
    ComponentPlacementID,
    LatticeID,
    ComponentID,
)
from pace.core.pace_object import PaceObject
from pace.core.versioned import Versioned
from enum import StrEnum


class ComponentType(StrEnum):
    """The kinds of object a composite or lattice can place.

    Values match ReferenceableType's, so converting between the two is a
    lookup by value (see ComponentRef.referenceable_type).
    """

    LCOMPONENT = "lcomponent"
    CCOMPONENT = "ccomponent"
    LATTICE = "lattice"


@dataclass(frozen=True, kw_only=True)
class ComponentRef(PaceObject):
    """A typed reference to a placeable object.

    Example — pointing at a registered pin-cell composite:
        ComponentRef(type=ComponentType.CCOMPONENT, id=pin_cell.id)
    """

    type: ComponentType
    id: ComponentID

    @property
    def referenceable_type(self) -> ReferenceableType:
        """The reference-index type tag for this ref's target."""
        return ReferenceableType(self.type.value)

    def validate(self) -> None:
        if not isinstance(self.type, ComponentType):
            raise TypeError(
                f"type must be a ComponentType, got {self.type!r} "
                f"({type(self.type).__name__})"
            )
        if not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> dict:
        return {"type": self.type.value, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> ComponentRef:
        type = ComponentType(data["type"])
        raw_id = data["id"]
        typed_id: ComponentID
        if type == ComponentType.LCOMPONENT:
            typed_id = LComponentID(raw_id)
        elif type == ComponentType.CCOMPONENT:
            typed_id = CComponentID(raw_id)
        else:
            typed_id = LatticeID(raw_id)
        return cls(type=type, id=typed_id)


@dataclass(frozen=True, kw_only=True)
class ComponentPlacement(PaceObject):
    """A component placement — a GPose plus a ComponentRef to the LComponent,
    CComponent, or Lattice placed at that position. Only ever appears
    as a member of a CComponent's `placements` list, or as a Reactor's
    root — has no independent registry, lifecycle, or lineage.

    The pose positions the placed object's own origin (the center of
    its bounds) in the parent's coordinate frame.

    id must be unique WITHIN its own parent's `placements` list, not
    globally — the same underlying component legitimately gets reused
    across many placements, each with its own locally-unique
    ComponentPlacement.id. This is what makes path-based region addressing
    (e.g. "/fuel_pin_cell-1/rod/") work: an id stays attached to its
    ComponentPlacement regardless of list order, unlike an array index.

    Example — a fuel rod placed at its pin cell's center:
        ComponentPlacement(
            id=ComponentPlacementID("rod"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            ref=ComponentRef(type=ComponentType.CCOMPONENT, id=fuel_rod.id),
        )
    """

    id: ComponentPlacementID
    pose: GPose
    ref: ComponentRef

    def validate(self) -> None:
        if not self.id:
            raise ValueError("ComponentPlacement id must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "ref": self.ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ComponentPlacement:
        return cls(
            id=ComponentPlacementID(data["id"]),
            pose=GPose.from_dict(data["pose"]),
            ref=ComponentRef.from_dict(data["ref"]),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class LComponent(Versioned[LComponentID]):
    """A leaf component — one geometry paired with one material, no
    position of its own.

    Identity/lineage (id/family_name/version_label/derived_from/
    gt_run_id/user_edit, build_id(), create(), the three-state
    lineage rule) are inherited from Versioned — see that class's
    docstring.

    family_name has no shape/composition of its own to anchor it to
    the way it does for Geometry or Material — an LComponent IS the
    pairing, so family_name is the caller-chosen name for the
    conceptual SLOT this pairing fills (e.g. "fuel_pellet"), not a
    description of any content that survives a version bump.

    A version minted as part of a cascading edit (RegistryService
    propagating a change up through every reference path) carries the
    SAME cause as the root edit that triggered the cascade: if a human
    hand-edited the referenced geometry, the new LComponent version
    this produces is also user_edit=True; if a GT run produced the
    root change, the new LComponent version carries that same
    gt_run_id. The human or the GT run is the root cause of the whole
    chain, not just the one object they directly touched.

    Example — a UO2 fuel pellet, referencing an already-defined
    cylinder geometry and enriched-UO2 material:
        LComponent.create(
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
    def from_dict(cls, data: dict) -> LComponent:
        return cls(
            **cls._base_fields_from_dict(data),
            geometry=GeometryID(data["geometry"]),
            material=MaterialID(data["material"]),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CComponent(Versioned[CComponentID]):
    """A composite — the region of space it owns (`bounds`), the
    positioned members inside it, and optionally the one material
    (`fill`) occupying whatever part of the bounds no member occupies.
    Has no position of its own; whoever places it supplies one.

    Identity and versioning mirror Geometry/Material exactly (see
    Versioned).

    bounds must reference a GRectanglePrism, GHexPrism or GCylinder —
    the shapes every translator can turn into a clipping region and the
    Reactor module can mesh. That check needs the referenced geometry,
    so it's enforced by RegistryService, not here. The same goes for
    "every member lies inside the bounds".

    Example — the VERA problem 1 pin cell: a fuel rod (itself a
    composite bounded by its clad's outer cylinder) placed in a 1.26 cm
    pitch box, with borated water filling the rest:
        CComponent.create(
            family_name="fuel_pin_cell",
            version_label="1",
            bounds=pin_cell_box.id,        # GRectanglePrism, 12.6 mm square
            fill=borated_water.id,
            placements=[
                ComponentPlacement(
                    id=ComponentPlacementID("rod"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    ref=ComponentRef(
                        type=ComponentType.CCOMPONENT, id=fuel_rod.id
                    ),
                ),
            ],
        )
    """

    bounds: GeometryID
    fill: MaterialID | None = None
    placements: list[ComponentPlacement]

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
            - at least 1 member. A composite with no members would be
                  only its fill; an empty lattice slot already covers
                  that case.
            - unique ComponentPlacement ids within this composite — ids are the
                  path segments region addresses are built from.
            - no duplicate (ref, pose) pairs — the same component
                  placed twice at the exact same pose contributes
                  nothing physically and indicates a construction
                  mistake. The ref's kind is part of the key, so an
                  LComponent and a CComponent that happen to share an
                  id string are not mistaken for one another.
            - no direct self-reference (a CCOMPONENT ref to this
                  composite's own id). Multi-hop cycles need registry
                  access and are RegistryService's job.
        """
        if len(self.placements) < 1:
            raise ValueError("CComponent must have at least one component.")

        seen_ids: set[str] = set()
        seen_placements: set[tuple] = set()
        for placement in self.placements:
            if placement.id in seen_ids:
                raise ValueError(
                    f"duplicate ComponentPlacement id {placement.id!r} in {self.id!r}"
                )
            seen_ids.add(placement.id)

            ref = placement.ref
            if ref.type == ComponentType.CCOMPONENT and ref.id == self.id:
                raise ValueError(
                    f"CComponent {self.id!r} cannot reference itself directly"
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
                raise ValueError(f"duplicate component+position: {ref.id} at {pose}")
            seen_placements.add(key)

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "bounds": self.bounds,
            "fill": self.fill,
            "placements": [component.to_dict() for component in self.placements],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CComponent:
        return cls(
            **cls._base_fields_from_dict(data),
            bounds=GeometryID(data["bounds"]),
            fill=MaterialID(data["fill"]) if data["fill"] is not None else None,
            placements=[
                ComponentPlacement.from_dict(component)
                for component in data["placements"]
            ],
        )
