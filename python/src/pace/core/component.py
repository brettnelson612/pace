"""
pace/core/component.py

Reactor structure objects:
    - LComponent — a leaf: one geometry paired with one material, no
          position of its own. Versioned like Geometry/Material/
          CComponent: family_name + version_label identity, and the
          same three-state derived_from/gt_run_id/user_edit lineage
          rule.
    - PComponent — a placement: a pose plus a ComponentRef to one
          LComponent, CComponent, or Lattice. Only ever a member of a
          CComponent's `components` list (or a Reactor's root); has no
          independent registry, lifecycle, or lineage of its own. Any
          change to a PComponent (a moved pose, a repointed ref) is a
          change to its parent's own serialized data, which is enough
          to mint a new parent version through the existing mechanism.
    - CComponent — a composite: a region of space it owns (`bounds`),
          the positioned members inside it, and optionally one material
          (`fill`) that occupies whatever part of the bounds no member
          occupies. Versioned the same way as Geometry/Material/
          LComponent.

Bounds and fill (see docs/design/0001-bounds-fill-lattices-reactor.md):
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

Structures are recursive: a PComponent's ref can point at another
CComponent (or a Lattice), so pellets -> rods -> pin cells -> lattices
-> assemblies are all built the same way, one nesting level at a time.
Bare ids are opaque strings with no runtime type information, which is
why placements go through ComponentRef (kind + id) rather than a bare id.

Reference direction follows PACE's bare-ids-always rule: LComponent,
CComponent and Lattice have their own registries, so a placement
references them by id. PComponent has no registry, so CComponent embeds
PComponent objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.component_ref import ComponentKind, ComponentRef
from pace.core.geometry import GPose
from pace.core.ids import (
    CComponentID,
    GeometryID,
    LComponentID,
    MaterialID,
    PComponentID,
)
from pace.core.pace_object import PaceObject
from pace.core.versioned import Versioned


@dataclass(frozen=True, kw_only=True)
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

    A version minted as part of a cascading edit (ComponentService
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

    # No override of validate() — LComponent has nothing beyond the
    # two bare references and Versioned's own id/lineage checks;
    # confirming those references actually resolve to something real
    # requires registry access, which is ComponentService's job, not
    # this object's own validate().

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


@dataclass(frozen=True, kw_only=True)
class PComponent(PaceObject):
    """A placement — a GPose plus a ComponentRef to the one LComponent,
    CComponent, or Lattice placed at that position. Only ever appears
    as a member of a CComponent's `components` list, or as a Reactor's
    root — has no independent registry, lifecycle, or lineage.

    The pose positions the placed object's own origin (the center of
    its bounds) in the parent's coordinate frame.

    id must be unique WITHIN its own parent's `components` list, not
    globally — the same underlying component legitimately gets reused
    across many placements, each with its own locally-unique
    PComponent.id. This is what makes path-based region addressing
    (e.g. "/fuel_pin_cell-1/rod/") work: an id stays attached to its
    PComponent regardless of list order, unlike an array index.

    Example — a fuel rod placed at its pin cell's center:
        PComponent(
            id=PComponentID("rod"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            ref=ComponentRef(kind=ComponentKind.CCOMPONENT, id=fuel_rod.id),
        )
    """

    id: PComponentID
    pose: GPose
    ref: ComponentRef

    def validate(self) -> None:
        if not self.id:
            raise ValueError("PComponent id must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "ref": self.ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PComponent:
        return cls(
            id=PComponentID(data["id"]),
            pose=GPose.from_dict(data["pose"]),
            ref=ComponentRef.from_dict(data["ref"]),
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
    so it's enforced by ComponentService, not here. The same goes for
    "every member lies inside the bounds".

    Example — the VERA problem 1 pin cell: a fuel rod (itself a
    composite bounded by its clad's outer cylinder) placed in a 1.26 cm
    pitch box, with borated water filling the rest:
        CComponent.create(
            family_name="fuel_pin_cell",
            version_label="1",
            bounds=pin_cell_box.id,        # GRectanglePrism, 12.6 mm square
            fill=borated_water.id,
            components=[
                PComponent(
                    id=PComponentID("rod"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    ref=ComponentRef(
                        kind=ComponentKind.CCOMPONENT, id=fuel_rod.id
                    ),
                ),
            ],
        )

    eq=False / id-based equality: `components` is a list, not hashable
    via the frozen-dataclass default — same pattern as GAddition/
    GSubtraction and MIsotopic/MMixture.
    """

    bounds: GeometryID
    fill: MaterialID | None = None
    components: list[PComponent]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CComponent) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self) -> None:
        """Check identity/lineage (via Versioned), then the
        composition-level rules."""
        super().validate()
        self._validate_composition()

    def _validate_composition(self) -> None:
        """Enforce the within-object composition rules — no registry
        access needed (dangling references, bounds shape and
        containment are ComponentService's job).

        Rules enforced:
            - at least 1 member. A composite with no members would be
                  only its fill; an empty lattice slot already covers
                  that case.
            - unique PComponent ids within this composite — ids are the
                  path segments region addresses are built from.
            - no duplicate (ref, pose) pairs — the same component
                  placed twice at the exact same pose contributes
                  nothing physically and indicates a construction
                  mistake. The ref's kind is part of the key, so an
                  LComponent and a CComponent that happen to share an
                  id string are not mistaken for one another.
            - no direct self-reference (a CCOMPONENT ref to this
                  composite's own id). Multi-hop cycles need registry
                  access and are ComponentService's job.
        """
        if len(self.components) < 1:
            raise ValueError("CComponent must have at least one component.")

        seen_ids: set[str] = set()
        seen_placements: set[tuple] = set()
        for pcomponent in self.components:
            if pcomponent.id in seen_ids:
                raise ValueError(
                    f"duplicate PComponent id {pcomponent.id!r} in {self.id!r}"
                )
            seen_ids.add(pcomponent.id)

            ref = pcomponent.ref
            if ref.kind == ComponentKind.CCOMPONENT and ref.id == self.id:
                raise ValueError(
                    f"CComponent {self.id!r} cannot reference itself directly"
                )

            pose = pcomponent.pose
            key = (
                ref.kind,
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
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CComponent:
        return cls(
            **cls._base_fields_from_dict(data),
            bounds=GeometryID(data["bounds"]),
            fill=MaterialID(data["fill"]) if data["fill"] is not None else None,
            components=[
                PComponent.from_dict(component) for component in data["components"]
            ],
        )
