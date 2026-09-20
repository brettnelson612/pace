"""
pace/core/component.py

Reactor structure objects:
    - LComponent — a leaf: one geometry paired with one material, no
          position of its own. Versioned like Geometry/Material/
          CComponent: family_name + version_label identity, and the
          same three-state derived_from/gt_run_id/user_edit lineage
          rule.
    - PComponent — a position plus a reference to one LComponent or
          CComponent. Only ever a member of a CComponent's
          `components` list; has no independent registry, lifecycle,
          or lineage of its own. Any change to a PComponent (a moved
          pose, a repointed component) is a change to its parent
          CComponent's own serialized data, since PComponent is
          embedded rather than referenced by id — that alone is
          enough to mint a new CComponent version through the
          existing mechanism, so PComponent needs no lineage
          machinery of its own.
    - CComponent — a composite: a collection of >=2 positioned
          (PComponent) members, no position of its own. Versioned the
          same way as Geometry/Material/LComponent.

PComponent.component being LComponentID | CComponentID (rather than
embedding either directly) is what makes structures recursive: a
CComponent's members can themselves be positioned CComponents, so an
arbitrarily deep hierarchy — pellets -> pins -> assemblies -> a full
core — is built the same way at every level, just nesting one more
PComponent/CComponent layer each time. LComponentID and CComponentID
are both opaque strings with no runtime type information (see
core/ids.py), so PComponent also carries an explicit component_type
field — there is no way to tell which registry `component` should be
looked up in from the bare id string alone.

Reference direction follows PACE's existing bare-IDs-always rule:
LComponent and CComponent each have their own registry ("lcomponents",
"ccomponents" — see Registry), so PComponent references them by ID.
PComponent itself has no registry, so CComponent embeds PComponent
objects directly rather than referencing them by ID, since there's
nowhere for such an ID to be looked up from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from pace.core.geometry import (
    GAddition,
    Geometry,
    GPose,
    GSubtraction,
    geometry_from_dict,
)
from pace.core.ids import (
    CComponentID,
    GeometryID,
    GTRunID,
    LComponentID,
    MaterialID,
    PComponentID,
)
from pace.core.material import Material, MMixture, material_from_dict
from pace.core.pace_object import PaceObject
from pace.core.reference_types import ReferenceableType


@dataclass(frozen=True, kw_only=True)
class LComponent(PaceObject):
    """A leaf component — one geometry paired with one material, no
    position of its own.

    Identity: id is a single opaque string built from family_name +
    version_label via build_id() — e.g. family_name="fuel_pellet",
    version_label="1" -> id="fuel_pellet-1". Same scheme, same
    _validate_id() consistency check, as Geometry/Material/CComponent.

    family_name has no shape/composition of its own to anchor it to
    the way it does for Geometry or Material — an LComponent IS the
    pairing, so family_name is the caller-chosen name for the
    conceptual SLOT this pairing fills (e.g. "fuel_pellet"), not a
    description of any content that survives a version bump.

    Versioning rule — same three-state rule as Geometry/Material/
    CComponent:
        - derived_from is None: version 1. gt_run_id must be None and
              user_edit must be False.
        - derived_from is set: a derived version. Exactly one of
              gt_run_id or user_edit must also be set — never neither,
              never both.
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

    id: LComponentID
    family_name: str
    version_label: str
    geometry: GeometryID
    material: MaterialID
    derived_from: LComponentID | None = None
    gt_run_id: GTRunID | None = None
    user_edit: bool = False

    @staticmethod
    def build_id(family_name: str, version_label: str) -> LComponentID:
        """The one canonical way an id is constructed from a
        family_name + version_label pair. Used both when constructing
        a new version and by _validate_id() to confirm an existing
        id actually matches its own family_name/version_label."""
        return LComponentID(f"{family_name}-{version_label}")

    @classmethod
    def create(cls, *, family_name: str, version_label: str, **kwargs) -> Self:
        """Named-constructor convenience: derives id via build_id() so
        callers never have to compute and pass it separately.

        Prefer this over calling the constructor directly when
        constructing brand-new versions. from_dict() should keep
        calling cls(...) directly — it already has a trusted, stored
        id and doesn't need one derived.
        """
        return cls(
            id=cls.build_id(family_name, version_label),
            family_name=family_name,
            version_label=version_label,
            **kwargs,
        )

    def validate(self) -> None:
        """Check id/family_name/version_label consistency, then the
        derived_from/gt_run_id/user_edit lineage rule.

        No shape/composition-specific check here — unlike Geometry/
        Material/CComponent, LComponent has nothing beyond the two
        bare references, and confirming those references actually
        resolve to something real requires registry access, which is
        ComponentService's job, not this object's own validate().
        """
        self._validate_id()
        self._validate_lineage()

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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family_name": self.family_name,
            "version_label": self.version_label,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "user_edit": self.user_edit,
            "geometry": self.geometry,
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LComponent:
        return cls(
            id=LComponentID(data["id"]),
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=LComponentID(data["derived_from"]),
            gt_run_id=GTRunID(data["gt_run_id"]),
            user_edit=data["user_edit"],
            geometry=GeometryID(data["geometry"]),
            material=MaterialID(data["material"]),
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class PComponent(PaceObject):
    """A positioned component — a GPose plus a reference to the one
    LComponent or CComponent placed at that position, and an explicit
    tag for which of the two it is. Only ever appears as a member of a
    CComponent's `components` list — has no independent registry,
    lifecycle, or lineage of its own.

    component_type records which registry `component` should be
    looked up in (lcomponents or ccomponents). This can't be inferred
    from the id string alone — LComponentID and CComponentID are both
    opaque strings with no runtime type distinction — so it's carried
    explicitly, the same reason to_dict() tags a Geometry/Material
    subclass with its own "type" key.

    id must be unique WITHIN its own parent's `components` list, not
    globally — the same underlying LComponent/CComponent legitimately
    gets reused across many different placements, each with its own
    locally-unique PComponent.id. This is what makes path-based
    addressing (e.g. "/fuel_pin-1/pc-b/") work: an id stays attached
    to its PComponent regardless of list order, unlike an array index.
    """

    id: PComponentID
    pose: GPose
    component: LComponentID | CComponentID
    component_type: ReferenceableType

    def validate(self) -> None:
        """component_type must actually be one of the two kinds
        PComponent can reference — Geometry/Material are never valid
        here, since neither can be placed directly as a member of a
        CComponent."""
        if self.component_type not in (
            ReferenceableType.LCOMPONENT,
            ReferenceableType.CCOMPONENT,
        ):
            raise ValueError(
                "component_type must be LCOMPONENT or CCOMPONENT, got "
                f"{self.component_type!r}"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "component": self.component,
            "component_type": self.component_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PComponent:
        return cls(
            id=PComponentID(data["id"]),
            pose=GPose.from_dict(data["pose"]),
            component=(
                LComponentID(data["component"])
                if data["component_type"] == ReferenceableType.LCOMPONENT
                else CComponentID(data["component"])
            ),
            component_type=ReferenceableType(data["component_type"]),
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True, eq=False)
class CComponent(PaceObject):
    """A composite component — a collection of >=2 positioned
    (PComponent) members, no position of its own.

    Identity and versioning mirror Geometry/Material exactly: id is a
    single opaque string built from family_name + version_label via
    build_id(); the same three-state derived_from/gt_run_id/user_edit
    lineage rule applies (v1 has none of the three set; a derived
    version has derived_from plus exactly one of gt_run_id or
    user_edit).

    Example — a simple 2-pellet fuel pin stack:
        CComponent.create(
            family_name="fuel_pin",
            version_label="1",
            components=[
                PComponent(
                    id=PComponentID("pc-1"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    component=LComponentID("fuel_pellet-1"),
                    component_type=ReferenceableType.LCOMPONENT,
                ),
                PComponent(
                    id=PComponentID("pc-2"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
                    component=LComponentID("fuel_pellet-1"),
                    component_type=ReferenceableType.LCOMPONENT,
                ),
            ],
        )
    A CComponent's own members can themselves be positioned
    CComponents (via PComponent.component being a CComponentID) — e.g.
    an assembly built from positioned pins, each of which is itself a
    CComponent of positioned pellets.

    eq=False / id-based equality: `components` is a list of PComponent
    objects, not hashable via the frozen-dataclass default — same
    reasoning, and same pattern (isinstance check + self.id ==
    other.id, hash(self.id)), as GAddition/GSubtraction and
    MIsotopic/MMixture.
    """

    id: CComponentID
    family_name: str
    version_label: str
    components: list[PComponent]
    derived_from: CComponentID | None = None
    gt_run_id: GTRunID | None = None
    user_edit: bool = False

    @staticmethod
    def build_id(family_name: str, version_label: str) -> CComponentID:
        """The one canonical way an id is constructed from a
        family_name + version_label pair. Used both when constructing
        a new version and by _validate_id() to confirm an existing
        id actually matches its own family_name/version_label."""
        return CComponentID(f"{family_name}-{version_label}")

    @classmethod
    def create(cls, *, family_name: str, version_label: str, **kwargs) -> Self:
        """Named-constructor convenience: derives id via build_id() so
        callers never have to compute and pass it separately.

        Prefer this over calling the constructor directly when
        constructing brand-new versions. from_dict() should keep
        calling cls(...) directly — it already has a trusted, stored
        id and doesn't need one derived.
        """
        return cls(
            id=cls.build_id(family_name, version_label),
            family_name=family_name,
            version_label=version_label,
            **kwargs,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CComponent) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self):
        """Check id/family_name/version_label consistency, then the
        derived_from/gt_run_id/user_edit lineage rule, then the
        composition-level rule (>=2 members)."""
        self._validate_id()
        self._validate_lineage()
        self._validate_composition()

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

    def _validate_composition(self) -> None:
        """Enforce the composition-level physical rules — pure
        within-object checks, no registry access needed (that's
        ComponentService's job for anything requiring resolving a
        reference — dangling-reference checks, radial/axial fit,
        boundary containment).

        Rules enforced:
            - at least 2 members — a composite of fewer than 2 isn't a
                  composite.
            - no duplicate (component_type, component, position)
                  triples — the same LComponent/CComponent placed
                  twice at the exact same GPose contributes nothing
                  physically and indicates a construction mistake,
                  same reasoning as GAddition/GSubtraction's own
                  duplicate checks. component_type is part of the key
                  because an LComponentID and a CComponentID could
                  coincidentally share the same string across their
                  two separate registries — comparing the bare id
                  alone could misidentify two different objects as one
                  duplicate.
            - no direct self-reference — a PComponent whose
                  component_type is CCOMPONENT and whose component id
                  equals this CComponent's own id. This needs no
                  registry access (unlike a multi-hop cycle, which
                  does, and is ComponentService's job), so it's
                  enforced here rather than at the service layer.
        """
        if len(self.components) < 2:
            raise ValueError(
                "CComponent must be provided a list of at least two components."
            )

        seen = set()
        for pcomponent in self.components:
            if (
                pcomponent.component_type == ReferenceableType.CCOMPONENT
                and pcomponent.component == self.id
            ):
                raise ValueError(
                    f"CComponent {self.id!r} cannot reference itself directly"
                )

            pose = pcomponent.pose
            key = (
                pcomponent.component_type,
                pcomponent.component,
                pose.x_m,
                pose.y_m,
                pose.z_m,
                pose.z_rotation_rad,
            )
            if key in seen:
                raise ValueError(
                    f"duplicate component+position: {pcomponent.component} at {pose}"
                )
            seen.add(key)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family_name": self.family_name,
            "version_label": self.version_label,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "user_edit": self.user_edit,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CComponent:
        return cls(
            id=CComponentID(data["id"]),
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=CComponentID(data["derived_from"]),
            gt_run_id=GTRunID(data["gt_run_id"]),
            user_edit=data["user_edit"],
            components=[
                PComponent.from_dict(component) for component in data["components"]
            ],
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedCComponent(PaceObject):
    """A fully hydrated CComponent tree: the root CComponent plus
    every Geometry/Material/LComponent/CComponent it transitively
    references, as flat id-keyed maps rather than a nested object
    graph with embedded content. A component reused at several
    positions (e.g. one pellet LComponent placed six times in a pin)
    appears once in its map, referenced by id from wherever it's
    used — never duplicated.

    Not itself a registered/versioned object — has no id, family_name,
    or lineage of its own, since it's a derived, transient view over
    already-persisted data, not something anyone creates or edits
    directly.
    """

    root: CComponentID
    ccomponents: dict[CComponentID, CComponent]
    lcomponents: dict[LComponentID, LComponent]
    geometries: dict[GeometryID, Geometry]
    materials: dict[MaterialID, Material]

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "ccomponents": {
                key: ccomponent.to_dict()
                for key, ccomponent in self.ccomponents.items()
            },
            "lcomponents": {
                key: lcomponent.to_dict()
                for key, lcomponent in self.lcomponents.items()
            },
            "geometries": {
                key: geometry.to_dict() for key, geometry in self.geometries.items()
            },
            "materials": {
                key: material.to_dict() for key, material in self.materials.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedCComponent:
        ccomponents = {
            CComponentID(key): CComponent.from_dict(ccomponent_dict)
            for key, ccomponent_dict in data["ccomponents"].items()
        }
        lcomponents = {
            LComponentID(key): LComponent.from_dict(lcomponent_dict)
            for key, lcomponent_dict in data["lcomponents"].items()
        }
        geometries = {
            GeometryID(key): geometry_from_dict(geometry_dict)
            for key, geometry_dict in data["geometries"].items()
        }
        materials = {
            MaterialID(key): material_from_dict(material_dict)
            for key, material_dict in data["materials"].items()
        }
        return cls(
            root=CComponentID(data["root"]),
            ccomponents=ccomponents,
            lcomponents=lcomponents,
            geometries=geometries,
            materials=materials,
        )

    def validate(self) -> None:
        self._validate_id_mapping()
        self._validate_root_provided()
        self._validate_completeness()

    def _validate_root_provided(self) -> None:
        """Check that the root component is provided."""
        if self.root not in self.ccomponents:
            raise ValueError(
                f"root CComponent {self.root!r} not present in ccomponents"
            )

    def _validate_id_mapping(self) -> None:
        """Every dict key must match the id of the object stored under
        it — catches a bundle assembled with a mismatched key/value
        pair (e.g. a bug in whatever hydration walk built this)."""
        for ccomponent_id, ccomponent in self.ccomponents.items():
            if ccomponent.id != ccomponent_id:
                raise ValueError(
                    f"CComponent {ccomponent.id!r} stored under mismatched "
                    f"key {ccomponent_id!r}"
                )
        for lcomponent_id, lcomponent in self.lcomponents.items():
            if lcomponent.id != lcomponent_id:
                raise ValueError(
                    f"LComponent {lcomponent.id!r} stored under mismatched "
                    f"key {lcomponent_id!r}"
                )
        for geometry_id, geometry in self.geometries.items():
            if geometry.id != geometry_id:
                raise ValueError(
                    f"Geometry {geometry.id!r} stored under mismatched key "
                    f"{geometry_id!r}"
                )
        for material_id, material in self.materials.items():
            if material.id != material_id:
                raise ValueError(
                    f"Material {material.id!r} stored under mismatched key "
                    f"{material_id!r}"
                )

    def _validate_completeness(self) -> None:
        """Every reference embedded in this bundle's own CComponents,
        LComponents, and CSG-composite geometries/mixtures must resolve to
        something also present in this bundle — a bundle missing an entry
        is only partially hydrated, which defeats the whole point of
        ResolvedCComponent. No registry access needed, since this only
        checks the bundle's own internal consistency.
        """
        for ccomponent in self.ccomponents.values():
            for pcomponent in ccomponent.components:
                target_map = (
                    self.lcomponents
                    if pcomponent.component_type == ReferenceableType.LCOMPONENT
                    else self.ccomponents
                )
                if pcomponent.component not in target_map:
                    raise ValueError(
                        f"CComponent {ccomponent.id!r} references "
                        f"{pcomponent.component!r}, which is not present in "
                        "this bundle."
                    )

        for lcomponent in self.lcomponents.values():
            if lcomponent.geometry not in self.geometries:
                raise ValueError(
                    f"LComponent {lcomponent.id!r} references geometry "
                    f"{lcomponent.geometry!r}, which is not present in this "
                    "bundle."
                )
            if lcomponent.material not in self.materials:
                raise ValueError(
                    f"LComponent {lcomponent.id!r} references material "
                    f"{lcomponent.material!r}, which is not present in this "
                    "bundle."
                )

        for geometry in self.geometries.values():
            if isinstance(geometry, GAddition):
                referenced_geometry_ids = {unit_id for unit_id, _ in geometry.units}
            elif isinstance(geometry, GSubtraction):
                base_id, _ = geometry.base
                referenced_geometry_ids = {base_id} | {
                    cut_id for cut_id, _ in geometry.cuts
                }
            else:
                referenced_geometry_ids = set()

            for geo_ref_id in referenced_geometry_ids:
                if geo_ref_id not in self.geometries:
                    raise ValueError(
                        f"Geometry {geometry.id!r} references geometry "
                        f"{geo_ref_id!r}, which is not present in this "
                        "bundle."
                    )

        for material in self.materials.values():
            if isinstance(material, MMixture):
                referenced_material_ids = {
                    material_id for material_id, _ in material.components
                }
            else:
                referenced_material_ids = set()

            for mat_ref_id in referenced_material_ids:
                if mat_ref_id not in self.materials:
                    raise ValueError(
                        f"Material {material.id!r} references material "
                        f"{mat_ref_id!r}, which is not present in this "
                        "bundle."
                    )
