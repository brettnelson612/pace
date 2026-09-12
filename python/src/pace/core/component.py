"""
pace/core/component.py

Reactor structure objects:
    - LComponent (a geometry + a material, versioned) — a leaf: a
          specific shape paired with a specific composition, e.g. a
          fuel pellet. Versioned exactly like Geometry/Material/
          CComponent: family_name + version_label identity via
          build_id(), and the same three-state derived_from/gt_run_id/
          user_edit lineage rule. Previously content-addressable and
          unversioned (a changed pairing just got a freely-assigned
          new id) — that meant "why did this pairing change" was only
          indirectly recoverable, by noticing which of the two
          references differs and chasing THAT reference's own lineage
          separately. Giving LComponent its own lineage makes it
          queryable the same way as every other layer (children_of(),
          refcounting, reverse-index) and lets ComponentService's
          cascading edit() treat every layer uniformly instead of
          special-casing the leaf.
    - PComponent (a position + one LComponent or CComponent; only
          ever a member of a CComponent) — a leaf or composite placed
          somewhere in space. Deliberately NOT versioned: it has no
          content beyond {pose, component ref}, and because it's
          embedded directly inside CComponent.to_dict()["components"]
          rather than referenced by id, any change to a PComponent
          (a moved pose, a repointed component) is already a change
          to its parent CComponent's own serialized data — that
          already mints a new CComponent version through the existing
          mechanism. Versioning PComponent independently would track
          the same fact twice and would force it to have its own
          registry, contradicting its "no independent registry/
          lifecycle" design.
    - CComponent (a collection of >=2 PComponents, versioned) — a
          composite assembled from multiple positioned components,
          e.g. a fuel pin (a stack of positioned pellets), or a full
          fuel assembly. Renamed from CComponentVersion: the "Version"
          suffix on GeometryVersion/MaterialVersion/CComponentVersion
          was originally there to distinguish the versioned class from
          a bare, unversioned identity class (Geometry/Material/
          CComponent) — but those bare identity classes were removed
          from the design entirely, so there's nothing left for the
          suffix to disambiguate against. Versioning isn't a special
          mode a minority of objects are in here; it's the default
          shape of every non-PComponent object in this module, so the
          name doesn't need to keep announcing it.

PComponent.component being LComponentID | CComponentID (rather than
embedding either directly) is what makes structures recursive: a
CComponent's members can themselves be positioned CComponents, so an
arbitrarily deep hierarchy — pellets -> pins -> assemblies -> a full
core — is built the same way at every level, just nesting one more
PComponent/CComponent layer each time.

Reference direction follows PACE's existing bare-IDs-always rule:
LComponent and CComponent each have their own registry ("lcomponents",
"ccomponents" — see Registry), so PComponent references them by ID.
PComponent itself has no registry ("only ever a member of a
CComponent" — no independent lifecycle), so CComponent embeds
PComponent objects directly rather than referencing them by ID, since
there's nowhere for such an ID to be looked up from.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.geometry import GPose
from pace.core.ids import (
    CComponentID,
    GeometryID,
    GTRunID,
    LComponentID,
    MaterialID,
    PComponentID,
)
from pace.core.pace_object import PaceObject


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
    A version minted here as part of a cascading edit (ComponentService
    propagating a change up through every reference path — see
    component.py's design notes) should carry the SAME cause as the
    root edit that triggered the cascade: if a human hand-edited the
    referenced geometry, the new LComponent version this produces is
    also user_edit=True; if a GT run produced the root change, the new
    LComponent version carries that same gt_run_id. The human or the
    GT run is the root cause of the whole chain, not just the one
    object they directly touched.

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
    def create(cls, *, family_name: str, version_label: str, **kwargs) -> LComponent:
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
            id=data["id"],
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
            geometry=data["geometry"],
            material=data["material"],
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class PComponent(PaceObject):
    """A positioned component — a GPose plus a reference to the one
    LComponent or CComponent placed at that position. Only ever
    appears as a member of a CComponent's `components` list — has no
    independent registry/lifecycle of its own, and deliberately has no
    lineage of its own (see this module's docstring for why).

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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "component": self.component,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PComponent:
        return cls(
            id=data["id"],
            pose=GPose.from_dict(data["pose"]),
            component=data["component"],
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
                ),
                PComponent(
                    id=PComponentID("pc-2"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
                    component=LComponentID("fuel_pellet-1"),
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
    def create(cls, *, family_name: str, version_label: str, **kwargs) -> CComponent:
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
            - no duplicate (component, position) pairs — the same
                  LComponent/CComponent placed twice at the exact same
                  GPose contributes nothing physically and indicates a
                  construction mistake, same reasoning as GAddition/
                  GSubtraction's own duplicate checks.
        """
        if len(self.components) < 2:
            raise ValueError(
                "CComponent must be provided a list of at least two components."
            )

        seen = set()
        for pcomponent in self.components:
            pose = pcomponent.pose
            key = (
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
            id=data["id"],
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
            components=[
                PComponent.from_dict(component) for component in data["components"]
            ],
        )

    def to_open_mc(self):
        raise NotImplementedError

    def to_moose(self):
        raise NotImplementedError
