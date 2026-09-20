"""
pace/services/component_service.py

ComponentService: business logic over RegistryDB — persistence rules
a DAO can't enforce on its own.

- register_*(): validates (family uniqueness, dangling refs) then
  persists and updates the reference index. Always runs the checks —
  there's no unvalidated save path.
- validate_*(): the same checks, standalone, for a check-before-save
  flow (e.g. a draft UI that shouldn't persist on every keystroke).
- get_*(): fetch one row, unhydrated (embedded refs stay as bare ids).
- get_resolved_ccomponent(): fetch the full hydrated tree. Walks
  RegistryDB.references (cheap — just edges) to find everything
  needed, then does one batched fetch per type. The resulting
  ResolvedCComponent self-validates on construction (PaceObject), so
  a mismatch between the reference index and what's actually in the
  DB fails loudly here rather than returning a partial tree.

Not handled here: CComponent self-reference (checked at construction,
see component.py), multi-hop cycles, edit/delete, batch mode — all
deferred for now.
"""

from __future__ import annotations

from collections.abc import Callable

from pace.core.component import CComponent, LComponent, ResolvedCComponent
from pace.core.geometry import Geometry
from pace.core.ids import CComponentID, GeometryID, LComponentID, MaterialID
from pace.core.material import Material
from pace.core.reference_types import ReferenceableType
from pace.db.registry_db import RegistryDB
from pace.db.relational.reference import Reference
from pace.services.reference_extraction import (
    extract_ccomponent_references,
    extract_geometry_references,
    extract_lcomponent_references,
    extract_material_references,
)


class FamilyAlreadyExistsError(ValueError):
    """Raised when registering a v1 (derived_from=None) under a
    family_name that already has at least one version on record."""


class DanglingReferenceError(ValueError):
    """Raised when a referenced id does not resolve to an existing
    record in the registry it targets."""


class ComponentService:
    def __init__(self, registry: RegistryDB):
        self._registry = registry

    # -------------------------------------------------------------
    # Creation
    # -------------------------------------------------------------

    def register_geometry(self, geometry: Geometry) -> Geometry:
        self._reject_duplicate_family(
            self._registry.geometries.family_exists,
            geometry.family_name,
            geometry.derived_from,
        )
        references = extract_geometry_references(geometry)
        self.validate_references(list(references))
        self._registry.geometries.save(geometry)
        self._registry.references.add_many(list(references))
        return geometry

    def register_material(self, material: Material) -> Material:
        self._reject_duplicate_family(
            self._registry.materials.family_exists,
            material.family_name,
            material.derived_from,
        )
        references = extract_material_references(material)
        self.validate_references(list(references))
        self._registry.materials.save(material)
        self._registry.references.add_many(list(references))
        return material

    def register_lcomponent(self, component: LComponent) -> LComponent:
        self._reject_duplicate_family(
            self._registry.lcomponents.family_exists,
            component.family_name,
            component.derived_from,
        )
        references = extract_lcomponent_references(component)
        self.validate_references(list(references))
        self._registry.lcomponents.save(component)
        self._registry.references.add_many(list(references))
        return component

    def register_ccomponent(self, component: CComponent) -> CComponent:
        # Direct self-reference is already rejected at construction
        # time by CComponent's own validate() — see component.py.
        self._reject_duplicate_family(
            self._registry.ccomponents.family_exists,
            component.family_name,
            component.derived_from,
        )
        references = extract_ccomponent_references(component)
        self.validate_references(list(references))
        self._registry.ccomponents.save(component)
        self._registry.references.add_many(list(references))
        return component

    # -------------------------------------------------------------
    # Retrieval — single row, unhydrated
    # -------------------------------------------------------------

    def get_geometry(self, geometry_id: GeometryID) -> Geometry | None:
        return self._registry.geometries.get(geometry_id)

    def get_material(self, material_id: MaterialID) -> Material | None:
        return self._registry.materials.get(material_id)

    def get_lcomponent(self, lcomponent_id: LComponentID) -> LComponent | None:
        return self._registry.lcomponents.get(lcomponent_id)

    def get_ccomponent(self, ccomponent_id: CComponentID) -> CComponent | None:
        return self._registry.ccomponents.get(ccomponent_id)

    # -------------------------------------------------------------
    # Retrieval — fully hydrated
    # -------------------------------------------------------------

    def get_resolved_ccomponent(self, root_id: CComponentID) -> ResolvedCComponent:
        """Hydrate the full transitive tree a root CComponent depends on.

        1. Walk RegistryDB.references to find every id needed (cheap —
           edges only, no object reconstruction).
        2. Batch-fetch the real objects, one query per aggregate type.

        Returns a ResolvedCComponent, which self-validates on construction.
        """
        needed: dict[ReferenceableType, set[str]] = {
            ReferenceableType.CCOMPONENT: {root_id},
            ReferenceableType.LCOMPONENT: set(),
            ReferenceableType.GEOMETRY: set(),
            ReferenceableType.MATERIAL: set(),
        }
        frontier: list[tuple[ReferenceableType, str]] = [
            (ReferenceableType.CCOMPONENT, root_id)
        ]

        while frontier:
            edges = self._registry.references.outgoing_many(frontier)
            frontier = []
            for edge_type, edge_id in edges:
                if edge_id not in needed[edge_type]:
                    needed[edge_type].add(edge_id)
                    frontier.append((edge_type, edge_id))

        ccomponents = self._registry.ccomponents.get_many(
            [CComponentID(i) for i in needed[ReferenceableType.CCOMPONENT]]
        )
        lcomponents = self._registry.lcomponents.get_many(
            [LComponentID(i) for i in needed[ReferenceableType.LCOMPONENT]]
        )
        geometries = self._registry.geometries.get_many(
            [GeometryID(i) for i in needed[ReferenceableType.GEOMETRY]]
        )
        materials = self._registry.materials.get_many(
            [MaterialID(i) for i in needed[ReferenceableType.MATERIAL]]
        )

        # ResolvedCComponent.validate() runs automatically at
        # construction (PaceObject.__post_init__) — it's the actual
        # completeness check. If the reference index and the fetched
        # objects ever disagree (e.g. something was deleted between
        # the graph walk above and this fetch), construction fails
        # loudly here rather than returning a silently partial tree.
        return ResolvedCComponent(
            root=root_id,
            ccomponents=ccomponents,
            lcomponents=lcomponents,
            geometries=geometries,
            materials=materials,
        )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    def validate_geometry(self, geometry: Geometry) -> None:
        """Confirm every id a Geometry references (GAddition.units /
        GSubtraction.base+cuts) resolves to an existing Geometry."""
        self.validate_references(list(extract_geometry_references(geometry)))

    def validate_material(self, material: Material) -> None:
        """Confirm every id a Material references (MMixture.components)
        resolves to an existing Material."""
        self.validate_references(list(extract_material_references(material)))

    def validate_lcomponent(self, lcomponent: LComponent) -> None:
        """Confirm the Geometry and Material an LComponent references
        both resolve to existing records."""
        self.validate_references(list(extract_lcomponent_references(lcomponent)))

    def validate_ccomponent(self, ccomponent: CComponent) -> None:
        """Confirm every LComponent/CComponent a CComponent's
        PComponent members reference resolves to an existing record.
        Direct self-reference is not checked here — see
        register_ccomponent()."""
        self.validate_references(list(extract_ccomponent_references(ccomponent)))

    def validate_references(self, references: list[Reference]) -> None:
        """Check that every reference in `references` resolves to an
        existing record in the registry it targets."""
        dangling = [
            reference.target_id
            for reference in references
            if not self._reference_resolves(reference)
        ]
        if dangling:
            raise DanglingReferenceError(
                f"the following referenced ids do not exist: {dangling}"
            )

    def _reference_resolves(self, reference: Reference) -> bool:
        """Whether a single reference resolves to an existing record
        in its respective table."""
        if reference.target_type == ReferenceableType.GEOMETRY:
            return self._registry.geometries.exists(GeometryID(reference.target_id))
        elif reference.target_type == ReferenceableType.MATERIAL:
            return self._registry.materials.exists(MaterialID(reference.target_id))
        elif reference.target_type == ReferenceableType.LCOMPONENT:
            return self._registry.lcomponents.exists(LComponentID(reference.target_id))
        elif reference.target_type == ReferenceableType.CCOMPONENT:
            return self._registry.ccomponents.exists(CComponentID(reference.target_id))
        else:
            raise ValueError(
                f"unrecognized ReferenceableType: {reference.target_type!r}"
            )

    @staticmethod
    def _reject_duplicate_family(
        family_exists: Callable[[str], bool],
        family_name: str,
        derived_from: str | None,
    ) -> None:
        """The one genuinely registry-wide check — as opposed to
        everything else here, which is scoped to the subtree touched
        by one edit: a brand-new v1 (derived_from is None) can't be
        registered under a family_name that already has at least one
        version on record. A derived version (v2+) legitimately shares
        its family_name with every other version in that family, so
        this only applies to v1s."""
        if derived_from is None and family_exists(family_name):
            raise FamilyAlreadyExistsError(
                f"family_name {family_name!r} already exists — cannot "
                "register a new v1 under an existing family"
            )
