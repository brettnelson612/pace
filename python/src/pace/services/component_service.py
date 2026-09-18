"""
pace/services/component_service.py

ComponentService — the layer above the registry that couples
persistence to business rules a DAO can't enforce on its own:
family_name uniqueness for a v1, dangling-reference validation, and
keeping the reference index (ReferenceDAO) in sync with every create.

register_*() enforces validate_*() internally before every save() —
persistence always goes through the check. The matching validate_*()
methods are also exposed standalone for the Workshop's in-memory draft
mode, so a "check without saving" call is available before a user
commits an edit.

Direct CComponent self-reference is NOT checked here — it needs no
registry access, so it's enforced at construction time by CComponent's
own validate() instead (see component.py). Multi-hop cycle detection
(A contains B contains C contains A) is a real gap, deliberately
deferred for now.

Also deferred to a later pass: refcount-gated edit/delete and
cascading propagation (needs ReferenceDAO's reverse-index queries
plumbed through), radial-fit/axial-fit validation, and batch mode.

Takes the real RegistryDB directly, not a structural stand-in —
ComponentService is always built against RegistryDB in practice, and
RegistryDB's own DAOs already grow in lockstep as new methods are
added, so there's nothing to keep in sync by hand. If ComponentService
ever needs a test double, an in-memory SQLite-backed RegistryDB
(SqlDB("sqlite:///:memory:")) is cheap and exercises real behavior.
"""

from __future__ import annotations

from collections.abc import Callable

from pace.core.component import CComponent, LComponent
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
            reference.referenced_id
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
        if reference.referenced_type == ReferenceableType.GEOMETRY:
            return self._registry.geometries.exists(GeometryID(reference.referenced_id))
        elif reference.referenced_type == ReferenceableType.MATERIAL:
            return self._registry.materials.exists(MaterialID(reference.referenced_id))
        elif reference.referenced_type == ReferenceableType.LCOMPONENT:
            return self._registry.lcomponents.exists(
                LComponentID(reference.referenced_id)
            )
        elif reference.referenced_type == ReferenceableType.CCOMPONENT:
            return self._registry.ccomponents.exists(
                CComponentID(reference.referenced_id)
            )
        else:
            raise ValueError(
                f"unrecognized ReferenceableType: {reference.referenced_type!r}"
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
