"""
pace/services/component_service.py

ComponentService: business logic over RegistryDB — persistence rules
a DAO can't enforce on its own.

- register_*(): validates (family uniqueness, dangling refs, and for
  CComponents/Reactors the bounds shape and face coverage) then
  persists and updates the reference index. Always runs the checks —
  there's no unvalidated save path.
- validate_*(): the same checks, standalone, for a check-before-save
  flow (e.g. a draft UI that shouldn't persist on every keystroke).
- get_*(): fetch one row, unhydrated (embedded refs stay as bare ids).
- get_resolved_component() / get_resolved_reactor(): fetch a full hydrated
  tree. Walks RegistryDB.references (cheap — just edges) to find
  everything needed, then does one batched fetch per kind. The result
  self-validates on construction (PaceObject), so a mismatch between
  the reference index and what's actually in the DB fails loudly here
  rather than returning a partial tree.

Not handled here yet: multi-hop cycles, geometric containment (members
inside bounds, lattice occupants inside their slot), atomic
register (save + index in one transaction), edit/delete, batch mode.
"""

from __future__ import annotations

from collections.abc import Callable

from pace.core.bounds import BOUNDS_GEOMETRY_TYPES, faces_for_geometry_type
from pace.core.component import CComponent, LComponent, ComponentRef
from pace.core.geometry import Geometry
from pace.core.ids import (
    CComponentID,
    GeometryID,
    LatticeID,
    LComponentID,
    MaterialID,
    ReactorBlueprintID,
)
from pace.core.lattice import Lattice
from pace.core.material import Material
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.core.reference_types import ReferenceableType
from pace.core.resolved import ResolvedComponent, ResolvedReactorBlueprint
from pace.db.registry_db import RegistryDB
from pace.db.relational.reference import Reference
from pace.services.reference_extraction import (
    extract_ccomponent_references,
    extract_geometry_references,
    extract_lattice_references,
    extract_lcomponent_references,
    extract_material_references,
    extract_reactor_references,
)


class FamilyAlreadyExistsError(ValueError):
    """Raised when registering a v1 (derived_from=None) under a
    family_name that already has at least one version on record."""


class DanglingReferenceError(ValueError):
    """Raised when a referenced id does not resolve to an existing
    record in the registry it targets."""


class InvalidBoundsError(ValueError):
    """Raised when a bounds geometry is a shape that can't serve as
    bounds, or a Reactor's boundary conditions don't match the faces of
    its bounds shape."""


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
        self.validate_ccomponent(component)
        self._registry.ccomponents.save(component)
        self._registry.references.add_many(
            list(extract_ccomponent_references(component))
        )
        return component

    def register_lattice(self, lattice: Lattice) -> Lattice:
        self._reject_duplicate_family(
            self._registry.lattices.family_exists,
            lattice.family_name,
            lattice.derived_from,
        )
        references = extract_lattice_references(lattice)
        self.validate_references(list(references))
        self._registry.lattices.save(lattice)
        self._registry.references.add_many(list(references))
        return lattice

    def register_reactor_blueprint(
        self, reactor_blueprint: ReactorBlueprint
    ) -> ReactorBlueprint:
        self._reject_duplicate_family(
            self._registry.reactors.family_exists,
            reactor_blueprint.family_name,
            reactor_blueprint.derived_from,
        )
        self.validate_reactor_blueprint(reactor_blueprint)
        self._registry.reactors.save(reactor_blueprint)
        self._registry.references.add_many(
            list(extract_reactor_references(reactor_blueprint))
        )
        return reactor_blueprint

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

    def get_lattice(self, lattice_id: LatticeID) -> Lattice | None:
        return self._registry.lattices.get(lattice_id)

    def get_reactor(self, reactor_id: ReactorBlueprintID) -> ReactorBlueprint | None:
        return self._registry.reactors.get(reactor_id)

    # -------------------------------------------------------------
    # Retrieval — fully hydrated
    # -------------------------------------------------------------

    def get_resolved_component(self, root: ComponentRef) -> ResolvedComponent:
        """Hydrate the full transitive tree a root component depends on.

        1. Walk RegistryDB.references to find every id needed (cheap —
           edges only, no object reconstruction).
        2. Batch-fetch the real objects, one query per kind.

        Returns a ResolvedComponent, which self-validates on construction:
        if the reference index and the fetched objects ever disagree
        (e.g. something was deleted between the walk and the fetch),
        construction fails loudly rather than returning a partial tree.
        """
        needed = self._walk_references(root.referenceable_type, root.id)
        return self._build_resolved_component(root, needed)

    def get_resolved_reactor_blueprint(
        self, reactor_blueprint_id: ReactorBlueprintID
    ) -> ResolvedReactorBlueprint:
        """Hydrate a Reactor and everything it transitively references —
        its root's whole tree plus its own bounds, fill, and every
        material named in its operating state. This is the input the
        solver adapters take."""
        reactor = self._registry.reactors.get(reactor_blueprint_id)
        if reactor is None:
            raise DanglingReferenceError(
                f"reactor {reactor_blueprint_id!r} does not exist"
            )
        needed = self._walk_references(
            ReferenceableType.REACTOR_BLUEPRINT, reactor_blueprint_id
        )
        return ResolvedReactorBlueprint(
            reactor_blueprint=reactor,
            rc=self._build_resolved_component(reactor.root.ref, needed),
        )

    def _walk_references(
        self, seed_type: ReferenceableType, seed_id: str
    ) -> dict[ReferenceableType, set[str]]:
        """Every id reachable from the seed through the reference index,
        grouped by kind (the seed itself included)."""
        needed: dict[ReferenceableType, set[str]] = {
            kind: set() for kind in ReferenceableType
        }
        needed[seed_type].add(seed_id)
        frontier: list[tuple[ReferenceableType, str]] = [(seed_type, seed_id)]

        while frontier:
            edges = self._registry.references.outgoing_many(frontier)
            frontier = []
            for edge_type, edge_id in edges:
                if edge_id not in needed[edge_type]:
                    needed[edge_type].add(edge_id)
                    frontier.append((edge_type, edge_id))
        return needed

    def _build_resolved_component(
        self, root: ComponentRef, needed: dict[ReferenceableType, set[str]]
    ) -> ResolvedComponent:
        return ResolvedComponent(
            root=root,
            ccomponents=self._registry.ccomponents.get_many(
                [CComponentID(i) for i in needed[ReferenceableType.CCOMPONENT]]
            ),
            lcomponents=self._registry.lcomponents.get_many(
                [LComponentID(i) for i in needed[ReferenceableType.LCOMPONENT]]
            ),
            lattices=self._registry.lattices.get_many(
                [LatticeID(i) for i in needed[ReferenceableType.LATTICE]]
            ),
            geometries=self._registry.geometries.get_many(
                [GeometryID(i) for i in needed[ReferenceableType.GEOMETRY]]
            ),
            materials=self._registry.materials.get_many(
                [MaterialID(i) for i in needed[ReferenceableType.MATERIAL]]
            ),
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
        """Confirm every reference resolves (bounds, fill, members), then
        that the bounds geometry is a shape that can serve as bounds."""
        self.validate_references(list(extract_ccomponent_references(ccomponent)))
        self._require_bounds_shape(ccomponent.bounds)

    def validate_lattice(self, lattice: Lattice) -> None:
        """Confirm the fill and every placed component resolve."""
        self.validate_references(list(extract_lattice_references(lattice)))

    def validate_reactor_blueprint(self, reactor_blueprint: ReactorBlueprint) -> None:
        """Confirm every reference resolves, the bounds geometry is a
        valid bounds shape, and the boundary conditions match its faces:
        neutron conditions on exactly the shape's faces; thermal and
        flow conditions only on faces the shape has."""
        self.validate_references(list(extract_reactor_references(reactor_blueprint)))
        geometry = self._require_bounds_shape(reactor_blueprint.bounds)
        faces = set(faces_for_geometry_type(geometry.geometry_type))

        neutron_faces = set(reactor_blueprint.neutron_bcs)
        if neutron_faces != faces:
            missing = sorted(f.value for f in faces - neutron_faces)
            extra = sorted(f.value for f in neutron_faces - faces)
            raise InvalidBoundsError(
                f"ReactorBlueprint {reactor_blueprint.id!r} neutron conditions must cover exactly "
                f"the faces of its bounds: missing {missing}, not a face {extra}"
            )
        for label, conditions in (
            ("thermal", reactor_blueprint.thermal_bcs),
            ("flow", reactor_blueprint.flow_bcs),
        ):
            extra_faces = set(conditions) - faces
            if extra_faces:
                raise InvalidBoundsError(
                    f"ReactorBlueprint {reactor_blueprint.id!r} has {label} conditions on faces its "
                    f"bounds don't have: {sorted(f.value for f in extra_faces)}"
                )

    def _require_bounds_shape(self, geometry_id: GeometryID) -> Geometry:
        """The bounds geometry, which must exist and be a GRectanglePrism,
        GHexPrism or GCylinder."""
        geometry = self._registry.geometries.get(geometry_id)
        if geometry is None:
            raise DanglingReferenceError(
                f"bounds geometry {geometry_id!r} does not exist"
            )
        if geometry.geometry_type not in BOUNDS_GEOMETRY_TYPES:
            raise InvalidBoundsError(
                f"geometry {geometry_id!r} is a {geometry.geometry_type.value!r}; "
                "bounds must be a rect_prism, hex_prism or cylinder"
            )
        return geometry

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
        target_id = reference.target_id
        exists_by_type: dict[ReferenceableType, Callable[[], bool]] = {
            ReferenceableType.GEOMETRY: lambda: self._registry.geometries.exists(
                GeometryID(target_id)
            ),
            ReferenceableType.MATERIAL: lambda: self._registry.materials.exists(
                MaterialID(target_id)
            ),
            ReferenceableType.LCOMPONENT: lambda: self._registry.lcomponents.exists(
                LComponentID(target_id)
            ),
            ReferenceableType.CCOMPONENT: lambda: self._registry.ccomponents.exists(
                CComponentID(target_id)
            ),
            ReferenceableType.LATTICE: lambda: self._registry.lattices.exists(
                LatticeID(target_id)
            ),
            ReferenceableType.REACTOR_BLUEPRINT: lambda: self._registry.reactors.exists(
                ReactorBlueprintID(target_id)
            ),
        }
        exists = exists_by_type.get(reference.target_type)
        if exists is None:
            raise ValueError(
                f"unrecognized ReferenceableType: {reference.target_type!r}"
            )
        return exists()

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
