"""
pace/services/registry_service.py

RegistryService: the rules for writing to and reading from the
registry that a DAO can't enforce on its own.

- register_*(): checks, then writes the object and its reference-index
  edges in one transaction. The checks, in order:
      1. the id is not already registered — a registered version is
         never overwritten;
      2. lineage — a v1 (derived_from=None) may not reuse an existing
         family_name; a v2+ must derive from an existing version of the
         same type and family;
      3. the object's validate_*() checks (dangling references, plus
         bounds shape and face coverage for CComponents and
         ReactorBlueprints).
- validate_*(): step 3 on its own, for a check-before-save flow (e.g. a
  draft UI that shouldn't persist on every keystroke).
- get_*(): fetch one object, unhydrated (embedded refs stay as bare
  ids).
- get_resolved_component() / get_resolved_reactor_blueprint(): fetch a
  fully hydrated tree. Walks the reference index (edges only) to find
  everything needed, then does one batched fetch per type. The result
  self-validates on construction, so a mismatch between the index and
  the stored objects fails loudly instead of returning a partial tree.

Not handled here yet: multi-hop cycles, geometric containment (members
inside bounds, lattice occupants inside their slot), edit/delete,
batch mode.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, TypeVar

from sqlalchemy.exc import IntegrityError

from pace.core.bounds import BOUNDS_GEOMETRY_TYPES, faces_for_geometry_type
from pace.core.component import CComponent, ComponentRef, LComponent
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
from pace.core.versioned import Versioned
from pace.db.registry_db import RegistryDB
from pace.db.relational.reference import Reference
from pace.db.relational.versioned_dao import VersionedDAO
from pace.services.reference_extraction import (
    extract_ccomponent_references,
    extract_geometry_references,
    extract_lattice_references,
    extract_lcomponent_references,
    extract_material_references,
    extract_reactor_blueprint_references,
)

ModelT = TypeVar("ModelT", bound=Versioned)


class VersionAlreadyExistsError(ValueError):
    """Raised when registering an object whose id is already
    registered."""


class FamilyAlreadyExistsError(ValueError):
    """Raised when registering a v1 (derived_from=None) under a
    family_name that already has at least one version on record."""


class InvalidPredecessorError(ValueError):
    """Raised when a v2+ object's derived_from does not name an
    existing version of the same type and family."""


class DanglingReferenceError(ValueError):
    """Raised when a referenced id does not resolve to an existing
    record in the registry it targets."""


class InvalidBoundsError(ValueError):
    """Raised when a bounds geometry is a shape that can't serve as
    bounds, or a ReactorBlueprint's boundary conditions don't match the
    faces of its bounds shape."""


class RegistryService:
    def __init__(self, registry: RegistryDB):
        self._registry = registry

    # -------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------

    def register_geometry(self, geometry: Geometry) -> Geometry:
        return self._register(
            geometry,
            ReferenceableType.GEOMETRY,
            self.validate_geometry,
            extract_geometry_references,
        )

    def register_material(self, material: Material) -> Material:
        return self._register(
            material,
            ReferenceableType.MATERIAL,
            self.validate_material,
            extract_material_references,
        )

    def register_lcomponent(self, lcomponent: LComponent) -> LComponent:
        return self._register(
            lcomponent,
            ReferenceableType.LCOMPONENT,
            self.validate_lcomponent,
            extract_lcomponent_references,
        )

    def register_ccomponent(self, ccomponent: CComponent) -> CComponent:
        return self._register(
            ccomponent,
            ReferenceableType.CCOMPONENT,
            self.validate_ccomponent,
            extract_ccomponent_references,
        )

    def register_lattice(self, lattice: Lattice) -> Lattice:
        return self._register(
            lattice,
            ReferenceableType.LATTICE,
            self.validate_lattice,
            extract_lattice_references,
        )

    def register_reactor_blueprint(
        self, reactor_blueprint: ReactorBlueprint
    ) -> ReactorBlueprint:
        return self._register(
            reactor_blueprint,
            ReferenceableType.REACTOR_BLUEPRINT,
            self.validate_reactor_blueprint,
            extract_reactor_blueprint_references,
        )

    def _register(
        self,
        model: ModelT,
        referenceable_type: ReferenceableType,
        validate: Callable[[ModelT], None],
        extract_references: Callable[[ModelT], set[Reference]],
    ) -> ModelT:
        """Check, then write `model` and its reference edges in one
        transaction. If another writer registers the same id between
        the check and the write, the primary key rejects the insert and
        that is reported as VersionAlreadyExistsError too."""

        dao = self._registry.dao_for(referenceable_type)
        self._reject_existing_id(dao, model)
        self._validate_lineage(dao, model)
        validate(model)

        try:
            with self._registry.transaction() as session:
                dao.insert(session, model)
                self._registry.references.add_many(session, extract_references(model))
        except IntegrityError as error:
            if dao.exists(model.id):
                raise VersionAlreadyExistsError(_already_registered(model)) from error
            raise
        return model

    @staticmethod
    def _reject_existing_id(dao: VersionedDAO[Any, Any, Any], model: Versioned) -> None:
        if dao.exists(model.id):
            raise VersionAlreadyExistsError(_already_registered(model))

    @staticmethod
    def _validate_lineage(dao: VersionedDAO[Any, Any, Any], model: Versioned) -> None:
        """A v1 may not reuse an existing family_name. A v2+ must derive
        from an existing version in the same table (so the same type)
        and the same family."""
        if model.derived_from is None:
            if dao.family_exists(model.family_name):
                raise FamilyAlreadyExistsError(
                    f"family_name {model.family_name!r} already exists — cannot "
                    "register a new v1 under an existing family"
                )
            return

        predecessor = dao.describe(model.derived_from)
        if predecessor is None:
            raise InvalidPredecessorError(
                f"{model.id!r} derives from {model.derived_from!r}, which is not "
                f"a registered {type(model).__name__}"
            )
        if predecessor["family_name"] != model.family_name:
            raise InvalidPredecessorError(
                f"{model.id!r} (family {model.family_name!r}) derives from "
                f"{model.derived_from!r}, which belongs to family "
                f"{predecessor['family_name']!r}"
            )

    # -------------------------------------------------------------
    # Retrieval — single object, unhydrated
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

    def get_reactor_blueprint(
        self, reactor_blueprint_id: ReactorBlueprintID
    ) -> ReactorBlueprint | None:
        return self._registry.reactor_blueprints.get(reactor_blueprint_id)

    # -------------------------------------------------------------
    # Retrieval — fully hydrated
    # -------------------------------------------------------------

    def get_resolved_component(self, root: ComponentRef) -> ResolvedComponent:
        """Hydrate the full transitive tree a root component depends on.

        1. Walk the reference index to find every id needed (edges
           only, no object reconstruction).
        2. Batch-fetch the objects, one query per type.

        The ResolvedComponent self-validates on construction: if the
        index and the fetched objects disagree (e.g. something was
        deleted between the walk and the fetch), construction fails
        rather than returning a partial tree.
        """
        needed = self._walk_references(root.referenceable_type, root.id)
        return self._build_resolved_component(root, needed)

    def get_resolved_reactor_blueprint(
        self, reactor_blueprint_id: ReactorBlueprintID
    ) -> ResolvedReactorBlueprint:
        """Hydrate a ReactorBlueprint and everything it transitively
        references — its root's whole tree plus its own bounds and
        fill. This is the model input the solver adapters take.
        """
        reactor_blueprint = self._registry.reactor_blueprints.get(reactor_blueprint_id)
        if reactor_blueprint is None:
            raise DanglingReferenceError(
                f"reactor blueprint {reactor_blueprint_id!r} does not exist"
            )
        needed = self._walk_references(
            ReferenceableType.REACTOR_BLUEPRINT, reactor_blueprint_id
        )
        return ResolvedReactorBlueprint(
            reactor_blueprint=reactor_blueprint,
            rc=self._build_resolved_component(reactor_blueprint.root.ref, needed),
        )

    def _walk_references(
        self, seed_type: ReferenceableType, seed_id: str
    ) -> dict[ReferenceableType, set[str]]:
        """Every id reachable from the seed through the reference index,
        grouped by type (the seed itself included)."""
        needed: dict[ReferenceableType, set[str]] = {
            referenceable_type: set() for referenceable_type in ReferenceableType
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
        def fetch(referenceable_type: ReferenceableType) -> dict:
            dao = self._registry.dao_for(referenceable_type)
            return dao.get_many(needed[referenceable_type])

        return ResolvedComponent(
            root=root,
            ccomponents=fetch(ReferenceableType.CCOMPONENT),
            lcomponents=fetch(ReferenceableType.LCOMPONENT),
            lattices=fetch(ReferenceableType.LATTICE),
            geometries=fetch(ReferenceableType.GEOMETRY),
            materials=fetch(ReferenceableType.MATERIAL),
        )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    def validate_geometry(self, geometry: Geometry) -> None:
        """Confirm every id a Geometry references (GAddition.units /
        GSubtraction.base+cuts) resolves to an existing Geometry."""
        self.validate_references(extract_geometry_references(geometry))

    def validate_material(self, material: Material) -> None:
        """Confirm every id a Material references (MMixture.components)
        resolves to an existing Material."""
        self.validate_references(extract_material_references(material))

    def validate_lcomponent(self, lcomponent: LComponent) -> None:
        """Confirm the Geometry and Material an LComponent references
        both resolve to existing records."""
        self.validate_references(extract_lcomponent_references(lcomponent))

    def validate_ccomponent(self, ccomponent: CComponent) -> None:
        """Confirm every reference resolves (bounds, fill, members),
        then that the bounds geometry is a shape that can serve as
        bounds. Direct self-reference is rejected earlier, by
        CComponent's own validate()."""
        self.validate_references(extract_ccomponent_references(ccomponent))
        self._require_bounds_shape(ccomponent.bounds)

    def validate_lattice(self, lattice: Lattice) -> None:
        """Confirm the fill and every placed component resolve."""
        self.validate_references(extract_lattice_references(lattice))

    def validate_reactor_blueprint(self, reactor_blueprint: ReactorBlueprint) -> None:
        """Confirm every reference resolves, the bounds geometry is a
        valid bounds shape, and the boundary conditions match its faces:
        neutron conditions on exactly the shape's faces; thermal and
        flow conditions only on faces the shape has."""
        self.validate_references(
            extract_reactor_blueprint_references(reactor_blueprint)
        )
        geometry = self._require_bounds_shape(reactor_blueprint.bounds)
        bounds_faces = set(faces_for_geometry_type(geometry.geometry_type))
        owner = f"ReactorBlueprint {reactor_blueprint.id!r}"

        neutron_faces = set(reactor_blueprint.neutron_bcs)
        if neutron_faces != bounds_faces:
            missing = sorted(face.value for face in bounds_faces - neutron_faces)
            extra = sorted(face.value for face in neutron_faces - bounds_faces)
            raise InvalidBoundsError(
                f"{owner} neutron conditions must cover exactly the faces of "
                f"its bounds: missing {missing}, not a face {extra}"
            )
        for label, conditions in (
            ("thermal", reactor_blueprint.thermal_bcs),
            ("flow", reactor_blueprint.flow_bcs),
        ):
            extra_faces = set(conditions) - bounds_faces
            if extra_faces:
                raise InvalidBoundsError(
                    f"{owner} has {label} conditions on faces its bounds don't "
                    f"have: {sorted(face.value for face in extra_faces)}"
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

    def validate_references(self, references: Collection[Reference]) -> None:
        """Check that every reference's target exists in the registry it
        targets — one query per target type."""
        ids_by_type: dict[ReferenceableType, set[str]] = {}
        for reference in references:
            ids_by_type.setdefault(reference.target_type, set()).add(
                reference.target_id
            )

        dangling: list[str] = []
        for referenceable_type, ids in ids_by_type.items():
            found = self._registry.dao_for(referenceable_type).existing_ids(ids)
            dangling.extend(
                f"{referenceable_type.value} {target_id!r}"
                for target_id in sorted(ids - found)
            )
        if dangling:
            raise DanglingReferenceError(
                f"the following referenced ids do not exist: {dangling}"
            )


def _already_registered(model: Versioned) -> str:
    return (
        f"{type(model).__name__} {model.id!r} is already registered — "
        "a registered version is never overwritten"
    )
