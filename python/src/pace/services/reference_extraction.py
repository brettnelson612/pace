"""
pace/services/reference_extraction.py

One function per versioned aggregate kind, each walking that object's
embedded structure and returning the deduplicated set of outgoing
Reference edges it should record in the registry. Used by
RegistryService on every register_*() call, before writing to
ReferenceDAO.

Every function here is pure — no DB access, no side effects. A row
represents an edge between two registered aggregates, not a
per-occurrence count: if a CComponent embeds the same LComponent at
several different ComponentPlacement positions, that collapses to one edge,
since that's what refcounting/propagation actually need to know —
"does this object currently depend on that one," not "how many times
does it appear inside it."
"""

from __future__ import annotations

from pace.core.component import CComponent, LComponent
from pace.core.geometry import GAddition, Geometry, GSubtraction
from pace.core.lattice import Lattice
from pace.core.material import Material, MMixture
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.core.reference_types import ReferenceableType
from pace.db.relational.reference import Reference


def extract_geometry_references(geometry: Geometry) -> set[Reference]:
    """GAddition/GSubtraction reference other Geometries via units/
    base/cuts; every other concrete Geometry subclass has no embedded
    references at all."""
    target_ids: set[str] = set()

    if isinstance(geometry, GAddition):
        target_ids.update(unit_id for unit_id, _ in geometry.units)
    elif isinstance(geometry, GSubtraction):
        base_id, _ = geometry.base
        target_ids.add(base_id)
        target_ids.update(cut_id for cut_id, _ in geometry.cuts)

    return {
        Reference(
            source_type=ReferenceableType.GEOMETRY,
            source_id=geometry.id,
            target_type=ReferenceableType.GEOMETRY,
            target_id=target_id,
        )
        for target_id in target_ids
    }


def extract_material_references(material: Material) -> set[Reference]:
    """MMixture references other Materials via components; MIsotopic/
    MVoid have no embedded references."""
    target_ids: set[str] = set()

    if isinstance(material, MMixture):
        target_ids.update(material_id for material_id, _ in material.components)

    return {
        Reference(
            source_type=ReferenceableType.MATERIAL,
            source_id=material.id,
            target_type=ReferenceableType.MATERIAL,
            target_id=target_id,
        )
        for target_id in target_ids
    }


def extract_lcomponent_references(component: LComponent) -> set[Reference]:
    """An LComponent always references exactly one Geometry and one
    Material — never zero, never more than one of each."""
    return {
        Reference(
            source_type=ReferenceableType.LCOMPONENT,
            source_id=component.id,
            target_type=ReferenceableType.GEOMETRY,
            target_id=component.geometry,
        ),
        Reference(
            source_type=ReferenceableType.LCOMPONENT,
            source_id=component.id,
            target_type=ReferenceableType.MATERIAL,
            target_id=component.material,
        ),
    }


def extract_ccomponent_references(component: CComponent) -> set[Reference]:
    """A CComponent references its bounds Geometry, its fill Material
    (if any), and every LComponent/CComponent/Lattice its ComponentPlacement
    members place. Each member's ComponentRef says which registry its
    id belongs to — no registry lookup needed here to disambiguate."""
    references = {
        Reference(
            source_type=ReferenceableType.CCOMPONENT,
            source_id=component.id,
            target_type=ReferenceableType.GEOMETRY,
            target_id=component.bounds,
        )
    }

    if component.fill is not None:
        references.add(
            Reference(
                source_type=ReferenceableType.CCOMPONENT,
                source_id=component.id,
                target_type=ReferenceableType.MATERIAL,
                target_id=component.fill,
            )
        )

    references.update(
        Reference(
            source_type=ReferenceableType.CCOMPONENT,
            source_id=component.id,
            target_type=placement.ref.referenceable_type,
            target_id=placement.ref.id,
        )
        for placement in component.placements
    )
    return references


def extract_lattice_references(lattice: Lattice) -> set[Reference]:
    """A Lattice references its fill Material and every component it
    places — one edge per placed component, however many addresses it
    occupies."""
    references = {
        Reference(
            source_type=ReferenceableType.LATTICE,
            source_id=lattice.id,
            target_type=ReferenceableType.MATERIAL,
            target_id=lattice.fill,
        )
    }
    references.update(
        Reference(
            source_type=ReferenceableType.LATTICE,
            source_id=lattice.id,
            target_type=element.ref.referenceable_type,
            target_id=element.ref.id,
        )
        for element in lattice.elements
    )
    return references


def extract_reactor_blueprint_references(
    reactor_blueprint: ReactorBlueprint,
) -> set[Reference]:
    """A ReactorBlueprint references its bounds Geometry, its fill
    Material (if any), and its root component."""
    references = {
        Reference(
            source_type=ReferenceableType.REACTOR_BLUEPRINT,
            source_id=reactor_blueprint.id,
            target_type=ReferenceableType.GEOMETRY,
            target_id=reactor_blueprint.bounds,
        ),
        Reference(
            source_type=ReferenceableType.REACTOR_BLUEPRINT,
            source_id=reactor_blueprint.id,
            target_type=reactor_blueprint.root.ref.referenceable_type,
            target_id=reactor_blueprint.root.ref.id,
        ),
    }
    if reactor_blueprint.fill is not None:
        references.add(
            Reference(
                source_type=ReferenceableType.REACTOR_BLUEPRINT,
                source_id=reactor_blueprint.id,
                target_type=ReferenceableType.MATERIAL,
                target_id=reactor_blueprint.fill,
            )
        )
    return references
