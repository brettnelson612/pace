"""
pace/services/reference_extraction.py

One function per versioned aggregate kind, each walking that object's
embedded structure and returning the deduplicated set of outgoing
Reference edges it should record in the registry. Used by
ComponentService on every register_*()/edit_*() call, before writing
to ReferenceDAO.

Every function here is pure — no DB access, no side effects. A row
represents an edge between two registered aggregates, not a
per-occurrence count: if a CComponent embeds the same LComponent at
several different PComponent positions, that collapses to one edge,
since that's what refcounting/propagation actually need to know —
"does this object currently depend on that one," not "how many times
does it appear inside it."
"""

from __future__ import annotations

from pace.core.component import CComponent, LComponent
from pace.core.geometry import GAddition, Geometry, GSubtraction
from pace.core.material import Material, MMixture
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
    """A CComponent references other LComponents/CComponents via its
    PComponent members. Each PComponent's own component_type says
    which registry its `component` id belongs to — no registry lookup
    needed here to disambiguate."""
    return {
        Reference(
            source_type=ReferenceableType.CCOMPONENT,
            source_id=component.id,
            target_type=pcomponent.component_type,
            target_id=pcomponent.component,
        )
        for pcomponent in component.components
    }
