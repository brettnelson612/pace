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
from pace.core.lattice import Lattice
from pace.core.material import Material, MMixture
from pace.core.reactor import Reactor
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


def _edge(
    source_type: ReferenceableType,
    source_id: str,
    target_type: ReferenceableType,
    target_id: str,
) -> Reference:
    return Reference(
        source_type=source_type,
        source_id=source_id,
        target_type=target_type,
        target_id=target_id,
    )


def extract_ccomponent_references(component: CComponent) -> set[Reference]:
    """A CComponent references its bounds Geometry, its fill Material
    (if any), and every LComponent/CComponent/Lattice its PComponent
    members place. Each member's ComponentRef says which registry its
    id belongs to — no registry lookup needed here to disambiguate."""
    source = ReferenceableType.CCOMPONENT
    edges = {_edge(source, component.id, ReferenceableType.GEOMETRY, component.bounds)}
    if component.fill is not None:
        edges.add(
            _edge(source, component.id, ReferenceableType.MATERIAL, component.fill)
        )
    edges.update(
        _edge(source, component.id, pc.ref.referenceable_type, pc.ref.id)
        for pc in component.components
    )
    return edges


def extract_lattice_references(lattice: Lattice) -> set[Reference]:
    """A Lattice references its fill Material and every component it
    places — one edge per placed component, however many addresses it
    occupies."""
    source = ReferenceableType.LATTICE
    edges = {_edge(source, lattice.id, ReferenceableType.MATERIAL, lattice.fill)}
    edges.update(
        _edge(source, lattice.id, placement.ref.referenceable_type, placement.ref.id)
        for placement in lattice.placements
    )
    return edges


def extract_reactor_references(reactor: Reactor) -> set[Reference]:
    """A Reactor references its bounds Geometry, its fill Material (if
    any), its root component, and every Material its operating state
    gives a starting temperature for."""
    source = ReferenceableType.REACTOR
    edges = {
        _edge(source, reactor.id, ReferenceableType.GEOMETRY, reactor.bounds),
        _edge(
            source,
            reactor.id,
            reactor.root.ref.referenceable_type,
            reactor.root.ref.id,
        ),
    }
    if reactor.fill is not None:
        edges.add(_edge(source, reactor.id, ReferenceableType.MATERIAL, reactor.fill))
    edges.update(
        _edge(source, reactor.id, ReferenceableType.MATERIAL, material_id)
        for material_id in reactor.operating_state.initial_temperatures_k
    )
    return edges
