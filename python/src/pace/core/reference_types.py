"""
pace/core/reference_types.py

ReferenceableType — any of the versioned pace objects which can be referenced:
Geometry, Material, LComponent, CComponent, Lattice, and Reactor.

Used as the type tag on each row of the persistence-layer referenceindex
(see db/relational/reference.py). Lives in core rather than the db layer
since it's a fact about the domain's aggregate kinds, not about how
references happen to be stored.

The subset of these kinds that can be *placed* inside a composite is
ComponentType (see component_ref.py).
"""

from __future__ import annotations

from enum import StrEnum


class ReferenceableType(StrEnum):
    GEOMETRY = "geometry"
    MATERIAL = "material"
    LCOMPONENT = "lcomponent"
    CCOMPONENT = "ccomponent"
    LATTICE = "lattice"
    REACTOR_BLUEPRINT = "reactor_blueprint"
