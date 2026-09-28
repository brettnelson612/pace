"""
pace/core/reference_types.py

ReferenceableType — every versioned aggregate kind that can participate
in a reference between registered objects: Geometry, Material,
LComponent, CComponent, Lattice, and Reactor. Used as the type tag on
each row of the persistence-layer reference index (see db/relational/
reference.py). Lives in core rather than the db layer since it's a
fact about the domain's aggregate kinds, not about how references
happen to be stored.

The subset of these kinds that can be *placed* inside a composite is
ComponentKind (see component_ref.py).
"""

from __future__ import annotations

from enum import Enum


class ReferenceableType(str, Enum):
    GEOMETRY = "geometry"
    MATERIAL = "material"
    LCOMPONENT = "lcomponent"
    CCOMPONENT = "ccomponent"
    LATTICE = "lattice"
    REACTOR = "reactor"
