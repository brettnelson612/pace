"""
pace/core/reference_types.py

ReferenceableType — the four aggregate kinds that can participate in a
reference between versioned objects: Geometry, Material, LComponent,
CComponent. Used both as PComponent's explicit discriminator for its
`component` field (see component.py) and as the type tag on each row
of the persistence-layer reference index (see db/relational/
reference.py). Lives in core rather than the db layer since it's a
fact about the domain's aggregate kinds, not about how references
happen to be stored.
"""

from __future__ import annotations

from enum import Enum


class ReferenceableType(str, Enum):
    GEOMETRY = "geometry"
    MATERIAL = "material"
    LCOMPONENT = "lcomponent"
    CCOMPONENT = "ccomponent"
