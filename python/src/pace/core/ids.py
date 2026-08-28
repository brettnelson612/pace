"""
Distinct, mypy-checkable ID types for PACE's modeling-domain objects.

Each is a typing.NewType wrapping str — zero runtime cost (a GeometryID
*is* a str at runtime, no wrapping/unwrapping needed for serialization),
but mypy will flag passing e.g. a MaterialVersionID where a
GeometryVersionID is expected. Kept in one shared file (rather than
colocated with each class) specifically to avoid circular imports, since
several classes (LComponent, GeometryVersion, ...) need ID types from
more than one domain area at once.
"""

from typing import NewType

# --- Geometry ---
GeometryID = NewType("GeometryID", str)
GeometryVersionID = NewType("GeometryVersionID", str)

# --- Material ---
MaterialID = NewType("MaterialID", str)
MaterialVersionID = NewType("MaterialVersionID", str)

# --- Components ---
LComponentID = NewType("LComponentID", str)
CComponentID = NewType("CComponentID", str)

# RTBlueprint has no distinct ID type of its own,
# an RTBlueprint's ID *is* the CComponent ID of the CComponent it wraps
# ("It has an ID (specifically, the CComponent ID)"). Use CComponentID
# wherever an RTBlueprint is referenced; do not introduce RTBlueprintID.

# --- Simulation provenance ---
GTRunID = NewType("GTRunID", str)

# --- PComponent addressing ---
# PComponent identity is path-style (relative to its parent CComponent),
# not a flat opaque ID like the above — deliberately not aliased to a
# plain NewType(str) yet. Revisit once the path/address representation
# itself is designed; a flat-address lookup utility may warrant its own
# type at that point (e.g. PComponentAddress), separate from the
# individual per-level position keys that compose it.
