"""
pace/core/ids.py

Distinct, mypy-checkable ID types for PACE's modeling-domain objects.

Each is a typing.NewType wrapping str — zero runtime cost (a GeometryID
*is* a str at runtime, no wrapping/unwrapping needed for serialization),
but mypy will flag passing e.g. a MaterialID where a
GeometryID is expected. Kept in one shared file (rather than
colocated with each class) specifically to avoid circular imports, since
several classes (LComponent, Geometry, ...) need ID types from
more than one domain area at once.

NOTE: NewType wrappers are purely static; calling one on None returns
None unchanged, so from_dict() never needs to guard an Optional field
before wrapping it.

"""

from typing import NewType

# --- Geometry ---
GeometryID = NewType("GeometryID", str)

# --- Material ---
MaterialID = NewType("MaterialID", str)

# --- Components ---
LComponentID = NewType("LComponentID", str)
PComponentID = NewType("PComponentID", str)
CComponentID = NewType("CComponentID", str)

# --- Lattice ---
LatticeID = NewType("LatticeID", str)

# --- Reactor ---
ReactorID = NewType("ReactorID", str)

# --- Simulation provenance ---
GTRunID = NewType("GTRunID", str)

# --- PComponent addressing ---
# PComponent identity is path-style (relative to its parent CComponent),
# not a flat opaque ID like the above — deliberately not aliased to a
# plain NewType(str) yet. Revisit once the path/address representation
# itself is designed; a flat-address lookup utility may warrant its own
# type at that point (e.g. PComponentAddress), separate from the
# individual per-level position keys that compose it.
