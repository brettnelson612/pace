"""
pace/core/ids.py

Distinct, mypy-checkable ID types for PACE's modeling-domain objects.

Each is a typing.NewType wrapping str — zero runtime cost (a GeometryID
*is* a str at runtime, no wrapping/unwrapping needed for serialization),
but mypy will flag passing e.g. a MaterialID where a
GeometryID is expected. Kept in one shared file (rather than
colocated with each class) specifically to avoid circular imports, since
several classes (Part, Geometry, ...) need ID types from
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

# --- Placeables ---
PartID = NewType("PartID", str)
EnclosureID = NewType("EnclosureID", str)
LatticeID = NewType("LatticeID", str)

type PlaceableID = PartID | EnclosureID | LatticeID

# --- Placement ---
PlacementID = NewType("PlacementID", str)

# --- Reactor ---
ReactorBlueprintID = NewType("ReactorBlueprintID", str)

# --- Simulation ---
SimulationCaseID = NewType("SimulationCaseID", str)
SimulationJobID = NewType("SimulationJobID", str)
