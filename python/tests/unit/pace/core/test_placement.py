"""
tests/unit/pace/core/test_placement.py

Covers: Placement round-trips for every placeable type, frozen
immutability and id validation.
"""

from dataclasses import FrozenInstanceError

import pytest

from pace.core.geometry import GPose
from pace.core.ids import (
    EnclosureID,
    LatticeID,
    PartID,
    PlacementID,
)
from pace.core.placeable import PlaceableRef, PlaceableType
from pace.core.placement import Placement

# =============================================================================
# Placement
# =============================================================================

PELLET_REF = PlaceableRef(type=PlaceableType.PART, id=PartID("fuel_pellet-1"))
ROD_REF = PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("fuel_rod-1"))
LATTICE_REF = PlaceableRef(type=PlaceableType.LATTICE, id=LatticeID("lattice_3x3-1"))
ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)


@pytest.mark.parametrize("ref", [PELLET_REF, ROD_REF, LATTICE_REF])
def test_placement_round_trip_every_placeable_type(ref):
    pc = Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=ref)
    assert Placement.from_dict(pc.to_dict()) == pc


def test_placement_is_frozen():
    pc = Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF)
    with pytest.raises(FrozenInstanceError):
        pc.pose = GPose(x_m=1.0, y_m=0.0, z_m=0.0)  # type: ignore[misc]


def test_placement_rejects_empty_id():
    with pytest.raises(ValueError):
        Placement(id=PlacementID(""), pose=ORIGIN, ref=PELLET_REF)
