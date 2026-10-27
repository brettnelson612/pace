"""
tests/unit/pace/core/test_placeable.py

Covers: PlaceableRef round-trips, the PlaceableType -> ReferenceableType
mapping, validation and value-based equality.
"""

import pytest

from pace.core.ids import (
    EnclosureID,
    PartID,
)
from pace.core.placeable import PlaceableRef, PlaceableType
from pace.core.reference_types import ReferenceableType

# =============================================================================
# PlaceableRef
# =============================================================================


@pytest.mark.parametrize("placeable_type", list(PlaceableType))
def test_placeable_ref_round_trip_every_type(placeable_type):
    ref = PlaceableRef(type=placeable_type, id=PartID("thing-1"))
    assert PlaceableRef.from_dict(ref.to_dict()) == ref


@pytest.mark.parametrize(
    ("placeable_type", "expected"),
    [
        (PlaceableType.PART, ReferenceableType.PART),
        (PlaceableType.ENCLOSURE, ReferenceableType.ENCLOSURE),
        (PlaceableType.LATTICE, ReferenceableType.LATTICE),
    ],
)
def test_placeable_ref_referenceable_type(placeable_type, expected):
    ref = PlaceableRef(type=placeable_type, id=PartID("thing-1"))
    assert ref.referenceable_type == expected


def test_every_placeable_type_is_a_referenceable_type():
    """PlaceableType must stay a subset of ReferenceableType, by value."""
    for placeable_type in PlaceableType:
        assert ReferenceableType(placeable_type.value)


def test_placeable_ref_rejects_a_plain_string_type():
    with pytest.raises(TypeError):
        PlaceableRef(type="enclosure", id=EnclosureID("thing-1"))  # type: ignore[arg-type]


def test_placeable_ref_rejects_empty_id():
    with pytest.raises(ValueError):
        PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID(""))


def test_placeable_ref_from_dict_rejects_unplaceable_type():
    with pytest.raises(ValueError):
        PlaceableRef.from_dict({"type": "geometry", "id": "cyl-1"})


def test_equal_placeable_refs_hash_equal():
    a = PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("pin-1"))
    b = PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("pin-1"))
    assert a == b
    assert len({a, b}) == 1


def test_same_id_different_type_is_a_different_placeable_ref():
    a = PlaceableRef(type=PlaceableType.PART, id=PartID("pin-1"))
    b = PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("pin-1"))
    assert a != b
