"""
tests/pace/core/test_component_ref.py

Covers: ComponentRef round-trips for every ComponentKind (including the
kind-specific id wrapping in from_dict), the kind -> ReferenceableType
mapping, validation of kind and id, and value-based equality/hashing
(lattice placements rely on refs being usable as dict keys).
"""

import pytest
from pace.core.component_ref import ComponentKind, ComponentRef
from pace.core.reference_types import ReferenceableType


@pytest.mark.parametrize("kind", list(ComponentKind))
def test_round_trip_every_kind(kind):
    ref = ComponentRef(kind=kind, id="thing-1")
    assert ComponentRef.from_dict(ref.to_dict()) == ref


@pytest.mark.parametrize(
    "kind,expected",
    [
        (ComponentKind.LCOMPONENT, ReferenceableType.LCOMPONENT),
        (ComponentKind.CCOMPONENT, ReferenceableType.CCOMPONENT),
        (ComponentKind.LATTICE, ReferenceableType.LATTICE),
    ],
)
def test_referenceable_type(kind, expected):
    assert ComponentRef(kind=kind, id="thing-1").referenceable_type == expected


def test_every_kind_has_a_referenceable_type():
    """ComponentKind must stay a subset of ReferenceableType, by value."""
    for kind in ComponentKind:
        assert ReferenceableType(kind.value)


def test_rejects_non_kind():
    with pytest.raises(TypeError):
        ComponentRef(kind="ccomponent", id="thing-1")  # type: ignore[arg-type]


def test_rejects_empty_id():
    with pytest.raises(ValueError):
        ComponentRef(kind=ComponentKind.CCOMPONENT, id="")


def test_from_dict_rejects_unplaceable_kind():
    with pytest.raises(ValueError):
        ComponentRef.from_dict({"kind": "geometry", "id": "cyl-1"})


def test_equal_refs_hash_equal():
    a = ComponentRef(kind=ComponentKind.CCOMPONENT, id="pin-1")
    b = ComponentRef(kind=ComponentKind.CCOMPONENT, id="pin-1")
    assert a == b
    assert len({a, b}) == 1


def test_same_id_different_kind_is_a_different_ref():
    a = ComponentRef(kind=ComponentKind.LCOMPONENT, id="pin-1")
    b = ComponentRef(kind=ComponentKind.CCOMPONENT, id="pin-1")
    assert a != b
