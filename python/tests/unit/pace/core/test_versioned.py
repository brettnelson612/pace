"""
tests/unit/pace/core/test_versioned.py

Covers: the equality contract every versioned class inherits from
Versioned (same concrete type and same id; hash by id), including a
guard that no subclass regenerates a field-based __eq__/__hash__ by
omitting eq=False from its @dataclass decorator.
"""

import pytest
import pace.core.component  # noqa: F401  (registers subclasses)
import pace.core.geometry  # noqa: F401
import pace.core.lattice  # noqa: F401
import pace.core.material  # noqa: F401
import pace.core.reactor_blueprint  # noqa: F401
from pace.core.geometry import GCylinder
from pace.core.material import MVoid
from pace.core.versioned import Versioned


def _all_subclasses(cls: type) -> list[type]:
    found = []
    for subclass in cls.__subclasses__():
        found.append(subclass)
        found.extend(_all_subclasses(subclass))
    return found


VERSIONED_CLASSES = _all_subclasses(Versioned)


def test_every_versioned_class_is_found():
    names = {cls.__name__ for cls in VERSIONED_CLASSES}
    assert {
        "GCylinder",
        "MIsotopic",
        "LComponent",
        "CComponent",
        "RectLattice",
        "HexLattice",
        "ReactorBlueprint",
    } <= names


@pytest.mark.parametrize("cls", VERSIONED_CLASSES, ids=lambda cls: cls.__name__)
def test_subclass_inherits_versioned_equality(cls):
    """A subclass declared without eq=False gets a dataclass-generated
    __eq__ (and __hash__) that silently replaces Versioned's."""
    assert cls.__eq__ is Versioned.__eq__
    assert cls.__hash__ is Versioned.__hash__


def test_same_type_same_id_is_equal_regardless_of_fields():
    a = GCylinder.create(family_name="rod", version_label="1", radius_m=0.1, height_m=1)
    b = GCylinder.create(family_name="rod", version_label="1", radius_m=0.2, height_m=1)
    assert a == b
    assert hash(a) == hash(b) == hash("rod-1")


def test_different_id_is_not_equal():
    a = GCylinder.create(family_name="rod", version_label="1", radius_m=0.1, height_m=1)
    b = GCylinder.create(family_name="rod", version_label="2", radius_m=0.1, height_m=1)
    assert a != b


def test_same_id_different_type_is_not_equal():
    cylinder = GCylinder.create(
        family_name="shared", version_label="1", radius_m=0.1, height_m=1
    )
    void = MVoid.create(family_name="shared", version_label="1")
    assert cylinder.id == void.id
    assert cylinder != void


def test_not_equal_to_a_non_versioned_object():
    cylinder = GCylinder.create(
        family_name="rod", version_label="1", radius_m=0.1, height_m=1
    )
    assert cylinder != "rod-1"


def test_usable_as_set_members_by_id():
    a = GCylinder.create(family_name="rod", version_label="1", radius_m=0.1, height_m=1)
    b = GCylinder.create(family_name="rod", version_label="1", radius_m=0.2, height_m=1)
    assert len({a, b}) == 1
