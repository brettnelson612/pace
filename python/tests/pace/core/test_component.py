"""
Unit tests for pace.core.component: LComponent, PComponent, and
CComponent.

Covers:
- to_dict()/from_dict() round-trips, including PComponent's nested
  GPose serialization
- CComponent's minimum-member-count rule (>=2)
- frozen immutability
- CComponent's id-based __eq__/__hash__ (matching GAddition/GSubtraction
  and MIsotopic/MMixture's pattern), including a regression guard for
  the hash(self.id) vs id(self.id) bug caught earlier in material.py
- PComponent referencing either an LComponentID or a CComponentID
"""

import dataclasses

import pytest
from pace.core.component import CComponent, LComponent, PComponent
from pace.core.geometry import GPose
from pace.core.ids import (
    CComponentID,
    GeometryVersionID,
    LComponentID,
    MaterialVersionID,
    PComponentID,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lcomponent(**overrides) -> LComponent:
    kwargs = {
        "id": LComponentID("lc-1"),
        "geometry_version": GeometryVersionID("gv-1"),
        "material_version": MaterialVersionID("mv-1"),
    }
    kwargs.update(overrides)
    return LComponent(**kwargs)


def make_pcomponent(**overrides) -> PComponent:
    kwargs = {
        "id": PComponentID("pc-1"),
        "pose": GPose(x_m=0.0, y_m=0.0, z_m=0.0),
        "component": LComponentID("lc-1"),
    }
    kwargs.update(overrides)
    return PComponent(**kwargs)


def make_ccomponent(**overrides) -> CComponent:
    kwargs = {
        "id": CComponentID("cc-1"),
        "components": [
            make_pcomponent(id=PComponentID("pc-1")),
            make_pcomponent(
                id=PComponentID("pc-2"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01, z_rotation_rad=0.0),
            ),
        ],
    }
    kwargs.update(overrides)
    return CComponent(**kwargs)


# ---------------------------------------------------------------------------
# LComponent
# ---------------------------------------------------------------------------


class TestLComponent:
    def test_round_trip(self):
        original = make_lcomponent()
        rebuilt = LComponent.from_dict(original.to_dict())
        assert rebuilt == original

    def test_frozen(self):
        instance = make_lcomponent()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.geometry_version = GeometryVersionID("gv-2")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PComponent
# ---------------------------------------------------------------------------


class TestPComponent:
    def test_round_trip_referencing_lcomponent(self):
        original = make_pcomponent(component=LComponentID("lc-1"))
        rebuilt = PComponent.from_dict(original.to_dict())
        assert rebuilt == original
        assert rebuilt.pose == original.pose

    def test_round_trip_referencing_ccomponent(self):
        original = make_pcomponent(component=CComponentID("cc-nested"))
        rebuilt = PComponent.from_dict(original.to_dict())
        assert rebuilt == original
        assert rebuilt.component == CComponentID("cc-nested")

    def test_pose_serialized_as_dict_not_object(self):
        # regression guard: to_dict() must call pose.to_dict(), not embed
        # the live GPose object directly (would break JSON-serializability)
        instance = make_pcomponent()
        assert instance.to_dict()["pose"] == {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "z_rotation_rad": 0.0,
        }

    def test_frozen(self):
        instance = make_pcomponent()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.pose = GPose(x_m=1.0, y_m=0.0, z_m=0.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CComponent
# ---------------------------------------------------------------------------


class TestCComponent:
    def test_valid_construction(self):
        make_ccomponent()  # sanity check — default valid kwargs should not raise

    def test_requires_at_least_two_components(self):
        with pytest.raises(ValueError):
            make_ccomponent(components=[make_pcomponent()])

    def test_empty_components_rejected(self):
        with pytest.raises(ValueError):
            make_ccomponent(components=[])

    def test_can_nest_a_ccomponent_via_pcomponent(self):
        # a PComponent's `component` may itself be a CComponentID —
        # this is what makes the structure recursive (pins of pellets,
        # assemblies of pins, etc.)
        nested = make_ccomponent(
            id=CComponentID("cc-outer"),
            components=[
                make_pcomponent(
                    id=PComponentID("pc-a"), component=CComponentID("cc-inner")
                ),
                make_pcomponent(
                    id=PComponentID("pc-b"), component=LComponentID("lc-1")
                ),
            ],
        )
        assert nested.components[0].component == CComponentID("cc-inner")

    def test_to_dict_from_dict_round_trip(self):
        original = make_ccomponent()
        rebuilt = CComponent.from_dict(original.to_dict())
        assert rebuilt == original
        assert rebuilt.components == original.components

    def test_frozen(self):
        instance = make_ccomponent()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.components = []  # type: ignore[misc]


class TestCComponentEquality:
    """CComponent's __eq__/__hash__ are id-based (isinstance check +
    self.id == other.id), matching GAddition/GSubtraction and
    MIsotopic/MMixture — NOT the frozen-dataclass field-based default
    (which would fail outright, since `components` is an unhashable
    list)."""

    def test_same_id_different_members_are_equal(self):
        a = make_ccomponent(id=CComponentID("cc-shared"))
        b = make_ccomponent(
            id=CComponentID("cc-shared"),
            components=[
                make_pcomponent(id=PComponentID("pc-x")),
                make_pcomponent(id=PComponentID("pc-y")),
            ],
        )
        # same id, different members — still equal/same-hash by design
        assert a == b
        assert hash(a) == hash(b)

    def test_different_id_is_not_equal(self):
        a = make_ccomponent(id=CComponentID("cc-a"))
        b = make_ccomponent(id=CComponentID("cc-b"))
        assert a != b

    def test_hash_matches_hash_of_id(self):
        # regression guard: __hash__ must be hash(self.id), not id(self.id)
        # (see the same bug caught in MIsotopic/MMixture)
        instance = make_ccomponent(id=CComponentID("cc-1"))
        assert hash(instance) == hash(CComponentID("cc-1"))

    def test_is_hashable_despite_list_field(self):
        # the actual bug this whole pattern exists to avoid: a plain
        # frozen-dataclass default __hash__ would try to hash the
        # `components` list and raise TypeError
        instance = make_ccomponent()
        hash(instance)  # should not raise
