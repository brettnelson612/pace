"""
tests/pace/core/test_geometry.py

Covers: Geometry identity (build_id/_validate_id), the three-state
lineage rule, ABC enforcement, per-shape field constraints via
validate_fields(), GAnnulus's inner<outer relational check,
GAddition/GSubtraction structural checks (min count, duplicate
(geometry, pose) detection including the z_rotation_rad
differentiation), round-trips, frozen immutability, GAddition/
GSubtraction id-based equality + the hash(self.id) regression guard,
and geometry_from_dict()'s type-based dispatch.
"""

import dataclasses
from typing import ClassVar

import pytest
from pace.core.geometry import (
    MAX_GEO_LENGTH_M,
    GAddition,
    GAnnulus,
    GCylinder,
    Geometry,
    GHexPrism,
    GNull,
    GPose,
    GRectanglePrism,
    GSphere,
    GSubtraction,
    geometry_from_dict,
)
from pace.core.ids import GeometryID, GTRunID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(
    family_name: str = "test_geo", version_label: str = "1", **overrides
) -> dict:
    kwargs = {
        "id": Geometry.build_id(family_name, version_label),
        "family_name": family_name,
        "version_label": version_label,
        "derived_from": None,
        "gt_run_id": None,
        "user_edit": False,
    }
    kwargs.update(overrides)
    return kwargs


VALID_SHAPE_KWARGS = {
    GCylinder: {"radius_m": 0.5, "height_m": 2.0},
    GHexPrism: {"circumradius_m": 0.5, "height_m": 2.0},
    GSphere: {"radius_m": 0.5},
    GRectanglePrism: {"length_m": 1.0, "width_m": 1.0, "height_m": 1.0},
}


def make(cls, family_name: str = "test_geo", version_label: str = "1", **overrides):
    """Construct a valid instance of a simple (non-relational) shape class."""
    kwargs = _base_kwargs(family_name=family_name, version_label=version_label)
    kwargs.update(VALID_SHAPE_KWARGS.get(cls, {}))
    kwargs.update(overrides)
    return cls(**kwargs)


def make_with_field_override(cls, field_name: str, value: float):
    kwargs = _base_kwargs()
    kwargs.update(VALID_SHAPE_KWARGS.get(cls, {}))
    kwargs[field_name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Identity: build_id() / _validate_id() / create()
# ---------------------------------------------------------------------------


class TestGeometryIdentity:
    def test_build_id_format(self):
        assert Geometry.build_id("uranium3.2", "1") == "uranium3.2-1"

    def test_build_id_with_string_version_label(self):
        assert (
            Geometry.build_id("fuel_pellet", "6month_depletion")
            == "fuel_pellet-6month_depletion"
        )

    def test_mismatched_id_rejected(self):
        with pytest.raises(ValueError):
            make(
                GCylinder,
                id=GeometryID("wrong_id"),
                family_name="test_geo",
                version_label="1",
            )

    def test_matching_id_is_allowed(self):
        make(GCylinder, family_name="test_geo", version_label="1")  # should not raise

    def test_create_derives_same_id_as_build_id(self):
        geo = GCylinder.create(
            family_name="test_geo", version_label="1", radius_m=0.5, height_m=2.0
        )
        assert geo.id == Geometry.build_id("test_geo", "1")


# ---------------------------------------------------------------------------
# Geometry — ABC enforcement + shared lineage invariant
# ---------------------------------------------------------------------------


class TestGeometryBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Geometry(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    def test_incomplete_subclass_cannot_instantiate(self):
        # a subclass missing _validate_shape/to_open_mc/to_moose should
        # still fail to instantiate, same as the base itself
        class Incomplete(Geometry):
            pass

        with pytest.raises(TypeError):
            Incomplete(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    # (derived_from, gt_run_id, user_edit, should_raise), parametrized over
    # all 8 lineage-rule cases
    LINEAGE_CASES: ClassVar[list] = [
        pytest.param(None, None, False, False, id="v1_valid"),
        pytest.param(None, GTRunID("run-1"), False, True, id="v1_with_gt_run_id"),
        pytest.param(None, None, True, True, id="v1_with_user_edit"),
        pytest.param(None, GTRunID("run-1"), True, True, id="v1_with_both"),
        pytest.param(
            GeometryID("gv-0"), GTRunID("run-1"), False, False, id="derived_gt_run_only"
        ),
        pytest.param(
            GeometryID("gv-0"), None, True, False, id="derived_user_edit_only"
        ),
        pytest.param(GeometryID("gv-0"), None, False, True, id="derived_with_neither"),
        pytest.param(
            GeometryID("gv-0"), GTRunID("run-1"), True, True, id="derived_with_both"
        ),
    ]

    @pytest.mark.parametrize(
        "derived_from,gt_run_id,user_edit,should_raise", LINEAGE_CASES
    )
    def test_lineage_rule(self, derived_from, gt_run_id, user_edit, should_raise):
        kwargs = _base_kwargs(
            derived_from=derived_from, gt_run_id=gt_run_id, user_edit=user_edit
        )
        kwargs.update(VALID_SHAPE_KWARGS[GCylinder])
        if should_raise:
            with pytest.raises(ValueError):
                GCylinder(**kwargs)
        else:
            GCylinder(**kwargs)  # should not raise


# ---------------------------------------------------------------------------
# Per-shape field constraints (positive / range), parametrized
# ---------------------------------------------------------------------------

SIMPLE_SHAPES_AND_FIELDS = [
    (GCylinder, "radius_m"),
    (GCylinder, "height_m"),
    (GHexPrism, "circumradius_m"),
    (GHexPrism, "height_m"),
    (GSphere, "radius_m"),
    (GRectanglePrism, "length_m"),
    (GRectanglePrism, "width_m"),
    (GRectanglePrism, "height_m"),
]


class TestShapeConstraints:
    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_valid_construction(self, cls, field_name):
        make(cls)  # sanity check — default valid kwargs should not raise

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_zero_is_rejected(self, cls, field_name):
        with pytest.raises(ValueError):
            make_with_field_override(cls, field_name, 0.0)

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_negative_is_rejected(self, cls, field_name):
        with pytest.raises(ValueError):
            make_with_field_override(cls, field_name, -1.0)

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_above_max_is_rejected(self, cls, field_name):
        with pytest.raises(ValueError):
            make_with_field_override(cls, field_name, MAX_GEO_LENGTH_M + 1)

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_at_max_is_allowed(self, cls, field_name):
        make_with_field_override(cls, field_name, MAX_GEO_LENGTH_M)


class TestGAnnulus:
    def test_valid_construction_and_round_trip(self):
        geo = make(GAnnulus, inner_radius_m=0.3, outer_radius_m=0.5, height_m=2.0)
        assert geo.inner_radius_m == 0.3
        assert GAnnulus.from_dict(geo.to_dict()) == geo

    def test_inner_equal_to_outer_rejected(self):
        with pytest.raises(ValueError):
            make(GAnnulus, inner_radius_m=0.5, outer_radius_m=0.5, height_m=2.0)

    def test_inner_greater_than_outer_rejected(self):
        with pytest.raises(ValueError):
            make(GAnnulus, inner_radius_m=0.6, outer_radius_m=0.5, height_m=2.0)

    def test_zero_inner_radius_rejected(self):
        # Constraint.POSITIVE still applies to inner_radius_m independent
        # of the inner < outer relational check
        with pytest.raises(ValueError):
            make(GAnnulus, inner_radius_m=0.0, outer_radius_m=0.5, height_m=2.0)


class TestGNull:
    def test_valid_construction_and_round_trip(self):
        null = make(GNull)
        assert GNull.from_dict(null.to_dict()) == null


# ---------------------------------------------------------------------------
# GAddition / GSubtraction — structural validation
# ---------------------------------------------------------------------------


class TestGAddition:
    @staticmethod
    def _unit(geo_id: str, x: float = 0.0, z_rotation_rad: float = 0.0):
        return (
            GeometryID(geo_id),
            GPose(x_m=x, y_m=0.0, z_m=0.0, z_rotation_rad=z_rotation_rad),
        )

    def test_valid_construction(self):
        make(GAddition, units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)])

    def test_requires_at_least_two_units(self):
        with pytest.raises(ValueError):
            make(GAddition, units=[self._unit("g-a", 0.0)])

    def test_rejects_duplicate_unit_and_position(self):
        dup = self._unit("g-a", 0.0)
        with pytest.raises(ValueError):
            make(GAddition, units=[dup, dup])

    def test_same_geometry_different_position_is_allowed(self):
        # per design: a duplicate is (geometry, position) together — the
        # same geometry reused at a different position is legitimate
        make(GAddition, units=[self._unit("g-a", 0.0), self._unit("g-a", 1.0)])

    def test_same_position_different_rotation_is_not_a_duplicate(self):
        # z_rotation_rad is part of the duplicate-detection key — two
        # units at the same geometry+position but different rotation are
        # NOT duplicates
        a = self._unit("g-a", 0.0, z_rotation_rad=0.0)
        b = self._unit("g-a", 0.0, z_rotation_rad=1.57)
        make(GAddition, units=[a, b])  # should not raise

    def test_hash_and_equality_are_id_based(self):
        a = make(
            GAddition,
            family_name="shared",
            version_label="1",
            units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)],
        )
        b = make(
            GAddition,
            family_name="shared",
            version_label="1",
            units=[self._unit("g-c", 2.0), self._unit("g-d", 3.0)],
        )
        # same id (same family_name/version_label), different units —
        # still equal/same-hash by design (id-based __eq__/__hash__, not
        # field-based)
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_matches_hash_of_id(self):
        # regression guard: __hash__ must be hash(self.id), not id(self.id)
        instance = make(
            GAddition,
            family_name="shared",
            version_label="1",
            units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)],
        )
        assert hash(instance) == hash(Geometry.build_id("shared", "1"))

    def test_different_family_is_not_equal(self):
        a = make(
            GAddition,
            family_name="fam-a",
            version_label="1",
            units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)],
        )
        b = make(
            GAddition,
            family_name="fam-b",
            version_label="1",
            units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)],
        )
        assert a != b

    def test_to_dict_from_dict_round_trip(self):
        addition = make(
            GAddition, units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)]
        )
        rebuilt = GAddition.from_dict(addition.to_dict())
        assert rebuilt.units == addition.units
        assert rebuilt.id == addition.id
        assert rebuilt.family_name == addition.family_name
        assert rebuilt.version_label == addition.version_label

    def test_to_dict_includes_type(self):
        addition = make(
            GAddition, units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)]
        )
        assert addition.to_dict()["type"] == "addition"

    def test_frozen(self):
        addition = make(
            GAddition, units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)]
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            addition.units = []  # type: ignore[misc]


class TestGSubtraction:
    @staticmethod
    def _pair(geo_id: str, x: float = 0.0, z_rotation_rad: float = 0.0):
        return (
            GeometryID(geo_id),
            GPose(x_m=x, y_m=0.0, z_m=0.0, z_rotation_rad=z_rotation_rad),
        )

    def test_valid_construction(self):
        make(GSubtraction, base=self._pair("g-base"), cuts=[self._pair("g-cut", 0.5)])

    def test_requires_at_least_one_cut(self):
        with pytest.raises(ValueError):
            make(GSubtraction, base=self._pair("g-base"), cuts=[])

    def test_rejects_duplicate_cuts(self):
        cut = self._pair("g-cut", 0.5)
        with pytest.raises(ValueError):
            make(GSubtraction, base=self._pair("g-base"), cuts=[cut, cut])

    def test_same_position_different_rotation_is_not_a_duplicate(self):
        a = self._pair("g-cut", 0.5, z_rotation_rad=0.0)
        b = self._pair("g-cut", 0.5, z_rotation_rad=1.57)
        make(GSubtraction, base=self._pair("g-base"), cuts=[a, b])  # should not raise

    def test_hash_and_equality_are_id_based(self):
        a = make(
            GSubtraction,
            family_name="shared",
            version_label="1",
            base=self._pair("g-base"),
            cuts=[self._pair("g-cut", 0.5)],
        )
        b = make(
            GSubtraction,
            family_name="shared",
            version_label="1",
            base=self._pair("g-other-base"),
            cuts=[self._pair("g-other-cut", 0.7)],
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_matches_hash_of_id(self):
        instance = make(
            GSubtraction,
            family_name="shared",
            version_label="1",
            base=self._pair("g-base"),
            cuts=[self._pair("g-cut", 0.5)],
        )
        assert hash(instance) == hash(Geometry.build_id("shared", "1"))

    def test_to_dict_from_dict_round_trip(self):
        sub = make(
            GSubtraction, base=self._pair("g-base"), cuts=[self._pair("g-cut", 0.5)]
        )
        rebuilt = GSubtraction.from_dict(sub.to_dict())
        assert rebuilt.base == sub.base
        assert rebuilt.cuts == sub.cuts

    def test_to_dict_includes_type(self):
        sub = make(
            GSubtraction, base=self._pair("g-base"), cuts=[self._pair("g-cut", 0.5)]
        )
        assert sub.to_dict()["type"] == "subtraction"

    def test_frozen(self):
        sub = make(
            GSubtraction, base=self._pair("g-base"), cuts=[self._pair("g-cut", 0.5)]
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sub.cuts = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# to_dict() / from_dict() round-trips + frozen immutability, simple shapes
# ---------------------------------------------------------------------------


class TestRoundTripAndImmutability:
    @pytest.mark.parametrize("cls", [GCylinder, GHexPrism, GSphere, GRectanglePrism])
    def test_round_trip(self, cls):
        original = make(cls)
        rebuilt = cls.from_dict(original.to_dict())
        assert rebuilt == original

    @pytest.mark.parametrize("cls", [GCylinder, GHexPrism, GSphere, GRectanglePrism])
    def test_frozen(self, cls):
        instance = make(cls)
        field_name = next(iter(VALID_SHAPE_KWARGS[cls]))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, 999.0)

    @pytest.mark.parametrize("cls", [GCylinder, GHexPrism, GSphere, GRectanglePrism])
    def test_to_dict_includes_type(self, cls):
        instance = make(cls)
        assert "type" in instance.to_dict()


class TestGPose:
    def test_rotation_defaults_to_zero(self):
        pos = GPose(x_m=0.0, y_m=0.0, z_m=0.0)
        assert pos.z_rotation_rad == 0.0

    def test_rotation_round_trip(self):
        pos = GPose(x_m=1.0, y_m=2.0, z_m=3.0, z_rotation_rad=1.5707963267948966)
        assert GPose.from_dict(pos.to_dict()) == pos

    def test_frozen(self):
        pos = GPose(x_m=0.0, y_m=0.0, z_m=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pos.x_m = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# geometry_from_dict() — type-based dispatch
# ---------------------------------------------------------------------------


class TestGeometryFromDict:
    def test_dispatches_addition(self):
        addition = make(
            GAddition,
            units=[
                (GeometryID("g-a"), GPose(x_m=0.0, y_m=0.0, z_m=0.0)),
                (GeometryID("g-b"), GPose(x_m=1.0, y_m=0.0, z_m=0.0)),
            ],
        )
        rebuilt = geometry_from_dict(addition.to_dict())
        assert type(rebuilt) is GAddition
        assert rebuilt.id == addition.id

    def test_unknown_type_raises_value_error_not_key_error(self):
        with pytest.raises(ValueError):
            geometry_from_dict({"type": "not_a_real_shape"})
