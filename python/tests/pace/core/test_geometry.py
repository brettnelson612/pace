"""
Unit tests for pace.core.geometry: GeometryVersion and its concrete
shape types (GCylinder, GAnnulus, GHexPrism, GSphere, GRectanglePrism,
GNull, GAddition, GSubtraction), plus the bare Geometry identity class.

Covers the invariants established during design, not just field coverage:
- the derived_from/gt_run_id pairing rule (version 1 vs version 2+ via GT run)
- GeometryVersion's ABC enforcement (can't instantiate directly, and a
  subclass missing an abstract method still can't instantiate)
- per-shape field constraints (positive/range) via validate_fields()
- relational checks not expressible via field metadata (GAnnulus)
- structural checks for GAddition/GSubtraction (min count, no duplicates)
- to_dict()/from_dict() round-trips
- frozen immutability
- GAddition/GSubtraction identity-based __eq__/__hash__
"""

import dataclasses

import pytest
from pace.core.geometry import (
    MAX_GEO_LENGTH_M,
    GAddition,
    GAnnulus,
    GCylinder,
    Geometry,
    GeometryVersion,
    GHexPrism,
    GNull,
    GPos,
    GRectanglePrism,
    GSphere,
    GSubtraction,
)
from pace.core.ids import GeometryID, GeometryVersionID, GTRunID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides) -> dict:
    """Common GeometryVersion base fields — valid version-1 (no lineage) by default."""
    kwargs = {
        "id": GeometryVersionID("gv-1"),
        "geometry_id": GeometryID("g-1"),
        "derived_from": None,
        "gt_run_id": None,
    }
    kwargs.update(overrides)
    return kwargs


VALID_SHAPE_KWARGS = {
    GCylinder: {"radius_m": 0.5, "height_m": 2.0},
    GHexPrism: {"circumradius_m": 0.5, "height_m": 2.0},
    GSphere: {"radius_m": 0.5},
    GRectanglePrism: {"length_m": 1.0, "width_m": 1.0, "height_m": 1.0},
}


def make(cls, **overrides):
    """Construct a valid instance of a simple (non-relational) shape class."""
    kwargs = _base_kwargs()
    kwargs.update(VALID_SHAPE_KWARGS.get(cls, {}))
    kwargs.update(overrides)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Geometry (bare identity — no shape data)
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_construct_and_round_trip(self):
        geo = Geometry(id=GeometryID("g-1"))
        assert geo.to_dict() == {"id": GeometryID("g-1")}
        assert Geometry.from_dict(geo.to_dict()) == geo

    def test_frozen(self):
        geo = Geometry(id=GeometryID("g-1"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            geo.id = GeometryID("g-2")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GeometryVersion — ABC enforcement + shared pairing invariant
# ---------------------------------------------------------------------------


class TestGeometryVersionBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            GeometryVersion(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    def test_incomplete_subclass_cannot_instantiate(self):
        # a subclass missing _validate_shape/to_open_mc/to_moose should
        # still fail to instantiate, same as the base itself
        class Incomplete(GeometryVersion):
            pass

        with pytest.raises(TypeError):
            Incomplete(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    @pytest.mark.parametrize(
        "derived_from,gt_run_id,should_raise",
        [
            (None, None, False),  # version 1: both unset
            (GeometryVersionID("gv-0"), GTRunID("run-1"), False),  # v2+: both set
            (GeometryVersionID("gv-0"), None, True),  # derived_from w/o gt_run_id
            (None, GTRunID("run-1"), True),  # gt_run_id w/o derived_from
        ],
    )
    def test_derived_from_gt_run_id_pairing(
        self, derived_from, gt_run_id, should_raise
    ):
        kwargs = _base_kwargs(derived_from=derived_from, gt_run_id=gt_run_id)
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
        # Constraint.POSITIVE — zero is degenerate, must be rejected
        with pytest.raises(ValueError):
            make(cls, **{field_name: 0.0})

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_negative_is_rejected(self, cls, field_name):
        with pytest.raises(ValueError):
            make(cls, **{field_name: -1.0})

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_above_max_is_rejected(self, cls, field_name):
        with pytest.raises(ValueError):
            make(cls, **{field_name: MAX_GEO_LENGTH_M + 1})

    @pytest.mark.parametrize("cls,field_name", SIMPLE_SHAPES_AND_FIELDS)
    def test_at_max_is_allowed(self, cls, field_name):
        # upper bound is inclusive per validate_fields()
        make(cls, **{field_name: MAX_GEO_LENGTH_M})


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
    def _unit(geo_id: str, x: float = 0.0):
        return (GeometryVersionID(geo_id), GPos(x_m=x, y_m=0.0, z_m=0.0))

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

    def test_hash_and_equality_are_id_based(self):
        a = make(
            GAddition,
            id=GeometryVersionID("gv-a"),
            units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)],
        )
        b = make(
            GAddition,
            id=GeometryVersionID("gv-a"),
            units=[self._unit("g-c", 2.0), self._unit("g-d", 3.0)],
        )
        # same id, different units — still equal/same-hash by design
        # (identity-based __eq__/__hash__, not field-based)
        assert a == b
        assert hash(a) == hash(b)

    def test_to_dict_from_dict_round_trip(self):
        addition = make(
            GAddition, units=[self._unit("g-a", 0.0), self._unit("g-b", 1.0)]
        )
        rebuilt = GAddition.from_dict(addition.to_dict())
        assert rebuilt.units == addition.units
        assert rebuilt.id == addition.id


class TestGSubtraction:
    @staticmethod
    def _pair(geo_id: str, x: float = 0.0):
        return (GeometryVersionID(geo_id), GPos(x_m=x, y_m=0.0, z_m=0.0))

    def test_valid_construction(self):
        make(GSubtraction, base=self._pair("g-base"), cuts=[self._pair("g-cut", 0.5)])

    def test_requires_at_least_one_cut(self):
        with pytest.raises(ValueError):
            make(GSubtraction, base=self._pair("g-base"), cuts=[])

    def test_rejects_duplicate_cuts(self):
        cut = self._pair("g-cut", 0.5)
        with pytest.raises(ValueError):
            make(GSubtraction, base=self._pair("g-base"), cuts=[cut, cut])

    def test_hash_and_equality_are_id_based(self):
        a = make(
            GSubtraction,
            id=GeometryVersionID("gv-a"),
            base=self._pair("g-base"),
            cuts=[self._pair("g-cut", 0.5)],
        )
        b = make(
            GSubtraction,
            id=GeometryVersionID("gv-a"),
            base=self._pair("g-base"),
            cuts=[self._pair("g-cut", 0.5)],
        )
        # same id, different units — still equal/same-hash by design
        # (identity-based __eq__/__hash__, not field-based)
        assert a == b
        assert hash(a) == hash(b)

    def test_to_dict_from_dict_round_trip(self):
        sub = make(
            GSubtraction, base=self._pair("g-base"), cuts=[self._pair("g-cut", 0.5)]
        )
        rebuilt = GSubtraction.from_dict(sub.to_dict())
        assert rebuilt.base == sub.base
        assert rebuilt.cuts == sub.cuts


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

    def test_gpos_frozen(self):
        pos = GPos(x_m=0.0, y_m=0.0, z_m=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pos.x_m = 1.0  # type: ignore[misc]
