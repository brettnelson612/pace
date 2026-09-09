"""
Unit tests for pace.core.material: MaterialVersion and its concrete
composition types (MIsotopic, MMixture), MaterialComponentEntry, and the
bare Material identity class.

Covers the invariants established during design, not just field coverage:
- the derived_from/gt_run_id pairing rule (version 1 vs version 2+ via GT run)
- MaterialVersion's ABC enforcement (can't instantiate directly, and a
  subclass missing an abstract method still can't instantiate)
- MaterialComponentEntry's all-or-none enrichment-field invariant
- field constraints (positive/range) via validate_fields()
- MIsotopic's composition-level rules: non-empty components, v2+
  (GT-run-derived) versions restricted to exact-nuclide entries only,
  enrichment-format entries restricted to bare-element keys
- MMixture's composition-level rules: minimum constituent count,
  fraction bounds (0, 1), fractions summing to exactly 1
- to_dict()/from_dict() round-trips
- frozen immutability
- MIsotopic/MMixture's identity-based __eq__/__hash__ (self is other,
  NOT id-field-based — see TestMIsotopicEquality/TestMMixtureEquality
  for why this differs from GAddition/GSubtraction in geometry.py, and
  why round-trip tests here compare fields rather than object equality)
"""

import dataclasses

import pytest
from pace.core.ids import GTRunID, MaterialID, MaterialVersionID
from pace.core.material import (
    MaterialComponentEntry,
    MaterialVersion,
    MIsotopic,
    MMixture,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides) -> dict:
    """Common MaterialVersion base fields — valid version-1 (no lineage) by default."""
    kwargs = {
        "id": MaterialVersionID("mv-1"),
        "material_id": MaterialID("m-1"),
        "derived_from": None,
        "gt_run_id": None,
    }
    kwargs.update(overrides)
    return kwargs


def _valid_isotopic_kwargs(**overrides) -> dict:
    """A valid, v1, bare-nuclide MIsotopic composition — 3.2/96.8 wo% U235/U238."""
    kwargs = {
        "components": {
            "U235": MaterialComponentEntry(percent=3.2),
            "U238": MaterialComponentEntry(percent=96.8),
        },
        "percent_type": "wo",
        "density_value": 10.3,
        "density_unit": "g/cm3",
    }
    kwargs.update(overrides)
    return kwargs


def _valid_mixture_kwargs(**overrides) -> dict:
    """A valid MMixture composition — 70/30 atom-fraction mix of two materials."""
    kwargs = {
        "components": [
            (MaterialVersionID("mv-a"), 0.7),
            (MaterialVersionID("mv-b"), 0.3),
        ],
        "percent_type": "ao",
    }
    kwargs.update(overrides)
    return kwargs


def make_isotopic(**overrides) -> MIsotopic:
    kwargs = _base_kwargs()
    kwargs.update(_valid_isotopic_kwargs())
    kwargs.update(overrides)
    return MIsotopic(**kwargs)


def make_mixture(**overrides) -> MMixture:
    kwargs = _base_kwargs()
    kwargs.update(_valid_mixture_kwargs())
    kwargs.update(overrides)
    return MMixture(**kwargs)


# ---------------------------------------------------------------------------
# MaterialComponentEntry — enrichment invariant + field constraints
# ---------------------------------------------------------------------------


class TestMaterialComponentEntry:
    def test_bare_percent_only_is_valid(self):
        entry = MaterialComponentEntry(percent=3.2)
        assert entry.enrichment is None
        assert entry.enrichment_target is None
        assert entry.enrichment_type is None

    def test_full_enrichment_is_valid(self):
        entry = MaterialComponentEntry(
            percent=1.0,
            enrichment=3.2,
            enrichment_target="U235",
            enrichment_type="wo",
        )
        assert entry.enrichment == 3.2

    @pytest.mark.parametrize(
        "overrides",
        [
            {"enrichment": 3.2},  # enrichment only
            {"enrichment_target": "U235"},  # target only
            {"enrichment_type": "wo"},  # type only
            {"enrichment": 3.2, "enrichment_target": "U235"},  # missing type
            {"enrichment": 3.2, "enrichment_type": "wo"},  # missing target
            {
                "enrichment_target": "U235",
                "enrichment_type": "wo",
            },  # missing enrichment
        ],
    )
    def test_partial_enrichment_fields_rejected(self, overrides):
        with pytest.raises(ValueError):
            MaterialComponentEntry(percent=1.0, **overrides)

    def test_percent_zero_rejected(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry(percent=0.0)

    def test_percent_negative_rejected(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry(percent=-1.0)

    def test_enrichment_zero_rejected(self):
        # Constraint.POSITIVE — zero enrichment is degenerate, must be rejected
        # even though the range's lower bound (0, 100) is inclusive of 0
        with pytest.raises(ValueError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=0.0,
                enrichment_target="U235",
                enrichment_type="wo",
            )

    def test_enrichment_negative_rejected(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=-5.0,
                enrichment_target="U235",
                enrichment_type="wo",
            )

    def test_enrichment_at_max_is_allowed(self):
        # upper bound is inclusive per validate_fields()
        MaterialComponentEntry(
            percent=1.0,
            enrichment=100.0,
            enrichment_target="U235",
            enrichment_type="wo",
        )

    def test_enrichment_above_max_rejected(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=100.1,
                enrichment_target="U235",
                enrichment_type="wo",
            )

    def test_round_trip(self):
        entry = MaterialComponentEntry(
            percent=1.0, enrichment=3.2, enrichment_target="U235", enrichment_type="wo"
        )
        assert MaterialComponentEntry.from_dict(entry.to_dict()) == entry

    def test_frozen(self):
        entry = MaterialComponentEntry(percent=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.percent = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MaterialVersion — ABC enforcement + shared pairing invariant
# ---------------------------------------------------------------------------


class TestMaterialVersionBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            MaterialVersion(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    def test_incomplete_subclass_cannot_instantiate(self):
        # a subclass missing _validate_composition/to_open_mc/to_moose should
        # still fail to instantiate, same as the base itself
        class Incomplete(MaterialVersion):
            pass

        with pytest.raises(TypeError):
            Incomplete(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    @pytest.mark.parametrize(
        "derived_from,gt_run_id,user_edit,should_raise",
        [
            (None, None, False, False),  # version 1: no source settings
            (
                MaterialVersionID("mv-0"),
                GTRunID("run-1"),
                False,
                False,
            ),  # v2+: only gt_run set
            (
                MaterialVersionID("mv-0"),
                None,
                True,
                False,
            ),  # v2+: only user_edit set
            (
                MaterialVersionID("mv-0"),
                GTRunID("run-1"),
                True,
                True,
            ),  # v2+: both version sources set, invalid
            (
                MaterialVersionID("mv-0"),
                None,
                False,
                True,
            ),  # derived_from w/o either version source set, invalid
            (None, GTRunID("run-1"), False, True),  # gt_run_id w/o derived_from
        ],
    )
    def test_derived_from_gt_run_id_pairing(
        self, derived_from, gt_run_id, user_edit, should_raise
    ):
        if should_raise:
            with pytest.raises(ValueError):
                make_isotopic(
                    derived_from=derived_from, gt_run_id=gt_run_id, user_edit=user_edit
                )
        else:
            make_isotopic(
                derived_from=derived_from, gt_run_id=gt_run_id, user_edit=user_edit
            )  # should not raise


# ---------------------------------------------------------------------------
# MIsotopic — composition-level rules
# ---------------------------------------------------------------------------


class TestMIsotopic:
    def test_valid_construction(self):
        make_isotopic()  # sanity check — default valid kwargs should not raise

    def test_empty_components_rejected(self):
        with pytest.raises(ValueError):
            make_isotopic(components={})

    def test_bare_nuclide_with_digits_no_enrichment_is_allowed(self):
        # the ordinary case — "U235"/"U238" are nuclide keys with digits,
        # but carry no enrichment fields, so the digit check never fires
        make_isotopic(
            components={
                "U235": MaterialComponentEntry(percent=3.2),
                "U238": MaterialComponentEntry(percent=96.8),
            }
        )

    def test_enrichment_on_bare_element_key_is_allowed(self):
        make_isotopic(
            components={
                "U": MaterialComponentEntry(
                    percent=1.0,
                    enrichment=3.2,
                    enrichment_target="U235",
                    enrichment_type="wo",
                ),
                "O": MaterialComponentEntry(percent=2.0),
            }
        )

    def test_enrichment_on_nuclide_key_rejected(self):
        # a specific isotope ("U235") can't itself carry enrichment fields —
        # only a bare element ("U") can
        with pytest.raises(ValueError):
            make_isotopic(
                components={
                    "U235": MaterialComponentEntry(
                        percent=1.0,
                        enrichment=3.2,
                        enrichment_target="U235",
                        enrichment_type="wo",
                    ),
                }
            )

    def test_v1_with_enrichment_is_allowed(self):
        make_isotopic(
            derived_from=None,
            gt_run_id=None,
            components={
                "U": MaterialComponentEntry(
                    percent=1.0,
                    enrichment=3.2,
                    enrichment_target="U235",
                    enrichment_type="wo",
                ),
            },
        )

    def test_v2_plus_with_enrichment_rejected(self):
        # depletion output must be exact nuclide fractions — enrichment
        # shorthand on a GT-run-derived version indicates an inconsistent
        # or incorrectly-constructed version
        with pytest.raises(ValueError):
            make_isotopic(
                derived_from=MaterialVersionID("mv-0"),
                gt_run_id=GTRunID("run-1"),
                components={
                    "U": MaterialComponentEntry(
                        percent=1.0,
                        enrichment=3.2,
                        enrichment_target="U235",
                        enrichment_type="wo",
                    ),
                },
            )

    def test_v2_plus_with_bare_nuclides_is_allowed(self):
        make_isotopic(
            derived_from=MaterialVersionID("mv-0"),
            gt_run_id=GTRunID("run-1"),
            components={
                "U235": MaterialComponentEntry(percent=3.0),
                "U236": MaterialComponentEntry(percent=0.5),
                "U238": MaterialComponentEntry(percent=96.5),
            },
        )

    def test_density_value_zero_rejected(self):
        with pytest.raises(ValueError):
            make_isotopic(density_value=0.0)

    def test_density_value_negative_rejected(self):
        with pytest.raises(ValueError):
            make_isotopic(density_value=-1.0)

    def test_to_dict_from_dict_round_trip(self):
        original = make_isotopic()
        rebuilt = MIsotopic.from_dict(original.to_dict())
        # id-field-based __eq__ (matching GAddition/GSubtraction) means
        # this holds directly now, but field-by-field checks are kept
        # too since they pin down exactly what round-trips correctly.
        assert rebuilt == original
        assert rebuilt.material_id == original.material_id
        assert rebuilt.derived_from == original.derived_from
        assert rebuilt.gt_run_id == original.gt_run_id
        assert rebuilt.percent_type == original.percent_type
        assert rebuilt.density_value == original.density_value
        assert rebuilt.density_unit == original.density_unit
        assert rebuilt.components == original.components

    def test_frozen(self):
        instance = make_isotopic()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.density_value = 999.0  # type: ignore[misc]


class TestMIsotopicEquality:
    """MIsotopic's __eq__/__hash__ are id-field-based (isinstance check +
    self.id == value.id), matching GAddition/GSubtraction in geometry.py —
    NOT pure object identity. Two separately-constructed instances with the
    same id are equal and must hash identically, even with different
    composition data."""

    def test_same_id_different_fields_are_equal(self):
        a = make_isotopic(
            id=MaterialVersionID("mv-shared"),
            components={"U235": MaterialComponentEntry(percent=1.0)},
        )
        b = make_isotopic(
            id=MaterialVersionID("mv-shared"),
            components={"O16": MaterialComponentEntry(percent=2.0)},
        )
        # same id, different composition — still equal/same-hash by design
        assert a == b
        assert hash(a) == hash(b)

    def test_different_id_is_not_equal(self):
        a = make_isotopic(id=MaterialVersionID("mv-a"))
        b = make_isotopic(id=MaterialVersionID("mv-b"))
        assert a != b

    def test_hash_matches_hash_of_id(self):
        # regression guard: __hash__ must be hash(self.id), not id(self.id)
        # (the latter hashes object identity/memory address, not the id's
        # value — silently breaks the a == b => hash(a) == hash(b) contract)
        instance = make_isotopic(id=MaterialVersionID("mv-1"))
        assert hash(instance) == hash(MaterialVersionID("mv-1"))


# ---------------------------------------------------------------------------
# MMixture — composition-level rules
# ---------------------------------------------------------------------------


class TestMMixture:
    def test_valid_construction(self):
        make_mixture()  # sanity check — default valid kwargs should not raise

    def test_requires_at_least_two_constituents(self):
        with pytest.raises(ValueError):
            make_mixture(components=[(MaterialVersionID("mv-a"), 1.0)])

    @pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.1])
    def test_fraction_out_of_range_rejected(self, bad_fraction):
        with pytest.raises(ValueError):
            make_mixture(
                components=[
                    (MaterialVersionID("mv-a"), bad_fraction),
                    (
                        MaterialVersionID("mv-b"),
                        1.0 - bad_fraction if 0.0 < bad_fraction < 1.0 else 0.5,
                    ),
                ]
            )

    def test_fractions_not_summing_to_one_rejected(self):
        with pytest.raises(ValueError):
            make_mixture(
                components=[
                    (MaterialVersionID("mv-a"), 0.5),
                    (MaterialVersionID("mv-b"), 0.4),
                ]
            )

    def test_fractions_summing_to_one_is_allowed(self):
        make_mixture(
            components=[
                (MaterialVersionID("mv-a"), 0.7),
                (MaterialVersionID("mv-b"), 0.3),
            ]
        )

    def test_fractions_summing_to_one_across_three_is_allowed(self):
        make_mixture(
            components=[
                (MaterialVersionID("mv-a"), 0.2),
                (MaterialVersionID("mv-b"), 0.3),
                (MaterialVersionID("mv-c"), 0.5),
            ]
        )

    def test_fractions_summing_to_one_within_float_tolerance_is_allowed(self):
        # 0.1 + 0.2 + 0.7 != 1.0 exactly in floating point —
        # math.isclose() should still accept this
        make_mixture(
            components=[
                (MaterialVersionID("mv-a"), 0.1),
                (MaterialVersionID("mv-b"), 0.2),
                (MaterialVersionID("mv-c"), 0.7),
            ]
        )

    def test_to_dict_from_dict_round_trip(self):
        original = make_mixture()
        rebuilt = MMixture.from_dict(original.to_dict())
        assert rebuilt == original
        assert rebuilt.material_id == original.material_id
        assert rebuilt.percent_type == original.percent_type
        assert rebuilt.components == original.components

    def test_frozen(self):
        instance = make_mixture()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.percent_type = "ao"  # type: ignore[misc]


class TestMMixtureEquality:
    """Same id-field-based __eq__/__hash__ as MIsotopic — see
    TestMIsotopicEquality."""

    def test_same_id_different_fields_are_equal(self):
        a = make_mixture(
            id=MaterialVersionID("mv-shared"),
            components=[
                (MaterialVersionID("mv-a"), 0.7),
                (MaterialVersionID("mv-b"), 0.3),
            ],
        )
        b = make_mixture(
            id=MaterialVersionID("mv-shared"),
            components=[
                (MaterialVersionID("mv-c"), 0.5),
                (MaterialVersionID("mv-d"), 0.5),
            ],
        )
        # same id, different components — still equal/same-hash by design
        assert a == b
        assert hash(a) == hash(b)

    def test_different_id_is_not_equal(self):
        a = make_mixture(id=MaterialVersionID("mv-a"))
        b = make_mixture(id=MaterialVersionID("mv-b"))
        assert a != b

    def test_hash_matches_hash_of_id(self):
        # regression guard: __hash__ must be hash(self.id), not id(self.id)
        instance = make_mixture(id=MaterialVersionID("mv-1"))
        assert hash(instance) == hash(MaterialVersionID("mv-1"))
