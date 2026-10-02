"""
tests/unit/pace/core/test_material.py

Covers: Material identity (build_id/_validate_id), the three-state
lineage rule, ABC enforcement, MaterialComponentEntry's all-or-none
enrichment invariant, field constraints, enum-typed fields (PercentType,
DensityUnit) rejecting plain strings and unknown values,
MIsotopic's composition-level rules (non-empty, GT-run-derived
restricted to exact nuclides, enrichment-format keys restricted to bare
elements), MMixture's composition-level rules (min constituents,
fraction bounds, fractions summing to 1), ThermalScatteringLibrary and
TabulatedProperty, round-trips, frozen immutability, id-based equality,
and material_from_dict()'s type-based dispatch.
"""

import dataclasses
from typing import ClassVar

import pytest
from pace.core.ids import GTRunID, MaterialID
from pace.core.material import (
    DensityUnit,
    Material,
    MaterialComponentEntry,
    MIsotopic,
    MMixture,
    MVoid,
    PercentType,
    TabulatedProperty,
    ThermalScatteringLibrary,
    material_from_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(
    family_name: str = "test_mat", version_label: str = "1", **overrides
) -> dict:
    """Common Material base fields — valid version-1 (no lineage)
    by default. id is computed from family_name/version_label via
    build_id(), matching _validate_id()'s requirement — override id
    directly only when deliberately testing a mismatch."""
    kwargs = {
        "id": Material.build_id(family_name, version_label),
        "family_name": family_name,
        "version_label": version_label,
        "derived_from": None,
        "gt_run_id": None,
        "user_edit": False,
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
        "percent_type": PercentType.WO,
        "density_value": 10.3,
        "density_unit": DensityUnit.G_PER_CM3,
    }
    kwargs.update(overrides)
    return kwargs


def _valid_mixture_kwargs(**overrides) -> dict:
    """A valid MMixture composition — 70/30 atom-fraction mix of two materials."""
    kwargs = {
        "components": [
            (MaterialID("mv-a"), 0.7),
            (MaterialID("mv-b"), 0.3),
        ],
        "percent_type": PercentType.AO,
    }
    kwargs.update(overrides)
    return kwargs


def make_isotopic(
    family_name: str = "test_mat", version_label: str = "1", **overrides
) -> MIsotopic:
    kwargs = _base_kwargs(family_name=family_name, version_label=version_label)
    kwargs.update(_valid_isotopic_kwargs())
    kwargs.update(overrides)
    return MIsotopic(**kwargs)


def make_mixture(
    family_name: str = "test_mat", version_label: str = "1", **overrides
) -> MMixture:
    kwargs = _base_kwargs(family_name=family_name, version_label=version_label)
    kwargs.update(_valid_mixture_kwargs())
    kwargs.update(overrides)
    return MMixture(**kwargs)


def make_void(
    family_name: str = "test_mat", version_label: str = "1", **overrides
) -> MVoid:
    kwargs = _base_kwargs(family_name=family_name, version_label=version_label)
    kwargs.update(overrides)
    return MVoid(**kwargs)


# ---------------------------------------------------------------------------
# Identity: build_id() / _validate_id() / create()
# ---------------------------------------------------------------------------


class TestMaterialIdentity:
    def test_build_id_format(self):
        assert Material.build_id("uranium3.2", "1") == "uranium3.2-1"

    def test_build_id_with_string_version_label(self):
        assert Material.build_id("fuel", "6month_depletion") == "fuel-6month_depletion"

    def test_mismatched_id_rejected(self):
        with pytest.raises(ValueError):
            make_isotopic(id=MaterialID("wrong_id"))

    def test_matching_id_is_allowed(self):
        make_isotopic()  # should not raise

    def test_create_derives_same_id_as_build_id(self):
        mat = MIsotopic.create(
            family_name="test_mat", version_label="1", **_valid_isotopic_kwargs()
        )
        assert mat.id == Material.build_id("test_mat", "1")


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
            enrichment_type=PercentType.WO,
        )
        assert entry.enrichment == 3.2

    @pytest.mark.parametrize(
        "overrides",
        [
            {"enrichment": 3.2},  # enrichment only
            {"enrichment_target": "U235"},  # target only
            {"enrichment_type": PercentType.WO},  # type only
            {"enrichment": 3.2, "enrichment_target": "U235"},  # missing type
            {"enrichment": 3.2, "enrichment_type": PercentType.WO},  # missing target
            {
                "enrichment_target": "U235",
                "enrichment_type": PercentType.WO,
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

    def test_enrichment_unset_does_not_raise(self):
        # enrichment carries Constraint.POSITIVE + a range, but is
        # Optional — validate_fields() must skip a None value rather
        # than treating it as failing those checks
        MaterialComponentEntry(percent=1.0)  # should not raise

    def test_enrichment_zero_rejected(self):
        # Constraint.POSITIVE — zero enrichment is degenerate, must be rejected
        # even though the range's lower bound (0, 100) is inclusive of 0
        with pytest.raises(ValueError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=0.0,
                enrichment_target="U235",
                enrichment_type=PercentType.WO,
            )

    def test_enrichment_negative_rejected(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=-5.0,
                enrichment_target="U235",
                enrichment_type=PercentType.WO,
            )

    def test_enrichment_at_max_is_allowed(self):
        # upper bound is inclusive per validate_fields()
        MaterialComponentEntry(
            percent=1.0,
            enrichment=100.0,
            enrichment_target="U235",
            enrichment_type=PercentType.WO,
        )

    def test_enrichment_above_max_rejected(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=100.1,
                enrichment_target="U235",
                enrichment_type=PercentType.WO,
            )

    def test_round_trip(self):
        entry = MaterialComponentEntry(
            percent=1.0,
            enrichment=3.2,
            enrichment_target="U235",
            enrichment_type=PercentType.WO,
        )
        assert MaterialComponentEntry.from_dict(entry.to_dict()) == entry

    def test_enrichment_type_rejects_plain_string(self):
        with pytest.raises(TypeError):
            MaterialComponentEntry(
                percent=1.0,
                enrichment=3.2,
                enrichment_target="U235",
                enrichment_type="wo",  # type: ignore[arg-type]
            )

    def test_from_dict_rejects_unknown_enrichment_type(self):
        with pytest.raises(ValueError):
            MaterialComponentEntry.from_dict(
                {
                    "percent": 1.0,
                    "enrichment": 3.2,
                    "enrichment_target": "U235",
                    "enrichment_type": "vo",
                }
            )

    def test_frozen(self):
        entry = MaterialComponentEntry(percent=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.percent = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Material — ABC enforcement + shared lineage invariant
# ---------------------------------------------------------------------------


class TestMaterialBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Material(**_base_kwargs())  # pyright: ignore[reportAbstractUsage]

    def test_incomplete_subclass_cannot_instantiate(self):
        # a subclass missing _validate_composition/from_dict should
        # still fail to instantiate, same as the base itself
        class Incomplete(Material):
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
            MaterialID("mv-0"), GTRunID("run-1"), False, False, id="derived_gt_run_only"
        ),
        pytest.param(
            MaterialID("mv-0"), None, True, False, id="derived_user_edit_only"
        ),
        pytest.param(MaterialID("mv-0"), None, False, True, id="derived_with_neither"),
        pytest.param(
            MaterialID("mv-0"), GTRunID("run-1"), True, True, id="derived_with_both"
        ),
    ]

    @pytest.mark.parametrize(
        "derived_from,gt_run_id,user_edit,should_raise", LINEAGE_CASES
    )
    def test_lineage_rule(self, derived_from, gt_run_id, user_edit, should_raise):
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
                    enrichment_type=PercentType.WO,
                ),
                "O": MaterialComponentEntry(percent=2.0),
            }
        )

    def test_enrichment_on_nuclide_key_rejected(self):
        # a specific isotope ("U235") can't itself carry enrichment fields —
        # only a bare element ("U") can; the check is key-format based
        # (digits present), not a lookup against a periodic table
        with pytest.raises(ValueError):
            make_isotopic(
                components={
                    "U235": MaterialComponentEntry(
                        percent=1.0,
                        enrichment=3.2,
                        enrichment_target="U235",
                        enrichment_type=PercentType.WO,
                    ),
                }
            )

    def test_v1_with_enrichment_is_allowed(self):
        make_isotopic(
            derived_from=None,
            gt_run_id=None,
            user_edit=False,
            components={
                "U": MaterialComponentEntry(
                    percent=1.0,
                    enrichment=3.2,
                    enrichment_target="U235",
                    enrichment_type=PercentType.WO,
                ),
            },
        )

    def test_user_edited_version_with_enrichment_is_allowed(self):
        # a manual edit (user_edit=True) is not GT-run output, so it can
        # still use enrichment shorthand — only gt_run_id restricts to
        # bare nuclides
        make_isotopic(
            derived_from=MaterialID("mv-0"),
            gt_run_id=None,
            user_edit=True,
            components={
                "U": MaterialComponentEntry(
                    percent=1.0,
                    enrichment=3.2,
                    enrichment_target="U235",
                    enrichment_type=PercentType.WO,
                ),
            },
        )

    def test_v2_plus_gt_derived_with_enrichment_rejected(self):
        # depletion output must be exact nuclide fractions — enrichment
        # shorthand on a GT-run-derived version indicates an inconsistent
        # or incorrectly-constructed version
        with pytest.raises(ValueError):
            make_isotopic(
                derived_from=MaterialID("mv-0"),
                gt_run_id=GTRunID("run-1"),
                user_edit=False,
                components={
                    "U": MaterialComponentEntry(
                        percent=1.0,
                        enrichment=3.2,
                        enrichment_target="U235",
                        enrichment_type=PercentType.WO,
                    ),
                },
            )

    def test_v2_plus_gt_derived_with_bare_nuclides_is_allowed(self):
        make_isotopic(
            derived_from=MaterialID("mv-0"),
            gt_run_id=GTRunID("run-1"),
            user_edit=False,
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
        assert rebuilt.to_dict() == original.to_dict()
        assert rebuilt.family_name == original.family_name
        assert rebuilt.version_label == original.version_label
        assert rebuilt.derived_from == original.derived_from
        assert rebuilt.gt_run_id == original.gt_run_id
        assert rebuilt.user_edit == original.user_edit
        assert rebuilt.percent_type == original.percent_type
        assert rebuilt.density_value == original.density_value
        assert rebuilt.density_unit == original.density_unit
        assert rebuilt.components == original.components

    def test_to_dict_includes_type(self):
        assert make_isotopic().to_dict()["type"] == "isotopic"

    def test_to_dict_stores_enum_values_as_plain_strings(self):
        data = make_isotopic().to_dict()
        assert type(data["percent_type"]) is str
        assert type(data["density_unit"]) is str

    def test_percent_type_rejects_plain_string(self):
        with pytest.raises(TypeError):
            make_isotopic(percent_type="wo")

    def test_density_unit_rejects_plain_string(self):
        with pytest.raises(TypeError):
            make_isotopic(density_unit="g/cm3")

    @pytest.mark.parametrize(
        "key,bad_value", [("percent_type", "vo"), ("density_unit", "lb/ft3")]
    )
    def test_from_dict_rejects_unknown_enum_value(self, key, bad_value):
        data = make_isotopic().to_dict()
        data[key] = bad_value
        with pytest.raises(ValueError):
            MIsotopic.from_dict(data)

    @pytest.mark.xfail(
        strict=True,
        reason="ThermalScatteringLibrary.from_dict reads nuclide from data['name']",
    )
    def test_thermal_scattering_round_trips(self):
        original = make_isotopic(
            thermal_scattering=(
                ThermalScatteringLibrary(name="c_H_in_H2O", nuclide="H1"),
            )
        )
        rebuilt = MIsotopic.from_dict(original.to_dict())
        assert rebuilt.thermal_scattering == original.thermal_scattering

    def test_duplicate_thermal_scattering_name_rejected(self):
        table = ThermalScatteringLibrary(name="c_H_in_H2O", nuclide="H1")
        with pytest.raises(ValueError):
            make_isotopic(thermal_scattering=(table, table))

    def test_frozen(self):
        instance = make_isotopic()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.density_value = 999.0  # type: ignore[misc]


class TestMIsotopicEquality:
    """Equality is inherited from Versioned: same concrete type and same
    id. Two separately-constructed instances with the same id are equal
    and hash identically, even with different composition data."""

    def test_same_id_different_fields_are_equal(self):
        a = make_isotopic(
            family_name="shared",
            version_label="1",
            components={"U235": MaterialComponentEntry(percent=1.0)},
        )
        b = make_isotopic(
            family_name="shared",
            version_label="1",
            components={"O16": MaterialComponentEntry(percent=2.0)},
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_different_id_is_not_equal(self):
        a = make_isotopic(family_name="fam-a", version_label="1")
        b = make_isotopic(family_name="fam-b", version_label="1")
        assert a != b

    def test_hash_matches_hash_of_id(self):
        # regression guard: __hash__ must be hash(self.id), not id(self.id)
        instance = make_isotopic(family_name="shared", version_label="1")
        assert hash(instance) == hash(Material.build_id("shared", "1"))


# ---------------------------------------------------------------------------
# MMixture — composition-level rules
# ---------------------------------------------------------------------------


class TestMMixture:
    def test_valid_construction(self):
        make_mixture()  # sanity check — default valid kwargs should not raise

    def test_requires_at_least_two_constituents(self):
        with pytest.raises(ValueError):
            make_mixture(components=[(MaterialID("mv-a"), 1.0)])

    @pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.1])
    def test_fraction_out_of_range_rejected(self, bad_fraction):
        with pytest.raises(ValueError):
            make_mixture(
                components=[
                    (MaterialID("mv-a"), bad_fraction),
                    (
                        MaterialID("mv-b"),
                        1.0 - bad_fraction if 0.0 < bad_fraction < 1.0 else 0.5,
                    ),
                ]
            )

    def test_fractions_not_summing_to_one_rejected(self):
        with pytest.raises(ValueError):
            make_mixture(
                components=[
                    (MaterialID("mv-a"), 0.5),
                    (MaterialID("mv-b"), 0.4),
                ]
            )

    def test_fractions_summing_to_one_is_allowed(self):
        make_mixture(
            components=[
                (MaterialID("mv-a"), 0.7),
                (MaterialID("mv-b"), 0.3),
            ]
        )

    def test_fractions_summing_to_one_across_three_is_allowed(self):
        make_mixture(
            components=[
                (MaterialID("mv-a"), 0.2),
                (MaterialID("mv-b"), 0.3),
                (MaterialID("mv-c"), 0.5),
            ]
        )

    def test_fractions_summing_to_one_within_float_tolerance_is_allowed(self):
        # 0.1 + 0.2 + 0.7 != 1.0 exactly in floating point —
        # math.isclose() should still accept this
        make_mixture(
            components=[
                (MaterialID("mv-a"), 0.1),
                (MaterialID("mv-b"), 0.2),
                (MaterialID("mv-c"), 0.7),
            ]
        )

    def test_to_dict_from_dict_round_trip(self):
        original = make_mixture()
        rebuilt = MMixture.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()
        assert rebuilt.family_name == original.family_name
        assert rebuilt.version_label == original.version_label
        assert rebuilt.percent_type == original.percent_type
        assert rebuilt.components == original.components

    def test_to_dict_includes_type(self):
        assert make_mixture().to_dict()["type"] == "mixture"

    def test_percent_type_rejects_plain_string(self):
        with pytest.raises(TypeError):
            make_mixture(percent_type="ao")

    def test_volume_percent_is_not_accepted(self):
        data = make_mixture().to_dict()
        data["percent_type"] = "vo"
        with pytest.raises(ValueError):
            MMixture.from_dict(data)

    def test_frozen(self):
        instance = make_mixture()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.percent_type = PercentType.WO  # type: ignore[misc]


class TestMMixtureEquality:
    """Same id-based equality as MIsotopic — see TestMIsotopicEquality."""

    def test_same_id_different_fields_are_equal(self):
        a = make_mixture(
            family_name="shared",
            version_label="1",
            components=[
                (MaterialID("mv-a"), 0.7),
                (MaterialID("mv-b"), 0.3),
            ],
        )
        b = make_mixture(
            family_name="shared",
            version_label="1",
            components=[
                (MaterialID("mv-c"), 0.5),
                (MaterialID("mv-d"), 0.5),
            ],
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_different_id_is_not_equal(self):
        a = make_mixture(family_name="fam-a", version_label="1")
        b = make_mixture(family_name="fam-b", version_label="1")
        assert a != b

    def test_hash_matches_hash_of_id(self):
        instance = make_mixture(family_name="shared", version_label="1")
        assert hash(instance) == hash(Material.build_id("shared", "1"))


# ---------------------------------------------------------------------------
# MVoid — no composition, no fields
# ---------------------------------------------------------------------------


class TestMVoid:
    def test_valid_construction(self):
        make_void()  # sanity check — should not raise

    def test_to_dict_from_dict_round_trip(self):
        original = make_void()
        rebuilt = MVoid.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()

    def test_to_dict_includes_type(self):
        assert make_void().to_dict()["type"] == "void"

    def test_lineage_rule_still_applies(self):
        # MVoid inherits Material's lineage validation same as
        # any other subclass — v1 (no derivation cause) is fine here
        make_void(derived_from=None, gt_run_id=None, user_edit=False)
        # derived with no cause is still rejected
        with pytest.raises(ValueError):
            make_void(derived_from=MaterialID("mv-0"), gt_run_id=None, user_edit=False)

    def test_frozen(self):
        instance = make_void()
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.family_name = "other"  # type: ignore[misc]

    def test_different_id_is_not_equal(self):
        a = make_void(family_name="fam-a", version_label="1")
        b = make_void(family_name="fam-b", version_label="1")
        assert a != b

    def test_same_id_different_fields_are_equal(self):
        a = make_void(family_name="shared", version_label="1")
        b = make_void(
            family_name="shared",
            version_label="1",
            thermal_conductivity=TabulatedProperty(points=((300.0, 1.0),)),
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_hashable_with_a_tabulated_property(self):
        instance = make_void(
            thermal_conductivity=TabulatedProperty(points=((300.0, 1.0),))
        )
        assert hash(instance) == hash(instance.id)


# ---------------------------------------------------------------------------
# material_from_dict() — type-based dispatch
# ---------------------------------------------------------------------------


class TestMaterialFromDict:
    def test_dispatches_isotopic(self):
        original = make_isotopic()
        rebuilt = material_from_dict(original.to_dict())
        assert type(rebuilt) is MIsotopic
        assert rebuilt.to_dict() == original.to_dict()

    def test_dispatches_mixture(self):
        original = make_mixture()
        rebuilt = material_from_dict(original.to_dict())
        assert type(rebuilt) is MMixture
        assert rebuilt.to_dict() == original.to_dict()

    def test_dispatches_void(self):
        original = make_void()
        rebuilt = material_from_dict(original.to_dict())
        assert type(rebuilt) is MVoid
        assert rebuilt.to_dict() == original.to_dict()

    def test_unknown_type_raises_value_error_not_key_error(self):
        with pytest.raises(ValueError):
            material_from_dict({"type": "not_a_real_material"})


def test_isotopic_and_void_with_same_id_are_not_equal():
    """Equality requires the same concrete type, not just the same id."""
    assert make_isotopic(family_name="shared") != make_void(family_name="shared")


# ---------------------------------------------------------------------------
# ThermalScatteringLibrary
# ---------------------------------------------------------------------------


class TestThermalScatteringLibrary:
    def test_valid_construction(self):
        table = ThermalScatteringLibrary(
            name="c_Be_in_BeO", nuclide="Be9", fraction=0.5
        )
        assert table.fraction == 0.5

    @pytest.mark.parametrize("bad_fraction", [0.0, -0.1, 1.1])
    def test_fraction_out_of_range_rejected(self, bad_fraction):
        with pytest.raises(ValueError):
            ThermalScatteringLibrary(
                name="c_H_in_H2O", nuclide="H1", fraction=bad_fraction
            )

    def test_fraction_of_one_is_allowed(self):
        ThermalScatteringLibrary(name="c_H_in_H2O", nuclide="H1", fraction=1.0)

    @pytest.mark.xfail(
        strict=True,
        reason="ThermalScatteringLibrary.from_dict reads nuclide from data['name']",
    )
    def test_round_trip_keeps_nuclide(self):
        table = ThermalScatteringLibrary(name="c_H_in_H2O", nuclide="H1")
        assert ThermalScatteringLibrary.from_dict(table.to_dict()) == table


# ---------------------------------------------------------------------------
# TabulatedProperty
# ---------------------------------------------------------------------------


class TestTabulatedProperty:
    def test_valid_construction(self):
        prop = TabulatedProperty(points=((300.0, 3.0), (600.0, 2.5)))
        assert len(prop.points) == 2

    def test_requires_at_least_one_point(self):
        with pytest.raises(ValueError):
            TabulatedProperty(points=())

    def test_unsorted_temperatures_rejected(self):
        with pytest.raises(ValueError):
            TabulatedProperty(points=((600.0, 2.5), (300.0, 3.0)))

    def test_round_trip(self):
        prop = TabulatedProperty(points=((300.0, 3.0), (600.0, 2.5)))
        assert TabulatedProperty.from_dict(prop.to_dict()) == prop

    def test_is_hashable(self):
        prop = TabulatedProperty(points=((300.0, 3.0),))
        assert hash(prop) == hash(TabulatedProperty(points=((300.0, 3.0),)))

    def test_material_round_trip_keeps_tabulated_properties(self):
        original = make_isotopic(
            thermal_conductivity=TabulatedProperty(points=((300.0, 3.0), (900.0, 2.4))),
            specific_heat=TabulatedProperty(points=((300.0, 240.0),)),
        )
        rebuilt = MIsotopic.from_dict(original.to_dict())
        assert rebuilt.thermal_conductivity == original.thermal_conductivity
        assert rebuilt.specific_heat == original.specific_heat
