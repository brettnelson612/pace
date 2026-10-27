"""
tests/unit/pace/core/test_part.py

Covers: Part identity (build_id/_validate_id), the three-state lineage
rule, round-trips, immutability and id-based equality.
"""

from dataclasses import FrozenInstanceError

import pytest

from pace.core.ids import (
    GeometryID,
    MaterialID,
    PartID,
    SimulationJobID,
)
from pace.core.part import Part

# =============================================================================
# Part — identity
# =============================================================================


def test_part_build_id():
    assert Part.build_id("fuel_pellet", "1") == "fuel_pellet-1"


def test_part_create_derives_id():
    lc = Part.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert lc.id == "fuel_pellet-1"


def test_part_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        Part(
            id=PartID("mismatched_id"),
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )


# =============================================================================
# Part — three-state lineage rule
# =============================================================================

LINEAGE_CASES = [
    # (derived_from, simulation_job_id, user_edit, should_raise)
    pytest.param(None, None, False, False, id="v1_valid"),
    pytest.param(
        None, SimulationJobID("job-1"), False, True, id="v1_with_simulation_job_id"
    ),
    pytest.param(None, None, True, True, id="v1_with_user_edit"),
    pytest.param(None, SimulationJobID("job-1"), True, True, id="v1_with_both"),
    pytest.param(
        "fuel_pellet-1",
        SimulationJobID("job-1"),
        False,
        False,
        id="derived_simulation_job_only",
    ),
    pytest.param("fuel_pellet-1", None, True, False, id="derived_user_edit_only"),
    pytest.param("fuel_pellet-1", None, False, True, id="derived_with_neither"),
    pytest.param(
        "fuel_pellet-1", SimulationJobID("job-1"), True, True, id="derived_with_both"
    ),
]


@pytest.mark.parametrize(
    ("derived_from", "simulation_job_id", "user_edit", "should_raise"), LINEAGE_CASES
)
def test_part_lineage_rule(derived_from, simulation_job_id, user_edit, should_raise):
    version_label = "2" if derived_from else "1"
    derived_from_id = PartID(derived_from) if derived_from else None

    def build() -> Part:
        return Part.create(
            family_name="fuel_pellet",
            version_label=version_label,
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
            derived_from=derived_from_id,
            simulation_job_id=simulation_job_id,
            user_edit=user_edit,
        )

    if should_raise:
        with pytest.raises(ValueError):
            build()
    else:
        assert build().derived_from == derived_from_id


# =============================================================================
# Part — round-trip, immutability, equality
# =============================================================================


def test_part_round_trip():
    lc = Part.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert Part.from_dict(lc.to_dict()).to_dict() == lc.to_dict()


def test_part_derived_round_trip():
    lc = Part.create(
        family_name="fuel_pellet",
        version_label="2",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
        derived_from=PartID("fuel_pellet-1"),
        user_edit=True,
    )
    round_tripped = Part.from_dict(lc.to_dict())
    assert round_tripped.to_dict() == lc.to_dict()
    assert round_tripped.derived_from == "fuel_pellet-1"
    assert round_tripped.user_edit is True


def test_part_is_frozen():
    lc = Part.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    with pytest.raises(FrozenInstanceError):
        lc.family_name = "other"  # type: ignore[misc]


def test_part_equality_is_id_based():
    """Same id, different fields: equal, since the id is the identity."""
    lc_a = Part.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    lc_b = Part.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("different_geometry-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert lc_a == lc_b
    assert hash(lc_a) == hash(lc_b) == hash(lc_a.id)
