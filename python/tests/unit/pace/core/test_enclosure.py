"""
tests/unit/pace/core/test_enclosure.py

Covers: Enclosure identity, lineage, bounds/fill (including fill-only
Enclosures), composition (unique placement ids, duplicate-placement
detection including the z_rotation_rad and ref-type differentiation,
self-reference), id-based equality and round-trips.
"""

from dataclasses import FrozenInstanceError

import pytest

from pace.core.enclosure import Enclosure
from pace.core.geometry import GPose
from pace.core.ids import (
    EnclosureID,
    GeometryID,
    LatticeID,
    MaterialID,
    PartID,
    PlacementID,
    SimulationJobID,
)
from pace.core.placeable import PlaceableRef, PlaceableType
from pace.core.placement import Placement

PELLET_REF = PlaceableRef(type=PlaceableType.PART, id=PartID("fuel_pellet-1"))
LATTICE_REF = PlaceableRef(type=PlaceableType.LATTICE, id=LatticeID("lattice_3x3-1"))
ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)

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


# =============================================================================
# Enclosure — identity
# =============================================================================

BOUNDS = GeometryID("pin_cell_box-1")
WATER = MaterialID("borated_water-1")


def _two_placements(z_rotation_rad_second: float = 0.0) -> list[Placement]:
    return [
        Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF),
        Placement(
            id=PlacementID("pc-2"),
            pose=GPose(
                x_m=0.0, y_m=0.0, z_m=0.01, z_rotation_rad=z_rotation_rad_second
            ),
            ref=PELLET_REF,
        ),
    ]


def _enclosure(**overrides) -> Enclosure:
    kwargs = {
        "family_name": "fuel_pin",
        "version_label": "1",
        "bounds": BOUNDS,
        "fill": WATER,
        "placements": _two_placements(),
    }
    kwargs.update(overrides)
    return Enclosure.create(**kwargs)


def test_enclosure_build_id():
    assert Enclosure.build_id("fuel_pin", "1") == "fuel_pin-1"


def test_enclosure_create_derives_id():
    assert _enclosure().id == "fuel_pin-1"


def test_enclosure_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        Enclosure(
            id=EnclosureID("mismatched_id"),
            family_name="fuel_pin",
            version_label="1",
            bounds=BOUNDS,
            placements=_two_placements(),
        )


# =============================================================================
# Enclosure — three-state lineage rule
# =============================================================================


@pytest.mark.parametrize(
    ("derived_from", "simulation_job_id", "user_edit", "should_raise"), LINEAGE_CASES
)
def test_enclosure_lineage_rule(
    derived_from, simulation_job_id, user_edit, should_raise
):
    version_label = "2" if derived_from else "1"
    derived_from_id = (
        EnclosureID(derived_from.replace("fuel_pellet", "fuel_pin"))
        if derived_from
        else None
    )

    def build() -> Enclosure:
        return _enclosure(
            version_label=version_label,
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
# Enclosure — bounds and fill
# =============================================================================


def test_enclosure_fill_is_optional():
    """fill=None means the members are expected to occupy the bounds
    exactly (e.g. concentric layers tiling a cylinder)."""
    assert _enclosure(fill=None).fill is None


def test_enclosure_single_member_is_allowed():
    """A guide-tube cell is one tube member plus fill — the minimum is
    one member, not two."""
    cc = _enclosure(placements=_two_placements()[:1])
    assert len(cc.placements) == 1


def test_enclosure_with_neither_fill_nor_placements_is_rejected():
    with pytest.raises(ValueError, match="needs a fill, at least one placement"):
        _enclosure(fill=None, placements=[])


def test_enclosure_with_only_a_fill_is_a_single_material_region():
    coolant_only = _enclosure(fill=WATER, placements=[])
    assert coolant_only.placements == []


@pytest.mark.parametrize("fill", [WATER, None])
def test_enclosure_round_trip_keeps_bounds_and_fill(fill):
    cc = _enclosure(fill=fill)
    restored = Enclosure.from_dict(cc.to_dict())
    assert restored.bounds == BOUNDS
    assert restored.fill == fill
    assert restored.to_dict() == cc.to_dict()


# =============================================================================
# Enclosure — composition (unique ids, duplicate placements, self-ref)
# =============================================================================


def test_enclosure_rejects_duplicate_placement_id():
    first, second = _two_placements()
    renamed = Placement(id=first.id, pose=second.pose, ref=second.ref)
    with pytest.raises(ValueError):
        _enclosure(placements=[first, renamed])


def test_enclosure_rejects_duplicate_placeable_position():
    duplicate = [
        Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF),
        Placement(id=PlacementID("pc-2"), pose=ORIGIN, ref=PELLET_REF),
    ]
    with pytest.raises(ValueError):
        _enclosure(placements=duplicate)


def test_enclosure_same_position_different_rotation_is_not_a_duplicate():
    """z_rotation_rad is part of the duplicate key — the same placeable
    at the same x/y/z but rotated is a distinct placement."""
    rotated = [
        Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF),
        Placement(
            id=PlacementID("pc-2"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0, z_rotation_rad=0.5),
            ref=PELLET_REF,
        ),
    ]
    assert len(_enclosure(placements=rotated).placements) == 2


def test_enclosure_same_id_string_different_type_is_not_a_duplicate():
    """The ref's type is part of the duplicate key — a Part and a
    Enclosure that coincidentally share an id string, placed at the
    same pose, are two different objects."""
    shared = "shared-1"
    members = [
        Placement(
            id=PlacementID("pc-1"),
            pose=ORIGIN,
            ref=PlaceableRef(type=PlaceableType.PART, id=PartID(shared)),
        ),
        Placement(
            id=PlacementID("pc-2"),
            pose=ORIGIN,
            ref=PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID(shared)),
        ),
    ]
    assert len(_enclosure(placements=members).placements) == 2


def test_enclosure_rejects_direct_self_reference():
    self_ref = Placement(
        id=PlacementID("pc-self"),
        pose=ORIGIN,
        ref=PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("fuel_pin-1")),
    )
    with pytest.raises(ValueError):
        _enclosure(placements=[self_ref])


def test_enclosure_can_place_a_lattice():
    cc = _enclosure(
        placements=[Placement(id=PlacementID("lattice"), pose=ORIGIN, ref=LATTICE_REF)]
    )
    assert cc.placements[0].ref.type == PlaceableType.LATTICE


# =============================================================================
# Enclosure — round-trip, immutability, equality
# =============================================================================


def test_enclosure_round_trip():
    cc = _enclosure()
    assert Enclosure.from_dict(cc.to_dict()).to_dict() == cc.to_dict()


def test_enclosure_is_frozen():
    cc = _enclosure()
    with pytest.raises(FrozenInstanceError):
        cc.fill = None  # type: ignore[misc]


def test_enclosure_equality_is_id_based_not_field_based():
    a = _enclosure()
    b = _enclosure(fill=None)
    assert a == b


def test_enclosure_hash_uses_hash_of_id_not_python_id():
    a = _enclosure()
    b = _enclosure()
    assert hash(a) == hash(b) == hash(a.id)
    assert len({a, b}) == 1
