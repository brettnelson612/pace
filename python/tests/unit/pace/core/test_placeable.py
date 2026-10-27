"""
tests/unit/pace/core/test_placeable.py

Covers: PlaceableRef round-trips, the PlaceableType -> ReferenceableType
mapping, validation and value-based equality; Part identity
(build_id/_validate_id), the three-state lineage rule and id-based
equality; Placement round-trips for every placeable type and
frozen immutability; Enclosure identity, lineage, bounds/fill,
composition (min-member, unique placement ids, duplicate-placement
detection including the z_rotation_rad and ref-type differentiation,
self-reference), id-based equality and round-trips.
ResolvedPlaceable/ResolvedReactorBlueprint are covered in test_resolved.py.
"""

from dataclasses import FrozenInstanceError

import pytest

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
from pace.core.placeable import (
    Enclosure,
    Part,
    PlaceableRef,
    PlaceableType,
    Placement,
)
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


# =============================================================================
# Placement
# =============================================================================

PELLET_REF = PlaceableRef(type=PlaceableType.PART, id=PartID("fuel_pellet-1"))
ROD_REF = PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("fuel_rod-1"))
LATTICE_REF = PlaceableRef(type=PlaceableType.LATTICE, id=LatticeID("lattice_3x3-1"))
ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)


@pytest.mark.parametrize("ref", [PELLET_REF, ROD_REF, LATTICE_REF])
def test_placement_round_trip_every_placeable_type(ref):
    pc = Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=ref)
    assert Placement.from_dict(pc.to_dict()) == pc


def test_placement_is_frozen():
    pc = Placement(id=PlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF)
    with pytest.raises(FrozenInstanceError):
        pc.pose = GPose(x_m=1.0, y_m=0.0, z_m=0.0)  # type: ignore[misc]


def test_placement_rejects_empty_id():
    with pytest.raises(ValueError):
        Placement(id=PlacementID(""), pose=ORIGIN, ref=PELLET_REF)


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


def test_enclosure_requires_at_least_one_placement():
    with pytest.raises(ValueError):
        _enclosure(placements=[])


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
