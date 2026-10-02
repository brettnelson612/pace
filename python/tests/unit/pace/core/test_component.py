"""
tests/pace/core/test_component.py

Covers: LComponent identity (build_id/_validate_id) and the three-state
lineage rule; ComponentPlacement round-trips for every placeable kind and frozen
immutability; CComponent identity, lineage, bounds/fill, composition
(min-member, unique member ids, duplicate-placement detection including
the z_rotation_rad and ref-kind differentiation, self-reference), id-based
equality + the hash(self.id) regression guard, and round-trips.
ResolvedComponent/ResolvedReactorBlueprint are covered in test_resolved.py.
"""

from dataclasses import FrozenInstanceError

import pytest
from pace.core.component import CComponent, LComponent, ComponentPlacement
from pace.core.component_ref import ComponentType, ComponentRef
from pace.core.geometry import GPose
from pace.core.ids import (
    CComponentID,
    GeometryID,
    GTRunID,
    LatticeID,
    LComponentID,
    MaterialID,
    ComponentPlacementID,
)

# =============================================================================
# LComponent — identity
# =============================================================================


def test_lcomponent_build_id():
    assert LComponent.build_id("fuel_pellet", "1") == "fuel_pellet-1"


def test_lcomponent_create_derives_id():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert lc.id == "fuel_pellet-1"


def test_lcomponent_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        LComponent(
            id=LComponentID("mismatched_id"),
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )


# =============================================================================
# LComponent — three-state lineage rule
# =============================================================================

LINEAGE_CASES = [
    # (derived_from, gt_run_id, user_edit, should_raise)
    pytest.param(None, None, False, False, id="v1_valid"),
    pytest.param(None, GTRunID("gt-1"), False, True, id="v1_with_gt_run_id"),
    pytest.param(None, None, True, True, id="v1_with_user_edit"),
    pytest.param(None, GTRunID("gt-1"), True, True, id="v1_with_both"),
    pytest.param(
        "fuel_pellet-1", GTRunID("gt-1"), False, False, id="derived_gt_run_only"
    ),
    pytest.param("fuel_pellet-1", None, True, False, id="derived_user_edit_only"),
    pytest.param("fuel_pellet-1", None, False, True, id="derived_with_neither"),
    pytest.param("fuel_pellet-1", GTRunID("gt-1"), True, True, id="derived_with_both"),
]


@pytest.mark.parametrize("derived_from,gt_run_id,user_edit,should_raise", LINEAGE_CASES)
def test_lcomponent_lineage_rule(derived_from, gt_run_id, user_edit, should_raise):
    version_label = "2" if derived_from else "1"
    derived_from_id = LComponentID(derived_from) if derived_from else None

    def build() -> LComponent:
        return LComponent.create(
            family_name="fuel_pellet",
            version_label=version_label,
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
            derived_from=derived_from_id,
            gt_run_id=gt_run_id,
            user_edit=user_edit,
        )

    if should_raise:
        with pytest.raises(ValueError):
            build()
    else:
        assert build().derived_from == derived_from_id


# =============================================================================
# LComponent — round-trip, immutability, equality
# =============================================================================


def test_lcomponent_round_trip():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert LComponent.from_dict(lc.to_dict()) == lc


def test_lcomponent_derived_round_trip():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="2",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
        derived_from=LComponentID("fuel_pellet-1"),
        user_edit=True,
    )
    round_tripped = LComponent.from_dict(lc.to_dict())
    assert round_tripped == lc
    assert round_tripped.derived_from == "fuel_pellet-1"
    assert round_tripped.user_edit is True


def test_lcomponent_is_frozen():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    with pytest.raises(FrozenInstanceError):
        lc.family_name = "other"  # type: ignore[misc]


def test_lcomponent_default_equality_is_value_based():
    """LComponent has no dict/list fields, so it keeps the plain
    frozen-dataclass default equality — unlike CComponent, it does NOT
    override __eq__/__hash__ to be id-based."""

    def build() -> LComponent:
        return LComponent.create(
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )

    lc_a = build()
    lc_b = build()
    assert lc_a == lc_b

    lc_c = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("different_geometry-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert lc_a != lc_c


# =============================================================================
# ComponentPlacement
# =============================================================================

PELLET_REF = ComponentRef(
    type=ComponentType.LCOMPONENT, id=LComponentID("fuel_pellet-1")
)
ROD_REF = ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("fuel_rod-1"))
LATTICE_REF = ComponentRef(type=ComponentType.LATTICE, id=LatticeID("lattice_3x3-1"))
ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)


@pytest.mark.parametrize("ref", [PELLET_REF, ROD_REF, LATTICE_REF])
def test_placement_round_trip_every_placeable_kind(ref):
    pc = ComponentPlacement(id=ComponentPlacementID("pc-1"), pose=ORIGIN, ref=ref)
    assert ComponentPlacement.from_dict(pc.to_dict()) == pc


def test_placement_is_frozen():
    pc = ComponentPlacement(
        id=ComponentPlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF
    )
    with pytest.raises(FrozenInstanceError):
        pc.pose = GPose(x_m=1.0, y_m=0.0, z_m=0.0)  # type: ignore[misc]


def test_placement_rejects_empty_id():
    with pytest.raises(ValueError):
        ComponentPlacement(id=ComponentPlacementID(""), pose=ORIGIN, ref=PELLET_REF)


# =============================================================================
# CComponent — identity
# =============================================================================

BOUNDS = GeometryID("pin_cell_box-1")
WATER = MaterialID("borated_water-1")


def _two_placements(z_rotation_rad_second: float = 0.0) -> list[ComponentPlacement]:
    return [
        ComponentPlacement(
            id=ComponentPlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF
        ),
        ComponentPlacement(
            id=ComponentPlacementID("pc-2"),
            pose=GPose(
                x_m=0.0, y_m=0.0, z_m=0.01, z_rotation_rad=z_rotation_rad_second
            ),
            ref=PELLET_REF,
        ),
    ]


def _ccomponent(**overrides) -> CComponent:
    kwargs = {
        "family_name": "fuel_pin",
        "version_label": "1",
        "bounds": BOUNDS,
        "fill": WATER,
        "components": _two_placements(),
    }
    kwargs.update(overrides)
    return CComponent.create(**kwargs)


def test_ccomponent_build_id():
    assert CComponent.build_id("fuel_pin", "1") == "fuel_pin-1"


def test_ccomponent_create_derives_id():
    assert _ccomponent().id == "fuel_pin-1"


def test_ccomponent_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        CComponent(
            id=CComponentID("mismatched_id"),
            family_name="fuel_pin",
            version_label="1",
            bounds=BOUNDS,
            components=_two_placements(),
        )


# =============================================================================
# CComponent — three-state lineage rule
# =============================================================================


@pytest.mark.parametrize("derived_from,gt_run_id,user_edit,should_raise", LINEAGE_CASES)
def test_ccomponent_lineage_rule(derived_from, gt_run_id, user_edit, should_raise):
    version_label = "2" if derived_from else "1"
    derived_from_id = (
        CComponentID(derived_from.replace("fuel_pellet", "fuel_pin"))
        if derived_from
        else None
    )

    def build() -> CComponent:
        return _ccomponent(
            version_label=version_label,
            derived_from=derived_from_id,
            gt_run_id=gt_run_id,
            user_edit=user_edit,
        )

    if should_raise:
        with pytest.raises(ValueError):
            build()
    else:
        assert build().derived_from == derived_from_id


# =============================================================================
# CComponent — bounds and fill
# =============================================================================


def test_ccomponent_fill_is_optional():
    """fill=None means the members are expected to occupy the bounds
    exactly (e.g. concentric layers tiling a cylinder)."""
    assert _ccomponent(fill=None).fill is None


def test_ccomponent_single_member_is_allowed():
    """A guide-tube cell is one tube member plus fill — the minimum is
    one member, not two."""
    cc = _ccomponent(components=_two_placements()[:1])
    assert len(cc.components) == 1


def test_ccomponent_requires_at_least_one_component():
    with pytest.raises(ValueError):
        _ccomponent(components=[])


@pytest.mark.parametrize("fill", [WATER, None])
def test_ccomponent_round_trip_keeps_bounds_and_fill(fill):
    cc = _ccomponent(fill=fill)
    restored = CComponent.from_dict(cc.to_dict())
    assert restored.bounds == BOUNDS
    assert restored.fill == fill
    assert restored.to_dict() == cc.to_dict()


# =============================================================================
# CComponent — composition (unique ids, duplicate placements, self-ref)
# =============================================================================


def test_ccomponent_rejects_duplicate_placement_id():
    first, second = _two_placements()
    renamed = ComponentPlacement(id=first.id, pose=second.pose, ref=second.ref)
    with pytest.raises(ValueError):
        _ccomponent(components=[first, renamed])


def test_ccomponent_rejects_duplicate_component_position():
    duplicate = [
        ComponentPlacement(
            id=ComponentPlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF
        ),
        ComponentPlacement(
            id=ComponentPlacementID("pc-2"), pose=ORIGIN, ref=PELLET_REF
        ),
    ]
    with pytest.raises(ValueError):
        _ccomponent(components=duplicate)


def test_ccomponent_same_position_different_rotation_is_not_a_duplicate():
    """z_rotation_rad is part of the duplicate key — the same component
    at the same x/y/z but rotated is a distinct placement."""
    rotated = [
        ComponentPlacement(
            id=ComponentPlacementID("pc-1"), pose=ORIGIN, ref=PELLET_REF
        ),
        ComponentPlacement(
            id=ComponentPlacementID("pc-2"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0, z_rotation_rad=0.5),
            ref=PELLET_REF,
        ),
    ]
    assert len(_ccomponent(components=rotated).components) == 2


def test_ccomponent_same_id_string_different_kind_is_not_a_duplicate():
    """The ref's kind is part of the duplicate key — an LComponent and a
    CComponent that coincidentally share an id string, placed at the
    same pose, are two different objects."""
    shared = "shared-1"
    members = [
        ComponentPlacement(
            id=ComponentPlacementID("pc-1"),
            pose=ORIGIN,
            ref=ComponentRef(type=ComponentType.LCOMPONENT, id=LComponentID(shared)),
        ),
        ComponentPlacement(
            id=ComponentPlacementID("pc-2"),
            pose=ORIGIN,
            ref=ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID(shared)),
        ),
    ]
    assert len(_ccomponent(components=members).components) == 2


def test_ccomponent_rejects_direct_self_reference():
    self_ref = ComponentPlacement(
        id=ComponentPlacementID("pc-self"),
        pose=ORIGIN,
        ref=ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("fuel_pin-1")),
    )
    with pytest.raises(ValueError):
        _ccomponent(components=[self_ref])


def test_ccomponent_can_place_a_lattice():
    cc = _ccomponent(
        components=[
            ComponentPlacement(
                id=ComponentPlacementID("lattice"), pose=ORIGIN, ref=LATTICE_REF
            )
        ]
    )
    assert cc.components[0].ref.type == ComponentType.LATTICE


# =============================================================================
# CComponent — round-trip, immutability, equality
# =============================================================================


def test_ccomponent_round_trip():
    cc = _ccomponent()
    assert CComponent.from_dict(cc.to_dict()).to_dict() == cc.to_dict()


def test_ccomponent_is_frozen():
    cc = _ccomponent()
    with pytest.raises(FrozenInstanceError):
        cc.fill = None  # type: ignore[misc]


def test_ccomponent_equality_is_id_based_not_field_based():
    a = _ccomponent()
    b = _ccomponent(fill=None)
    assert a == b


def test_ccomponent_hash_uses_hash_of_id_not_python_id():
    a = _ccomponent()
    b = _ccomponent()
    assert hash(a) == hash(b) == hash(a.id)
    assert len({a, b}) == 1
