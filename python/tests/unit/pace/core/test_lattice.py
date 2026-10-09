"""
tests/unit/pace/core/test_lattice.py

Covers: LatticeElement validation and round-trip; the Lattice base
(ABC enforcement, pitch constraint, lineage via Versioned, element
rules: at least one element, one element per ref, addresses in range,
no address occupied twice, no self-reference); RectLattice
addressing (row 0 on top), from_grid/to_grid, occupant_at and
empty_addresses; HexLattice ring sizes, (ring, index) addressing and
validation; lattice_from_dict dispatch; id-based equality.
"""

import pytest
from pace.core.component import ComponentRef, ComponentType
from pace.core.ids import LatticeID, MaterialID, SimulationJobID
from pace.core.lattice import (
    HexLattice,
    HexOrientation,
    Lattice,
    LatticeElement,
    RectLattice,
    lattice_from_dict,
)

WATER = MaterialID("borated_water-1")
SODIUM = MaterialID("sodium-1")
FUEL = ComponentRef(type=ComponentType.CCOMPONENT, id="fuel_pin_cell-1")
GUIDE = ComponentRef(type=ComponentType.CCOMPONENT, id="guide_tube_cell-1")
PIN = ComponentRef(type=ComponentType.CCOMPONENT, id="sfr_pin-1")


def _rect(**overrides) -> RectLattice:
    kwargs = {
        "family_name": "mini_lattice",
        "version_label": "1",
        "pitch_m": 0.0126,
        "fill": WATER,
        "shape": (3, 3),
        "elements": [
            LatticeElement(
                ref=FUEL,
                addresses=((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
            ),
            LatticeElement(ref=GUIDE, addresses=((1, 1),)),
        ],
    }
    kwargs.update(overrides)
    return RectLattice.create(**kwargs)


def _hex(**overrides) -> HexLattice:
    kwargs = {
        "family_name": "seven_pin_bundle",
        "version_label": "1",
        "pitch_m": 0.009,
        "fill": SODIUM,
        "num_rings": 2,
        "orientation": HexOrientation.FLAT_TOP,
        "elements": [
            LatticeElement(
                ref=PIN, addresses=((0, 0),) + tuple((1, i) for i in range(6))
            )
        ],
    }
    kwargs.update(overrides)
    return HexLattice.create(**kwargs)


# =============================================================================
# LatticeElement
# =============================================================================


def test_element_round_trip():
    element = LatticeElement(ref=FUEL, addresses=((0, 0), (2, 1)))
    assert LatticeElement.from_dict(element.to_dict()) == element


def test_element_requires_an_address():
    with pytest.raises(ValueError):
        LatticeElement(ref=FUEL, addresses=())


def test_element_rejects_repeated_address():
    with pytest.raises(ValueError):
        LatticeElement(ref=FUEL, addresses=((0, 0), (0, 0)))


def test_element_serializes_addresses_as_lists():
    """JSON object keys must be strings, so elements are a list of
    {ref, addresses} entries with addresses as [a, b] pairs."""
    data = LatticeElement(ref=FUEL, addresses=((0, 1),)).to_dict()
    assert data["addresses"] == [[0, 1]]


# =============================================================================
# Lattice base — ABC, fields, lineage
# =============================================================================


def test_lattice_base_cannot_instantiate():
    with pytest.raises(TypeError):
        Lattice(  # type: ignore[abstract]
            id=LatticeID("l-1"),
            family_name="l",
            version_label="1",
            pitch_m=0.01,
            fill=WATER,
            elements=[],
        )


@pytest.mark.parametrize("pitch_m", [0.0, -0.01])
def test_pitch_must_be_positive(pitch_m):
    with pytest.raises(ValueError):
        _rect(pitch_m=pitch_m)


def test_lineage_rule_applies():
    with pytest.raises(ValueError):
        _rect(simulation_job_id=SimulationJobID("job-1"))


# =============================================================================
# Lattice base — element rules
# =============================================================================


def test_requires_at_least_one_element():
    with pytest.raises(ValueError):
        _rect(elements=[])


def test_rejects_same_ref_in_two_elements():
    with pytest.raises(ValueError):
        _rect(
            elements=[
                LatticeElement(ref=FUEL, addresses=((0, 0),)),
                LatticeElement(ref=FUEL, addresses=((0, 1),)),
            ]
        )


def test_rejects_address_outside_lattice():
    with pytest.raises(ValueError):
        _rect(elements=[LatticeElement(ref=FUEL, addresses=((3, 0),))])


def test_rejects_address_occupied_twice():
    with pytest.raises(ValueError):
        _rect(
            elements=[
                LatticeElement(ref=FUEL, addresses=((0, 0),)),
                LatticeElement(ref=GUIDE, addresses=((0, 0),)),
            ]
        )


def test_rejects_direct_self_reference():
    self_ref = ComponentRef(type=ComponentType.LATTICE, id="mini_lattice-1")
    with pytest.raises(ValueError):
        _rect(elements=[LatticeElement(ref=self_ref, addresses=((0, 0),))])


def test_can_place_a_nested_lattice():
    inner = ComponentRef(type=ComponentType.LATTICE, id="inner_lattice-1")
    lattice = _rect(elements=[LatticeElement(ref=inner, addresses=((0, 0),))])
    assert lattice.occupant_at((0, 0)) == inner


# =============================================================================
# RectLattice — addressing, grid conversion, queries
# =============================================================================


def test_rect_addresses_are_row_major_top_row_first():
    lattice = _rect(
        shape=(2, 3),
        elements=[LatticeElement(ref=FUEL, addresses=((0, 0),))],
    )
    assert lattice.addresses() == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (-1, 2)])
def test_rect_shape_must_be_positive(shape):
    with pytest.raises(ValueError):
        _rect(
            shape=shape,
            elements=[LatticeElement(ref=FUEL, addresses=((0, 0),))],
        )


def test_occupant_at_and_empty_slots():
    lattice = _rect(
        elements=[
            LatticeElement(ref=FUEL, addresses=((0, 0), (0, 1))),
            LatticeElement(ref=GUIDE, addresses=((1, 1),)),
        ]
    )
    assert lattice.occupant_at((0, 1)) == FUEL
    assert lattice.occupant_at((1, 1)) == GUIDE
    assert lattice.occupant_at((2, 2)) is None
    assert lattice.empty_addresses() == [(0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


def test_occupant_at_rejects_address_outside_lattice():
    with pytest.raises(ValueError):
        _rect().occupant_at((5, 5))


def test_from_grid_groups_by_ref_in_first_appearance_order():
    lattice = RectLattice.from_grid(
        family_name="mini_lattice",
        version_label="1",
        pitch_m=0.0126,
        fill=WATER,
        grid=[
            [FUEL, FUEL, None],
            [FUEL, GUIDE, FUEL],
        ],
    )
    assert lattice.shape == (2, 3)
    assert [p.ref for p in lattice.elements] == [FUEL, GUIDE]
    assert lattice.elements[0].addresses == ((0, 0), (0, 1), (1, 0), (1, 2))
    assert lattice.empty_addresses() == [(0, 2)]


def test_to_grid_inverts_from_grid():
    grid = [
        [FUEL, FUEL, FUEL],
        [FUEL, GUIDE, None],
        [FUEL, FUEL, FUEL],
    ]
    lattice = RectLattice.from_grid(
        family_name="mini_lattice",
        version_label="1",
        pitch_m=0.0126,
        fill=WATER,
        grid=grid,
    )
    assert lattice.to_grid() == grid


@pytest.mark.parametrize("grid", [[], [[]], [[FUEL, FUEL], [FUEL]]])
def test_from_grid_rejects_malformed_grid(grid):
    with pytest.raises(ValueError):
        RectLattice.from_grid(
            family_name="bad",
            version_label="1",
            pitch_m=0.0126,
            fill=WATER,
            grid=grid,
        )


def test_rect_round_trip():
    lattice = _rect()
    restored = RectLattice.from_dict(lattice.to_dict())
    assert restored.to_dict() == lattice.to_dict()
    assert restored.shape == (3, 3)


# =============================================================================
# HexLattice
# =============================================================================


@pytest.mark.parametrize("ring,size", [(0, 1), (1, 6), (2, 12), (8, 48)])
def test_hex_ring_size(ring, size):
    assert HexLattice.ring_size(ring) == size


def test_hex_addresses_center_out_index_zero_first():
    lattice = _hex()
    assert lattice.addresses() == [(0, 0)] + [(1, i) for i in range(6)]


def test_hex_three_rings_has_nineteen_slots():
    lattice = _hex(
        num_rings=3,
        elements=[LatticeElement(ref=PIN, addresses=((0, 0),))],
    )
    assert len(lattice.addresses()) == 19
    assert len(lattice.empty_addresses()) == 18


def test_hex_rejects_index_past_ring_size():
    with pytest.raises(ValueError):
        _hex(elements=[LatticeElement(ref=PIN, addresses=((1, 6),))])


def test_hex_rejects_ring_past_num_rings():
    with pytest.raises(ValueError):
        _hex(elements=[LatticeElement(ref=PIN, addresses=((2, 0),))])


def test_hex_requires_at_least_one_ring():
    with pytest.raises(ValueError):
        _hex(num_rings=0)


def test_hex_rejects_non_orientation():
    with pytest.raises(TypeError):
        _hex(orientation="flat_top")


def test_hex_round_trip():
    lattice = _hex(orientation=HexOrientation.POINT_TOP)
    restored = HexLattice.from_dict(lattice.to_dict())
    assert restored.to_dict() == lattice.to_dict()
    assert restored.orientation == HexOrientation.POINT_TOP


# =============================================================================
# Dispatch and equality
# =============================================================================


@pytest.mark.parametrize("build,cls", [(_rect, RectLattice), (_hex, HexLattice)])
def test_lattice_from_dict_dispatches_on_type(build, cls):
    lattice = build()
    restored = lattice_from_dict(lattice.to_dict())
    assert isinstance(restored, cls)
    assert restored.to_dict() == lattice.to_dict()


def test_lattice_from_dict_rejects_unknown_type():
    data = _rect().to_dict()
    data["type"] = "triangular"
    with pytest.raises(ValueError):
        lattice_from_dict(data)


def test_equality_and_hash_are_id_based():
    a = _rect()
    b = _rect(fill=MaterialID("other_water-1"))
    assert a == b
    assert hash(a) == hash(b) == hash(a.id)


def test_rect_and_hex_with_the_same_id_are_not_equal():
    rect = _rect(family_name="shared")
    hexagonal = _hex(family_name="shared")
    assert rect.id == hexagonal.id
    assert rect != hexagonal


def test_hex_addresses_run_center_first_then_each_ring_in_order():
    lattice = _hex(
        num_rings=3,
        elements=[LatticeElement(ref=PIN, addresses=((0, 0),))],
    )
    assert lattice.addresses() == (
        [(0, 0)] + [(1, i) for i in range(6)] + [(2, i) for i in range(12)]
    )
