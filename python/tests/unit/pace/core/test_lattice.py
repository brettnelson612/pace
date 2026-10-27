"""
tests/unit/pace/core/test_lattice.py

Covers: LatticeOccupant validation and round-trip; the Lattice base
(ABC enforcement, pitch constraint, lineage via Versioned, occupant
rules: one occupant per Enclosure, addresses in range, no address
occupied twice, every address occupied); RectLattice addressing
(row 0 on top), from_grid/to_grid, occupant_at, slot centers and slot
matching; HexLattice ring sizes, (ring, index) addressing, slot centers
for both orientations (OpenMC's within-ring order) and slot matching;
lattice_from_dict dispatch; id-based equality.
"""

import math

import pytest

from pace.core.geometry import GCylinder, GHexPrism, GRectanglePrism
from pace.core.ids import EnclosureID, LatticeID, SimulationJobID
from pace.core.lattice import (
    HexLattice,
    HexOrientation,
    Lattice,
    LatticeOccupant,
    RectLattice,
    lattice_from_dict,
)

FUEL = EnclosureID("fuel_pin_cell-1")
GUIDE = EnclosureID("guide_tube_cell-1")
PIN = EnclosureID("sfr_pin-1")

RECT_PITCH_M = 0.0126
HEX_PITCH_M = 0.009


def _rect(**overrides) -> RectLattice:
    kwargs = {
        "family_name": "mini_lattice",
        "version_label": "1",
        "pitch_m": RECT_PITCH_M,
        "shape": (3, 3),
        "occupants": [
            LatticeOccupant(
                enclosure=FUEL,
                addresses=(
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 2),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                ),
            ),
            LatticeOccupant(enclosure=GUIDE, addresses=((1, 1),)),
        ],
    }
    kwargs.update(overrides)
    return RectLattice.create(**kwargs)


def _filled_hex(num_rings: int, orientation: HexOrientation) -> HexLattice:
    addresses = tuple(
        (ring, index)
        for ring in range(num_rings)
        for index in range(HexLattice.ring_size(ring))
    )
    return HexLattice.create(
        family_name="hex_bundle",
        version_label="1",
        pitch_m=HEX_PITCH_M,
        num_rings=num_rings,
        orientation=orientation,
        occupants=[LatticeOccupant(enclosure=PIN, addresses=addresses)],
    )


def _hex(**overrides) -> HexLattice:
    kwargs = {
        "family_name": "seven_pin_bundle",
        "version_label": "1",
        "pitch_m": HEX_PITCH_M,
        "num_rings": 2,
        "orientation": HexOrientation.FLAT_TOP,
        "occupants": [
            LatticeOccupant(
                enclosure=PIN, addresses=((0, 0), *((1, i) for i in range(6)))
            )
        ],
    }
    kwargs.update(overrides)
    return HexLattice.create(**kwargs)


def _assert_point_close(actual: tuple[float, float], expected: tuple[float, float]):
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert actual[1] == pytest.approx(expected[1], abs=1e-12)


# =============================================================================
# LatticeOccupant
# =============================================================================


def test_occupant_round_trip():
    occupant = LatticeOccupant(enclosure=FUEL, addresses=((0, 0), (2, 1)))
    assert LatticeOccupant.from_dict(occupant.to_dict()) == occupant


def test_occupant_requires_an_address():
    with pytest.raises(ValueError, match="lists no addresses"):
        LatticeOccupant(enclosure=FUEL, addresses=())


def test_occupant_rejects_repeated_address():
    with pytest.raises(ValueError, match="repeats an address"):
        LatticeOccupant(enclosure=FUEL, addresses=((0, 0), (0, 0)))


def test_occupant_requires_an_enclosure_id():
    with pytest.raises(ValueError, match="non-empty id"):
        LatticeOccupant(enclosure=EnclosureID(""), addresses=((0, 0),))


def test_occupant_serializes_addresses_as_lists():
    """JSON object keys must be strings, so occupants are a list of
    {enclosure, addresses} entries with addresses as [a, b] pairs."""
    data = LatticeOccupant(enclosure=FUEL, addresses=((0, 1),)).to_dict()
    assert data == {"enclosure": FUEL, "addresses": [[0, 1]]}


# =============================================================================
# Lattice base — ABC, fields, lineage
# =============================================================================


def test_lattice_base_cannot_instantiate():
    with pytest.raises(TypeError, match="abstract"):
        Lattice(  # type: ignore[abstract]
            id=LatticeID("l-1"),
            family_name="l",
            version_label="1",
            pitch_m=0.01,
            occupants=[],
        )


@pytest.mark.parametrize("pitch_m", [0.0, -0.01])
def test_pitch_must_be_positive(pitch_m):
    with pytest.raises(ValueError, match="pitch_m"):
        _rect(pitch_m=pitch_m)


def test_lineage_rule_applies():
    with pytest.raises(ValueError, match="simulation_job_id must be unset"):
        _rect(simulation_job_id=SimulationJobID("job-1"))


# =============================================================================
# Lattice base — occupant rules
# =============================================================================


def test_rejects_same_enclosure_in_two_occupants():
    with pytest.raises(ValueError, match="more than one occupant"):
        _rect(
            occupants=[
                LatticeOccupant(enclosure=FUEL, addresses=((0, 0),)),
                LatticeOccupant(enclosure=FUEL, addresses=((0, 1),)),
            ]
        )


def test_rejects_address_outside_lattice():
    with pytest.raises(ValueError, match="outside lattice"):
        _rect(occupants=[LatticeOccupant(enclosure=FUEL, addresses=((3, 0),))])


def test_rejects_address_occupied_twice():
    with pytest.raises(ValueError, match="occupied twice"):
        _rect(
            occupants=[
                LatticeOccupant(enclosure=FUEL, addresses=((0, 0),)),
                LatticeOccupant(enclosure=GUIDE, addresses=((0, 0),)),
            ]
        )


def test_rejects_an_empty_slot():
    with pytest.raises(ValueError, match=r"must be occupied; empty: \[\(2, 2\)\]"):
        _rect(
            occupants=[
                LatticeOccupant(
                    enclosure=FUEL,
                    addresses=(
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (1, 0),
                        (1, 2),
                        (2, 0),
                        (2, 1),
                    ),
                ),
                LatticeOccupant(enclosure=GUIDE, addresses=((1, 1),)),
            ]
        )


def test_rejects_no_occupants():
    with pytest.raises(ValueError, match="must be occupied"):
        _rect(occupants=[])


# =============================================================================
# RectLattice — addressing, grid conversion, queries
# =============================================================================


def test_rect_addresses_are_row_major_top_row_first():
    lattice = _rect(
        shape=(2, 3),
        occupants=[
            LatticeOccupant(
                enclosure=FUEL,
                addresses=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
            )
        ],
    )
    assert lattice.addresses() == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (-1, 2)])
def test_rect_shape_must_be_positive(shape):
    with pytest.raises(ValueError, match="shape must be positive"):
        _rect(
            shape=shape,
            occupants=[LatticeOccupant(enclosure=FUEL, addresses=((0, 0),))],
        )


def test_occupant_at():
    lattice = _rect()
    assert lattice.occupant_at((0, 1)) == FUEL
    assert lattice.occupant_at((1, 1)) == GUIDE


def test_occupant_at_rejects_address_outside_lattice():
    with pytest.raises(ValueError, match="outside lattice"):
        _rect().occupant_at((5, 5))


def test_from_grid_groups_by_enclosure_in_first_appearance_order():
    lattice = RectLattice.from_grid(
        family_name="mini_lattice",
        version_label="1",
        pitch_m=RECT_PITCH_M,
        grid=[
            [FUEL, FUEL, GUIDE],
            [FUEL, GUIDE, FUEL],
        ],
    )
    assert lattice.shape == (2, 3)
    assert [occupant.enclosure for occupant in lattice.occupants] == [FUEL, GUIDE]
    assert lattice.occupants[0].addresses == ((0, 0), (0, 1), (1, 0), (1, 2))
    assert lattice.occupants[1].addresses == ((0, 2), (1, 1))


def test_to_grid_inverts_from_grid():
    grid = [
        [FUEL, FUEL, FUEL],
        [FUEL, GUIDE, FUEL],
        [FUEL, FUEL, FUEL],
    ]
    lattice = RectLattice.from_grid(
        family_name="mini_lattice",
        version_label="1",
        pitch_m=RECT_PITCH_M,
        grid=grid,
    )
    assert lattice.to_grid() == grid


@pytest.mark.parametrize("grid", [[], [[]], [[FUEL, FUEL], [FUEL]]])
def test_from_grid_rejects_malformed_grid(grid):
    with pytest.raises(ValueError, match="grid"):
        RectLattice.from_grid(
            family_name="bad",
            version_label="1",
            pitch_m=RECT_PITCH_M,
            grid=grid,
        )


def test_rect_round_trip():
    lattice = _rect()
    restored = RectLattice.from_dict(lattice.to_dict())
    assert restored.to_dict() == lattice.to_dict()
    assert restored.shape == (3, 3)


# =============================================================================
# RectLattice — slot centers and slot matching
# =============================================================================


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ((0, 0), (-RECT_PITCH_M, RECT_PITCH_M)),  # top left
        ((0, 2), (RECT_PITCH_M, RECT_PITCH_M)),  # top right
        ((1, 1), (0.0, 0.0)),  # center
        ((2, 0), (-RECT_PITCH_M, -RECT_PITCH_M)),  # bottom left
    ],
)
def test_rect_slot_center_puts_row_zero_on_top(address, expected):
    _assert_point_close(_rect().slot_center(address), expected)


def test_rect_slot_center_of_even_grid_straddles_the_origin():
    lattice = _rect(
        shape=(2, 2),
        occupants=[
            LatticeOccupant(enclosure=FUEL, addresses=((0, 0), (0, 1), (1, 0), (1, 1)))
        ],
    )
    half = RECT_PITCH_M / 2
    _assert_point_close(lattice.slot_center((0, 0)), (-half, half))
    _assert_point_close(lattice.slot_center((1, 1)), (half, -half))


def test_rect_slot_center_rejects_address_outside_lattice():
    with pytest.raises(ValueError, match="outside lattice"):
        _rect().slot_center((3, 0))


def test_rect_slot_is_a_square_prism_of_the_pitch():
    lattice = _rect()
    square = GRectanglePrism.create(
        family_name="slot",
        version_label="1",
        length_m=RECT_PITCH_M,
        width_m=RECT_PITCH_M,
        height_m=1.0,
    )
    too_wide = GRectanglePrism.create(
        family_name="wide",
        version_label="1",
        length_m=RECT_PITCH_M * 2,
        width_m=RECT_PITCH_M,
        height_m=1.0,
    )
    cylinder = GCylinder.create(
        family_name="cyl", version_label="1", radius_m=RECT_PITCH_M / 2, height_m=1.0
    )
    assert lattice.matches_slot(square)
    assert not lattice.matches_slot(too_wide)
    assert not lattice.matches_slot(cylinder)


# =============================================================================
# HexLattice — addressing and validation
# =============================================================================


@pytest.mark.parametrize(("ring", "size"), [(0, 1), (1, 6), (2, 12), (8, 48)])
def test_hex_ring_size(ring, size):
    assert HexLattice.ring_size(ring) == size


def test_hex_addresses_run_center_first_then_each_ring_in_order():
    lattice = _filled_hex(3, HexOrientation.FLAT_TOP)
    assert lattice.addresses() == (
        [(0, 0)] + [(1, i) for i in range(6)] + [(2, i) for i in range(12)]
    )


def test_hex_three_rings_has_nineteen_slots():
    assert len(_filled_hex(3, HexOrientation.FLAT_TOP).addresses()) == 19


def test_hex_rejects_index_past_ring_size():
    with pytest.raises(ValueError, match="outside lattice"):
        _hex(
            occupants=[
                LatticeOccupant(
                    enclosure=PIN,
                    addresses=((0, 0), *((1, i) for i in range(7))),
                )
            ]
        )


def test_hex_rejects_ring_past_num_rings():
    with pytest.raises(ValueError, match="outside lattice"):
        _hex(
            occupants=[
                LatticeOccupant(
                    enclosure=PIN,
                    addresses=((0, 0), (2, 0), *((1, i) for i in range(6))),
                )
            ]
        )


def test_hex_requires_at_least_one_ring():
    with pytest.raises(ValueError, match="num_rings must be at least 1"):
        _hex(num_rings=0)


def test_hex_rejects_non_orientation():
    with pytest.raises(TypeError, match="orientation must be a HexOrientation"):
        _hex(orientation="flat_top")


def test_hex_round_trip():
    lattice = _hex(orientation=HexOrientation.POINT_TOP)
    restored = HexLattice.from_dict(lattice.to_dict())
    assert restored.to_dict() == lattice.to_dict()
    assert restored.orientation == HexOrientation.POINT_TOP


# =============================================================================
# HexLattice — slot centers and slot matching
# =============================================================================

P = HEX_PITCH_M
COS_30 = math.cos(math.pi / 6)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ((0, 0), (0.0, 0.0)),
        ((1, 0), (0.0, P)),  # top
        ((1, 1), (P * COS_30, P / 2)),  # upper right
        ((1, 2), (P * COS_30, -P / 2)),  # lower right
        ((1, 3), (0.0, -P)),  # bottom
        ((1, 4), (-P * COS_30, -P / 2)),  # lower left
        ((1, 5), (-P * COS_30, P / 2)),  # upper left
        ((2, 0), (0.0, 2 * P)),  # first corner of ring 2
        ((2, 1), (P * COS_30, 1.5 * P)),  # between corners 0 and 1
        ((2, 2), (2 * P * COS_30, P)),  # second corner of ring 2
    ],
)
def test_flat_top_slots_start_at_the_top_and_run_clockwise(address, expected):
    lattice = _filled_hex(3, HexOrientation.FLAT_TOP)
    _assert_point_close(lattice.slot_center(address), expected)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ((1, 0), (P, 0.0)),  # right
        ((1, 1), (P / 2, -P * COS_30)),  # lower right
        ((1, 2), (-P / 2, -P * COS_30)),  # lower left
        ((1, 3), (-P, 0.0)),  # left
        ((1, 4), (-P / 2, P * COS_30)),  # upper left
        ((1, 5), (P / 2, P * COS_30)),  # upper right
        ((2, 1), (1.5 * P, -P * COS_30)),  # between corners 0 and 1
    ],
)
def test_point_top_slots_start_at_the_right_and_run_clockwise(address, expected):
    """Matches OpenMC's orientation 'x', whose rings start on the +x
    axis."""
    lattice = _filled_hex(3, HexOrientation.POINT_TOP)
    _assert_point_close(lattice.slot_center(address), expected)


@pytest.mark.parametrize("orientation", list(HexOrientation))
def test_hex_slots_are_distinct_and_one_pitch_from_their_nearest_neighbor(
    orientation,
):
    lattice = _filled_hex(4, orientation)
    centers = [lattice.slot_center(address) for address in lattice.addresses()]
    for i, (x, y) in enumerate(centers):
        nearest = min(
            math.hypot(x - other_x, y - other_y)
            for j, (other_x, other_y) in enumerate(centers)
            if j != i
        )
        assert nearest == pytest.approx(P)


def test_hex_slot_center_rejects_address_outside_lattice():
    with pytest.raises(ValueError, match="outside lattice"):
        _hex().slot_center((2, 0))


def test_hex_slot_is_a_hex_prism_with_the_pitch_across_flats():
    lattice = _hex()
    slot = GHexPrism.create(
        family_name="slot",
        version_label="1",
        circumradius_m=P / math.sqrt(3),
        height_m=1.0,
    )
    across_corners = GHexPrism.create(
        family_name="corners", version_label="1", circumradius_m=P / 2, height_m=1.0
    )
    square = GRectanglePrism.create(
        family_name="square", version_label="1", length_m=P, width_m=P, height_m=1.0
    )
    assert lattice.matches_slot(slot)
    assert not lattice.matches_slot(across_corners)
    assert not lattice.matches_slot(square)


# =============================================================================
# Dispatch and equality
# =============================================================================


@pytest.mark.parametrize(("build", "cls"), [(_rect, RectLattice), (_hex, HexLattice)])
def test_lattice_from_dict_dispatches_on_type(build, cls):
    lattice = build()
    restored = lattice_from_dict(lattice.to_dict())
    assert isinstance(restored, cls)
    assert restored.to_dict() == lattice.to_dict()


def test_lattice_from_dict_rejects_unknown_type():
    data = _rect().to_dict()
    data["type"] = "triangular"
    with pytest.raises(ValueError, match="unknown lattice type"):
        lattice_from_dict(data)


def test_equality_and_hash_are_id_based():
    a = _rect()
    b = _rect(pitch_m=0.0130)
    assert a == b
    assert hash(a) == hash(b) == hash(a.id)


def test_rect_and_hex_with_the_same_id_are_not_equal():
    rect = _rect(family_name="shared")
    hexagonal = _hex(family_name="shared")
    assert rect.id == hexagonal.id
    assert rect != hexagonal
