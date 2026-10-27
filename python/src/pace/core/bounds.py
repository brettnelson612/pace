"""
pace/core/bounds.py

What may serve as bounds, and the named faces of each bounds shape.

Bounds — the region of space an Enclosure owns — are limited to the
three shapes every translator can turn into a clipping region and the
MOOSE Reactor module can mesh: a rectangular prism, a hexagonal prism,
or a cylinder. A ReactorBlueprint's problem boundary is its root
Enclosure's bounds, and a lattice occupant's bounds must be exactly its
slot (a square or hexagonal prism of the lattice's pitch).

BoundsFace names each face of those shapes. Boundary conditions on a
ReactorBlueprint are keyed by face (see reactor_blueprint.py).
"""

from __future__ import annotations

from enum import StrEnum

from pace.core.geometry import GeometryType


class BoundsFace(StrEnum):
    """A named face of a bounds shape.

    Which faces exist depends on the bounds geometry type — see
    faces_for_geometry_type():
        - rect prism: x_min, x_max, y_min, y_max, z_min, z_max
        - hex prism: side_0 ... side_5, plus z_min, z_max. side_0 is
              the top face (centered on +y, since a GHexPrism is
              flat-top at zero rotation); the rest follow clockwise.
        - cylinder: radial, z_min, z_max
    """

    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"
    Z_MAX = "z_max"
    SIDE_0 = "side_0"
    SIDE_1 = "side_1"
    SIDE_2 = "side_2"
    SIDE_3 = "side_3"
    SIDE_4 = "side_4"
    SIDE_5 = "side_5"
    RADIAL = "radial"


_AXIAL_FACES = (BoundsFace.Z_MIN, BoundsFace.Z_MAX)
_HEX_SIDES = (
    BoundsFace.SIDE_0,
    BoundsFace.SIDE_1,
    BoundsFace.SIDE_2,
    BoundsFace.SIDE_3,
    BoundsFace.SIDE_4,
    BoundsFace.SIDE_5,
)
_FACES_BY_GEOMETRY_TYPE: dict[GeometryType, tuple[BoundsFace, ...]] = {
    GeometryType.RECT_PRISM: (
        BoundsFace.X_MIN,
        BoundsFace.X_MAX,
        BoundsFace.Y_MIN,
        BoundsFace.Y_MAX,
        *_AXIAL_FACES,
    ),
    GeometryType.HEX_PRISM: (*_HEX_SIDES, *_AXIAL_FACES),
    GeometryType.CYLINDER: (BoundsFace.RADIAL, *_AXIAL_FACES),
}

# The geometry types allowed as an Enclosure's bounds (including a
# ReactorBlueprint's root and every Lattice occupant).
BOUNDS_GEOMETRY_TYPES = frozenset(_FACES_BY_GEOMETRY_TYPE)

# The face each face is paired with under a PERIODIC condition.
PERIODIC_PARTNER: dict[BoundsFace, BoundsFace] = {
    BoundsFace.X_MIN: BoundsFace.X_MAX,
    BoundsFace.X_MAX: BoundsFace.X_MIN,
    BoundsFace.Y_MIN: BoundsFace.Y_MAX,
    BoundsFace.Y_MAX: BoundsFace.Y_MIN,
    BoundsFace.Z_MIN: BoundsFace.Z_MAX,
    BoundsFace.Z_MAX: BoundsFace.Z_MIN,
    BoundsFace.SIDE_0: BoundsFace.SIDE_3,
    BoundsFace.SIDE_3: BoundsFace.SIDE_0,
    BoundsFace.SIDE_1: BoundsFace.SIDE_4,
    BoundsFace.SIDE_4: BoundsFace.SIDE_1,
    BoundsFace.SIDE_2: BoundsFace.SIDE_5,
    BoundsFace.SIDE_5: BoundsFace.SIDE_2,
}


def faces_for_geometry_type(geometry_type: GeometryType) -> tuple[BoundsFace, ...]:
    """The faces a bounds shape of this type has. Raises for geometry
    types that can't be bounds."""
    try:
        return _FACES_BY_GEOMETRY_TYPE[geometry_type]
    except KeyError:
        raise ValueError(
            f"{geometry_type.value!r} geometry cannot be used as bounds; "
            f"allowed: {sorted(t.value for t in BOUNDS_GEOMETRY_TYPES)}"
        ) from None
