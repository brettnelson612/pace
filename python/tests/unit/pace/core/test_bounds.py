"""
tests/pace/core/test_bounds.py

Covers: which geometry types may serve as bounds, the faces each bounds
shape has, and the periodic face pairing.
"""

import pytest
from pace.core.bounds import (
    BOUNDS_GEOMETRY_TYPES,
    PERIODIC_PARTNER,
    BoundsFace,
    faces_for_geometry_type,
)
from pace.core.geometry import GeometryType


def test_bounds_shapes_are_box_hex_and_cylinder():
    assert BOUNDS_GEOMETRY_TYPES == {
        GeometryType.RECT_PRISM,
        GeometryType.HEX_PRISM,
        GeometryType.CYLINDER,
    }


@pytest.mark.parametrize(
    "geometry_type,count",
    [
        (GeometryType.RECT_PRISM, 6),
        (GeometryType.HEX_PRISM, 8),
        (GeometryType.CYLINDER, 3),
    ],
)
def test_face_counts(geometry_type, count):
    faces = faces_for_geometry_type(geometry_type)
    assert len(faces) == count
    assert len(set(faces)) == count


@pytest.mark.parametrize("geometry_type", list(BOUNDS_GEOMETRY_TYPES))
def test_every_bounds_shape_has_top_and_bottom(geometry_type):
    faces = faces_for_geometry_type(geometry_type)
    assert BoundsFace.Z_MIN in faces
    assert BoundsFace.Z_MAX in faces


@pytest.mark.parametrize(
    "geometry_type",
    [
        GeometryType.SPHERE,
        GeometryType.ANNULUS,
        GeometryType.ADDITION,
        GeometryType.SUBTRACTION,
    ],
)
def test_non_bounds_shapes_are_rejected(geometry_type):
    with pytest.raises(ValueError):
        faces_for_geometry_type(geometry_type)


def test_periodic_pairing_is_symmetric():
    for face, partner in PERIODIC_PARTNER.items():
        assert PERIODIC_PARTNER[partner] == face
        assert partner != face


def test_radial_face_has_no_periodic_partner():
    assert BoundsFace.RADIAL not in PERIODIC_PARTNER
