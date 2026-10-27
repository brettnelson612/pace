"""
tests/unit/pace/core/test_resolved.py

Covers: ResolvedPlaceable's internal-consistency validation — id-mapping,
root presence for every root type, and completeness for every kind of
reference a bundle can hold (Enclosure bounds/fill/members, Lattice
occupants, Part geometry/material, GAddition units, GSubtraction
base/cuts, MMixture constituents) — plus round-trip;
ResolvedReactorBlueprint's check that the blueprint's root Enclosure is
the model root, plus round-trip.

The fixture is a miniature VERA-style assembly: a pellet+clad rod in a
pin cell, a 2x2 lattice of three pin cells and one coolant-only cell,
and an assembly Enclosure holding the lattice.
"""

from typing import Any

import pytest

from pace.core.boundary_conditions import NeutronBC
from pace.core.bounds import faces_for_geometry_type
from pace.core.enclosure import Enclosure
from pace.core.geometry import (
    GAddition,
    GAnnulus,
    GCylinder,
    GeometryType,
    GPose,
    GRectanglePrism,
    GSubtraction,
)
from pace.core.ids import EnclosureID, GeometryID, MaterialID, PlacementID
from pace.core.lattice import LatticeOccupant, RectLattice
from pace.core.material import (
    DensityUnit,
    MaterialComponentEntry,
    MIsotopic,
    MMixture,
    PercentType,
)
from pace.core.part import Part
from pace.core.placeable import PlaceableRef, PlaceableType
from pace.core.placement import Placement
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.core.resolved import ResolvedPlaceable, ResolvedReactorBlueprint

ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)


def _material(name: str) -> MIsotopic:
    return MIsotopic.create(
        family_name=name,
        version_label="1",
        components={"H1": MaterialComponentEntry(percent=1.0)},
        percent_type=PercentType.AO,
        density_value=1.0,
        density_unit=DensityUnit.G_PER_CM3,
    )


def _place(
    placement_id: str, placeable_type: PlaceableType, target_id: str
) -> Placement:
    return Placement(
        id=PlacementID(placement_id),
        pose=ORIGIN,
        ref=PlaceableRef(type=placeable_type, id=target_id),
    )


def _bundle_fields() -> dict[str, Any]:
    uo2, zirc, water = _material("uo2"), _material("zirc"), _material("water")
    pellet_cyl = GCylinder.create(
        family_name="pellet_cyl", version_label="1", radius_m=0.004, height_m=1.0
    )
    clad_ring = GAnnulus.create(
        family_name="clad_ring",
        version_label="1",
        inner_radius_m=0.0042,
        outer_radius_m=0.00475,
        height_m=1.0,
    )
    rod_cyl = GCylinder.create(
        family_name="rod_cyl", version_label="1", radius_m=0.00475, height_m=1.0
    )
    pin_box = GRectanglePrism.create(
        family_name="pin_box",
        version_label="1",
        length_m=0.0126,
        width_m=0.0126,
        height_m=1.0,
    )
    assembly_box = GRectanglePrism.create(
        family_name="assembly_box",
        version_label="1",
        length_m=0.026,
        width_m=0.026,
        height_m=1.0,
    )
    pellet = Part.create(
        family_name="pellet", version_label="1", geometry=pellet_cyl.id, material=uo2.id
    )
    clad = Part.create(
        family_name="clad", version_label="1", geometry=clad_ring.id, material=zirc.id
    )
    rod = Enclosure.create(
        family_name="rod",
        version_label="1",
        bounds=rod_cyl.id,
        fill=zirc.id,
        placements=[
            _place("pellet", PlaceableType.PART, pellet.id),
            _place("clad", PlaceableType.PART, clad.id),
        ],
    )
    pin_cell = Enclosure.create(
        family_name="pin_cell",
        version_label="1",
        bounds=pin_box.id,
        fill=water.id,
        placements=[_place("rod", PlaceableType.ENCLOSURE, rod.id)],
    )
    water_cell = Enclosure.create(
        family_name="water_cell",
        version_label="1",
        bounds=pin_box.id,
        fill=water.id,
        placements=[],
    )
    lattice = RectLattice.create(
        family_name="lattice_2x2",
        version_label="1",
        pitch_m=0.0126,
        shape=(2, 2),
        occupants=[
            LatticeOccupant(enclosure=pin_cell.id, addresses=((0, 0), (0, 1), (1, 0))),
            LatticeOccupant(enclosure=water_cell.id, addresses=((1, 1),)),
        ],
    )
    assembly = Enclosure.create(
        family_name="assembly",
        version_label="1",
        bounds=assembly_box.id,
        fill=water.id,
        placements=[_place("lattice", PlaceableType.LATTICE, lattice.id)],
    )
    return {
        "root": PlaceableRef(type=PlaceableType.ENCLOSURE, id=assembly.id),
        "enclosures": {c.id: c for c in (rod, pin_cell, water_cell, assembly)},
        "parts": {c.id: c for c in (pellet, clad)},
        "lattices": {lattice.id: lattice},
        "geometries": {
            g.id: g for g in (pellet_cyl, clad_ring, rod_cyl, pin_box, assembly_box)
        },
        "materials": {m.id: m for m in (uo2, zirc, water)},
    }


def _without(fields: dict[str, Any], map_name: str, key: str) -> dict[str, Any]:
    trimmed = dict(fields)
    trimmed[map_name] = {k: v for k, v in fields[map_name].items() if k != key}
    return trimmed


# =============================================================================
# ResolvedPlaceable
# =============================================================================


def test_valid_bundle_round_trips():
    bundle = ResolvedPlaceable(**_bundle_fields())
    assert ResolvedPlaceable.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()


@pytest.mark.parametrize(
    ("placeable_type", "target_id"),
    [
        (PlaceableType.ENCLOSURE, "pin_cell-1"),
        (PlaceableType.LATTICE, "lattice_2x2-1"),
        (PlaceableType.PART, "pellet-1"),
    ],
)
def test_any_placeable_type_can_be_root(placeable_type, target_id):
    fields = _bundle_fields()
    fields["root"] = PlaceableRef(type=placeable_type, id=target_id)
    assert ResolvedPlaceable(**fields).root.id == target_id


def test_root_missing_raises():
    fields = _bundle_fields()
    fields["root"] = PlaceableRef(type=PlaceableType.ENCLOSURE, id="nope-1")
    with pytest.raises(ValueError):
        ResolvedPlaceable(**fields)


def test_mismatched_key_raises():
    fields = _bundle_fields()
    uo2 = fields["materials"][MaterialID("uo2-1")]
    fields["materials"] = {**fields["materials"], MaterialID("wrong-1"): uo2}
    with pytest.raises(ValueError):
        ResolvedPlaceable(**fields)


@pytest.mark.parametrize(
    ("map_name", "key"),
    [
        ("geometries", "pin_box-1"),  # an Enclosure's bounds
        ("geometries", "pellet_cyl-1"),  # a Part's geometry
        ("materials", "uo2-1"),  # a Part's material
        ("enclosures", "rod-1"),  # an Enclosure member
        ("parts", "clad-1"),  # a Part member
        ("enclosures", "pin_cell-1"),  # a lattice occupant
        ("enclosures", "water_cell-1"),  # a coolant-only lattice occupant
        ("lattices", "lattice_2x2-1"),  # a lattice placed in an Enclosure
    ],
)
def test_missing_dependency_raises(map_name, key):
    with pytest.raises(ValueError):
        ResolvedPlaceable(**_without(_bundle_fields(), map_name, key))


def test_missing_fill_material_raises():
    """water is only referenced as fill (pin cell, water cell, assembly)."""
    with pytest.raises(ValueError):
        ResolvedPlaceable(**_without(_bundle_fields(), "materials", "water-1"))


def test_gaddition_unit_must_be_present():
    fields = _bundle_fields()
    addition = GAddition.create(
        family_name="joined",
        version_label="1",
        units=[
            (GeometryID("pellet_cyl-1"), ORIGIN),
            (GeometryID("missing_cyl-1"), GPose(x_m=0.0, y_m=0.0, z_m=1.0)),
        ],
    )
    fields["geometries"] = {**fields["geometries"], addition.id: addition}
    with pytest.raises(ValueError):
        ResolvedPlaceable(**fields)


def test_mmixture_constituent_must_be_present():
    fields = _bundle_fields()
    mixture = MMixture.create(
        family_name="blend",
        version_label="1",
        components=[(MaterialID("uo2-1"), 0.5), (MaterialID("missing-1"), 0.5)],
        percent_type=PercentType.AO,
    )
    fields["materials"] = {**fields["materials"], mixture.id: mixture}
    with pytest.raises(ValueError):
        ResolvedPlaceable(**fields)


def test_gsubtraction_base_and_cuts_must_be_present():
    fields = _bundle_fields()
    subtraction = GSubtraction.create(
        family_name="notched",
        version_label="1",
        base=(GeometryID("rod_cyl-1"), ORIGIN),
        cuts=[(GeometryID("missing_cut-1"), ORIGIN)],
    )
    fields["geometries"] = {**fields["geometries"], subtraction.id: subtraction}
    with pytest.raises(ValueError):
        ResolvedPlaceable(**fields)


@pytest.mark.parametrize(
    ("placeable_type", "target_id", "expected"),
    [
        (PlaceableType.ENCLOSURE, "rod-1", True),
        (PlaceableType.PART, "clad-1", True),
        (PlaceableType.LATTICE, "lattice_2x2-1", True),
        (PlaceableType.PART, "rod-1", False),
        (PlaceableType.ENCLOSURE, "missing-1", False),
    ],
)
def test_contains_placeable_checks_the_map_for_the_ref_type(
    placeable_type, target_id, expected
):
    bundle = ResolvedPlaceable(**_bundle_fields())
    ref = PlaceableRef(type=placeable_type, id=target_id)
    assert bundle.contains_placeable(ref) is expected


# =============================================================================
# ResolvedReactorBlueprint
# =============================================================================


def _blueprint(**overrides) -> ReactorBlueprint:
    kwargs: dict[str, Any] = {
        "family_name": "mini_assembly",
        "version_label": "1",
        "root": EnclosureID("assembly-1"),
        "neutron_bcs": dict.fromkeys(
            faces_for_geometry_type(GeometryType.RECT_PRISM), NeutronBC.REFLECTIVE
        ),
    }
    kwargs.update(overrides)
    return ReactorBlueprint.create(**kwargs)


def _resolve(blueprint: ReactorBlueprint) -> ResolvedReactorBlueprint:
    return ResolvedReactorBlueprint(
        reactor_blueprint=blueprint,
        resolved_placeable=ResolvedPlaceable(**_bundle_fields()),
    )


def test_resolved_reactor_blueprint_valid_round_trip():
    resolved = _resolve(_blueprint())
    restored = ResolvedReactorBlueprint.from_dict(resolved.to_dict())
    assert restored.to_dict() == resolved.to_dict()


def test_resolved_reactor_blueprint_root_must_match_model_root():
    blueprint = _blueprint(root=EnclosureID("pin_cell-1"))
    with pytest.raises(ValueError, match="does not match the model root"):
        _resolve(blueprint)
