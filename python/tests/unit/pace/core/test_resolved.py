"""
tests/pace/core/test_resolved.py

Covers: ResolvedModel's internal-consistency validation — id-mapping,
root presence for every root kind, and completeness for every kind of
reference a bundle can hold (CComponent bounds/fill/members, Lattice
fill/placements, LComponent geometry/material, GAddition units, MMixture
constituents) — plus round-trip; ResolvedReactor's checks that the
Reactor's root, bounds, fill and operating-state materials are all in
the model, plus round-trip.

The fixture is a miniature VERA-style assembly: a pellet+gap+clad rod in
a pin cell, a 2x2 lattice of pin cells with one empty slot, and an
assembly composite holding the lattice.
"""

from typing import Any

import pytest
from pace.core.bounds import faces_for_geometry_type
from pace.core.component import CComponent, LComponent, Placement
from pace.core.component_ref import ComponentType, ComponentRef
from pace.core.geometry import (
    GAddition,
    GAnnulus,
    GCylinder,
    GeometryType,
    GPose,
    GRectanglePrism,
)
from pace.core.ids import GeometryID, MaterialID, PlacementID
from pace.core.lattice import LatticePlacement, RectLattice
from pace.core.material import MaterialComponentEntry, MIsotopic, MMixture
from pace.core.reactor import NeutronBC, OperatingState, Reactor
from pace.core.resolved import ResolvedModel, ResolvedReactor

ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)


def _material(name: str) -> MIsotopic:
    return MIsotopic.create(
        family_name=name,
        version_label="1",
        components={"H1": MaterialComponentEntry(percent=1.0)},
        percent_type="ao",
        density_value=1.0,
        density_unit="g/cm3",
    )


def _place(pc_id: str, kind: ComponentType, target_id: str) -> Placement:
    return Placement(
        id=PlacementID(pc_id), pose=ORIGIN, ref=ComponentRef(type=kind, id=target_id)
    )


def _bundle_parts() -> dict[str, Any]:
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
    pellet = LComponent.create(
        family_name="pellet", version_label="1", geometry=pellet_cyl.id, material=uo2.id
    )
    clad = LComponent.create(
        family_name="clad", version_label="1", geometry=clad_ring.id, material=zirc.id
    )
    rod = CComponent.create(
        family_name="rod",
        version_label="1",
        bounds=rod_cyl.id,
        fill=zirc.id,
        components=[
            _place("pellet", ComponentType.LCOMPONENT, pellet.id),
            _place("clad", ComponentType.LCOMPONENT, clad.id),
        ],
    )
    pin_cell = CComponent.create(
        family_name="pin_cell",
        version_label="1",
        bounds=pin_box.id,
        fill=water.id,
        components=[_place("rod", ComponentType.CCOMPONENT, rod.id)],
    )
    lattice = RectLattice.create(
        family_name="lattice_2x2",
        version_label="1",
        pitch_m=0.0126,
        fill=water.id,
        shape=(2, 2),
        placements=[
            LatticePlacement(
                ref=ComponentRef(type=ComponentType.CCOMPONENT, id=pin_cell.id),
                addresses=((0, 0), (0, 1), (1, 0)),
            )
        ],
    )
    assembly = CComponent.create(
        family_name="assembly",
        version_label="1",
        bounds=assembly_box.id,
        fill=water.id,
        components=[_place("lattice", ComponentType.LATTICE, lattice.id)],
    )
    return {
        "root": ComponentRef(type=ComponentType.CCOMPONENT, id=assembly.id),
        "ccomponents": {c.id: c for c in (rod, pin_cell, assembly)},
        "lcomponents": {c.id: c for c in (pellet, clad)},
        "lattices": {lattice.id: lattice},
        "geometries": {
            g.id: g for g in (pellet_cyl, clad_ring, rod_cyl, pin_box, assembly_box)
        },
        "materials": {m.id: m for m in (uo2, zirc, water)},
    }


def _without(parts: dict[str, Any], map_name: str, key: str) -> dict[str, Any]:
    trimmed = dict(parts)
    trimmed[map_name] = {k: v for k, v in parts[map_name].items() if k != key}
    return trimmed


# =============================================================================
# ResolvedModel
# =============================================================================


def test_valid_bundle_round_trips():
    bundle = ResolvedModel(**_bundle_parts())
    assert ResolvedModel.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()


@pytest.mark.parametrize(
    "kind,target_id",
    [
        (ComponentType.CCOMPONENT, "pin_cell-1"),
        (ComponentType.LATTICE, "lattice_2x2-1"),
        (ComponentType.LCOMPONENT, "pellet-1"),
    ],
)
def test_any_placeable_kind_can_be_root(kind, target_id):
    parts = _bundle_parts()
    parts["root"] = ComponentRef(type=kind, id=target_id)
    assert ResolvedModel(**parts).root.id == target_id


def test_root_missing_raises():
    parts = _bundle_parts()
    parts["root"] = ComponentRef(type=ComponentType.CCOMPONENT, id="nope-1")
    with pytest.raises(ValueError):
        ResolvedModel(**parts)


def test_mismatched_key_raises():
    parts = _bundle_parts()
    uo2 = parts["materials"][MaterialID("uo2-1")]
    parts["materials"] = {**parts["materials"], MaterialID("wrong-1"): uo2}
    with pytest.raises(ValueError):
        ResolvedModel(**parts)


@pytest.mark.parametrize(
    "map_name,key",
    [
        ("geometries", "pin_box-1"),  # a CComponent's bounds
        ("geometries", "pellet_cyl-1"),  # an LComponent's geometry
        ("materials", "uo2-1"),  # an LComponent's material
        ("ccomponents", "rod-1"),  # a CComponent member
        ("lcomponents", "clad-1"),  # an LComponent member
        ("ccomponents", "pin_cell-1"),  # a lattice placement target
        ("lattices", "lattice_2x2-1"),  # a lattice placed in a CComponent
    ],
)
def test_missing_dependency_raises(map_name, key):
    with pytest.raises(ValueError):
        ResolvedModel(**_without(_bundle_parts(), map_name, key))


def test_missing_fill_material_raises():
    """water is only referenced as fill (pin cell, lattice, assembly)."""
    with pytest.raises(ValueError):
        ResolvedModel(**_without(_bundle_parts(), "materials", "water-1"))


def test_gaddition_unit_must_be_present():
    parts = _bundle_parts()
    addition = GAddition.create(
        family_name="joined",
        version_label="1",
        units=[
            (GeometryID("pellet_cyl-1"), ORIGIN),
            (GeometryID("missing_cyl-1"), GPose(x_m=0.0, y_m=0.0, z_m=1.0)),
        ],
    )
    parts["geometries"] = {**parts["geometries"], addition.id: addition}
    with pytest.raises(ValueError):
        ResolvedModel(**parts)


def test_mmixture_constituent_must_be_present():
    parts = _bundle_parts()
    mixture = MMixture.create(
        family_name="blend",
        version_label="1",
        components=[(MaterialID("uo2-1"), 0.5), (MaterialID("missing-1"), 0.5)],
        percent_type="ao",
    )
    parts["materials"] = {**parts["materials"], mixture.id: mixture}
    with pytest.raises(ValueError):
        ResolvedModel(**parts)


# =============================================================================
# ResolvedReactor
# =============================================================================


def _reactor(**overrides) -> Reactor:
    kwargs: dict[str, Any] = {
        "family_name": "mini_assembly",
        "version_label": "1",
        "bounds": GeometryID("assembly_box-1"),
        "root": _place("assembly", ComponentType.CCOMPONENT, "assembly-1"),
        "neutron_bcs": dict.fromkeys(
            faces_for_geometry_type(GeometryType.RECT_PRISM), NeutronBC.REFLECTIVE
        ),
        "operating_state": OperatingState(
            initial_temperatures_k={
                MaterialID("uo2-1"): 565.0,
                MaterialID("zirc-1"): 565.0,
                MaterialID("water-1"): 565.0,
            }
        ),
    }
    kwargs.update(overrides)
    return Reactor.create(**kwargs)


def test_resolved_reactor_valid_round_trip():
    resolved = ResolvedReactor(
        reactor=_reactor(), model=ResolvedModel(**_bundle_parts())
    )
    restored = ResolvedReactor.from_dict(resolved.to_dict())
    assert restored.to_dict() == resolved.to_dict()


def test_resolved_reactor_root_must_match_model_root():
    reactor = _reactor(root=_place("pin", ComponentType.CCOMPONENT, "pin_cell-1"))
    with pytest.raises(ValueError):
        ResolvedReactor(reactor=reactor, model=ResolvedModel(**_bundle_parts()))


def test_resolved_reactor_bounds_must_be_present():
    reactor = _reactor(bounds=GeometryID("missing_box-1"))
    with pytest.raises(ValueError):
        ResolvedReactor(reactor=reactor, model=ResolvedModel(**_bundle_parts()))


def test_resolved_reactor_fill_must_be_present():
    reactor = _reactor(fill=MaterialID("missing_water-1"))
    with pytest.raises(ValueError):
        ResolvedReactor(reactor=reactor, model=ResolvedModel(**_bundle_parts()))


def test_resolved_reactor_operating_state_materials_must_be_present():
    reactor = _reactor(
        operating_state=OperatingState(
            initial_temperatures_k={MaterialID("missing-1"): 565.0}
        )
    )
    with pytest.raises(ValueError):
        ResolvedReactor(reactor=reactor, model=ResolvedModel(**_bundle_parts()))
