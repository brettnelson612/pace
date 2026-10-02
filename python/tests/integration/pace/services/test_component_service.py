"""
tests/integration/pace/services/test_component_service.py

Integration tests for ComponentService against a real (in-memory)
PaceDB — confirms the domain objects, DAOs, RegistryDB, and reference
index actually work together, not just correct in isolation. Covers
register_*()/get_*() for every kind (including the bounds-shape and
boundary-condition face checks), reference-index bookkeeping, and full
hydration via get_resolved_component() / get_resolved_reactor().
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pace.core.bounds import BoundsFace, faces_for_geometry_type
from pace.core.component import CComponent, LComponent, ComponentPlacement
from pace.core.component_ref import ComponentType, ComponentRef
from pace.core.geometry import GAnnulus, GCylinder, GeometryType, GPose, GRectanglePrism
from pace.core.ids import (
    CComponentID,
    GeometryID,
    LComponentID,
    MaterialID,
    ComponentPlacementID,
    ReactorBlueprintID,
)
from pace.core.lattice import LatticeElement, RectLattice
from pace.core.material import MaterialComponentEntry, MIsotopic
from pace.core.reactor_blueprint import NeutronBC, OperatingState, Reactor, ThermalBC
from pace.core.reference_types import ReferenceableType
from pace.db.pace_db import PaceDB
from pace.services.component_service import (
    ComponentService,
    DanglingReferenceError,
    FamilyAlreadyExistsError,
    InvalidBoundsError,
)

ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)
BOX_FACES = faces_for_geometry_type(GeometryType.RECT_PRISM)


@pytest.fixture
def pace_db() -> PaceDB:
    db = PaceDB(db_url="sqlite:///:memory:")
    db.create_all()
    return db


@pytest.fixture
def component_service(pace_db: PaceDB) -> ComponentService:
    return ComponentService(pace_db.registry)


def _make_cylinder(
    family_name: str = "test_cylinder", version_label: str = "1"
) -> GCylinder:
    return GCylinder.create(
        family_name=family_name,
        version_label=version_label,
        radius_m=0.005,
        height_m=0.01,
    )


def _make_box(family_name: str, side_m: float) -> GRectanglePrism:
    return GRectanglePrism.create(
        family_name=family_name,
        version_label="1",
        length_m=side_m,
        width_m=side_m,
        height_m=0.01,
    )


def _make_isotopic(
    family_name: str = "test_uo2", version_label: str = "1"
) -> MIsotopic:
    return MIsotopic.create(
        family_name=family_name,
        version_label=version_label,
        components={"U235": MaterialComponentEntry(percent=100.0)},
        percent_type="wo",
        density_value=10.3,
        density_unit="g/cm3",
    )


def _place(pc_id: str, kind: ComponentType, target_id: str) -> ComponentPlacement:
    return ComponentPlacement(
        id=ComponentPlacementID(pc_id),
        pose=ORIGIN,
        ref=ComponentRef(type=kind, id=target_id),
    )


@dataclass
class MiniAssembly:
    """Everything registered by _register_mini_assembly()."""

    pellet_cyl: GCylinder
    pin_box: GRectanglePrism
    assembly_box: GRectanglePrism
    uo2: MIsotopic
    water: MIsotopic
    pellet: LComponent
    pin_cell: CComponent
    lattice: RectLattice
    assembly: CComponent


def _register_mini_assembly(service: ComponentService) -> MiniAssembly:
    """pellet -> pin cell (box + water) -> 2x2 lattice (one empty slot,
    water fill) -> assembly (box + water margin)."""
    pellet_cyl = service.register_geometry(_make_cylinder("pellet_cyl"))
    pin_box = service.register_geometry(_make_box("pin_box", 0.0126))
    assembly_box = service.register_geometry(_make_box("assembly_box", 0.026))
    uo2 = service.register_material(_make_isotopic("uo2"))
    water = service.register_material(_make_isotopic("water"))
    pellet = service.register_lcomponent(
        LComponent.create(
            family_name="pellet",
            version_label="1",
            geometry=pellet_cyl.id,
            material=uo2.id,
        )
    )
    pin_cell = service.register_ccomponent(
        CComponent.create(
            family_name="pin_cell",
            version_label="1",
            bounds=pin_box.id,
            fill=water.id,
            components=[_place("pellet", ComponentType.LCOMPONENT, pellet.id)],
        )
    )
    lattice = service.register_lattice(
        RectLattice.create(
            family_name="lattice_2x2",
            version_label="1",
            pitch_m=0.0126,
            fill=water.id,
            shape=(2, 2),
            placements=[
                LatticeElement(
                    ref=ComponentRef(type=ComponentType.CCOMPONENT, id=pin_cell.id),
                    addresses=((0, 0), (0, 1), (1, 0)),
                )
            ],
        )
    )
    assembly = service.register_ccomponent(
        CComponent.create(
            family_name="assembly",
            version_label="1",
            bounds=assembly_box.id,
            fill=water.id,
            components=[_place("lattice", ComponentType.LATTICE, lattice.id)],
        )
    )
    return MiniAssembly(
        pellet_cyl=pellet_cyl,
        pin_box=pin_box,
        assembly_box=assembly_box,
        uo2=uo2,
        water=water,
        pellet=pellet,
        pin_cell=pin_cell,
        lattice=lattice,
        assembly=assembly,
    )


def _make_reactor(mini: MiniAssembly, **overrides) -> Reactor:
    kwargs = {
        "family_name": "mini_assembly_2d",
        "version_label": "1",
        "bounds": mini.assembly_box.id,
        "root": _place("assembly", ComponentType.CCOMPONENT, mini.assembly.id),
        "neutron_bcs": dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE),
        "operating_state": OperatingState(
            initial_temperatures_k={mini.uo2.id: 565.0, mini.water.id: 565.0}
        ),
    }
    kwargs.update(overrides)
    return Reactor.create(**kwargs)


# =============================================================================
# register_geometry / register_material — creation + family uniqueness
# =============================================================================


def test_register_geometry_round_trips(pace_db, component_service):
    geometry = _make_cylinder()
    component_service.register_geometry(geometry)
    assert pace_db.registry.geometries.get(geometry.id) == geometry


def test_register_geometry_rejects_duplicate_family(component_service):
    component_service.register_geometry(_make_cylinder())
    with pytest.raises(FamilyAlreadyExistsError):
        component_service.register_geometry(_make_cylinder())


def test_register_material_round_trips(pace_db, component_service):
    material = _make_isotopic()
    component_service.register_material(material)
    assert pace_db.registry.materials.get(material.id) == material


def test_register_material_rejects_duplicate_family(component_service):
    component_service.register_material(_make_isotopic())
    with pytest.raises(FamilyAlreadyExistsError):
        component_service.register_material(_make_isotopic())


# =============================================================================
# register_lcomponent — creation + dangling-reference validation
# =============================================================================


def test_register_lcomponent_round_trips(pace_db, component_service):
    geometry = component_service.register_geometry(_make_cylinder())
    material = component_service.register_material(_make_isotopic())
    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    component_service.register_lcomponent(lcomponent)
    assert pace_db.registry.lcomponents.get(lcomponent.id) == lcomponent


def test_register_lcomponent_rejects_dangling_geometry(pace_db, component_service):
    material = component_service.register_material(_make_isotopic())
    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=GeometryID("nonexistent-1"),
        material=material.id,
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_lcomponent(lcomponent)
    # Validation runs before save() — a rejected LComponent must not
    # have been persisted.
    assert pace_db.registry.lcomponents.get(lcomponent.id) is None


# =============================================================================
# register_ccomponent — references, bounds shape, dedupe
# =============================================================================


def test_register_ccomponent_records_bounds_fill_and_member_edges(
    pace_db, component_service
):
    mini = _register_mini_assembly(component_service)
    assert pace_db.registry.ccomponents.get(mini.pin_cell.id) == mini.pin_cell
    outgoing = set(
        pace_db.registry.references.outgoing(
            ReferenceableType.CCOMPONENT, mini.pin_cell.id
        )
    )
    assert outgoing == {
        (ReferenceableType.GEOMETRY, mini.pin_box.id),
        (ReferenceableType.MATERIAL, mini.water.id),
        (ReferenceableType.LCOMPONENT, mini.pellet.id),
    }


def test_register_ccomponent_dedupes_repeated_placements(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    doubled = CComponent.create(
        family_name="two_pellets",
        version_label="1",
        bounds=mini.pin_box.id,
        fill=mini.water.id,
        components=[
            _place("a", ComponentType.LCOMPONENT, mini.pellet.id),
            ComponentPlacement(
                id=ComponentPlacementID("b"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.005),
                ref=ComponentRef(type=ComponentType.LCOMPONENT, id=mini.pellet.id),
            ),
        ],
    )
    component_service.register_ccomponent(doubled)
    outgoing = pace_db.registry.references.outgoing(
        ReferenceableType.CCOMPONENT, doubled.id
    )
    assert outgoing.count((ReferenceableType.LCOMPONENT, mini.pellet.id)) == 1


def test_register_ccomponent_rejects_dangling_member(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    ccomponent = CComponent.create(
        family_name="bad_pin",
        version_label="1",
        bounds=mini.pin_box.id,
        components=[
            _place("x", ComponentType.LCOMPONENT, LComponentID("nonexistent-1"))
        ],
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_ccomponent(ccomponent)
    assert pace_db.registry.ccomponents.get(ccomponent.id) is None


def test_register_ccomponent_rejects_dangling_bounds(component_service):
    mini = _register_mini_assembly(component_service)
    ccomponent = CComponent.create(
        family_name="bad_pin",
        version_label="1",
        bounds=GeometryID("nonexistent-1"),
        components=[_place("p", ComponentType.LCOMPONENT, mini.pellet.id)],
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_ccomponent(ccomponent)


def test_register_ccomponent_rejects_non_bounds_shape(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    ring = component_service.register_geometry(
        GAnnulus.create(
            family_name="ring",
            version_label="1",
            inner_radius_m=0.001,
            outer_radius_m=0.002,
            height_m=0.01,
        )
    )
    ccomponent = CComponent.create(
        family_name="ring_bounded",
        version_label="1",
        bounds=ring.id,
        components=[_place("p", ComponentType.LCOMPONENT, mini.pellet.id)],
    )
    with pytest.raises(InvalidBoundsError):
        component_service.register_ccomponent(ccomponent)
    assert pace_db.registry.ccomponents.get(ccomponent.id) is None


def test_ccomponent_cannot_self_reference_at_construction():
    """Not a ComponentService call — the self-reference guard lives on
    CComponent's own construction, before ComponentService is involved."""
    self_id = CComponent.build_id("test_pin", "1")
    with pytest.raises(ValueError):
        CComponent(
            id=self_id,
            family_name="test_pin",
            version_label="1",
            bounds=GeometryID("pin_box-1"),
            components=[_place("self", ComponentType.CCOMPONENT, self_id)],
        )


# =============================================================================
# register_lattice
# =============================================================================


def test_register_lattice_round_trips_with_one_edge_per_ref(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    assert pace_db.registry.lattices.get(mini.lattice.id) == mini.lattice
    outgoing = set(
        pace_db.registry.references.outgoing(ReferenceableType.LATTICE, mini.lattice.id)
    )
    # three slots of the same pin cell collapse to one edge
    assert outgoing == {
        (ReferenceableType.MATERIAL, mini.water.id),
        (ReferenceableType.CCOMPONENT, mini.pin_cell.id),
    }


def test_register_lattice_rejects_dangling_placement(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    lattice = RectLattice.create(
        family_name="bad_lattice",
        version_label="1",
        pitch_m=0.0126,
        fill=mini.water.id,
        shape=(1, 1),
        placements=[
            LatticeElement(
                ref=ComponentRef(
                    type=ComponentType.CCOMPONENT, id=CComponentID("nope-1")
                ),
                addresses=((0, 0),),
            )
        ],
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_lattice(lattice)
    assert pace_db.registry.lattices.get(lattice.id) is None


def test_register_lattice_rejects_duplicate_family(component_service):
    mini = _register_mini_assembly(component_service)
    with pytest.raises(FamilyAlreadyExistsError):
        component_service.register_lattice(mini.lattice)


# =============================================================================
# register_reactor — references and face checks
# =============================================================================


def test_register_reactor_round_trips(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    reactor = component_service.register_reactor(_make_reactor(mini))
    assert pace_db.registry.reactors.get(reactor.id).to_dict() == reactor.to_dict()
    outgoing = set(
        pace_db.registry.references.outgoing(ReferenceableType.REACTOR, reactor.id)
    )
    assert outgoing == {
        (ReferenceableType.GEOMETRY, mini.assembly_box.id),
        (ReferenceableType.CCOMPONENT, mini.assembly.id),
        (ReferenceableType.MATERIAL, mini.uo2.id),
        (ReferenceableType.MATERIAL, mini.water.id),
    }


def test_register_reactor_rejects_missing_neutron_face(pace_db, component_service):
    mini = _register_mini_assembly(component_service)
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    del bcs[BoundsFace.Z_MAX]
    reactor = _make_reactor(mini, neutron_bcs=bcs)
    with pytest.raises(InvalidBoundsError):
        component_service.register_reactor(reactor)
    assert pace_db.registry.reactors.get(reactor.id) is None


def test_register_reactor_rejects_faces_of_the_wrong_shape(component_service):
    mini = _register_mini_assembly(component_service)
    cylinder_faces = faces_for_geometry_type(GeometryType.CYLINDER)
    reactor = _make_reactor(
        mini, neutron_bcs=dict.fromkeys(cylinder_faces, NeutronBC.VACUUM)
    )
    with pytest.raises(InvalidBoundsError):
        component_service.register_reactor(reactor)


def test_register_reactor_rejects_thermal_condition_on_missing_face(
    component_service,
):
    mini = _register_mini_assembly(component_service)
    reactor = _make_reactor(mini, thermal_bcs={BoundsFace.RADIAL: ThermalBC.ADIABATIC})
    with pytest.raises(InvalidBoundsError):
        component_service.register_reactor(reactor)


def test_register_reactor_rejects_dangling_root(component_service):
    mini = _register_mini_assembly(component_service)
    reactor = _make_reactor(
        mini, root=_place("x", ComponentType.CCOMPONENT, CComponentID("nope-1"))
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_reactor(reactor)


def test_register_reactor_rejects_unknown_operating_state_material(
    component_service,
):
    mini = _register_mini_assembly(component_service)
    reactor = _make_reactor(
        mini,
        operating_state=OperatingState(
            initial_temperatures_k={MaterialID("nope-1"): 565.0}
        ),
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_reactor(reactor)


# =============================================================================
# get_* — single row, unhydrated
# =============================================================================


def test_get_geometry_returns_none_when_absent(component_service):
    assert component_service.get_geometry(GeometryID("nonexistent-1")) is None


def test_get_reactor_returns_none_when_absent(component_service):
    assert component_service.get_reactor(ReactorBlueprintID("nonexistent-1")) is None


def test_get_ccomponent_returns_unhydrated_row(component_service):
    """get_ccomponent() returns the bare row — refs stay as ids."""
    mini = _register_mini_assembly(component_service)
    fetched = component_service.get_ccomponent(mini.pin_cell.id)
    assert fetched == mini.pin_cell
    assert fetched.components[0].ref.id == mini.pellet.id


def test_get_lattice_returns_persisted_row(component_service):
    mini = _register_mini_assembly(component_service)
    fetched = component_service.get_lattice(mini.lattice.id)
    assert fetched.to_dict() == mini.lattice.to_dict()


# =============================================================================
# Hydration
# =============================================================================


def test_get_resolved_component_hydrates_full_tree_once_per_object(component_service):
    """assembly -> lattice -> pin cell (3 slots) -> pellet: every
    distinct object appears exactly once, however often it's placed."""
    mini = _register_mini_assembly(component_service)
    root = ComponentRef(type=ComponentType.CCOMPONENT, id=mini.assembly.id)
    resolved = component_service.get_resolved_component(root)

    assert resolved.root == root
    assert resolved.ccomponents.keys() == {mini.assembly.id, mini.pin_cell.id}
    assert resolved.lattices.keys() == {mini.lattice.id}
    assert resolved.lcomponents.keys() == {mini.pellet.id}
    assert resolved.geometries.keys() == {
        mini.pellet_cyl.id,
        mini.pin_box.id,
        mini.assembly_box.id,
    }
    assert resolved.materials.keys() == {mini.uo2.id, mini.water.id}


def test_get_resolved_component_with_lattice_root(component_service):
    mini = _register_mini_assembly(component_service)
    root = ComponentRef(type=ComponentType.LATTICE, id=mini.lattice.id)
    resolved = component_service.get_resolved_component(root)
    assert resolved.ccomponents.keys() == {mini.pin_cell.id}
    assert mini.assembly_box.id not in resolved.geometries


def test_get_resolved_component_raises_for_nonexistent_root(component_service):
    with pytest.raises(ValueError):
        component_service.get_resolved_component(
            ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("nope-1"))
        )


def test_get_resolved_reactor_includes_reactor_only_references(component_service):
    """A Reactor bounds geometry that nothing in the root's tree uses
    must still be hydrated."""
    mini = _register_mini_assembly(component_service)
    reactor_box = component_service.register_geometry(_make_box("reactor_box", 0.03))
    reactor = component_service.register_reactor(
        _make_reactor(mini, bounds=reactor_box.id, fill=mini.water.id)
    )
    resolved = component_service.get_resolved_reactor(reactor.id)

    assert resolved.reactor.to_dict() == reactor.to_dict()
    assert reactor_box.id in resolved.model.geometries
    assert resolved.model.root == reactor.root.ref
    assert resolved.model.lattices.keys() == {mini.lattice.id}


def test_get_resolved_reactor_raises_for_nonexistent_reactor(component_service):
    with pytest.raises(DanglingReferenceError):
        component_service.get_resolved_reactor(ReactorBlueprintID("nope-1"))
