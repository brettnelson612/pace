"""
tests/integration/pace/services/test_registry_service.py

Integration tests for RegistryService against a real (in-memory)
PaceDB — confirms the domain objects, DAOs, RegistryDB and reference
index work together. Covers register_*()/get_*() for every type
(including the bounds-shape and boundary-condition face checks),
version immutability, lineage checks, atomic registration,
reference-index bookkeeping, batched dangling-reference reporting, and
full hydration via get_resolved_placeable() /
get_resolved_reactor_blueprint().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from sqlalchemy.exc import IntegrityError

from pace.core.boundary_conditions import FlowBC, NeutronBC, ThermalBC
from pace.core.bounds import BoundsFace, faces_for_geometry_type
from pace.core.enclosure import Enclosure
from pace.core.geometry import (
    GAnnulus,
    GCylinder,
    GeometryType,
    GHexPrism,
    GPose,
    GRectanglePrism,
)
from pace.core.ids import (
    EnclosureID,
    GeometryID,
    MaterialID,
    PartID,
    PlacementID,
    ReactorBlueprintID,
)
from pace.core.lattice import HexLattice, HexOrientation, LatticeOccupant, RectLattice
from pace.core.material import (
    DensityUnit,
    MaterialComponentEntry,
    MIsotopic,
    MVoid,
    PercentType,
)
from pace.core.part import Part
from pace.core.placeable import PlaceableRef, PlaceableType
from pace.core.placement import Placement
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.core.reference_types import ReferenceableType
from pace.db.pace_db import PaceDB
from pace.services.registry_service import (
    DanglingReferenceError,
    FamilyAlreadyExistsError,
    InvalidBoundsError,
    InvalidPredecessorError,
    RegistryService,
    VersionAlreadyExistsError,
)

ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)
BOX_FACES = faces_for_geometry_type(GeometryType.RECT_PRISM)


@pytest.fixture
def registry_service(pace_db: PaceDB) -> RegistryService:
    return RegistryService(pace_db.registry)


def _make_cylinder(
    family_name: str = "test_cylinder", version_label: str = "1", **overrides
) -> GCylinder:
    kwargs = {"radius_m": 0.005, "height_m": 0.01}
    kwargs.update(overrides)
    return GCylinder.create(
        family_name=family_name, version_label=version_label, **kwargs
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
        percent_type=PercentType.WO,
        density_value=10.3,
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


@dataclass
class MiniAssembly:
    """Everything registered by _register_mini_assembly()."""

    pellet_cyl: GCylinder
    pin_box: GRectanglePrism
    assembly_box: GRectanglePrism
    uo2: MIsotopic
    water: MIsotopic
    pellet: Part
    pin_cell: Enclosure
    water_cell: Enclosure
    lattice: RectLattice
    assembly: Enclosure


def _register_mini_assembly(service: RegistryService) -> MiniAssembly:
    """pellet -> pin cell (box + water) -> 2x2 lattice (three pin cells
    and one coolant-only water cell) -> assembly (box + water margin)."""
    pellet_cyl = service.register_geometry(_make_cylinder("pellet_cyl"))
    pin_box = service.register_geometry(_make_box("pin_box", 0.0126))
    assembly_box = service.register_geometry(_make_box("assembly_box", 0.026))
    uo2 = service.register_material(_make_isotopic("uo2"))
    water = service.register_material(_make_isotopic("water"))
    pellet = service.register_part(
        Part.create(
            family_name="pellet",
            version_label="1",
            geometry=pellet_cyl.id,
            material=uo2.id,
        )
    )
    pin_cell = service.register_enclosure(
        Enclosure.create(
            family_name="pin_cell",
            version_label="1",
            bounds=pin_box.id,
            fill=water.id,
            placements=[_place("pellet", PlaceableType.PART, pellet.id)],
        )
    )
    water_cell = service.register_enclosure(
        Enclosure.create(
            family_name="water_cell",
            version_label="1",
            bounds=pin_box.id,
            fill=water.id,
            placements=[],
        )
    )
    lattice = service.register_lattice(
        RectLattice.create(
            family_name="lattice_2x2",
            version_label="1",
            pitch_m=0.0126,
            shape=(2, 2),
            occupants=[
                LatticeOccupant(
                    enclosure=pin_cell.id, addresses=((0, 0), (0, 1), (1, 0))
                ),
                LatticeOccupant(enclosure=water_cell.id, addresses=((1, 1),)),
            ],
        )
    )
    assembly = service.register_enclosure(
        Enclosure.create(
            family_name="assembly",
            version_label="1",
            bounds=assembly_box.id,
            fill=water.id,
            placements=[_place("lattice", PlaceableType.LATTICE, lattice.id)],
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
        water_cell=water_cell,
        lattice=lattice,
        assembly=assembly,
    )


def _make_blueprint(mini: MiniAssembly, **overrides) -> ReactorBlueprint:
    kwargs = {
        "family_name": "mini_assembly_2d",
        "version_label": "1",
        "root": mini.assembly.id,
        "neutron_bcs": dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE),
    }
    kwargs.update(overrides)
    return ReactorBlueprint.create(**kwargs)


# =============================================================================
# register_geometry / register_material — creation + family uniqueness
# =============================================================================


def test_register_geometry_round_trips(pace_db, registry_service):
    geometry = _make_cylinder()
    registry_service.register_geometry(geometry)
    fetched = pace_db.registry.geometries.get(geometry.id)
    assert fetched.to_dict() == geometry.to_dict()


def test_register_geometry_rejects_duplicate_family(registry_service):
    registry_service.register_geometry(_make_cylinder(version_label="1"))
    with pytest.raises(FamilyAlreadyExistsError):
        registry_service.register_geometry(_make_cylinder(version_label="other"))


def test_register_material_round_trips(pace_db, registry_service):
    material = _make_isotopic()
    registry_service.register_material(material)
    fetched = pace_db.registry.materials.get(material.id)
    assert fetched.to_dict() == material.to_dict()


def test_register_material_rejects_duplicate_family(registry_service):
    registry_service.register_material(_make_isotopic(version_label="1"))
    with pytest.raises(FamilyAlreadyExistsError):
        registry_service.register_material(_make_isotopic(version_label="other"))


# =============================================================================
# Version immutability
# =============================================================================


def test_reregistering_a_v1_is_rejected(registry_service):
    registry_service.register_geometry(_make_cylinder())
    with pytest.raises(VersionAlreadyExistsError):
        registry_service.register_geometry(_make_cylinder())


def test_registered_v2_is_never_overwritten(pace_db, registry_service):
    v1 = registry_service.register_geometry(_make_cylinder())
    v2 = _make_cylinder(
        version_label="2", derived_from=v1.id, user_edit=True, radius_m=0.006
    )
    registry_service.register_geometry(v2)
    imposter = _make_cylinder(
        version_label="2", derived_from=v1.id, user_edit=True, radius_m=0.009
    )
    with pytest.raises(VersionAlreadyExistsError):
        registry_service.register_geometry(imposter)
    assert pace_db.registry.geometries.get(v2.id).radius_m == 0.006


def test_concurrent_duplicate_is_reported_as_already_registered(
    pace_db, registry_service, monkeypatch
):
    """If another writer registers the same id between the existence
    check and the insert, the primary key rejects the insert and the
    service reports it the same way."""
    registry_service.register_geometry(_make_cylinder())
    monkeypatch.setattr(
        RegistryService, "_reject_existing_id", staticmethod(lambda dao, model: None)
    )
    monkeypatch.setattr(
        RegistryService, "_validate_lineage", staticmethod(lambda dao, model: None)
    )
    with pytest.raises(VersionAlreadyExistsError):
        registry_service.register_geometry(_make_cylinder(radius_m=0.009))
    assert pace_db.registry.geometries.get("test_cylinder-1").radius_m == 0.005


# =============================================================================
# Lineage
# =============================================================================


def test_register_v2_records_lineage(pace_db, registry_service):
    v1 = registry_service.register_geometry(_make_cylinder())
    v2 = registry_service.register_geometry(
        _make_cylinder(version_label="2", derived_from=v1.id, user_edit=True)
    )
    children = pace_db.registry.geometries.children_of(v1.id)
    assert [child.id for child in children] == [v2.id]


def test_register_v2_with_missing_predecessor_is_rejected(pace_db, registry_service):
    orphan = _make_cylinder(
        version_label="2", derived_from=GeometryID("missing-1"), user_edit=True
    )
    with pytest.raises(InvalidPredecessorError):
        registry_service.register_geometry(orphan)
    assert not pace_db.registry.geometries.exists(orphan.id)


def test_register_v2_from_another_family_is_rejected(registry_service):
    other = registry_service.register_geometry(_make_cylinder(family_name="other"))
    with pytest.raises(InvalidPredecessorError):
        registry_service.register_geometry(
            _make_cylinder(version_label="2", derived_from=other.id, user_edit=True)
        )


def test_register_v2_from_another_type_is_rejected(registry_service):
    """The predecessor must be in the same table: a Material can't
    derive from a Geometry, even one whose family_name matches."""
    cylinder = registry_service.register_geometry(_make_cylinder(family_name="shared"))
    with pytest.raises(InvalidPredecessorError):
        registry_service.register_material(
            MVoid.create(
                family_name="shared",
                version_label="2",
                derived_from=MaterialID(cylinder.id),
                user_edit=True,
            )
        )


# =============================================================================
# Atomic registration
# =============================================================================


def test_failed_reference_write_leaves_nothing_behind(
    pace_db, registry_service, monkeypatch
):
    def fail(session, references):
        raise RuntimeError("reference index unavailable")

    monkeypatch.setattr(pace_db.registry.references, "add_many", fail)
    geometry = _make_cylinder()
    with pytest.raises(RuntimeError):
        registry_service.register_geometry(geometry)
    assert not pace_db.registry.geometries.exists(geometry.id)


def test_rejected_registration_writes_no_edges(pace_db, registry_service):
    mini = _register_mini_assembly(registry_service)
    bad = Enclosure.create(
        family_name="bad_pin",
        version_label="1",
        bounds=mini.pin_box.id,
        placements=[_place("x", PlaceableType.PART, PartID("nope-1"))],
    )
    with pytest.raises(DanglingReferenceError):
        registry_service.register_enclosure(bad)
    assert (
        pace_db.registry.references.outgoing(ReferenceableType.ENCLOSURE, bad.id) == []
    )


def test_database_integrity_errors_other_than_a_duplicate_id_propagate(
    pace_db, registry_service, monkeypatch
):
    """With the service's own reference check bypassed, SQLite's foreign
    key still rejects a Part pointing at a missing Geometry —
    and that error is not misreported as a duplicate version."""
    material = registry_service.register_material(_make_isotopic())
    monkeypatch.setattr(registry_service, "validate_part", lambda model: None)
    part = Part.create(
        family_name="test_pellet",
        version_label="1",
        geometry=GeometryID("missing_cyl-1"),
        material=material.id,
    )
    with pytest.raises(IntegrityError):
        registry_service.register_part(part)
    assert not pace_db.registry.parts.exists(part.id)


# =============================================================================
# register_part — creation + dangling-reference validation
# =============================================================================


def test_register_part_round_trips(pace_db, registry_service):
    geometry = registry_service.register_geometry(_make_cylinder())
    material = registry_service.register_material(_make_isotopic())
    part = Part.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    registry_service.register_part(part)
    fetched = pace_db.registry.parts.get(part.id)
    assert fetched.to_dict() == part.to_dict()


def test_register_part_rejects_dangling_geometry(pace_db, registry_service):
    material = registry_service.register_material(_make_isotopic())
    part = Part.create(
        family_name="test_pellet",
        version_label="1",
        geometry=GeometryID("nonexistent-1"),
        material=material.id,
    )
    with pytest.raises(DanglingReferenceError):
        registry_service.register_part(part)
    assert pace_db.registry.parts.get(part.id) is None


def test_dangling_reference_error_names_every_missing_target(registry_service):
    part = Part.create(
        family_name="test_pellet",
        version_label="1",
        geometry=GeometryID("missing_cyl-1"),
        material=MaterialID("missing_uo2-1"),
    )
    with pytest.raises(DanglingReferenceError) as error:
        registry_service.register_part(part)
    assert "missing_cyl-1" in str(error.value)
    assert "missing_uo2-1" in str(error.value)


def test_validate_references_queries_once_per_target_type(
    pace_db, registry_service, monkeypatch
):
    mini = _register_mini_assembly(registry_service)
    calls: list[str] = []
    for attribute in ("geometries", "materials", "parts", "enclosures"):
        dao = getattr(pace_db.registry, attribute)
        original = dao.existing_ids

        def counting(ids, original=original, attribute=attribute):
            calls.append(attribute)
            return original(ids)

        monkeypatch.setattr(dao, "existing_ids", counting)

    two_pellets = Enclosure.create(
        family_name="two_pellets",
        version_label="1",
        bounds=mini.pin_box.id,
        fill=mini.water.id,
        placements=[
            _place("a", PlaceableType.PART, mini.pellet.id),
            _place("b", PlaceableType.ENCLOSURE, mini.pin_cell.id),
        ],
    )
    registry_service.validate_enclosure(two_pellets)
    assert sorted(calls) == ["enclosures", "geometries", "materials", "parts"]


# =============================================================================
# register_enclosure — references, bounds shape, dedupe
# =============================================================================


def test_register_enclosure_records_bounds_fill_and_member_edges(
    pace_db, registry_service
):
    mini = _register_mini_assembly(registry_service)
    outgoing = set(
        pace_db.registry.references.outgoing(
            ReferenceableType.ENCLOSURE, mini.pin_cell.id
        )
    )
    assert outgoing == {
        (ReferenceableType.GEOMETRY, mini.pin_box.id),
        (ReferenceableType.MATERIAL, mini.water.id),
        (ReferenceableType.PART, mini.pellet.id),
    }


def test_register_enclosure_dedupes_repeated_placements(pace_db, registry_service):
    mini = _register_mini_assembly(registry_service)
    doubled = Enclosure.create(
        family_name="two_pellets",
        version_label="1",
        bounds=mini.pin_box.id,
        fill=mini.water.id,
        placements=[
            _place("a", PlaceableType.PART, mini.pellet.id),
            Placement(
                id=PlacementID("b"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.005),
                ref=PlaceableRef(type=PlaceableType.PART, id=mini.pellet.id),
            ),
        ],
    )
    registry_service.register_enclosure(doubled)
    outgoing = pace_db.registry.references.outgoing(
        ReferenceableType.ENCLOSURE, doubled.id
    )
    assert outgoing.count((ReferenceableType.PART, mini.pellet.id)) == 1


def test_register_enclosure_rejects_dangling_member(pace_db, registry_service):
    mini = _register_mini_assembly(registry_service)
    enclosure = Enclosure.create(
        family_name="bad_pin",
        version_label="1",
        bounds=mini.pin_box.id,
        placements=[_place("x", PlaceableType.PART, PartID("nonexistent-1"))],
    )
    with pytest.raises(DanglingReferenceError):
        registry_service.register_enclosure(enclosure)
    assert pace_db.registry.enclosures.get(enclosure.id) is None


def test_register_enclosure_rejects_dangling_bounds(registry_service):
    mini = _register_mini_assembly(registry_service)
    enclosure = Enclosure.create(
        family_name="bad_pin",
        version_label="1",
        bounds=GeometryID("nonexistent-1"),
        placements=[_place("p", PlaceableType.PART, mini.pellet.id)],
    )
    with pytest.raises(DanglingReferenceError):
        registry_service.register_enclosure(enclosure)


def test_register_enclosure_rejects_non_bounds_shape(pace_db, registry_service):
    mini = _register_mini_assembly(registry_service)
    ring = registry_service.register_geometry(
        GAnnulus.create(
            family_name="ring",
            version_label="1",
            inner_radius_m=0.001,
            outer_radius_m=0.002,
            height_m=0.01,
        )
    )
    enclosure = Enclosure.create(
        family_name="ring_bounded",
        version_label="1",
        bounds=ring.id,
        placements=[_place("p", PlaceableType.PART, mini.pellet.id)],
    )
    with pytest.raises(InvalidBoundsError):
        registry_service.register_enclosure(enclosure)
    assert pace_db.registry.enclosures.get(enclosure.id) is None


def test_validate_enclosure_does_not_persist(pace_db, registry_service):
    mini = _register_mini_assembly(registry_service)
    draft = Enclosure.create(
        family_name="draft_pin",
        version_label="1",
        bounds=mini.pin_box.id,
        placements=[_place("p", PlaceableType.PART, mini.pellet.id)],
    )
    registry_service.validate_enclosure(draft)
    assert not pace_db.registry.enclosures.exists(draft.id)


# =============================================================================
# register_lattice
# =============================================================================


def test_register_lattice_round_trips_with_one_edge_per_occupant(
    pace_db, registry_service
):
    mini = _register_mini_assembly(registry_service)
    fetched = pace_db.registry.lattices.get(mini.lattice.id)
    assert fetched.to_dict() == mini.lattice.to_dict()
    outgoing = set(
        pace_db.registry.references.outgoing(ReferenceableType.LATTICE, mini.lattice.id)
    )
    # three slots of the same pin cell collapse to one edge
    assert outgoing == {
        (ReferenceableType.ENCLOSURE, mini.pin_cell.id),
        (ReferenceableType.ENCLOSURE, mini.water_cell.id),
    }


def test_register_lattice_rejects_dangling_occupant(pace_db, registry_service):
    _register_mini_assembly(registry_service)
    lattice = RectLattice.create(
        family_name="bad_lattice",
        version_label="1",
        pitch_m=0.0126,
        shape=(1, 1),
        occupants=[
            LatticeOccupant(enclosure=EnclosureID("nope-1"), addresses=((0, 0),))
        ],
    )
    with pytest.raises(DanglingReferenceError, match="nope-1"):
        registry_service.register_lattice(lattice)
    assert pace_db.registry.lattices.get(lattice.id) is None


def _single_slot_lattice(occupant: EnclosureID, pitch_m: float = 0.0126) -> RectLattice:
    return RectLattice.create(
        family_name="single_slot",
        version_label="1",
        pitch_m=pitch_m,
        shape=(1, 1),
        occupants=[LatticeOccupant(enclosure=occupant, addresses=((0, 0),))],
    )


def test_register_lattice_rejects_occupant_bounds_of_the_wrong_size(
    pace_db, registry_service
):
    mini = _register_mini_assembly(registry_service)
    lattice = _single_slot_lattice(mini.assembly.id)
    with pytest.raises(InvalidBoundsError, match="not the lattice's slot shape"):
        registry_service.register_lattice(lattice)
    assert pace_db.registry.lattices.get(lattice.id) is None


def test_register_lattice_rejects_occupant_bounds_of_the_wrong_shape(
    registry_service,
):
    mini = _register_mini_assembly(registry_service)
    round_cell = registry_service.register_enclosure(
        Enclosure.create(
            family_name="round_cell",
            version_label="1",
            bounds=registry_service.register_geometry(
                _make_cylinder("round_bounds", radius_m=0.0063)
            ).id,
            fill=mini.water.id,
            placements=[],
        )
    )
    with pytest.raises(InvalidBoundsError, match="cylinder"):
        registry_service.register_lattice(_single_slot_lattice(round_cell.id))


def test_register_lattice_rejects_occupants_of_different_heights(registry_service):
    mini = _register_mini_assembly(registry_service)
    tall_box = registry_service.register_geometry(
        GRectanglePrism.create(
            family_name="tall_pin_box",
            version_label="1",
            length_m=0.0126,
            width_m=0.0126,
            height_m=0.02,
        )
    )
    tall_cell = registry_service.register_enclosure(
        Enclosure.create(
            family_name="tall_water_cell",
            version_label="1",
            bounds=tall_box.id,
            fill=mini.water.id,
            placements=[],
        )
    )
    lattice = RectLattice.create(
        family_name="mixed_heights",
        version_label="1",
        pitch_m=0.0126,
        shape=(1, 2),
        occupants=[
            LatticeOccupant(enclosure=mini.water_cell.id, addresses=((0, 0),)),
            LatticeOccupant(enclosure=tall_cell.id, addresses=((0, 1),)),
        ],
    )
    with pytest.raises(InvalidBoundsError, match="must share one height"):
        registry_service.register_lattice(lattice)


def test_register_hex_lattice_accepts_hex_cells_of_the_pitch(registry_service):
    mini = _register_mini_assembly(registry_service)
    pitch_m = 0.009
    hex_cell = registry_service.register_enclosure(
        Enclosure.create(
            family_name="hex_water_cell",
            version_label="1",
            bounds=registry_service.register_geometry(
                GHexPrism.create(
                    family_name="hex_slot",
                    version_label="1",
                    circumradius_m=pitch_m / math.sqrt(3),
                    height_m=0.01,
                )
            ).id,
            fill=mini.water.id,
            placements=[],
        )
    )
    lattice = registry_service.register_lattice(
        HexLattice.create(
            family_name="hex_bundle",
            version_label="1",
            pitch_m=pitch_m,
            num_rings=2,
            orientation=HexOrientation.FLAT_TOP,
            occupants=[
                LatticeOccupant(
                    enclosure=hex_cell.id,
                    addresses=((0, 0), *((1, i) for i in range(6))),
                )
            ],
        )
    )
    assert registry_service.get_lattice(lattice.id) is not None


def test_reregistering_a_lattice_is_rejected(registry_service):
    mini = _register_mini_assembly(registry_service)
    with pytest.raises(VersionAlreadyExistsError):
        registry_service.register_lattice(mini.lattice)


# =============================================================================
# register_reactor_blueprint — references and face checks
# =============================================================================


def test_register_reactor_blueprint_round_trips(pace_db, registry_service):
    mini = _register_mini_assembly(registry_service)
    blueprint = registry_service.register_reactor_blueprint(_make_blueprint(mini))
    fetched = pace_db.registry.reactor_blueprints.get(blueprint.id)
    assert fetched.to_dict() == blueprint.to_dict()
    outgoing = set(
        pace_db.registry.references.outgoing(
            ReferenceableType.REACTOR_BLUEPRINT, blueprint.id
        )
    )
    assert outgoing == {(ReferenceableType.ENCLOSURE, mini.assembly.id)}


def test_register_reactor_blueprint_accepts_flow_on_its_faces(registry_service):
    mini = _register_mini_assembly(registry_service)
    blueprint = registry_service.register_reactor_blueprint(
        _make_blueprint(
            mini,
            flow_bcs={BoundsFace.Z_MIN: FlowBC.INLET, BoundsFace.Z_MAX: FlowBC.OUTLET},
        )
    )
    assert blueprint.flow_bcs[BoundsFace.Z_MIN] == FlowBC.INLET


def test_register_reactor_blueprint_rejects_missing_neutron_face(
    pace_db, registry_service
):
    mini = _register_mini_assembly(registry_service)
    bcs = dict.fromkeys(BOX_FACES, NeutronBC.REFLECTIVE)
    del bcs[BoundsFace.Z_MAX]
    blueprint = _make_blueprint(mini, neutron_bcs=bcs)
    with pytest.raises(InvalidBoundsError):
        registry_service.register_reactor_blueprint(blueprint)
    assert pace_db.registry.reactor_blueprints.get(blueprint.id) is None


def test_register_reactor_blueprint_rejects_faces_of_the_wrong_shape(
    registry_service,
):
    mini = _register_mini_assembly(registry_service)
    cylinder_faces = faces_for_geometry_type(GeometryType.CYLINDER)
    blueprint = _make_blueprint(
        mini, neutron_bcs=dict.fromkeys(cylinder_faces, NeutronBC.VACUUM)
    )
    with pytest.raises(InvalidBoundsError):
        registry_service.register_reactor_blueprint(blueprint)


@pytest.mark.parametrize(
    "overrides",
    [
        {"thermal_bcs": {BoundsFace.RADIAL: ThermalBC.ADIABATIC}},
        {
            "flow_bcs": {
                BoundsFace.RADIAL: FlowBC.INLET,
                BoundsFace.Z_MAX: FlowBC.OUTLET,
            }
        },
    ],
)
def test_register_reactor_blueprint_rejects_conditions_on_missing_faces(
    registry_service, overrides
):
    mini = _register_mini_assembly(registry_service)
    with pytest.raises(InvalidBoundsError):
        registry_service.register_reactor_blueprint(_make_blueprint(mini, **overrides))


def test_register_reactor_blueprint_uses_the_faces_of_its_roots_bounds(
    registry_service,
):
    mini = _register_mini_assembly(registry_service)
    round_root = registry_service.register_enclosure(
        Enclosure.create(
            family_name="round_root",
            version_label="1",
            bounds=registry_service.register_geometry(
                _make_cylinder("round_root_bounds", radius_m=0.02)
            ).id,
            fill=mini.water.id,
            placements=[_place("pin", PlaceableType.ENCLOSURE, mini.pin_cell.id)],
        )
    )
    cylinder_faces = faces_for_geometry_type(GeometryType.CYLINDER)
    blueprint = registry_service.register_reactor_blueprint(
        _make_blueprint(
            mini,
            root=round_root.id,
            neutron_bcs=dict.fromkeys(cylinder_faces, NeutronBC.VACUUM),
        )
    )
    assert set(blueprint.neutron_bcs) == set(cylinder_faces)

    with pytest.raises(InvalidBoundsError, match="faces of its root's bounds"):
        registry_service.register_reactor_blueprint(
            _make_blueprint(
                mini, family_name="box_faces_on_cylinder", root=round_root.id
            )
        )


def test_register_reactor_blueprint_rejects_dangling_root(registry_service):
    mini = _register_mini_assembly(registry_service)
    blueprint = _make_blueprint(mini, root=EnclosureID("nope-1"))
    with pytest.raises(DanglingReferenceError, match="nope-1"):
        registry_service.register_reactor_blueprint(blueprint)


# =============================================================================
# get_* — single row, unhydrated
# =============================================================================


def test_get_geometry_returns_none_when_absent(registry_service):
    assert registry_service.get_geometry(GeometryID("nonexistent-1")) is None


def test_get_reactor_blueprint_returns_none_when_absent(registry_service):
    assert (
        registry_service.get_reactor_blueprint(ReactorBlueprintID("nonexistent-1"))
        is None
    )


def test_get_material_and_part_return_persisted_rows(registry_service):
    mini = _register_mini_assembly(registry_service)
    assert registry_service.get_material(mini.uo2.id).to_dict() == mini.uo2.to_dict()
    fetched = registry_service.get_part(mini.pellet.id)
    assert fetched.to_dict() == mini.pellet.to_dict()


def test_get_enclosure_returns_unhydrated_row(registry_service):
    """get_enclosure() returns the bare row — refs stay as ids."""
    mini = _register_mini_assembly(registry_service)
    fetched = registry_service.get_enclosure(mini.pin_cell.id)
    assert fetched.to_dict() == mini.pin_cell.to_dict()
    assert fetched.placements[0].ref.id == mini.pellet.id


def test_get_lattice_returns_persisted_row(registry_service):
    mini = _register_mini_assembly(registry_service)
    fetched = registry_service.get_lattice(mini.lattice.id)
    assert fetched.to_dict() == mini.lattice.to_dict()


# =============================================================================
# Hydration
# =============================================================================


def test_get_resolved_placeable_hydrates_full_tree_once_per_object(registry_service):
    """assembly -> lattice -> pin cell (3 slots) and water cell ->
    pellet: every distinct object appears exactly once, however often
    it's placed."""
    mini = _register_mini_assembly(registry_service)
    root = PlaceableRef(type=PlaceableType.ENCLOSURE, id=mini.assembly.id)
    resolved = registry_service.get_resolved_placeable(root)

    assert resolved.root == root
    assert resolved.enclosures.keys() == {
        mini.assembly.id,
        mini.pin_cell.id,
        mini.water_cell.id,
    }
    assert resolved.lattices.keys() == {mini.lattice.id}
    assert resolved.parts.keys() == {mini.pellet.id}
    assert resolved.geometries.keys() == {
        mini.pellet_cyl.id,
        mini.pin_box.id,
        mini.assembly_box.id,
    }
    assert resolved.materials.keys() == {mini.uo2.id, mini.water.id}


def test_get_resolved_placeable_with_lattice_root(registry_service):
    mini = _register_mini_assembly(registry_service)
    root = PlaceableRef(type=PlaceableType.LATTICE, id=mini.lattice.id)
    resolved = registry_service.get_resolved_placeable(root)
    assert resolved.enclosures.keys() == {mini.pin_cell.id, mini.water_cell.id}
    assert resolved.parts.keys() == {mini.pellet.id}
    assert mini.assembly_box.id not in resolved.geometries


def test_get_resolved_placeable_with_part_root(registry_service):
    mini = _register_mini_assembly(registry_service)
    root = PlaceableRef(type=PlaceableType.PART, id=mini.pellet.id)
    resolved = registry_service.get_resolved_placeable(root)
    assert resolved.enclosures == {}
    assert resolved.geometries.keys() == {mini.pellet_cyl.id}
    assert resolved.materials.keys() == {mini.uo2.id}


def test_get_resolved_placeable_raises_for_nonexistent_root(registry_service):
    with pytest.raises(ValueError):
        registry_service.get_resolved_placeable(
            PlaceableRef(type=PlaceableType.ENCLOSURE, id=EnclosureID("nope-1"))
        )


def test_get_resolved_reactor_blueprint_hydrates_the_root_enclosures_tree(
    registry_service,
):
    mini = _register_mini_assembly(registry_service)
    blueprint = registry_service.register_reactor_blueprint(_make_blueprint(mini))
    resolved = registry_service.get_resolved_reactor_blueprint(blueprint.id)

    assert resolved.reactor_blueprint.to_dict() == blueprint.to_dict()
    assert resolved.resolved_placeable.root == PlaceableRef(
        type=PlaceableType.ENCLOSURE, id=mini.assembly.id
    )
    assert mini.assembly_box.id in resolved.resolved_placeable.geometries
    assert resolved.resolved_placeable.lattices.keys() == {mini.lattice.id}


def test_get_resolved_reactor_blueprint_raises_for_nonexistent_blueprint(
    registry_service,
):
    with pytest.raises(DanglingReferenceError):
        registry_service.get_resolved_reactor_blueprint(ReactorBlueprintID("nope-1"))
