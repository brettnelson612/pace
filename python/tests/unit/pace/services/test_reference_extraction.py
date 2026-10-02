"""
tests/unit/pace/services/test_reference_extraction.py

Covers: the pure reference-extraction functions — the exact set of
edges each versioned type records, with the right source type, target
types, and deduplication when the same target is used several times.
"""

from pace.core.boundary_conditions import NeutronBC
from pace.core.bounds import faces_for_geometry_type
from pace.core.component import (
    CComponent,
    ComponentPlacement,
    ComponentRef,
    ComponentType,
    LComponent,
)
from pace.core.geometry import (
    GAddition,
    GCylinder,
    GeometryType,
    GPose,
    GSubtraction,
)
from pace.core.ids import (
    CComponentID,
    ComponentPlacementID,
    GeometryID,
    LatticeID,
    LComponentID,
    MaterialID,
)
from pace.core.lattice import LatticeElement, RectLattice
from pace.core.material import MMixture, MVoid, PercentType
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.core.reference_types import ReferenceableType as RT
from pace.db.relational.reference import Reference
from pace.services.reference_extraction import (
    extract_ccomponent_references,
    extract_geometry_references,
    extract_lattice_references,
    extract_lcomponent_references,
    extract_material_references,
    extract_reactor_blueprint_references,
)

ORIGIN = GPose(x_m=0.0, y_m=0.0, z_m=0.0)


def _edges(references: set[Reference]) -> set[tuple]:
    return {
        (r.source_type, r.source_id, r.target_type, r.target_id) for r in references
    }


def _placement(placement_id: str, ref: ComponentRef, z_m: float = 0.0):
    return ComponentPlacement(
        id=ComponentPlacementID(placement_id),
        pose=GPose(x_m=0.0, y_m=0.0, z_m=z_m),
        ref=ref,
    )


def test_simple_geometry_has_no_references():
    cylinder = GCylinder.create(
        family_name="cyl", version_label="1", radius_m=0.1, height_m=1.0
    )
    assert extract_geometry_references(cylinder) == set()


def test_gaddition_references_each_distinct_unit_once():
    addition = GAddition.create(
        family_name="joined",
        version_label="1",
        units=[
            (GeometryID("a-1"), ORIGIN),
            (GeometryID("a-1"), GPose(x_m=1.0, y_m=0.0, z_m=0.0)),
            (GeometryID("b-1"), ORIGIN),
        ],
    )
    assert _edges(extract_geometry_references(addition)) == {
        (RT.GEOMETRY, "joined-1", RT.GEOMETRY, "a-1"),
        (RT.GEOMETRY, "joined-1", RT.GEOMETRY, "b-1"),
    }


def test_gsubtraction_references_base_and_cuts():
    subtraction = GSubtraction.create(
        family_name="notched",
        version_label="1",
        base=(GeometryID("base-1"), ORIGIN),
        cuts=[(GeometryID("cut-1"), ORIGIN)],
    )
    assert _edges(extract_geometry_references(subtraction)) == {
        (RT.GEOMETRY, "notched-1", RT.GEOMETRY, "base-1"),
        (RT.GEOMETRY, "notched-1", RT.GEOMETRY, "cut-1"),
    }


def test_void_material_has_no_references():
    assert (
        extract_material_references(MVoid.create(family_name="v", version_label="1"))
        == set()
    )


def test_mixture_references_its_constituents():
    mixture = MMixture.create(
        family_name="blend",
        version_label="1",
        components=[(MaterialID("a-1"), 0.5), (MaterialID("b-1"), 0.5)],
        percent_type=PercentType.AO,
    )
    assert _edges(extract_material_references(mixture)) == {
        (RT.MATERIAL, "blend-1", RT.MATERIAL, "a-1"),
        (RT.MATERIAL, "blend-1", RT.MATERIAL, "b-1"),
    }


def test_lcomponent_references_geometry_and_material():
    pellet = LComponent.create(
        family_name="pellet",
        version_label="1",
        geometry=GeometryID("cyl-1"),
        material=MaterialID("uo2-1"),
    )
    assert _edges(extract_lcomponent_references(pellet)) == {
        (RT.LCOMPONENT, "pellet-1", RT.GEOMETRY, "cyl-1"),
        (RT.LCOMPONENT, "pellet-1", RT.MATERIAL, "uo2-1"),
    }


def test_ccomponent_references_bounds_fill_and_each_distinct_member():
    pellet_ref = ComponentRef(
        type=ComponentType.LCOMPONENT, id=LComponentID("pellet-1")
    )
    lattice_ref = ComponentRef(type=ComponentType.LATTICE, id=LatticeID("grid-1"))
    rod = CComponent.create(
        family_name="rod",
        version_label="1",
        bounds=GeometryID("rod_cyl-1"),
        fill=MaterialID("helium-1"),
        placements=[
            _placement("lower", pellet_ref),
            _placement("upper", pellet_ref, z_m=0.01),
            _placement("grid", lattice_ref),
        ],
    )
    assert _edges(extract_ccomponent_references(rod)) == {
        (RT.CCOMPONENT, "rod-1", RT.GEOMETRY, "rod_cyl-1"),
        (RT.CCOMPONENT, "rod-1", RT.MATERIAL, "helium-1"),
        (RT.CCOMPONENT, "rod-1", RT.LCOMPONENT, "pellet-1"),
        (RT.CCOMPONENT, "rod-1", RT.LATTICE, "grid-1"),
    }


def test_ccomponent_without_fill_has_no_material_edge():
    rod = CComponent.create(
        family_name="rod",
        version_label="1",
        bounds=GeometryID("rod_cyl-1"),
        placements=[
            _placement(
                "pellet",
                ComponentRef(type=ComponentType.LCOMPONENT, id=LComponentID("p-1")),
            )
        ],
    )
    assert {r.target_type for r in extract_ccomponent_references(rod)} == {
        RT.GEOMETRY,
        RT.LCOMPONENT,
    }


def test_lattice_edges_are_sourced_from_the_lattice():
    pin_ref = ComponentRef(type=ComponentType.CCOMPONENT, id=CComponentID("pin-1"))
    lattice = RectLattice.create(
        family_name="grid",
        version_label="1",
        pitch_m=0.0126,
        fill=MaterialID("water-1"),
        shape=(2, 2),
        elements=[LatticeElement(ref=pin_ref, addresses=((0, 0), (1, 1)))],
    )
    assert _edges(extract_lattice_references(lattice)) == {
        (RT.LATTICE, "grid-1", RT.MATERIAL, "water-1"),
        (RT.LATTICE, "grid-1", RT.CCOMPONENT, "pin-1"),
    }


def test_reactor_blueprint_references_bounds_fill_and_root():
    blueprint = ReactorBlueprint.create(
        family_name="core",
        version_label="1",
        bounds=GeometryID("core_box-1"),
        fill=MaterialID("water-1"),
        root=_placement(
            "root", ComponentRef(type=ComponentType.LATTICE, id=LatticeID("grid-1"))
        ),
        neutron_bcs=dict.fromkeys(
            faces_for_geometry_type(GeometryType.RECT_PRISM), NeutronBC.VACUUM
        ),
    )
    assert _edges(extract_reactor_blueprint_references(blueprint)) == {
        (RT.REACTOR_BLUEPRINT, "core-1", RT.GEOMETRY, "core_box-1"),
        (RT.REACTOR_BLUEPRINT, "core-1", RT.MATERIAL, "water-1"),
        (RT.REACTOR_BLUEPRINT, "core-1", RT.LATTICE, "grid-1"),
    }
