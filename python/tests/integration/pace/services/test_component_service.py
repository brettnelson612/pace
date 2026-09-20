"""
tests/integration/pace/services/test_component_service.py

Integration tests for ComponentService against a real (in-memory)
PaceDB — confirms the domain objects, DAOs, RegistryDB, and reference
index actually work together, not just correct in isolation. Covers
register_*()/get_*() and get_resolved_ccomponent() (full hydration).
"""

from __future__ import annotations

import pytest
from pace.core.component import CComponent, LComponent, PComponent
from pace.core.geometry import GCylinder, GPose
from pace.core.ids import CComponentID, GeometryID, LComponentID, PComponentID
from pace.core.material import MaterialComponentEntry, MIsotopic
from pace.core.reference_types import ReferenceableType
from pace.db.pace_db import PaceDB
from pace.services.component_service import (
    ComponentService,
    DanglingReferenceError,
    FamilyAlreadyExistsError,
)


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


def _two_pellet_placements(lcomponent_id: LComponentID) -> list[PComponent]:
    return [
        PComponent(
            id=PComponentID("pc-1"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            component=lcomponent_id,
            component_type=ReferenceableType.LCOMPONENT,
        ),
        PComponent(
            id=PComponentID("pc-2"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
            component=lcomponent_id,
            component_type=ReferenceableType.LCOMPONENT,
        ),
    ]


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
    geometry = _make_cylinder()
    material = _make_isotopic()
    component_service.register_geometry(geometry)
    component_service.register_material(material)

    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    component_service.register_lcomponent(lcomponent)

    assert pace_db.registry.lcomponents.get(lcomponent.id) == lcomponent


def test_register_lcomponent_rejects_dangling_geometry(pace_db, component_service):
    material = _make_isotopic()
    component_service.register_material(material)

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
# register_ccomponent — creation, dangling-reference validation, and
# reference-index bookkeeping (deduplication across repeated placements)
# =============================================================================


def test_register_ccomponent_round_trips_and_dedupes_references(
    pace_db, component_service
):
    geometry = _make_cylinder()
    material = _make_isotopic()
    component_service.register_geometry(geometry)
    component_service.register_material(material)

    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    component_service.register_lcomponent(lcomponent)

    ccomponent = CComponent.create(
        family_name="test_pin",
        version_label="1",
        components=_two_pellet_placements(lcomponent.id),
    )
    component_service.register_ccomponent(ccomponent)

    assert pace_db.registry.ccomponents.get(ccomponent.id) == ccomponent

    # Two placements of the same LComponent inside one CComponent
    # collapse to ONE reference edge, not two.
    assert (
        pace_db.registry.references.count_incoming(
            ReferenceableType.LCOMPONENT, lcomponent.id
        )
        == 1
    )
    assert (
        pace_db.registry.references.count_incoming(
            ReferenceableType.GEOMETRY, geometry.id
        )
        == 1
    )
    assert (
        pace_db.registry.references.count_incoming(
            ReferenceableType.MATERIAL, material.id
        )
        == 1
    )


def test_register_ccomponent_rejects_dangling_component(pace_db, component_service):
    ccomponent = CComponent.create(
        family_name="test_pin",
        version_label="1",
        components=_two_pellet_placements(LComponentID("nonexistent-1")),
    )
    with pytest.raises(DanglingReferenceError):
        component_service.register_ccomponent(ccomponent)

    assert pace_db.registry.ccomponents.get(ccomponent.id) is None


def test_ccomponent_cannot_self_reference_at_construction():
    """Not a ComponentService call — confirms the self-reference guard
    lives where register_ccomponent()'s docstring says it does: at
    CComponent's own construction time, before ComponentService is
    ever involved."""
    self_id = CComponent.build_id("test_pin", "1")
    with pytest.raises(ValueError):
        CComponent(
            id=self_id,
            family_name="test_pin",
            version_label="1",
            components=[
                PComponent(
                    id=PComponentID("pc-1"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    component=self_id,
                    component_type=ReferenceableType.CCOMPONENT,
                ),
                PComponent(
                    id=PComponentID("pc-2"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
                    component=self_id,
                    component_type=ReferenceableType.CCOMPONENT,
                ),
            ],
        )


# =============================================================================
# get_* — single row, unhydrated
# =============================================================================


def test_get_geometry_returns_none_when_absent(component_service):
    assert component_service.get_geometry(GeometryID("nonexistent-1")) is None


def test_get_geometry_returns_persisted_row(component_service):
    geometry = _make_cylinder()
    component_service.register_geometry(geometry)
    assert component_service.get_geometry(geometry.id) == geometry


def test_get_ccomponent_returns_unhydrated_row(component_service):
    """get_ccomponent() returns the bare row — embedded PComponent
    references stay as ids, nothing is resolved. Contrast with
    get_resolved_ccomponent() below."""
    geometry = _make_cylinder()
    material = _make_isotopic()
    component_service.register_geometry(geometry)
    component_service.register_material(material)
    lcomponent = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    component_service.register_lcomponent(lcomponent)
    ccomponent = CComponent.create(
        family_name="test_pin",
        version_label="1",
        components=_two_pellet_placements(lcomponent.id),
    )
    component_service.register_ccomponent(ccomponent)

    fetched = component_service.get_ccomponent(ccomponent.id)
    assert fetched == ccomponent
    assert fetched.components[0].component == lcomponent.id


# =============================================================================
# get_resolved_ccomponent — full hydration
# =============================================================================


def test_get_resolved_ccomponent_hydrates_full_tree(pace_db, component_service):
    """A two-level tree (assembly -> pin -> pellet -> geometry/
    material), each level placed twice for the dedup check, hydrates
    to exactly one entry per distinct object — not one per
    placement."""
    geometry = _make_cylinder()
    material = _make_isotopic()
    component_service.register_geometry(geometry)
    component_service.register_material(material)

    pellet = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    component_service.register_lcomponent(pellet)

    pin = CComponent.create(
        family_name="test_pin",
        version_label="1",
        components=_two_pellet_placements(pellet.id),
    )
    component_service.register_ccomponent(pin)

    assembly = CComponent.create(
        family_name="test_assembly",
        version_label="1",
        components=[
            PComponent(
                id=PComponentID("pc-1"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                component=pin.id,
                component_type=ReferenceableType.CCOMPONENT,
            ),
            PComponent(
                id=PComponentID("pc-2"),
                pose=GPose(x_m=0.02, y_m=0.0, z_m=0.0),
                component=pin.id,
                component_type=ReferenceableType.CCOMPONENT,
            ),
        ],
    )
    component_service.register_ccomponent(assembly)

    resolved = component_service.get_resolved_ccomponent(assembly.id)

    assert resolved.root == assembly.id
    assert resolved.ccomponents.keys() == {assembly.id, pin.id}
    assert resolved.lcomponents.keys() == {pellet.id}
    assert resolved.geometries.keys() == {geometry.id}
    assert resolved.materials.keys() == {material.id}

    assert resolved.ccomponents[pin.id] == pin
    assert resolved.lcomponents[pellet.id] == pellet
    assert resolved.geometries[geometry.id] == geometry
    assert resolved.materials[material.id] == material


def test_get_resolved_ccomponent_single_level_tree(pace_db, component_service):
    """A flat (non-nested) tree hydrates correctly too — the graph
    walk isn't only exercised by the multi-level case above."""
    geometry = _make_cylinder()
    material = _make_isotopic()
    component_service.register_geometry(geometry)
    component_service.register_material(material)

    pellet = LComponent.create(
        family_name="test_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    component_service.register_lcomponent(pellet)

    pin = CComponent.create(
        family_name="test_pin",
        version_label="1",
        components=_two_pellet_placements(pellet.id),
    )
    component_service.register_ccomponent(pin)

    resolved = component_service.get_resolved_ccomponent(pin.id)

    assert resolved.root == pin.id
    assert resolved.ccomponents.keys() == {pin.id}
    assert resolved.lcomponents.keys() == {pellet.id}
    assert resolved.geometries.keys() == {geometry.id}
    assert resolved.materials.keys() == {material.id}


def test_get_resolved_ccomponent_raises_for_nonexistent_root(component_service):
    """An empty/nonexistent root means outgoing_many() finds no edges,
    and get_many() returns nothing for that id -- so the root never
    lands in ccomponents, and ResolvedCComponent's own
    _validate_root_provided() catches it at construction."""
    with pytest.raises(ValueError):
        component_service.get_resolved_ccomponent(CComponentID("nonexistent-1"))
