"""
tests/pace/core/test_component.py

Covers: LComponent identity (build_id/_validate_id) and the three-state
lineage rule now that LComponent is versioned; PComponent round-trips,
frozen immutability, and component_type validation; CComponent
identity, lineage, composition (min-member and duplicate-detection,
including the z_rotation_rad and component_type differentiation
fixes), id-based equality + the hash(self.id) regression guard, and
round-trips; ResolvedCComponent's internal-consistency validation
(id-mapping, root presence, dangling-reference completeness).
"""

from dataclasses import FrozenInstanceError

import pytest
from pace.core.component import CComponent, LComponent, PComponent, ResolvedCComponent
from pace.core.geometry import GAddition, GCylinder, GPose
from pace.core.ids import (
    CComponentID,
    GeometryID,
    GTRunID,
    LComponentID,
    MaterialID,
    PComponentID,
)
from pace.core.material import MMixture, MVoid
from pace.core.reference_types import ReferenceableType

# =============================================================================
# LComponent — identity
# =============================================================================


def test_lcomponent_build_id():
    assert LComponent.build_id("fuel_pellet", "1") == "fuel_pellet-1"


def test_lcomponent_create_derives_id():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert lc.id == "fuel_pellet-1"


def test_lcomponent_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        LComponent(
            id=LComponentID("mismatched_id"),
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )


# =============================================================================
# LComponent — three-state lineage rule
# =============================================================================

LINEAGE_CASES = [
    # (derived_from, gt_run_id, user_edit, should_raise)
    pytest.param(None, None, False, False, id="v1_valid"),
    pytest.param(None, GTRunID("gt-1"), False, True, id="v1_with_gt_run_id"),
    pytest.param(None, None, True, True, id="v1_with_user_edit"),
    pytest.param(None, GTRunID("gt-1"), True, True, id="v1_with_both"),
    pytest.param(
        "fuel_pellet-1", GTRunID("gt-1"), False, False, id="derived_gt_run_only"
    ),
    pytest.param("fuel_pellet-1", None, True, False, id="derived_user_edit_only"),
    pytest.param("fuel_pellet-1", None, False, True, id="derived_with_neither"),
    pytest.param("fuel_pellet-1", GTRunID("gt-1"), True, True, id="derived_with_both"),
]


@pytest.mark.parametrize("derived_from,gt_run_id,user_edit,should_raise", LINEAGE_CASES)
def test_lcomponent_lineage_rule(derived_from, gt_run_id, user_edit, should_raise):
    version_label = "2" if derived_from else "1"
    derived_from_id = LComponentID(derived_from) if derived_from else None

    def build() -> LComponent:
        return LComponent.create(
            family_name="fuel_pellet",
            version_label=version_label,
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
            derived_from=derived_from_id,
            gt_run_id=gt_run_id,
            user_edit=user_edit,
        )

    if should_raise:
        with pytest.raises(ValueError):
            build()
    else:
        assert build().derived_from == derived_from_id


# =============================================================================
# LComponent — round-trip, immutability, equality
# =============================================================================


def test_lcomponent_round_trip():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert LComponent.from_dict(lc.to_dict()) == lc


def test_lcomponent_derived_round_trip():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="2",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
        derived_from=LComponentID("fuel_pellet-1"),
        user_edit=True,
    )
    round_tripped = LComponent.from_dict(lc.to_dict())
    assert round_tripped == lc
    assert round_tripped.derived_from == "fuel_pellet-1"
    assert round_tripped.user_edit is True


def test_lcomponent_is_frozen():
    lc = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("fuel_pellet_cyl-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    with pytest.raises(FrozenInstanceError):
        lc.family_name = "other"  # type: ignore[misc]


def test_lcomponent_default_equality_is_value_based():
    """LComponent has no dict/list fields, so it keeps the plain
    frozen-dataclass default equality — unlike CComponent, it does NOT
    override __eq__/__hash__ to be id-based."""

    def build() -> LComponent:
        return LComponent.create(
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )

    lc_a = build()
    lc_b = build()
    assert lc_a == lc_b

    lc_c = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=GeometryID("different_geometry-1"),
        material=MaterialID("uranium3.2_uo2-1"),
    )
    assert lc_a != lc_c


# =============================================================================
# PComponent
# =============================================================================


def test_pcomponent_round_trip_lcomponent_ref():
    pc = PComponent(
        id=PComponentID("pc-1"),
        pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01, z_rotation_rad=0.0),
        component=LComponentID("fuel_pellet-1"),
        component_type=ReferenceableType.LCOMPONENT,
    )
    assert PComponent.from_dict(pc.to_dict()) == pc


def test_pcomponent_round_trip_ccomponent_ref():
    pc = PComponent(
        id=PComponentID("pc-1"),
        pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
        component=CComponentID("fuel_pin-1"),
        component_type=ReferenceableType.CCOMPONENT,
    )
    assert PComponent.from_dict(pc.to_dict()) == pc


def test_pcomponent_is_frozen():
    pc = PComponent(
        id=PComponentID("pc-1"),
        pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
        component=LComponentID("fuel_pellet-1"),
        component_type=ReferenceableType.LCOMPONENT,
    )
    with pytest.raises(FrozenInstanceError):
        pc.pose = GPose(x_m=1.0, y_m=0.0, z_m=0.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "component_type",
    [ReferenceableType.GEOMETRY, ReferenceableType.MATERIAL],
)
def test_pcomponent_rejects_non_component_type(component_type):
    """A PComponent can only reference an LComponent or a CComponent —
    Geometry/Material are never valid here, since neither can be
    placed directly as a member of a CComponent."""
    with pytest.raises(ValueError):
        PComponent(
            id=PComponentID("pc-1"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            component=LComponentID("fuel_pellet-1"),
            component_type=component_type,
        )


# =============================================================================
# CComponent — identity
# =============================================================================


def _two_pcomponents(z_rotation_rad_second: float = 0.0) -> list[PComponent]:
    return [
        PComponent(
            id=PComponentID("pc-1"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            component=LComponentID("fuel_pellet-1"),
            component_type=ReferenceableType.LCOMPONENT,
        ),
        PComponent(
            id=PComponentID("pc-2"),
            pose=GPose(
                x_m=0.0, y_m=0.0, z_m=0.01, z_rotation_rad=z_rotation_rad_second
            ),
            component=LComponentID("fuel_pellet-1"),
            component_type=ReferenceableType.LCOMPONENT,
        ),
    ]


def test_ccomponent_build_id():
    assert CComponent.build_id("fuel_pin", "1") == "fuel_pin-1"


def test_ccomponent_create_derives_id():
    cc = CComponent.create(
        family_name="fuel_pin", version_label="1", components=_two_pcomponents()
    )
    assert cc.id == "fuel_pin-1"


def test_ccomponent_validate_id_mismatch_raises():
    with pytest.raises(ValueError):
        CComponent(
            id=CComponentID("mismatched_id"),
            family_name="fuel_pin",
            version_label="1",
            components=_two_pcomponents(),
        )


# =============================================================================
# CComponent — three-state lineage rule
# =============================================================================


@pytest.mark.parametrize("derived_from,gt_run_id,user_edit,should_raise", LINEAGE_CASES)
def test_ccomponent_lineage_rule(derived_from, gt_run_id, user_edit, should_raise):
    version_label = "2" if derived_from else "1"
    derived_from_id = (
        CComponentID(derived_from.replace("fuel_pellet", "fuel_pin"))
        if derived_from
        else None
    )

    def build() -> CComponent:
        return CComponent.create(
            family_name="fuel_pin",
            version_label=version_label,
            components=_two_pcomponents(),
            derived_from=derived_from_id,
            gt_run_id=gt_run_id,
            user_edit=user_edit,
        )

    if should_raise:
        with pytest.raises(ValueError):
            build()
    else:
        assert build().derived_from == derived_from_id


# =============================================================================
# CComponent — composition (min-member, duplicate detection)
# =============================================================================


def test_ccomponent_requires_at_least_two_components():
    with pytest.raises(ValueError):
        CComponent.create(
            family_name="fuel_pin",
            version_label="1",
            components=[
                PComponent(
                    id=PComponentID("pc-1"),
                    pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                    component=LComponentID("fuel_pellet-1"),
                    component_type=ReferenceableType.LCOMPONENT,
                )
            ],
        )


def test_ccomponent_rejects_duplicate_component_position():
    duplicate = PComponent(
        id=PComponentID("pc-1"),
        pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
        component=LComponentID("fuel_pellet-1"),
        component_type=ReferenceableType.LCOMPONENT,
    )
    same_again = PComponent(
        id=PComponentID("pc-2"),
        pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
        component=LComponentID("fuel_pellet-1"),
        component_type=ReferenceableType.LCOMPONENT,
    )
    with pytest.raises(ValueError):
        CComponent.create(
            family_name="fuel_pin",
            version_label="1",
            components=[duplicate, same_again],
        )


def test_ccomponent_same_position_different_rotation_is_not_a_duplicate():
    """Regression guard: the duplicate-detection key must include
    z_rotation_rad, or two placements at the same x/y/z but different
    rotation would be wrongly flagged as duplicates."""
    same_xyz_different_rotation = [
        PComponent(
            id=PComponentID("pc-1"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0, z_rotation_rad=0.0),
            component=LComponentID("fuel_pellet-1"),
            component_type=ReferenceableType.LCOMPONENT,
        ),
        PComponent(
            id=PComponentID("pc-2"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0, z_rotation_rad=1.0),
            component=LComponentID("fuel_pellet-1"),
            component_type=ReferenceableType.LCOMPONENT,
        ),
    ]
    cc = CComponent.create(
        family_name="fuel_pin",
        version_label="1",
        components=same_xyz_different_rotation,
    )
    assert len(cc.components) == 2


def test_ccomponent_rejects_direct_self_reference():
    """A PComponent whose component_type is CCOMPONENT and whose
    component id equals the parent CComponent's own id is a direct
    cycle — caught here without any registry access."""
    self_ref = [
        PComponent(
            id=PComponentID("pc-1"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            component=CComponentID("fuel_pin-1"),
            component_type=ReferenceableType.CCOMPONENT,
        ),
        PComponent(
            id=PComponentID("pc-2"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
            component=LComponentID("fuel_pellet-1"),
            component_type=ReferenceableType.LCOMPONENT,
        ),
    ]
    with pytest.raises(ValueError):
        CComponent.create(
            family_name="fuel_pin", version_label="1", components=self_ref
        )


def test_ccomponent_same_id_string_different_component_type_is_not_a_duplicate():
    """Regression guard: the duplicate-detection key must include
    component_type, not just the bare component id string — an
    LComponentID and a CComponentID live in separate registries and
    could coincidentally share the same string, and must not be
    conflated as the same reference."""
    same_id_string_different_registry = [
        PComponent(
            id=PComponentID("pc-1"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            component=LComponentID("fuel_pin-1"),
            component_type=ReferenceableType.LCOMPONENT,
        ),
        PComponent(
            id=PComponentID("pc-2"),
            pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
            component=CComponentID("fuel_pin-1"),
            component_type=ReferenceableType.CCOMPONENT,
        ),
    ]
    cc = CComponent.create(
        family_name="assembly",
        version_label="1",
        components=same_id_string_different_registry,
    )
    assert len(cc.components) == 2


# =============================================================================
# CComponent — round-trip, immutability, id-based equality
# =============================================================================


def test_ccomponent_round_trip():
    cc = CComponent.create(
        family_name="fuel_pin", version_label="1", components=_two_pcomponents()
    )
    assert CComponent.from_dict(cc.to_dict()) == cc


def test_ccomponent_is_frozen():
    cc = CComponent.create(
        family_name="fuel_pin", version_label="1", components=_two_pcomponents()
    )
    with pytest.raises(FrozenInstanceError):
        cc.family_name = "other"  # type: ignore[misc]


def test_ccomponent_equality_is_id_based_not_field_based():
    """Regression guard, same reasoning as GAddition/MIsotopic: two
    CComponents with the same id but DIFFERENT underlying component
    lists must still compare equal, since __eq__ is overridden to
    check self.id == other.id only, not full field equality."""
    cc_a = CComponent(
        id=CComponentID("fuel_pin-1"),
        family_name="fuel_pin",
        version_label="1",
        components=_two_pcomponents(z_rotation_rad_second=0.0),
    )
    cc_b = CComponent(
        id=CComponentID("fuel_pin-1"),
        family_name="fuel_pin",
        version_label="1",
        components=_two_pcomponents(z_rotation_rad_second=0.5),
    )
    assert cc_a == cc_b


def test_ccomponent_hash_uses_hash_of_id_not_python_id():
    """Regression guard for the hash(self.id) vs id(self.id) bug —
    two equal-by-id CComponents must share a hash, which only holds if
    __hash__ uses the built-in hash() on self.id, not the built-in
    id() (memory address)."""
    cc_a = CComponent(
        id=CComponentID("fuel_pin-1"),
        family_name="fuel_pin",
        version_label="1",
        components=_two_pcomponents(),
    )
    cc_b = CComponent(
        id=CComponentID("fuel_pin-1"),
        family_name="fuel_pin",
        version_label="1",
        components=_two_pcomponents(),
    )
    assert hash(cc_a) == hash(cc_b)
    assert {cc_a, cc_b} == {cc_a}


# =============================================================================
# ResolvedCComponent — bundle internal-consistency validation
# =============================================================================


def _resolved_bundle():
    """A minimal, fully-consistent 2-pellet fuel-pin bundle: one
    geometry, one void material, one LComponent, one CComponent."""
    geometry = GCylinder.create(
        family_name="pellet_cyl", version_label="1", radius_m=0.5, height_m=1.0
    )
    material = MVoid.create(family_name="pellet_mat", version_label="1")
    lcomponent = LComponent.create(
        family_name="fuel_pellet",
        version_label="1",
        geometry=geometry.id,
        material=material.id,
    )
    ccomponent = CComponent.create(
        family_name="fuel_pin",
        version_label="1",
        components=[
            PComponent(
                id=PComponentID("pc-1"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                component=lcomponent.id,
                component_type=ReferenceableType.LCOMPONENT,
            ),
            PComponent(
                id=PComponentID("pc-2"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.01),
                component=lcomponent.id,
                component_type=ReferenceableType.LCOMPONENT,
            ),
        ],
    )
    return ResolvedCComponent(
        root=ccomponent.id,
        ccomponents={ccomponent.id: ccomponent},
        lcomponents={lcomponent.id: lcomponent},
        geometries={geometry.id: geometry},
        materials={material.id: material},
    )


def test_resolved_ccomponent_valid_bundle_round_trips():
    # ResolvedCComponent is eq=False with no __eq__ override (it has no
    # id of its own to key equality on, unlike CComponent/GAddition) —
    # compare via to_dict() rather than == on the instances themselves.
    bundle = _resolved_bundle()
    assert ResolvedCComponent.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()


def test_resolved_ccomponent_root_not_in_ccomponents_raises():
    bundle = _resolved_bundle()
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=CComponentID("not_the_root-1"),
            ccomponents=bundle.ccomponents,
            lcomponents=bundle.lcomponents,
            geometries=bundle.geometries,
            materials=bundle.materials,
        )


def test_resolved_ccomponent_mismatched_dict_key_raises():
    bundle = _resolved_bundle()
    (ccomponent,) = bundle.ccomponents.values()
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=bundle.root,
            ccomponents={CComponentID("wrong_key-1"): ccomponent},
            lcomponents=bundle.lcomponents,
            geometries=bundle.geometries,
            materials=bundle.materials,
        )


def test_resolved_ccomponent_dangling_lcomponent_reference_raises():
    bundle = _resolved_bundle()
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=bundle.root,
            ccomponents=bundle.ccomponents,
            lcomponents={},  # the CComponent references an LComponent not here
            geometries=bundle.geometries,
            materials=bundle.materials,
        )


def test_resolved_ccomponent_dangling_geometry_reference_raises():
    bundle = _resolved_bundle()
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=bundle.root,
            ccomponents=bundle.ccomponents,
            lcomponents=bundle.lcomponents,
            geometries={},  # the LComponent references a geometry not here
            materials=bundle.materials,
        )


def test_resolved_ccomponent_dangling_material_reference_raises():
    bundle = _resolved_bundle()
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=bundle.root,
            ccomponents=bundle.ccomponents,
            lcomponents=bundle.lcomponents,
            geometries=bundle.geometries,
            materials={},  # the LComponent references a material not here
        )


def test_resolved_ccomponent_dangling_ccomponent_to_ccomponent_reference_raises():
    """A CComponent member referencing another CComponent (not just an
    LComponent) must also resolve within the bundle."""
    bundle = _resolved_bundle()
    (inner_ccomponent,) = bundle.ccomponents.values()
    outer = CComponent.create(
        family_name="assembly",
        version_label="1",
        components=[
            PComponent(
                id=PComponentID("pc-1"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=0.0),
                component=CComponentID("missing_pin-1"),
                component_type=ReferenceableType.CCOMPONENT,
            ),
            PComponent(
                id=PComponentID("pc-2"),
                pose=GPose(x_m=0.0, y_m=0.0, z_m=1.0),
                component=inner_ccomponent.id,
                component_type=ReferenceableType.CCOMPONENT,
            ),
        ],
    )
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=outer.id,
            ccomponents={outer.id: outer, inner_ccomponent.id: inner_ccomponent},
            lcomponents=bundle.lcomponents,
            geometries=bundle.geometries,
            materials=bundle.materials,
        )


def test_resolved_ccomponent_gaddition_dangling_unit_reference_raises():
    """A CSG-composite geometry (GAddition) referencing a unit
    geometry not present in the bundle fails completeness, same as a
    direct LComponent/CComponent dangling reference."""
    bundle = _resolved_bundle()
    (base_geometry,) = bundle.geometries.values()
    addition = GAddition.create(
        family_name="union_shape",
        version_label="1",
        units=[
            (base_geometry.id, GPose(x_m=0.0, y_m=0.0, z_m=0.0)),
            (GeometryID("missing_geo-1"), GPose(x_m=1.0, y_m=0.0, z_m=0.0)),
        ],
    )
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=bundle.root,
            ccomponents=bundle.ccomponents,
            lcomponents=bundle.lcomponents,
            geometries={**bundle.geometries, addition.id: addition},
            materials=bundle.materials,
        )


def test_resolved_ccomponent_mmixture_dangling_component_reference_raises():
    """An MMixture referencing a constituent Material not present in
    the bundle fails completeness, mirroring GAddition/GSubtraction's
    geometry-reference completeness check."""
    bundle = _resolved_bundle()
    (existing_material,) = bundle.materials.values()
    mixture = MMixture.create(
        family_name="blend",
        version_label="1",
        components=[
            (existing_material.id, 0.5),
            (MaterialID("missing_mat-1"), 0.5),
        ],
        percent_type="ao",
    )
    with pytest.raises(ValueError):
        ResolvedCComponent(
            root=bundle.root,
            ccomponents=bundle.ccomponents,
            lcomponents=bundle.lcomponents,
            geometries=bundle.geometries,
            materials={**bundle.materials, mixture.id: mixture},
        )
