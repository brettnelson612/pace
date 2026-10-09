"""
pace/core/resolved.py

Fully hydrated views over registered objects — what the solver
adapters consume:

    - ResolvedComponent — a root component (any placeable kind) plus every
          CComponent/LComponent/Lattice/Geometry/Material it transitively
          references, as flat id-keyed maps. A component reused at many
          positions (one pin cell in 264 lattice slots) appears once in
          its map, referenced by id from wherever it's used.
    - ResolvedReactorBlueprint — a ReactorBlueprint plus the
          ResolvedComponent of its root, whose maps also hold the
          blueprint's own bounds geometry and fill material.

Neither is registered or versioned: they're derived, transient views
over already-persisted data. Both self-validate on construction
(PaceObject), so a bundle missing anything it references fails loudly
rather than reaching a translator half-built.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.component import CComponent, LComponent, ComponentType, ComponentRef
from pace.core.geometry import GAddition, Geometry, GSubtraction, geometry_from_dict
from pace.core.ids import CComponentID, GeometryID, LatticeID, LComponentID, MaterialID
from pace.core.lattice import Lattice, lattice_from_dict
from pace.core.material import Material, MMixture, material_from_dict
from pace.core.pace_object import PaceObject
from pace.core.reactor_blueprint import ReactorBlueprint


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedComponent(PaceObject):
    """A root component and everything it transitively references."""

    root: ComponentRef
    ccomponents: dict[CComponentID, CComponent]
    lcomponents: dict[LComponentID, LComponent]
    lattices: dict[LatticeID, Lattice]
    geometries: dict[GeometryID, Geometry]
    materials: dict[MaterialID, Material]

    def contains_component(self, ref: ComponentRef) -> bool:
        """Whether the placeable object `ref` points at is in this
        bundle."""
        if ref.type == ComponentType.LCOMPONENT:
            return ref.id in self.lcomponents
        if ref.type == ComponentType.CCOMPONENT:
            return ref.id in self.ccomponents
        if ref.type == ComponentType.LATTICE:
            return ref.id in self.lattices
        raise ValueError(f"Invalid ComponentType provided: {ref.type}")

    def validate(self) -> None:
        self._validate_id_mapping()
        if not self.contains_component(self.root):
            raise ValueError(f"root {self.root.id!r} is not present in this bundle")
        self._validate_completeness()

    def _validate_id_mapping(self) -> None:
        """Every dict key must match the id of the object stored under
        it."""
        maps: dict[str, dict] = {
            "ccomponents": self.ccomponents,
            "lcomponents": self.lcomponents,
            "lattices": self.lattices,
            "geometries": self.geometries,
            "materials": self.materials,
        }
        for name, mapping in maps.items():
            for key, value in mapping.items():
                if value.id != key:
                    raise ValueError(
                        f"{name}: {value.id!r} stored under mismatched key {key!r}"
                    )

    def _require_geometry(self, owner: str, geometry_id: GeometryID) -> None:
        if geometry_id not in self.geometries:
            raise ValueError(
                f"{owner} references geometry {geometry_id!r}, which is not "
                "present in this bundle."
            )

    def _require_material(self, owner: str, material_id: MaterialID) -> None:
        if material_id not in self.materials:
            raise ValueError(
                f"{owner} references material {material_id!r}, which is not "
                "present in this bundle."
            )

    def _require_component(self, owner: str, ref: ComponentRef) -> None:
        if not self.contains_component(ref):
            raise ValueError(
                f"{owner} references {ref.type.value} {ref.id!r}, which is not "
                "present in this bundle."
            )

    def _validate_completeness(self) -> None:
        """Every reference held by an object in this bundle must resolve
        within the bundle."""
        for ccomponent in self.ccomponents.values():
            owner = f"CComponent {ccomponent.id!r}"
            self._require_geometry(owner, ccomponent.bounds)
            if ccomponent.fill is not None:
                self._require_material(owner, ccomponent.fill)
            for placement in ccomponent.placements:
                self._require_component(owner, placement.ref)

        for lattice in self.lattices.values():
            owner = f"Lattice {lattice.id!r}"
            self._require_material(owner, lattice.fill)
            for element in lattice.elements:
                self._require_component(owner, element.ref)

        for lcomponent in self.lcomponents.values():
            owner = f"LComponent {lcomponent.id!r}"
            self._require_geometry(owner, lcomponent.geometry)
            self._require_material(owner, lcomponent.material)

        for geometry in self.geometries.values():
            owner = f"Geometry {geometry.id!r}"
            if isinstance(geometry, GAddition):
                for unit_id, _ in geometry.units:
                    self._require_geometry(owner, unit_id)
            elif isinstance(geometry, GSubtraction):
                base_id, _ = geometry.base
                self._require_geometry(owner, base_id)
                for cut_id, _ in geometry.cuts:
                    self._require_geometry(owner, cut_id)

        for material in self.materials.values():
            if isinstance(material, MMixture):
                for material_id, _ in material.components:
                    self._require_material(f"Material {material.id!r}", material_id)

    def to_dict(self) -> dict:
        return {
            "root": self.root.to_dict(),
            "ccomponents": {k: v.to_dict() for k, v in self.ccomponents.items()},
            "lcomponents": {k: v.to_dict() for k, v in self.lcomponents.items()},
            "lattices": {k: v.to_dict() for k, v in self.lattices.items()},
            "geometries": {k: v.to_dict() for k, v in self.geometries.items()},
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedComponent:
        return cls(
            root=ComponentRef.from_dict(data["root"]),
            ccomponents={
                CComponentID(k): CComponent.from_dict(v)
                for k, v in data["ccomponents"].items()
            },
            lcomponents={
                LComponentID(k): LComponent.from_dict(v)
                for k, v in data["lcomponents"].items()
            },
            lattices={
                LatticeID(k): lattice_from_dict(v) for k, v in data["lattices"].items()
            },
            geometries={
                GeometryID(k): geometry_from_dict(v)
                for k, v in data["geometries"].items()
            },
            materials={
                MaterialID(k): material_from_dict(v)
                for k, v in data["materials"].items()
            },
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedReactorBlueprint(PaceObject):
    """A Reactor and the fully hydrated model it runs on — the single
    input every solver adapter takes (alongside the rest of the simulation
    case)."""

    reactor_blueprint: ReactorBlueprint
    rc: ResolvedComponent

    def validate(self) -> None:
        owner = f"Reactor {self.reactor_blueprint.id!r}"
        if self.reactor_blueprint.root.ref != self.rc.root:
            raise ValueError(
                f"{owner} root {self.reactor_blueprint.root.ref.id!r} does not match the "
                f"model root {self.rc.root.id!r}"
            )
        self.rc._require_geometry(owner, self.reactor_blueprint.bounds)
        if self.reactor_blueprint.fill is not None:
            self.rc._require_material(owner, self.reactor_blueprint.fill)

    def to_dict(self) -> dict:
        return {
            "reactor_blueprint": self.reactor_blueprint.to_dict(),
            "rc": self.rc.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedReactorBlueprint:
        return cls(
            reactor_blueprint=ReactorBlueprint.from_dict(data["reactor_blueprint"]),
            rc=ResolvedComponent.from_dict(data["rc"]),
        )
