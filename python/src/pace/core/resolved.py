"""
pace/core/resolved.py

Fully hydrated views over registered objects — what the solver
adapters consume:

    - ResolvedModel — a root component (any placeable kind) plus every
          CComponent/LComponent/Lattice/Geometry/Material it transitively
          references, as flat id-keyed maps. A component reused at many
          positions (one pin cell in 264 lattice slots) appears once in
          its map, referenced by id from wherever it's used.
    - ResolvedReactor — a Reactor plus the ResolvedModel of its root,
          whose maps also hold the Reactor's own bounds geometry, fill
          material, and every material its operating state names.

Neither is registered or versioned: they're derived, transient views
over already-persisted data. Both self-validate on construction
(PaceObject), so a bundle missing anything it references fails loudly
rather than reaching a translator half-built.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.component import CComponent, LComponent
from pace.core.component_ref import ComponentKind, ComponentRef
from pace.core.geometry import GAddition, Geometry, GSubtraction, geometry_from_dict
from pace.core.ids import CComponentID, GeometryID, LatticeID, LComponentID, MaterialID
from pace.core.lattice import Lattice, lattice_from_dict
from pace.core.material import Material, MMixture, material_from_dict
from pace.core.pace_object import PaceObject
from pace.core.reactor import Reactor


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedModel(PaceObject):
    """A root component and everything it transitively references."""

    root: ComponentRef
    ccomponents: dict[CComponentID, CComponent]
    lcomponents: dict[LComponentID, LComponent]
    lattices: dict[LatticeID, Lattice]
    geometries: dict[GeometryID, Geometry]
    materials: dict[MaterialID, Material]

    def contains(self, ref: ComponentRef) -> bool:
        """Whether the placeable object `ref` points at is in this
        bundle."""
        if ref.kind == ComponentKind.LCOMPONENT:
            return ref.id in self.lcomponents
        if ref.kind == ComponentKind.CCOMPONENT:
            return ref.id in self.ccomponents
        return ref.id in self.lattices

    def validate(self) -> None:
        self._validate_id_mapping()
        if not self.contains(self.root):
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

    def _require_ref(self, owner: str, ref: ComponentRef) -> None:
        if not self.contains(ref):
            raise ValueError(
                f"{owner} references {ref.kind.value} {ref.id!r}, which is not "
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
            for pcomponent in ccomponent.components:
                self._require_ref(owner, pcomponent.ref)

        for lattice in self.lattices.values():
            owner = f"Lattice {lattice.id!r}"
            self._require_material(owner, lattice.fill)
            for placement in lattice.placements:
                self._require_ref(owner, placement.ref)

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
    def from_dict(cls, data: dict) -> ResolvedModel:
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
class ResolvedReactor(PaceObject):
    """A Reactor and the fully hydrated model it runs on — the single
    input every solver adapter takes (alongside a run spec)."""

    reactor: Reactor
    model: ResolvedModel

    def validate(self) -> None:
        owner = f"Reactor {self.reactor.id!r}"
        if self.reactor.root.ref != self.model.root:
            raise ValueError(
                f"{owner} root {self.reactor.root.ref.id!r} does not match the "
                f"model root {self.model.root.id!r}"
            )
        self.model._require_geometry(owner, self.reactor.bounds)
        if self.reactor.fill is not None:
            self.model._require_material(owner, self.reactor.fill)
        for material_id in self.reactor.operating_state.initial_temperatures_k:
            self.model._require_material(owner, material_id)

    def to_dict(self) -> dict:
        return {"reactor": self.reactor.to_dict(), "model": self.model.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedReactor:
        return cls(
            reactor=Reactor.from_dict(data["reactor"]),
            model=ResolvedModel.from_dict(data["model"]),
        )
