"""
pace/core/resolved.py

Fully hydrated views over registered objects — what the solver
adapters consume:

    - ResolvedPlaceable — a root placeable (any kind) plus every
          Enclosure/Part/Lattice/Geometry/Material it transitively
          references, as flat id-keyed maps. A placeable reused at many
          positions (one pin cell in 264 lattice slots) appears once in
          its map, referenced by id from wherever it's used.
    - ResolvedReactorBlueprint — a ReactorBlueprint plus the
          ResolvedPlaceable of its root, whose maps also hold the
          blueprint's own bounds geometry and fill material.

Neither is registered or versioned: they're derived, transient views
over already-persisted data. Both self-validate on construction
(PaceObject), so a bundle missing anything it references fails loudly
rather than reaching a translator half-built.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.geometry import GAddition, Geometry, GSubtraction, geometry_from_dict
from pace.core.ids import EnclosureID, GeometryID, LatticeID, MaterialID, PartID
from pace.core.lattice import Lattice, lattice_from_dict
from pace.core.material import Material, MMixture, material_from_dict
from pace.core.pace_object import PaceObject
from pace.core.placeable import Enclosure, Part, PlaceableRef, PlaceableType
from pace.core.reactor_blueprint import ReactorBlueprint


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedPlaceable(PaceObject):
    """A root placeable and everything it transitively references."""

    root: PlaceableRef
    enclosures: dict[EnclosureID, Enclosure]
    parts: dict[PartID, Part]
    lattices: dict[LatticeID, Lattice]
    geometries: dict[GeometryID, Geometry]
    materials: dict[MaterialID, Material]

    def contains_placeable(self, ref: PlaceableRef) -> bool:
        """Whether the placeable object `ref` points at is in this
        bundle."""
        if ref.type == PlaceableType.PART:
            return ref.id in self.parts
        if ref.type == PlaceableType.ENCLOSURE:
            return ref.id in self.enclosures
        if ref.type == PlaceableType.LATTICE:
            return ref.id in self.lattices
        raise ValueError(f"Invalid PlaceableType provided: {ref.type}")

    def validate(self) -> None:
        self._validate_id_mapping()
        if not self.contains_placeable(self.root):
            raise ValueError(f"root {self.root.id!r} is not present in this bundle")
        self._validate_completeness()

    def _validate_id_mapping(self) -> None:
        """Every dict key must match the id of the object stored under
        it."""
        maps: dict[str, dict] = {
            "enclosures": self.enclosures,
            "parts": self.parts,
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

    def _require_placeable(self, owner: str, ref: PlaceableRef) -> None:
        if not self.contains_placeable(ref):
            raise ValueError(
                f"{owner} references {ref.type.value} {ref.id!r}, which is not "
                "present in this bundle."
            )

    def _validate_completeness(self) -> None:
        """Every reference held by an object in this bundle must resolve
        within the bundle."""
        for enclosure in self.enclosures.values():
            owner = f"Enclosure {enclosure.id!r}"
            self._require_geometry(owner, enclosure.bounds)
            if enclosure.fill is not None:
                self._require_material(owner, enclosure.fill)
            for placement in enclosure.placements:
                self._require_placeable(owner, placement.ref)

        for lattice in self.lattices.values():
            owner = f"Lattice {lattice.id!r}"
            self._require_material(owner, lattice.fill)
            for occupant in lattice.occupants:
                self._require_placeable(owner, occupant.ref)

        for part in self.parts.values():
            owner = f"Part {part.id!r}"
            self._require_geometry(owner, part.geometry)
            self._require_material(owner, part.material)

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
            "enclosures": {k: v.to_dict() for k, v in self.enclosures.items()},
            "parts": {k: v.to_dict() for k, v in self.parts.items()},
            "lattices": {k: v.to_dict() for k, v in self.lattices.items()},
            "geometries": {k: v.to_dict() for k, v in self.geometries.items()},
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedPlaceable:
        return cls(
            root=PlaceableRef.from_dict(data["root"]),
            enclosures={
                EnclosureID(k): Enclosure.from_dict(v)
                for k, v in data["enclosures"].items()
            },
            parts={PartID(k): Part.from_dict(v) for k, v in data["parts"].items()},
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
    resolved_placeable: ResolvedPlaceable

    def validate(self) -> None:
        owner = f"Reactor {self.reactor_blueprint.id!r}"
        if self.reactor_blueprint.root.ref != self.resolved_placeable.root:
            raise ValueError(
                f"{owner} root {self.reactor_blueprint.root.ref.id!r} does not match the "
                f"model root {self.resolved_placeable.root.id!r}"
            )
        self.resolved_placeable._require_geometry(owner, self.reactor_blueprint.bounds)
        if self.reactor_blueprint.fill is not None:
            self.resolved_placeable._require_material(
                owner, self.reactor_blueprint.fill
            )

    def to_dict(self) -> dict:
        return {
            "reactor_blueprint": self.reactor_blueprint.to_dict(),
            "resolved_placeable": self.resolved_placeable.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedReactorBlueprint:
        return cls(
            reactor_blueprint=ReactorBlueprint.from_dict(data["reactor_blueprint"]),
            resolved_placeable=ResolvedPlaceable.from_dict(data["resolved_placeable"]),
        )
