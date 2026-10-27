"""
pace/core/part.py

Part — a leaf: one geometry paired with one material, no position of
its own. Versioned like Geometry/Material/Enclosure: family_name +
version_label identity, and the same three-state
derived_from/simulation_job_id/user_edit lineage rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from pace.core.ids import GeometryID, MaterialID, PartID
from pace.core.versioned import Versioned


@dataclass(frozen=True, kw_only=True, eq=False)
class Part(Versioned[PartID]):
    """A leaf — one geometry paired with one material, no position of
    its own.

    Identity/lineage (id/family_name/version_label/derived_from/
    simulation_job_id/user_edit, build_id(), create(), the three-state
    lineage rule) are inherited from Versioned — see that class's
    docstring.

    family_name has no shape/composition of its own to anchor it to
    the way it does for Geometry or Material — a Part IS the
    pairing, so family_name is the caller-chosen name for the
    conceptual SLOT this pairing fills (e.g. "fuel_pellet"), not a
    description of any content that survives a version bump.

    A version minted as part of a cascading edit (RegistryService
    propagating a change up through every reference path) carries the
    SAME cause as the root edit that triggered the cascade: if a human
    hand-edited the referenced geometry, the new Part version
    this produces is also user_edit=True; if a simulation job produced
    the root change, the new Part version carries that same
    simulation_job_id. The human or the simulation job is the root
    cause of the whole chain, not just the one object they directly
    touched.

    Example — a UO2 fuel pellet, referencing an already-defined
    cylinder geometry and enriched-UO2 material:
        Part.create(
            family_name="fuel_pellet",
            version_label="1",
            geometry=GeometryID("fuel_pellet_cyl-1"),
            material=MaterialID("uranium3.2_uo2-1"),
        )
    """

    geometry: GeometryID
    material: MaterialID

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "geometry": self.geometry,
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Part:
        return cls(
            **cls._base_versioned_fields_from_dict(data),
            geometry=GeometryID(data["geometry"]),
            material=MaterialID(data["material"]),
        )
