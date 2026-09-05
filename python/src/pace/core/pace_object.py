"""
PaceObject: the shared ABC for PACE's modeling-domain objects
(geometries, materials, versions, components, blueprints) — to_dict()
and from_dict() only. Deliberately minimal: no display/rendering logic
(that's the registry's job, not the data's), no solver-translation
methods (those live on GeometryVersion specifically, not this base).

to_dict() always represents references to other PaceObjects as
bare IDs, never embedded/hydrated content. from_dict() fails loudly on
missing/malformed fields — no partial/best-effort reconstruction.
"""

from abc import ABC, abstractmethod


class PaceObject(ABC):
    """Shared interface for PACE's modeling-domain objects — geometries,
    materials, versions, components, and blueprints."""

    def __post_init__(self):
        self.validate()

    @abstractmethod
    def to_dict(self) -> dict:
        """Plain-field representation. References to other PaceObjects
        are always represented as bare IDs, never embedded/hydrated content."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "PaceObject":
        """Reconstruct an instance from its to_dict() representation.
        Raises on missing or malformed fields — no partial/best-effort
        reconstruction."""

    def validate(self):
        """Override this class to add class-specific validation logic."""
