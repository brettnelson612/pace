"""
pace/core/versioned.py

Versioned — shared identity + three-state lineage for every versioned
aggregate (Geometry, Material, LComponent, CComponent, Lattice,
ReactorBlueprint).
Generic over the concrete NewType each subclass uses for its own id
(GeometryID, MaterialID, ...) — id/derived_from resolve to the right
type per subclass via Versioned[GeometryID], etc.

Versioning rule: a version is either v1 (derived_from=None, no
gt_run_id, user_edit=False) or a derived version, in which case
exactly one of gt_run_id or user_edit must also be set — never
neither, never both.

    - user_edit=True: a person (or code acting for them, e.g. variant
          generation) changed the object.
    - gt_run_id set: the new version was produced from a GT run's
          output — a deliberate, labeled promotion of run state into
          the registry (e.g. averaging a depletion snapshot into one
          uniform material to reuse as a design input).

What a GT-derived version is NOT: the default home of run state.
Spatially varying results (temperature/density fields, burned
compositions per pin/layer/ring) live in run output as state
snapshots keyed by region address. The registry holds designs that
are uniform within each region; promoting run state into it is an
explicit, lossy choice, never a side effect of running.

Equality: two versioned objects are equal when they are the same
concrete type with the same id, and hash by id. Every subclass must be
declared with @dataclass(..., eq=False); with the default eq=True the
dataclass decorator generates a field-based __eq__ on the subclass that
replaces this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Self, TypeVar, cast

from pace.core.ids import GTRunID
from pace.core.pace_object import PaceObject

IDType = TypeVar("IDType")


@dataclass(frozen=True, kw_only=True, eq=False)
class Versioned(PaceObject, Generic[IDType]):
    id: IDType
    family_name: str
    version_label: str
    derived_from: IDType | None = None
    gt_run_id: GTRunID | None = None
    user_edit: bool = False

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Versioned)
            and type(self) is type(other)
            and self.id == other.id
        )

    def __hash__(self) -> int:
        return hash(self.id)

    @staticmethod
    def build_id(family_name: str, version_label: str) -> IDType:
        """The one canonical way an id is constructed from a
        family_name + version_label pair. cast(), like NewType
        wrapping, is a no-op at runtime — this returns the same plain
        string either way, just typed as IDType for the subclass it's
        called through."""
        return cast(IDType, f"{family_name}-{version_label}")

    @classmethod
    def create(cls, *, family_name: str, version_label: str, **kwargs) -> Self:
        """Named-constructor convenience: derives id via build_id() so
        callers never have to compute and pass it separately.

        Prefer this over calling the constructor directly when
        constructing brand-new versions. from_dict() should keep
        calling cls(...) directly — it already has a trusted, stored
        id and doesn't need one derived.
        """
        return cls(
            id=cls.build_id(family_name, version_label),
            family_name=family_name,
            version_label=version_label,
            **kwargs,
        )

    def validate(self) -> None:
        self._validate_id()
        self._validate_lineage()

    def _validate_id(self) -> None:
        expected = self.build_id(self.family_name, self.version_label)
        if self.id != expected:
            raise ValueError(
                f"id {self.id!r} does not match family_name/version_label "
                f"(expected {expected!r})"
            )

    def _validate_lineage(self) -> None:
        has_predecessor = self.derived_from is not None
        has_gt_run = self.gt_run_id is not None

        if not has_predecessor:
            if has_gt_run or self.user_edit:
                raise ValueError(
                    "gt_run_id must be unset and user_edit must be False "
                    "when derived_from is unset (version 1 has no "
                    "derivation cause)"
                )
        elif has_gt_run == self.user_edit:
            # both set, or both unset — either way, invalid
            raise ValueError(
                "when derived_from is set, exactly one of gt_run_id or "
                "user_edit must also be set (got gt_run_id="
                f"{self.gt_run_id!r}, user_edit={self.user_edit!r})"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family_name": self.family_name,
            "version_label": self.version_label,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "user_edit": self.user_edit,
        }

    @classmethod
    def _base_fields_from_dict(cls, data: dict) -> dict:
        """Shared identity/lineage parsing — every leaf class's
        from_dict() splices this in via **cls._base_fields_from_dict(data)
        (or, for Geometry/Material, via a subclass-level extension of
        it — see Material._material_fields_from_dict), then adds only
        its own type-specific fields on top."""
        return {
            "id": cast(IDType, data["id"]),
            "family_name": data["family_name"],
            "version_label": data["version_label"],
            "derived_from": (
                cast(IDType, data["derived_from"])
                if data["derived_from"] is not None
                else None
            ),
            "gt_run_id": (
                GTRunID(data["gt_run_id"]) if data["gt_run_id"] is not None else None
            ),
            "user_edit": data["user_edit"],
        }
