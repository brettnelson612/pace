"""
pace/db/relational/lcomponent.py

LComponentRow (the `lcomponents` table) and LComponentDAO, the
persistence layer for LComponent.

LComponentRow uses VersionMixin for its id/family_name/version_label/
gt_run_id/user_edit/type/data/created_at columns, plus its own
self-referential derived_from foreign key — the same shape as every
other versioned row (GeometryRow, MaterialRow, CComponentRow).
`type` is unused beyond the single LCOMPONENT_TYPE constant:
LComponent has no polymorphic subclasses, so there's nothing to
dispatch on, but the column is kept for shape consistency with the
other *Row tables rather than introducing a variant mixin for one
case.

geometry and material are real foreign keys (not JSON-embedded),
since an LComponent's whole content IS those two references —
normalizing them gives real referential integrity (can't save an
LComponent pointing at a nonexistent Geometry/Material) essentially
for free. `data` still mirrors the JSONB-hybrid pattern for
consistency with every other DAO's save()/_row_to_domain() shape.

LComponentDAO's method set mirrors GeometryDAO/MaterialDAO/
CComponentDAO exactly, since LComponent is versioned the same way as
those three.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.component import LComponent
from pace.core.ids import LComponentID
from pace.db.relational.base import Base
from pace.db.relational.mixins import VersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import LCOMPONENTS_TABLE_NAME


class LComponentRow(VersionMixin, Base):
    __tablename__ = LCOMPONENTS_TABLE_NAME
    __table_args__ = (Index("ix_lcomponents_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey("lcomponents.id"), nullable=True
    )
    geometry: Mapped[str] = mapped_column(ForeignKey("geometries.id"), nullable=False)
    material: Mapped[str] = mapped_column(ForeignKey("materials.id"), nullable=False)


class LComponentDAO:
    def __init__(self, sql_db: SqlDB):
        self._db = sql_db

    def exists(self, lcomponent_id: LComponentID) -> bool:
        with self._db.session() as session:
            return session.get(LComponentRow, lcomponent_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(LComponentRow)
                .filter(LComponentRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, lcomponent_id: LComponentID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction."""
        with self._db.session() as session:
            row = session.get(LComponentRow, lcomponent_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "family_name": row.family_name,
                "version_label": row.version_label,
                "derived_from": row.derived_from,
                "gt_run_id": row.gt_run_id,
                "user_edit": row.user_edit,
                "geometry": row.geometry,
                "material": row.material,
                "created_at": row.created_at,
            }

    def get(self, lcomponent_id: LComponentID) -> LComponent | None:
        with self._db.session() as session:
            row = session.get(LComponentRow, lcomponent_id)
            if row is None:
                return None
            return LComponent.from_dict(row.data)

    def versions_of_family(self, family_name: str) -> list[LComponent]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(LComponentRow)
                .filter(LComponentRow.family_name == family_name)
                .all()
            )
            return [LComponent.from_dict(row.data) for row in rows]

    def latest_version_of_family(self, family_name: str) -> LComponent | None:
        """The most recently created version in this family, by
        created_at — a fallback default when no explicit version has
        been selected. NOT a substitute for explicit version selection
        anywhere a user is actually choosing which version to act on
        (e.g. starting a GT run)."""
        with self._db.session() as session:
            row = (
                session.query(LComponentRow)
                .filter(LComponentRow.family_name == family_name)
                .order_by(LComponentRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return LComponent.from_dict(row.data)

    def save(self, component: LComponent) -> None:
        """Validates, then upserts (merge).

        component.validate() already ran once, automatically, at
        construction via PaceObject's __post_init__ wiring — calling
        it again here is a deliberate defensive re-check at the
        persistence boundary, not redundant: it catches anything built
        before a stricter rule existed, or mutated via
        object.__setattr__ bypassing __init__ (frozen dataclasses can
        still be bypassed that way).
        """
        component.validate()

        row = LComponentRow(
            id=component.id,
            family_name=component.family_name,
            version_label=component.version_label,
            derived_from=component.derived_from,
            gt_run_id=component.gt_run_id,
            user_edit=component.user_edit,
            geometry=component.geometry,
            material=component.material,
            data=component.to_dict(),
        )
        with self._db.session() as session:
            session.merge(row)

    def delete(self, lcomponent_id: LComponentID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(LComponentRow, lcomponent_id)
            if row is not None:
                session.delete(row)

    def children_of(self, lcomponent_id: LComponentID) -> list[LComponent]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(LComponentRow)
                .filter(LComponentRow.derived_from == lcomponent_id)
                .all()
            )
            return [LComponent.from_dict(row.data) for row in rows]
