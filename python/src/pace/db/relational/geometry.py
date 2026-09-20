"""
pace/db/relational/geometry.py

GeometryDAO — the persistence abstraction sitting on top
of SqlDB's raw sessions for Geometry. Part of the DAO
pattern decision from design: pure persistence, no business logic.

Enforcing the refcount-gated edit/delete rule (refcount 0 -> true
edit/delete, refcount > 0 -> mint a new version) AND the "can't create
a v1 whose family_name already exists" rule is ComponentService's job,
sitting on top of this dao's query methods — family_exists()
here answers the question, but the decision to reject based on it
belongs one layer up.

Structural mirror of MaterialDAO — same method set, same shape,
differing only in which domain module/row it's wired to.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.geometry import Geometry, geometry_from_dict
from pace.core.ids import GeometryID
from pace.db.relational.base import Base
from pace.db.relational.mixins import PolymorphicVersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import GEOMETRIES_TABLE_NAME


class GeometryRow(PolymorphicVersionMixin, Base):
    __tablename__ = GEOMETRIES_TABLE_NAME
    __table_args__ = (Index("ix_geometries_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey(f"{GEOMETRIES_TABLE_NAME}.id"), nullable=True
    )


class GeometryDAO:
    def __init__(self, db: SqlDB):
        self._db = db

    def exists(self, geometry_id: GeometryID) -> bool:
        with self._db.session() as session:
            return session.get(GeometryRow, geometry_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(GeometryRow)
                .filter(GeometryRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, geometry_id: GeometryID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction.
        Cheap even for a version whose `data` blob is large."""
        with self._db.session() as session:
            row = session.get(GeometryRow, geometry_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "family_name": row.family_name,
                "version_label": row.version_label,
                "type": row.type,
                "derived_from": row.derived_from,
                "gt_run_id": row.gt_run_id,
                "user_edit": row.user_edit,
                "created_at": row.created_at,
            }

    def get(self, geometry_id: GeometryID) -> Geometry | None:
        with self._db.session() as session:
            row = session.get(GeometryRow, geometry_id)
            if row is None:
                return None
            return geometry_from_dict(row.data)

    def get_many(self, ids: list[GeometryID]) -> dict[GeometryID, Geometry]:
        if not ids:
            return {}
        with self._db.session() as session:
            rows = session.query(GeometryRow).filter(GeometryRow.id.in_(ids)).all()
            return {GeometryID(row.id): geometry_from_dict(row.data) for row in rows}

    def versions_of_family(self, family_name: str) -> list[Geometry]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(GeometryRow)
                .filter(GeometryRow.family_name == family_name)
                .all()
            )
            return [geometry_from_dict(row.data) for row in rows]

    def latest_version_of_family(self, family_name: str) -> Geometry | None:
        """The most recently created version in this family, by
        created_at — the RT-page fallback default when no cookie/last-
        viewed state exists. NOT a substitute for explicit version
        selection anywhere a user is actually choosing which version
        to act on (e.g. starting a GT run) — see the design note that
        "latest" is deliberately never used for that."""
        with self._db.session() as session:
            row = (
                session.query(GeometryRow)
                .filter(GeometryRow.family_name == family_name)
                .order_by(GeometryRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return geometry_from_dict(row.data)

    def save(self, geometry: Geometry) -> None:
        """Validates, then upserts (merge).

        geometry.validate() already ran once, automatically, at
        construction via PaceObject's __post_init__ wiring — calling it
        again here is a deliberate defensive re-check at the
        persistence boundary, not redundant: it catches anything built
        before a stricter rule existed, or mutated via
        object.__setattr__ bypassing __init__ (frozen dataclasses can
        still be bypassed that way).
        """
        geometry.validate()

        data = geometry.to_dict()
        row = GeometryRow(
            id=geometry.id,
            family_name=geometry.family_name,
            version_label=geometry.version_label,
            derived_from=geometry.derived_from,
            gt_run_id=geometry.gt_run_id,
            user_edit=geometry.user_edit,
            type=data["type"],
            data=data,
        )
        with self._db.session() as session:
            session.merge(row)

    def delete(self, geometry_id: GeometryID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(GeometryRow, geometry_id)
            if row is not None:
                session.delete(row)

    def children_of(self, geometry_id: GeometryID) -> list[Geometry]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(GeometryRow)
                .filter(GeometryRow.derived_from == geometry_id)
                .all()
            )
            return [geometry_from_dict(row.data) for row in rows]
