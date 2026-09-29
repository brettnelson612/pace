"""
pace/db/relational/reactor.py

ReactorRow (the `reactors` table) and ReactorDAO — structural mirror of
ReactorDAO (same versioned method set, no polymorphic subclasses, so
no `type` column). See GeometryDAO's module docstring for the shared
reasoning behind the DAO/ComponentService split.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.ids import ReactorID
from pace.core.reactor import Reactor
from pace.db.relational.base import Base
from pace.db.relational.mixins import VersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import REACTORS_TABLE_NAME


class ReactorRow(VersionMixin, Base):
    __tablename__ = REACTORS_TABLE_NAME
    __table_args__ = (Index("ix_reactors_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey(f"{REACTORS_TABLE_NAME}.id"), nullable=True
    )


class ReactorDAO:
    def __init__(self, db: SqlDB):
        self._db = db

    def exists(self, reactor_id: ReactorID) -> bool:
        with self._db.session() as session:
            return session.get(ReactorRow, reactor_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(ReactorRow)
                .filter(ReactorRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, reactor_id: ReactorID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction.
        Cheap even for a version whose `data` blob is large."""
        with self._db.session() as session:
            row = session.get(ReactorRow, reactor_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "family_name": row.family_name,
                "version_label": row.version_label,
                "derived_from": row.derived_from,
                "gt_run_id": row.gt_run_id,
                "user_edit": row.user_edit,
                "created_at": row.created_at,
            }

    def get(self, reactor_id: ReactorID) -> Reactor | None:
        with self._db.session() as session:
            row = session.get(ReactorRow, reactor_id)
            if row is None:
                return None
            return self._row_to_domain(row)

    def get_many(self, ids: list[ReactorID]) -> dict[ReactorID, Reactor]:
        if not ids:
            return {}
        with self._db.session() as session:
            rows = session.query(ReactorRow).filter(ReactorRow.id.in_(ids)).all()
            return {ReactorID(row.id): Reactor.from_dict(row.data) for row in rows}

    def versions_of_family(self, family_name: str) -> list[Reactor]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(ReactorRow)
                .filter(ReactorRow.family_name == family_name)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    def latest_version_of_family(self, family_name: str) -> Reactor | None:
        """The most recently created version in this family, by
        created_at — the Reactor-page fallback default when no cookie/last-
        viewed state exists. NOT a substitute for explicit version
        selection anywhere a user is actually choosing which version
        to act on (e.g. starting a GT run)."""
        with self._db.session() as session:
            row = (
                session.query(ReactorRow)
                .filter(ReactorRow.family_name == family_name)
                .order_by(ReactorRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return self._row_to_domain(row)

    def save(self, reactor: Reactor) -> None:
        """Validates, then upserts (merge). See
        GeometryDAO.save() for why reactor.validate() is
        called again here despite already running once at construction."""
        reactor.validate()

        row = ReactorRow(
            id=reactor.id,
            family_name=reactor.family_name,
            version_label=reactor.version_label,
            derived_from=reactor.derived_from,
            gt_run_id=reactor.gt_run_id,
            user_edit=reactor.user_edit,
            data=reactor.to_dict(),
        )
        with self._db.session() as session:
            session.merge(row)

    def delete(self, reactor_id: ReactorID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(ReactorRow, reactor_id)
            if row is not None:
                session.delete(row)

    def children_of(self, reactor_id: ReactorID) -> list[Reactor]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(ReactorRow)
                .filter(ReactorRow.derived_from == reactor_id)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    @staticmethod
    def _row_to_domain(row: ReactorRow) -> Reactor:
        return Reactor.from_dict(row.data)
