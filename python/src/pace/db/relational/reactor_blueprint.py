"""
pace/db/relational/reactor_blueprint.py

ReactorBlueprintRow (the `reactor_blueprints` table) and ReactorBlueprintDAO
See GeometryDAO's module docstring for the shared
reasoning behind the DAO/ComponentService split.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.ids import ReactorBlueprintID
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.db.relational.base import Base
from pace.db.relational.mixins import VersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import REACTOR_BLUEPRINTS_TABLE_NAME


class ReactorBlueprintRow(VersionMixin, Base):
    __tablename__ = REACTOR_BLUEPRINTS_TABLE_NAME
    __table_args__ = (Index("ix_reactor_blueprints_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey(f"{REACTOR_BLUEPRINTS_TABLE_NAME}.id"), nullable=True
    )


class ReactorBlueprintDAO:
    def __init__(self, db: SqlDB):
        self._db = db

    def exists(self, reactor_blueprint_id: ReactorBlueprintID) -> bool:
        with self._db.session() as session:
            return session.get(ReactorBlueprintRow, reactor_blueprint_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(ReactorBlueprintRow)
                .filter(ReactorBlueprintRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, reactor_id: ReactorBlueprintID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction.
        Cheap even for a version whose `data` blob is large."""
        with self._db.session() as session:
            row = session.get(ReactorBlueprintRow, reactor_id)
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

    def get(self, reactor_id: ReactorBlueprintID) -> ReactorBlueprint | None:
        with self._db.session() as session:
            row = session.get(ReactorBlueprintRow, reactor_id)
            if row is None:
                return None
            return self._row_to_domain(row)

    def get_many(
        self, ids: list[ReactorBlueprintID]
    ) -> dict[ReactorBlueprintID, ReactorBlueprint]:
        if not ids:
            return {}
        with self._db.session() as session:
            rows = (
                session.query(ReactorBlueprintRow)
                .filter(ReactorBlueprintRow.id.in_(ids))
                .all()
            )
            return {
                ReactorBlueprintID(row.id): ReactorBlueprint.from_dict(row.data)
                for row in rows
            }

    def versions_of_family(self, family_name: str) -> list[ReactorBlueprint]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(ReactorBlueprintRow)
                .filter(ReactorBlueprintRow.family_name == family_name)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    def latest_version_of_family(self, family_name: str) -> ReactorBlueprint | None:
        """The most recently created version in this family, by
        created_at — the ReactorBlueprint-page fallback default when no cookie/last-
        viewed state exists. NOT a substitute for explicit version
        selection anywhere a user is actually choosing which version
        to act on (e.g. starting a GT run)."""
        with self._db.session() as session:
            row = (
                session.query(ReactorBlueprintRow)
                .filter(ReactorBlueprintRow.family_name == family_name)
                .order_by(ReactorBlueprintRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return self._row_to_domain(row)

    def save(self, reactor: ReactorBlueprint) -> None:
        """Validates, then upserts (merge). See
        GeometryDAO.save() for why reactor.validate() is
        called again here despite already running once at construction."""
        reactor.validate()

        row = ReactorBlueprintRow(
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

    def delete(self, reactor_id: ReactorBlueprintID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(ReactorBlueprintRow, reactor_id)
            if row is not None:
                session.delete(row)

    def children_of(self, reactor_id: ReactorBlueprintID) -> list[ReactorBlueprint]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(ReactorBlueprintRow)
                .filter(ReactorBlueprintRow.derived_from == reactor_id)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    @staticmethod
    def _row_to_domain(row: ReactorBlueprintRow) -> ReactorBlueprint:
        return ReactorBlueprint.from_dict(row.data)
