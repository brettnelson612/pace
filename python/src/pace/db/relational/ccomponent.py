"""
pace/db/relational/ccomponent.py

CComponentRow (the `ccomponents` table) and CComponentDAO, the
persistence layer for CComponent.

CComponentDAO is a structural mirror of GeometryDAO/MaterialDAO/
LComponentDAO (same versioned method set), except CComponent has no
polymorphic subclasses to dispatch on — every row is reconstructed via
CComponent.from_dict() directly, with no `type` column needed (see
mixins.py: CComponentRow uses VersionMixin, not
PolymorphicVersionMixin).
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.component import CComponent
from pace.core.ids import (
    CComponentID,
)
from pace.db.relational.base import Base
from pace.db.relational.mixins import VersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import CCOMPONENTS_TABLE_NAME


class CComponentRow(VersionMixin, Base):
    __tablename__ = CCOMPONENTS_TABLE_NAME
    __table_args__ = (Index("ix_ccomponents_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey("ccomponents.id"), nullable=True
    )


class CComponentDAO:
    def __init__(self, db: SqlDB):
        self._db = db

    def exists(self, ccomponent_id: CComponentID) -> bool:
        with self._db.session() as session:
            return session.get(CComponentRow, ccomponent_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(CComponentRow)
                .filter(CComponentRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, ccomponent_id: CComponentID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction.
        Cheap even for a version whose `data` blob is large."""
        with self._db.session() as session:
            row = session.get(CComponentRow, ccomponent_id)
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

    def get(self, ccomponent_id: CComponentID) -> CComponent | None:
        with self._db.session() as session:
            row = session.get(CComponentRow, ccomponent_id)
            if row is None:
                return None
            return self._row_to_domain(row)

    def versions_of_family(self, family_name: str) -> list[CComponent]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(CComponentRow)
                .filter(CComponentRow.family_name == family_name)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    def latest_version_of_family(self, family_name: str) -> CComponent | None:
        """The most recently created version in this family, by
        created_at — the RT-page fallback default when no cookie/last-
        viewed state exists. NOT a substitute for explicit version
        selection anywhere a user is actually choosing which version
        to act on (e.g. starting a GT run)."""
        with self._db.session() as session:
            row = (
                session.query(CComponentRow)
                .filter(CComponentRow.family_name == family_name)
                .order_by(CComponentRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return self._row_to_domain(row)

    def save(self, version: CComponent) -> None:
        """Validates, then upserts (merge). See
        GeometryDAO.save() for why version.validate() is
        called again here despite already running once at construction."""
        version.validate()

        row = CComponentRow(
            id=version.id,
            family_name=version.family_name,
            version_label=version.version_label,
            derived_from=version.derived_from,
            gt_run_id=version.gt_run_id,
            user_edit=version.user_edit,
            data=version.to_dict(),
        )
        with self._db.session() as session:
            session.merge(row)

    def delete(self, ccomponent_id: CComponentID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(CComponentRow, ccomponent_id)
            if row is not None:
                session.delete(row)

    def children_of(self, ccomponent_id: CComponentID) -> list[CComponent]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(CComponentRow)
                .filter(CComponentRow.derived_from == ccomponent_id)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    @staticmethod
    def _row_to_domain(row: CComponentRow) -> CComponent:
        return CComponent.from_dict(row.data)
