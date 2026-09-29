"""
pace/db/relational/lattice.py

LatticeRow (the `lattices` table) and LatticeDAO — structural mirror of
GeometryDAO/MaterialDAO. See GeometryDAO's module docstring for the
shared reasoning (DAO pattern, refcount/family_exists split of
responsibility with ComponentService). Lattice is polymorphic
(RectLattice/HexLattice), so rows carry a `type` column and are
reconstructed via lattice_from_dict().
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.ids import LatticeID
from pace.core.lattice import Lattice, lattice_from_dict
from pace.db.relational.base import Base
from pace.db.relational.mixins import PolymorphicVersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import LATTICES_TABLE_NAME


class LatticeRow(PolymorphicVersionMixin, Base):
    __tablename__ = LATTICES_TABLE_NAME
    __table_args__ = (Index("ix_lattices_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey(f"{LATTICES_TABLE_NAME}.id"), nullable=True
    )


class LatticeDAO:
    def __init__(self, db: SqlDB):
        self._db = db

    def exists(self, lattice_id: LatticeID) -> bool:
        with self._db.session() as session:
            return session.get(LatticeRow, lattice_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(LatticeRow)
                .filter(LatticeRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, lattice_id: LatticeID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction.
        Cheap even for a version whose `data` blob is large."""
        with self._db.session() as session:
            row = session.get(LatticeRow, lattice_id)
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

    def get(self, lattice_id: LatticeID) -> Lattice | None:
        with self._db.session() as session:
            row = session.get(LatticeRow, lattice_id)
            if row is None:
                return None
            return lattice_from_dict(row.data)

    def get_many(self, ids: list[LatticeID]) -> dict[LatticeID, Lattice]:
        if not ids:
            return {}
        with self._db.session() as session:
            rows = session.query(LatticeRow).filter(LatticeRow.id.in_(ids)).all()
            return {LatticeID(row.id): lattice_from_dict(row.data) for row in rows}

    def versions_of_family(self, family_name: str) -> list[Lattice]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(LatticeRow)
                .filter(LatticeRow.family_name == family_name)
                .all()
            )
            return [lattice_from_dict(row.data) for row in rows]

    def latest_version_of_family(self, family_name: str) -> Lattice | None:
        """The most recently created version in this family, by
        created_at — the RT-page fallback default when no cookie/last-
        viewed state exists. NOT a substitute for explicit version
        selection anywhere a user is actually choosing which version
        to act on (e.g. starting a GT run)."""
        with self._db.session() as session:
            row = (
                session.query(LatticeRow)
                .filter(LatticeRow.family_name == family_name)
                .order_by(LatticeRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return lattice_from_dict(row.data)

    def save(self, lattice: Lattice) -> None:
        """Validates, then upserts (merge). See
        GeometryDAO.save() for why lattice.validate() is
        called again here despite already running once at construction."""
        lattice.validate()

        data = lattice.to_dict()
        row = LatticeRow(
            id=lattice.id,
            family_name=lattice.family_name,
            version_label=lattice.version_label,
            derived_from=lattice.derived_from,
            gt_run_id=lattice.gt_run_id,
            user_edit=lattice.user_edit,
            type=data["type"],
            data=data,
        )
        with self._db.session() as session:
            session.merge(row)

    def delete(self, lattice_id: LatticeID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(LatticeRow, lattice_id)
            if row is not None:
                session.delete(row)

    def children_of(self, lattice_id: LatticeID) -> list[Lattice]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(LatticeRow)
                .filter(LatticeRow.derived_from == lattice_id)
                .all()
            )
            return [lattice_from_dict(row.data) for row in rows]
