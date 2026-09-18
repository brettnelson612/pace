"""
pace/db/relational/material.py

MaterialDAO — structural mirror of GeometryDAO. See that module's
docstring for the shared reasoning (DAO pattern, refcount/
family_exists split of responsibility with ComponentService); this one
differs only in which domain module/row/dispatch-table it's wired to.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.ids import MaterialID
from pace.core.material import (
    CLASS_TO_MATERIAL_TYPE,
    MATERIAL_TYPE_TO_CLASS,
    Material,
)
from pace.db.relational.base import Base
from pace.db.relational.mixins import PolymorphicVersionMixin
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import MATERIALS_TABLE_NAME


class MaterialRow(PolymorphicVersionMixin, Base):
    __tablename__ = MATERIALS_TABLE_NAME
    __table_args__ = (Index("ix_materials_family_name", "family_name"),)

    derived_from: Mapped[str | None] = mapped_column(
        ForeignKey("materials.id"), nullable=True
    )


class MaterialDAO:
    def __init__(self, db: SqlDB):
        self._db = db

    def exists(self, material_id: MaterialID) -> bool:
        with self._db.session() as session:
            return session.get(MaterialRow, material_id) is not None

    def family_exists(self, family_name: str) -> bool:
        """Whether ANY version currently exists under this family_name.
        Used by ComponentService to reject creating a brand-new family
        (v1) under a name that's already taken — does not itself decide
        whether that's an error, just answers the question."""
        with self._db.session() as session:
            return (
                session.query(MaterialRow)
                .filter(MaterialRow.family_name == family_name)
                .first()
                is not None
            )

    def describe(self, material_id: MaterialID) -> dict | None:
        """Lightweight metadata only — no full object reconstruction.
        Cheap even for a version whose `data` blob is large."""
        with self._db.session() as session:
            row = session.get(MaterialRow, material_id)
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

    def get(self, material_id: MaterialID) -> Material | None:
        with self._db.session() as session:
            row = session.get(MaterialRow, material_id)
            if row is None:
                return None
            return self._row_to_domain(row)

    def versions_of_family(self, family_name: str) -> list[Material]:
        """Every version ever recorded under this family_name — the
        query the version-tree UI / version dropdown is built on."""
        with self._db.session() as session:
            rows = (
                session.query(MaterialRow)
                .filter(MaterialRow.family_name == family_name)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    def latest_version_of_family(self, family_name: str) -> Material | None:
        """The most recently created version in this family, by
        created_at — the RT-page fallback default when no cookie/last-
        viewed state exists. NOT a substitute for explicit version
        selection anywhere a user is actually choosing which version
        to act on (e.g. starting a GT run)."""
        with self._db.session() as session:
            row = (
                session.query(MaterialRow)
                .filter(MaterialRow.family_name == family_name)
                .order_by(MaterialRow.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return self._row_to_domain(row)

    def save(self, version: Material) -> None:
        """Validates, then upserts (merge). See
        GeometryDAO.save() for why version.validate() is
        called again here despite already running once at construction."""
        version.validate()

        row = MaterialRow(
            id=version.id,
            family_name=version.family_name,
            version_label=version.version_label,
            derived_from=version.derived_from,
            gt_run_id=version.gt_run_id,
            user_edit=version.user_edit,
            type=self._domain_type_name(version),
            data=version.to_dict(),
        )
        with self._db.session() as session:
            session.merge(row)

    def delete(self, material_id: MaterialID) -> None:
        """Pure delete — no refcount check. Callers (ComponentService)
        must confirm refcount == 0 via the ReferenceRow index before
        calling this."""
        with self._db.session() as session:
            row = session.get(MaterialRow, material_id)
            if row is not None:
                session.delete(row)

    def children_of(self, material_id: MaterialID) -> list[Material]:
        """Reverse-index query — 'what versions were derived from this
        one' — computed on demand rather than stored on the version
        itself (a frozen dataclass can't hold a field that grows after
        construction, since its children don't exist yet when it's
        created)."""
        with self._db.session() as session:
            rows = (
                session.query(MaterialRow)
                .filter(MaterialRow.derived_from == material_id)
                .all()
            )
            return [self._row_to_domain(row) for row in rows]

    @staticmethod
    def _domain_type_name(version: Material) -> str:
        type_value = CLASS_TO_MATERIAL_TYPE.get(type(version))
        if type_value is None:
            raise ValueError(f"unknown Material subclass: {type(version)!r}")
        return type_value

    @staticmethod
    def _row_to_domain(row: MaterialRow) -> Material:
        cls = MATERIAL_TYPE_TO_CLASS.get(row.type)
        if cls is None:
            raise ValueError(f"unknown material version type in row: {row.type!r}")
        return cls.from_dict(row.data)
