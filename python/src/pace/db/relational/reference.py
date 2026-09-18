"""
pace/db/relational/reference.py

ReferenceRow — a generic reverse-index of "what references what"
across the whole registry. This is what makes reference-count checks
(can this be deleted?) and "what points at this version" lookups
cheap, indexed queries instead of full scans.

Deliberately maintained by ComponentService on every create/delete
that establishes or removes a reference — NOT derived automatically by
the database. A reference can live in a normalized FK column
(LComponentRow.geometry) or inside a JSON blob
(CComponentRow.data["components"]), and the database can't see into
the latter to enforce this on its own.

ReferenceableType enumerates the four aggregate kinds that can
participate in a reference, on either side. This is a coarser tag than
GeometryType/MaterialType — it identifies which registry an id belongs
to, not which concrete subclass a Geometry/Material version is.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Index, String
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.reference_types import ReferenceableType
from pace.db.relational.base import Base
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import REFERENCES_TABLE_NAME


@dataclass(frozen=True, kw_only=True)
class Reference:
    """One directed edge: referencing_id (of referencing_type) points
    at referenced_id (of referenced_type). A plain 4-string tuple
    would be easy to pass in the wrong order at a call site; this
    names each position."""

    referencing_type: ReferenceableType
    referencing_id: str
    referenced_type: ReferenceableType
    referenced_id: str


class ReferenceRow(Base):
    __tablename__ = REFERENCES_TABLE_NAME
    __table_args__ = (
        Index("ix_references_referencing", "referencing_type", "referencing_id"),
        Index("ix_references_referenced", "referenced_type", "referenced_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    referencing_type: Mapped[str] = mapped_column(String, nullable=False)
    referencing_id: Mapped[str] = mapped_column(String, nullable=False)
    referenced_type: Mapped[str] = mapped_column(String, nullable=False)
    referenced_id: Mapped[str] = mapped_column(String, nullable=False)


class ReferenceDAO:
    def __init__(self, sql_db: SqlDB):
        self._db = sql_db

    def add(self, reference: Reference) -> None:
        self.add_many([reference])

    def add_many(self, references: list[Reference]) -> None:
        """Bulk insert — a single save() (e.g. a CComponent with 6
        PComponents) can establish several references at once; this
        writes them in one transaction rather than one session per
        reference."""
        if not references:
            return
        rows = [
            ReferenceRow(
                referencing_type=ref.referencing_type.value,
                referencing_id=ref.referencing_id,
                referenced_type=ref.referenced_type.value,
                referenced_id=ref.referenced_id,
            )
            for ref in references
        ]
        with self._db.session() as session:
            session.add_all(rows)

    def remove_all_from(
        self, referencing_type: ReferenceableType, referencing_id: str
    ) -> None:
        """Delete every reference where referencing_id is the source.
        Called before re-establishing an object's outgoing references
        after an in-place edit (refcount == 0 case), so the old set is
        replaced rather than appended to, and before deleting an
        object outright, so no dangling forward-reference rows are
        left pointing from an id that no longer exists."""
        with self._db.session() as session:
            session.query(ReferenceRow).filter(
                ReferenceRow.referencing_type == referencing_type.value,
                ReferenceRow.referencing_id == referencing_id,
            ).delete()

    def count_referencing(
        self, referenced_type: ReferenceableType, referenced_id: str
    ) -> int:
        """The actual refcount check: how many other objects currently
        reference this one. 0 means an in-place edit/delete is legal;
        >0 means an edit must mint a new version instead."""
        with self._db.session() as session:
            return (
                session.query(ReferenceRow)
                .filter(
                    ReferenceRow.referenced_type == referenced_type.value,
                    ReferenceRow.referenced_id == referenced_id,
                )
                .count()
            )

    def referencing(
        self, referenced_type: ReferenceableType, referenced_id: str
    ) -> list[tuple[ReferenceableType, str]]:
        """Reverse direction — who points AT this object. Used by the
        refcount gate and by anything surfacing 'used in N places' to
        a user before they edit/delete something."""
        with self._db.session() as session:
            rows = (
                session.query(ReferenceRow)
                .filter(
                    ReferenceRow.referenced_type == referenced_type.value,
                    ReferenceRow.referenced_id == referenced_id,
                )
                .all()
            )
            return [
                (ReferenceableType(row.referencing_type), row.referencing_id)
                for row in rows
            ]

    def referenced_by(
        self, referencing_type: ReferenceableType, referencing_id: str
    ) -> list[tuple[ReferenceableType, str]]:
        """Forward direction — what does this object point AT. This is
        the edge ComponentService walks outward from a root id to
        hydrate a full model (e.g. get_ccomponent()) without a
        separate materialized closure table."""
        with self._db.session() as session:
            rows = (
                session.query(ReferenceRow)
                .filter(
                    ReferenceRow.referencing_type == referencing_type.value,
                    ReferenceRow.referencing_id == referencing_id,
                )
                .all()
            )
            return [
                (ReferenceableType(row.referenced_type), row.referenced_id)
                for row in rows
            ]
