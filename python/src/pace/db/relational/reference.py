"""
pace/db/relational/reference.py

ReferenceRow — a generic index of "what references what" across the
whole registry (this is what makes refcount checks and hydration
cheap, indexed queries instead of full scans). Maintained by
ComponentService on every create/delete, not derived automatically —
the database can't see into a JSON blob to enforce it on its own.

Each row is one directed edge: source (the object holding the
pointer) -> target (the object being pointed at).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Index, String, UniqueConstraint, or_
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.reference_types import ReferenceableType
from pace.db.relational.base import Base
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import REFERENCES_TABLE_NAME


@dataclass(frozen=True, kw_only=True)
class Reference:
    """One directed edge: source points at target."""

    source_type: ReferenceableType
    source_id: str
    target_type: ReferenceableType
    target_id: str


class ReferenceRow(Base):
    __tablename__ = REFERENCES_TABLE_NAME
    __table_args__ = (
        Index("ix_references_source", "source_type", "source_id"),
        Index("ix_references_target", "target_type", "target_id"),
        UniqueConstraint(
            "source_type",
            "source_id",
            "target_type",
            "target_id",
            name="uniq_references_edge",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    target_type: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)


class ReferenceDAO:
    def __init__(self, sql_db: SqlDB):
        self._db = sql_db

    def add(self, reference: Reference) -> None:
        self.add_many([reference])

    def add_many(self, references: list[Reference]) -> None:
        """Bulk insert — one save() can establish several edges at
        once (e.g. a CComponent's several ComponentPlacement's).

        Callers must dedupe to unique (source, target) pairs first —
        this writes an edge per occurrence in `references`, not per
        placement in the caller's own object"""
        if not references:
            return
        rows = [
            ReferenceRow(
                source_type=ref.source_type.value,
                source_id=ref.source_id,
                target_type=ref.target_type.value,
                target_id=ref.target_id,
            )
            for ref in references
        ]
        with self._db.session() as session:
            session.add_all(rows)

    def remove_outgoing(self, source_type: ReferenceableType, source_id: str) -> None:
        """Delete every edge sourced from this node — used before
        re-establishing an object's edges after an in-place edit (so
        the old set is replaced, not appended to), and before
        deleting an object outright (so no edge is left pointing from
        an id that no longer exists)."""
        with self._db.session() as session:
            session.query(ReferenceRow).filter(
                ReferenceRow.source_type == source_type.value,
                ReferenceRow.source_id == source_id,
            ).delete()

    def count_incoming(self, target_type: ReferenceableType, target_id: str) -> int:
        """The refcount check: how many other objects currently
        reference this one. 0 -> in-place edit/delete is legal; >0 ->
        an edit must mint a new version instead."""
        with self._db.session() as session:
            return (
                session.query(ReferenceRow)
                .filter(
                    ReferenceRow.target_type == target_type.value,
                    ReferenceRow.target_id == target_id,
                )
                .count()
            )

    def incoming(
        self, target_type: ReferenceableType, target_id: str
    ) -> list[tuple[ReferenceableType, str]]:
        """Who points AT this node — the refcount gate, and anything
        surfacing 'used in N places' to a user."""
        with self._db.session() as session:
            rows = (
                session.query(ReferenceRow)
                .filter(
                    ReferenceRow.target_type == target_type.value,
                    ReferenceRow.target_id == target_id,
                )
                .all()
            )
            return [(ReferenceableType(row.source_type), row.source_id) for row in rows]

    def outgoing(
        self, source_type: ReferenceableType, source_id: str
    ) -> list[tuple[ReferenceableType, str]]:
        """What this node points AT — the edge hydration walks
        outward from a root id."""
        with self._db.session() as session:
            rows = (
                session.query(ReferenceRow)
                .filter(
                    ReferenceRow.source_type == source_type.value,
                    ReferenceRow.source_id == source_id,
                )
                .all()
            )
            return [(ReferenceableType(row.target_type), row.target_id) for row in rows]

    def outgoing_many(
        self, sources: list[tuple[ReferenceableType, str]]
    ) -> list[tuple[ReferenceableType, str]]:
        """Batched outgoing() — the edges of several source nodes
        (possibly different types) in one query."""
        if not sources:
            return []

        by_type: dict[str, list[str]] = {}
        for source_type, source_id in sources:
            by_type.setdefault(source_type.value, []).append(source_id)

        with self._db.session() as session:
            conditions = [
                (ReferenceRow.source_type == type_value)
                & (ReferenceRow.source_id.in_(ids))
                for type_value, ids in by_type.items()
            ]
            rows = session.query(ReferenceRow).filter(or_(*conditions)).all()
            return [(ReferenceableType(row.target_type), row.target_id) for row in rows]
