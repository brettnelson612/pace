"""
pace/db/relational/reference.py

ReferenceRow — a generic index of "what references what" across the
whole registry (this is what makes refcount checks and hydration
cheap, indexed queries instead of full scans). Maintained by
RegistryService on every register, not derived automatically — the
database can't see into a JSON blob to enforce it on its own.

Reads open their own short session. Writes take the caller's Session,
so an object and its edges are written in one transaction.

Each row is one directed edge: source (the object holding the
pointer) -> target (the object being pointed at).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from sqlalchemy import Index, String, UniqueConstraint, and_, func, or_, select
from sqlalchemy.orm import Mapped, Session, mapped_column

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

    def add(self, session: Session, reference: Reference) -> None:
        self.add_many(session, [reference])

    def add_many(self, session: Session, references: Collection[Reference]) -> None:
        """Insert one edge per element of `references`. Callers pass
        unique (source, target) pairs — a repeated pair violates
        uniq_references_edge."""
        session.add_all(
            ReferenceRow(
                source_type=reference.source_type.value,
                source_id=reference.source_id,
                target_type=reference.target_type.value,
                target_id=reference.target_id,
            )
            for reference in references
        )
        session.flush()

    def remove_outgoing(
        self, session: Session, source_type: ReferenceableType, source_id: str
    ) -> None:
        """Delete every edge sourced from this node — before
        re-recording an object's edges after an in-place edit, or
        before deleting the object."""
        for row in session.scalars(
            select(ReferenceRow).where(
                ReferenceRow.source_type == source_type.value,
                ReferenceRow.source_id == source_id,
            )
        ):
            session.delete(row)

    def count_incoming(self, target_type: ReferenceableType, target_id: str) -> int:
        """How many objects currently reference this one."""
        with self._db.session() as session:
            return (
                session.scalar(
                    select(func.count())
                    .select_from(ReferenceRow)
                    .where(
                        ReferenceRow.target_type == target_type.value,
                        ReferenceRow.target_id == target_id,
                    )
                )
                or 0
            )

    def incoming(
        self, target_type: ReferenceableType, target_id: str
    ) -> list[tuple[ReferenceableType, str]]:
        """Who points AT this node."""
        with self._db.session() as session:
            rows = session.scalars(
                select(ReferenceRow).where(
                    ReferenceRow.target_type == target_type.value,
                    ReferenceRow.target_id == target_id,
                )
            ).all()
            return [(ReferenceableType(row.source_type), row.source_id) for row in rows]

    def outgoing(
        self, source_type: ReferenceableType, source_id: str
    ) -> list[tuple[ReferenceableType, str]]:
        """What this node points AT."""
        return self.outgoing_many([(source_type, source_id)])

    def outgoing_many(
        self, sources: Collection[tuple[ReferenceableType, str]]
    ) -> list[tuple[ReferenceableType, str]]:
        """The edges of several source nodes (possibly of different
        types) in one query."""
        if not sources:
            return []

        ids_by_type: dict[str, list[str]] = {}
        for source_type, source_id in sources:
            ids_by_type.setdefault(source_type.value, []).append(source_id)

        conditions = [
            and_(
                ReferenceRow.source_type == type_value, ReferenceRow.source_id.in_(ids)
            )
            for type_value, ids in ids_by_type.items()
        ]
        with self._db.session() as session:
            rows = session.scalars(select(ReferenceRow).where(or_(*conditions))).all()
            return [(ReferenceableType(row.target_type), row.target_id) for row in rows]
