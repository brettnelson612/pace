"""
pace/db/relational/reference.py

ReferenceRow — a generic reverse-index of "what references what"
across the whole registry. This is what makes reference-count checks
(can this be deleted?) and "what points at this version" lookups
cheap, indexed queries instead of full scans.

Deliberately maintained by the repository/service layer on every
create/delete that establishes or removes a reference — NOT derived
automatically by the database. A reference can live in a normalized
FK column (LComponentRow.geometry_version) or inside a JSON blob
(CComponentRow.data.components), and the database can't see
into the latter to enforce this on its own.
"""

from __future__ import annotations

from sqlalchemy import Index, String
from sqlalchemy.orm import Mapped, mapped_column

from pace.db.relational.base import Base


class ReferenceRow(Base):
    __tablename__ = "references"
    __table_args__ = (
        Index("ix_references_referencing", "referencing_type", "referencing_id"),
        Index("ix_references_referenced", "referenced_type", "referenced_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    referencing_type: Mapped[str] = mapped_column(String, nullable=False)
    referencing_id: Mapped[str] = mapped_column(String, nullable=False)
    referenced_type: Mapped[str] = mapped_column(String, nullable=False)
    referenced_id: Mapped[str] = mapped_column(String, nullable=False)
