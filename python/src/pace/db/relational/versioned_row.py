"""
pace/db/relational/versioned_row.py

VersionedRow — the shared column shape for every versioned table, and
PolymorphicVersionedRow, which adds the `type` discriminator column for
tables holding several concrete subclasses (geometries, materials,
lattices).

Both are SQLAlchemy abstract bases: they produce no table of their own,
and every concrete row class inheriting from one gets these columns.

Follows the JSON-hybrid pattern: normalize only what needs
querying/joining (family_name, version_label, lineage columns) and
store the rest of each domain object via its own to_dict()/from_dict()
in the `data` JSON column.

family_name is indexed but NOT unique — every version in a family
shares it. Rejecting a new family whose name already exists is
RegistryService's job, not a database constraint.

derived_from is a self-referential foreign key to the concrete row's
own table.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from pace.db.relational.base import Base


class VersionedRow(Base):
    __abstract__ = True

    id: Mapped[str] = mapped_column(String, primary_key=True)
    family_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version_label: Mapped[str] = mapped_column(String, nullable=False)
    simulation_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    user_edit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    @declared_attr
    def derived_from(cls) -> Mapped[str | None]:
        return mapped_column(ForeignKey(f"{cls.__tablename__}.id"), nullable=True)


class PolymorphicVersionedRow(VersionedRow):
    __abstract__ = True

    type: Mapped[str] = mapped_column(String, nullable=False)
