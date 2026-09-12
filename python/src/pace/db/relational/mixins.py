"""
pace/db/relational/mixins.py

VersionMixin — the shared column shape for every *Version table
(GeometryRow, MaterialRow, CComponentRow).

Follows the JSONB-hybrid pattern: normalize only what needs
querying/joining (family_name, version_label, lineage columns), store
the rest of each domain object's data via its own to_dict()/from_dict()
in the `data` JSON column.

family_name is NOT unique per row — every version in a family shares
it. Rejecting a duplicate family_name (someone trying to create a
brand-new family whose name already exists) is application logic
(ComponentService), not a database constraint — the column legitimately
repeats across many rows, so a UNIQUE constraint on it would be wrong.

derived_from is intentionally NOT declared here — it's a
self-referential foreign key, and the target table name differs per
concrete table, so each row model declares its own.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column


class VersionMixin:
    id: Mapped[str] = mapped_column(String, primary_key=True)
    family_name: Mapped[str] = mapped_column(String, nullable=False)
    version_label: Mapped[str] = mapped_column(String, nullable=False)
    gt_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    user_edit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PolymorphicVersionMixin(VersionMixin):
    type: Mapped[str] = mapped_column(String, nullable=False)
