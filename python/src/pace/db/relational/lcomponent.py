"""
pace/db/relational/lcomponent.py

LComponentRow (the `lcomponents` table) and LComponentDAO.

geometry and material are real foreign-key columns as well as being in
`data`: an LComponent's whole content is those two references, so
normalizing them gives database-level referential integrity — a row
cannot point at a missing Geometry/Material, and a referenced
Geometry/Material cannot be deleted.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.component import LComponent
from pace.core.ids import LComponentID
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import (
    GEOMETRIES_TABLE_NAME,
    LCOMPONENTS_TABLE_NAME,
    MATERIALS_TABLE_NAME,
)
from pace.db.relational.versioned_dao import VersionedDAO
from pace.db.relational.versioned_row import VersionedRow


class LComponentRow(VersionedRow):
    __tablename__ = LCOMPONENTS_TABLE_NAME

    geometry: Mapped[str] = mapped_column(
        ForeignKey(f"{GEOMETRIES_TABLE_NAME}.id"), nullable=False
    )
    material: Mapped[str] = mapped_column(
        ForeignKey(f"{MATERIALS_TABLE_NAME}.id"), nullable=False
    )


class LComponentDAO(VersionedDAO[LComponentRow, LComponent, LComponentID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, LComponentRow, LComponent.from_dict)

    def _extra_columns(self, model: LComponent, data: dict) -> dict[str, Any]:
        return {"geometry": model.geometry, "material": model.material}
