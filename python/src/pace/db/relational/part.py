"""
pace/db/relational/part.py

PartRow (the `parts` table) and PartDAO.

geometry and material are real foreign-key columns as well as being in
`data`: a Part's whole content is those two references, so
normalizing them gives database-level referential integrity — a row
cannot point at a missing Geometry/Material, and a referenced
Geometry/Material cannot be deleted.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.ids import PartID
from pace.core.placeable import Part
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import (
    GEOMETRIES_TABLE_NAME,
    MATERIALS_TABLE_NAME,
    PARTS_TABLE_NAME,
)
from pace.db.relational.versioned_dao import VersionedDAO
from pace.db.relational.versioned_row import VersionedRow


class PartRow(VersionedRow):
    __tablename__ = PARTS_TABLE_NAME

    geometry: Mapped[str] = mapped_column(
        ForeignKey(f"{GEOMETRIES_TABLE_NAME}.id"), nullable=False
    )
    material: Mapped[str] = mapped_column(
        ForeignKey(f"{MATERIALS_TABLE_NAME}.id"), nullable=False
    )


class PartDAO(VersionedDAO[PartRow, Part, PartID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, PartRow, Part.from_dict)

    def _extra_columns(self, model: Part, data: dict) -> dict[str, Any]:
        return {"geometry": model.geometry, "material": model.material}
