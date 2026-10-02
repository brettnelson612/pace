"""
pace/db/relational/material.py

MaterialRow (the `materials` table) and MaterialDAO. Material is
polymorphic, so rows carry a `type` column and are rebuilt via
material_from_dict().
"""

from __future__ import annotations

from pace.core.ids import MaterialID
from pace.core.material import Material, material_from_dict
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import MATERIALS_TABLE_NAME
from pace.db.relational.versioned_dao import PolymorphicVersionedDAO
from pace.db.relational.versioned_row import PolymorphicVersionedRow


class MaterialRow(PolymorphicVersionedRow):
    __tablename__ = MATERIALS_TABLE_NAME


class MaterialDAO(PolymorphicVersionedDAO[MaterialRow, Material, MaterialID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, MaterialRow, material_from_dict)
