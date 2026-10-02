"""
pace/db/relational/ccomponent.py

CComponentRow (the `ccomponents` table) and CComponentDAO.
"""

from __future__ import annotations

from pace.core.component import CComponent
from pace.core.ids import CComponentID
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import CCOMPONENTS_TABLE_NAME
from pace.db.relational.versioned_dao import VersionedDAO
from pace.db.relational.versioned_row import VersionedRow


class CComponentRow(VersionedRow):
    __tablename__ = CCOMPONENTS_TABLE_NAME


class CComponentDAO(VersionedDAO[CComponentRow, CComponent, CComponentID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, CComponentRow, CComponent.from_dict)
