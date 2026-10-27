"""
pace/db/relational/enclosure.py

EnclosureRow (the `enclosures` table) and EnclosureDAO.
"""

from __future__ import annotations

from pace.core.enclosure import Enclosure
from pace.core.ids import EnclosureID
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import ENCLOSURES_TABLE_NAME
from pace.db.relational.versioned_dao import VersionedDAO
from pace.db.relational.versioned_row import VersionedRow


class EnclosureRow(VersionedRow):
    __tablename__ = ENCLOSURES_TABLE_NAME


class EnclosureDAO(VersionedDAO[EnclosureRow, Enclosure, EnclosureID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, EnclosureRow, Enclosure.from_dict)
