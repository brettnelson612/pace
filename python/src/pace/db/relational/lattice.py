"""
pace/db/relational/lattice.py

LatticeRow (the `lattices` table) and LatticeDAO. Lattice is
polymorphic (RectLattice/HexLattice), so rows carry a `type` column and
are rebuilt via lattice_from_dict().
"""

from __future__ import annotations

from pace.core.ids import LatticeID
from pace.core.lattice import Lattice, lattice_from_dict
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import LATTICES_TABLE_NAME
from pace.db.relational.versioned_dao import PolymorphicVersionedDAO
from pace.db.relational.versioned_row import PolymorphicVersionedRow


class LatticeRow(PolymorphicVersionedRow):
    __tablename__ = LATTICES_TABLE_NAME


class LatticeDAO(PolymorphicVersionedDAO[LatticeRow, Lattice, LatticeID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, LatticeRow, lattice_from_dict)
