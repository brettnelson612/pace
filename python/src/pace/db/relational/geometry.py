"""
pace/db/relational/geometry.py

GeometryRow (the `geometries` table) and GeometryDAO. Geometry is
polymorphic, so rows carry a `type` column and are rebuilt via
geometry_from_dict().
"""

from __future__ import annotations

from pace.core.geometry import Geometry, geometry_from_dict
from pace.core.ids import GeometryID
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import GEOMETRIES_TABLE_NAME
from pace.db.relational.versioned_dao import PolymorphicVersionedDAO
from pace.db.relational.versioned_row import PolymorphicVersionedRow


class GeometryRow(PolymorphicVersionedRow):
    __tablename__ = GEOMETRIES_TABLE_NAME


class GeometryDAO(PolymorphicVersionedDAO[GeometryRow, Geometry, GeometryID]):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, GeometryRow, geometry_from_dict)
