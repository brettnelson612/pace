from pace.db.relational import (
    CComponentDAO,
    GeometryDAO,
    LComponentDAO,
    MaterialDAO,
)
from pace.db.relational.sql_db import SqlDB


class RegistryDB:
    def __init__(self, sql_db: SqlDB):
        self.geometries = GeometryDAO(sql_db)
        self.materials = MaterialDAO(sql_db)
        self.lcomponents = LComponentDAO(sql_db)
        self.ccomponents = CComponentDAO(sql_db)
