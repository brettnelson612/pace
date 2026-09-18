from pace.db.relational import (
    CComponentDAO,
    GeometryDAO,
    LComponentDAO,
    MaterialDAO,
    ReferenceDAO,
)
from pace.db.relational.sql_db import SqlDB


class RegistryDB:
    """Facade over the four versioned-aggregate DAOs plus the
    reference index — everything ComponentService needs to persist
    and cross-check Geometry/Material/LComponent/CComponent, all
    sharing one SqlDB connection."""

    def __init__(self, sql_db: SqlDB):
        self.geometries = GeometryDAO(sql_db)
        self.materials = MaterialDAO(sql_db)
        self.lcomponents = LComponentDAO(sql_db)
        self.ccomponents = CComponentDAO(sql_db)
        self.references = ReferenceDAO(sql_db)
