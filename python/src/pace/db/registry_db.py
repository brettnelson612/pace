"""
pace/db/registry_db.py

RegistryDB — the versioned-aggregate DAOs plus the reference index,
all sharing one SqlDB connection. Everything RegistryService needs to
persist and cross-check the registry.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from pace.core.reference_types import ReferenceableType
from pace.db.relational import (
    CComponentDAO,
    GeometryDAO,
    LatticeDAO,
    LComponentDAO,
    MaterialDAO,
    ReactorBlueprintDAO,
    ReferenceDAO,
    VersionedDAO,
)
from pace.db.relational.sql_db import SqlDB


class RegistryDB:
    def __init__(self, sql_db: SqlDB):
        self._sql_db = sql_db
        self.geometries = GeometryDAO(sql_db)
        self.materials = MaterialDAO(sql_db)
        self.lcomponents = LComponentDAO(sql_db)
        self.ccomponents = CComponentDAO(sql_db)
        self.lattices = LatticeDAO(sql_db)
        self.reactor_blueprints = ReactorBlueprintDAO(sql_db)
        self.references = ReferenceDAO(sql_db)

        self._dao_by_type: dict[ReferenceableType, VersionedDAO[Any, Any, Any]] = {
            ReferenceableType.GEOMETRY: self.geometries,
            ReferenceableType.MATERIAL: self.materials,
            ReferenceableType.LCOMPONENT: self.lcomponents,
            ReferenceableType.CCOMPONENT: self.ccomponents,
            ReferenceableType.LATTICE: self.lattices,
            ReferenceableType.REACTOR_BLUEPRINT: self.reactor_blueprints,
        }

    def dao_for(
        self, referenceable_type: ReferenceableType
    ) -> VersionedDAO[Any, Any, Any]:
        """The DAO storing objects of this type."""
        return self._dao_by_type[referenceable_type]

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """One transaction for the DAO writes made inside the block:
        commits on clean exit, rolls back on any exception."""
        with self._sql_db.session() as session:
            yield session
