"""
pace/db/relational/reactor_blueprint.py

ReactorBlueprintRow (the `reactor_blueprints` table) and
ReactorBlueprintDAO.
"""

from __future__ import annotations

from pace.core.ids import ReactorBlueprintID
from pace.core.reactor_blueprint import ReactorBlueprint
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.table_names import REACTOR_BLUEPRINTS_TABLE_NAME
from pace.db.relational.versioned_dao import VersionedDAO
from pace.db.relational.versioned_row import VersionedRow


class ReactorBlueprintRow(VersionedRow):
    __tablename__ = REACTOR_BLUEPRINTS_TABLE_NAME


class ReactorBlueprintDAO(
    VersionedDAO[ReactorBlueprintRow, ReactorBlueprint, ReactorBlueprintID]
):
    def __init__(self, sql_db: SqlDB):
        super().__init__(sql_db, ReactorBlueprintRow, ReactorBlueprint.from_dict)
