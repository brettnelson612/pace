"""
pace/db/pace_db.py

PaceDB — the top-level composition root. Owns the one shared SqlDB
connection and constructs every domain-specific DB from it, so nothing
downstream ever builds its own separate engine pointed at the same
database.

Currently exposes only `.registry` (RegistryDB) — `.reactor_twins`,
`.surrogates`, `.gt`, and `.data` are deliberately not stubbed in yet:
each needs its own DAOs (and, for the hybrid ones, an ObjectStore)
before there's anything real to compose. They get added here as those
land, not before.
"""

from __future__ import annotations

from pace.db.registry_db import RegistryDB
from pace.db.relational.sql_db import SqlDB


class PaceDB:
    def __init__(self, db_url: str = "sqlite:///pace.db", echo: bool = False):
        self._sql_db = SqlDB(db_url=db_url, echo=echo)
        self.registry = RegistryDB(self._sql_db)

        # Not yet built — no domain model exists for these areas.
        self.reactor_twins_db = None
        self.surrogates = None
        self.data = None

    def create_all(self) -> None:
        """Create every table that doesn't exist yet. Dev/v1
        convenience only — see SqlDB.create_all()."""
        self._sql_db.create_all()
