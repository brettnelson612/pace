"""
pace/db/pace_db.py

PaceDB — the composition root for all of PACE's persistence. Wires up
the underlying storage engines (SqlDB, ObjectStore) exactly once and
hands out the higher-level facades (Registry, ...) that everything else
depends on, rather than each part of the app constructing its own
connections.

Target accessor tree (per the datastore design — built out
incrementally, unbuilt branches are None until their domain model
exists):

    PaceDB.registry                    # Registry — see registry.py
    PaceDB.registry.geometries
    PaceDB.registry.materials
    PaceDB.registry.lcomponents
    PaceDB.registry.ccomponents
    PaceDB.reactor_twins                # not yet built — no domain model
    PaceDB.surrogates                   # not yet built — no domain model
    PaceDB.surrogates.models
    PaceDB.gt                           # GT-run persistence
    PaceDB.gt.gt_runs                   # not yet built — no GTRun domain model
    PaceDB.gt.gt_run_specs              # not yet built — no GTRun domain model
    PaceDB.gt.raw                       # GTRawRepository — see repositories/gt_raw_repository.py
    PaceDB.data                         # not yet built — no domain model
    PaceDB.data.datasets
    PaceDB.data.sample_specs
    PaceDB.data.charts
"""

from __future__ import annotations

from pace.db.object_store.object_store import ObjectStore
from pace.db.registry_db import RegistryDB
from pace.db.relational.sql_db import SqlDB


class PaceDB:
    """
    This class instantiates at maintains low-level data access connections
    and provides them to consuming DAOs.
    """

    def __init__(
        self,
        db_url: str = "sqlite:///pace.db",
        object_store_root: str = "pace_object_store",
    ):
        self._db = SqlDB(db_url)
        self._object_store = ObjectStore(object_store_root)

        self.registry = RegistryDB(self._db)

        # Not yet built — no domain model exists for these areas.
        self.reactor_twins_db = None
        self.surrogates = None
        self.data = None

    def create_all(self) -> None:
        """Create every relational table that doesn't exist yet. See
        SqlDB.create_all() for the dev/v1-only caveat (no migrations)."""
        self._db.create_all()


"""
class ReactorTwinsDB: ...
class SurrogatesDB: ...
class RegistryDB: ...
class GroundTruthDB: ...
class DataDB: ...
"""
