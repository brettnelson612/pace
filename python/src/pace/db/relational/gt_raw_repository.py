"""
pace/db/repositories/gt_raw_repository.py

GTRawRepository — the real hybrid case, as opposed to the JSONB-in-a-
single-row hybrid pattern the other repositories use (normalize a few
columns, stash the rest of one domain object in that same row's `data`
column). This repository straddles two separate storage systems:

  - the relational store (SqlDB), holding one small, queryable index
    row per raw object — which GT run it belongs to, what kind of
    artifact it is, and where the actual bytes live; and
  - the object store (ObjectStore), holding the raw bytes themselves
    (solver checkpoints, large raw output) that have no business being
    stuffed into a SQL column.

Corresponds to PaceDB.gt.raw.metadata / PaceDB.gt.raw.checkpoints in
the datastore design's accessor tree — "metadata" and "checkpoints" are
both raw-object *kinds* held here, distinguished by the `kind` column,
not two different repositories.

Known gap: there is no GTRun domain object in pace.core yet, so
GTRawObjectRow is defined locally rather than living in relational/
alongside the other row models — promote it there once a GT domain
module exists to import types from. Because it's defined here rather
than being imported for its side effect by sql_db.py (unlike the five
relational/ row modules), this module must be imported at least once
before SqlDB.create_all() runs, or its table won't be created.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from pace.core.ids import GTRunID
from pace.db.object_store.object_store import ObjectStore
from pace.db.relational.base import Base
from pace.db.relational.sql_db import SqlDB


class GTRawObjectRow(Base):
    __tablename__ = "gt_raw_objects"
    __table_args__ = (Index("ix_gt_raw_objects_gt_run_id", "gt_run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gt_run_id: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    object_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class GTRawRepository:
    def __init__(self, db: SqlDB, object_store: ObjectStore):
        self._db = db
        self._object_store = object_store

    def put(self, gt_run_id: GTRunID, kind: str, name: str, data: bytes) -> str:
        """Store a raw object's bytes in the object store and index it
        in the relational store. Returns the object key so the caller
        can hand it back to get()/delete() directly, without needing to
        re-derive it."""
        object_key = self._object_key(gt_run_id, kind, name)
        self._object_store.put(object_key, data)

        row = GTRawObjectRow(gt_run_id=gt_run_id, kind=kind, object_key=object_key)
        with self._db.session() as session:
            session.merge(row)
        return object_key

    def get(self, object_key: str) -> bytes | None:
        return self._object_store.get(object_key)

    def delete(self, object_key: str) -> None:
        """Removes both the indexed row and the underlying bytes —
        callers never have to remember to clean up both sides."""
        with self._db.session() as session:
            row = (
                session.query(GTRawObjectRow)
                .filter(GTRawObjectRow.object_key == object_key)
                .first()
            )
            if row is not None:
                session.delete(row)
        self._object_store.delete(object_key)

    def list_for_run(self, gt_run_id: GTRunID, kind: str | None = None) -> list[str]:
        """Object keys recorded for this GT run, optionally filtered to
        one kind (e.g. "checkpoint" vs "metadata") — the indexed query
        this repository exists to make possible, versus a full
        object-store prefix scan."""
        with self._db.session() as session:
            query = session.query(GTRawObjectRow).filter(
                GTRawObjectRow.gt_run_id == gt_run_id
            )
            if kind is not None:
                query = query.filter(GTRawObjectRow.kind == kind)
            return [row.object_key for row in query.all()]

    @staticmethod
    def _object_key(gt_run_id: GTRunID, kind: str, name: str) -> str:
        return f"gt_runs/{gt_run_id}/{kind}/{name}"
