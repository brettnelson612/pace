"""
pace/db/relational/versioned_dao.py

VersionedDAO — persistence for one versioned aggregate's table: pure
storage and lookup, no business rules. Deciding whether a write is
allowed (family uniqueness, lineage, refcounts, dangling references)
is RegistryService's job; the DAO only answers questions and performs
the writes it is told to.

Reads open their own short session. Writes take the caller's Session,
so RegistryService can put an object and its reference-index edges in
one transaction.

Each concrete DAO is a thin subclass that names its row class and how
to rebuild a domain object from a row's `data` column.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from pace.core.versioned import Versioned
from pace.db.relational.sql_db import SqlDB
from pace.db.relational.versioned_row import VersionedRow

RowT = TypeVar("RowT", bound=VersionedRow)
ModelT = TypeVar("ModelT", bound=Versioned)
IDT = TypeVar("IDT", bound=str)


class VersionedDAO(Generic[RowT, ModelT, IDT]):
    """Storage for one versioned table.

    - row_type: the concrete row class (one table).
    - to_model: rebuilds a domain object from a row's `data` dict —
          the class's from_dict(), or a dispatcher such as
          geometry_from_dict() for polymorphic tables.
    """

    def __init__(
        self,
        sql_db: SqlDB,
        row_type: type[RowT],
        to_model: Callable[[dict], ModelT],
    ):
        self._db = sql_db
        self._row_type = row_type
        self._to_model = to_model

    # -------------------------------------------------------------
    # Reads
    # -------------------------------------------------------------

    def exists(self, model_id: IDT) -> bool:
        with self._db.session() as session:
            return session.get(self._row_type, model_id) is not None

    def existing_ids(self, model_ids: Collection[IDT]) -> set[IDT]:
        """The subset of `model_ids` that has a row, in one query."""
        if not model_ids:
            return set()
        with self._db.session() as session:
            found = set(
                session.scalars(
                    select(self._row_type.id).where(self._row_type.id.in_(model_ids))
                ).all()
            )
        return {model_id for model_id in model_ids if model_id in found}

    def family_exists(self, family_name: str) -> bool:
        """Whether any version exists under this family_name."""
        with self._db.session() as session:
            row_id = session.scalars(
                select(self._row_type.id)
                .where(self._row_type.family_name == family_name)
                .limit(1)
            ).first()
            return row_id is not None

    def describe(self, model_id: IDT) -> dict[str, Any] | None:
        """Every column except `data` — identity, lineage, and any
        table-specific columns — without rebuilding the domain
        object."""
        with self._db.session() as session:
            row = session.get(self._row_type, model_id)
            if row is None:
                return None
            return {
                column.key: getattr(row, column.key)
                for column in self._row_type.__table__.columns
                if column.key != "data"
            }

    def get(self, model_id: IDT) -> ModelT | None:
        with self._db.session() as session:
            row = session.get(self._row_type, model_id)
            if row is None:
                return None
            return self._to_model(row.data)

    def get_many(self, model_ids: Collection[IDT]) -> dict[IDT, ModelT]:
        """The objects for every id in `model_ids` that has a row,
        keyed by id. Missing ids are absent from the result."""
        if not model_ids:
            return {}
        with self._db.session() as session:
            rows = session.scalars(
                select(self._row_type).where(self._row_type.id.in_(model_ids))
            ).all()
            models = [self._to_model(row.data) for row in rows]
        return {model.id: model for model in models}

    def versions_of_family(self, family_name: str) -> list[ModelT]:
        """Every version recorded under this family_name, oldest
        first."""
        with self._db.session() as session:
            rows = session.scalars(
                select(self._row_type)
                .where(self._row_type.family_name == family_name)
                .order_by(self._row_type.created_at)
            ).all()
            return [self._to_model(row.data) for row in rows]

    def latest_version_of_family(self, family_name: str) -> ModelT | None:
        """The most recently created version in this family. A display
        default only — never a substitute for explicit version
        selection where a user chooses which version to act on (e.g.
        starting a simulation job)."""
        with self._db.session() as session:
            row = session.scalars(
                select(self._row_type)
                .where(self._row_type.family_name == family_name)
                .order_by(self._row_type.created_at.desc())
                .limit(1)
            ).first()
            if row is None:
                return None
            return self._to_model(row.data)

    def children_of(self, model_id: IDT) -> list[ModelT]:
        """The versions whose derived_from is `model_id`."""
        with self._db.session() as session:
            rows = session.scalars(
                select(self._row_type).where(self._row_type.derived_from == model_id)
            ).all()
            return [self._to_model(row.data) for row in rows]

    # -------------------------------------------------------------
    # Writes — caller supplies the transaction
    # -------------------------------------------------------------

    def insert(self, session: Session, model: ModelT) -> None:
        """Add a new row for `model`. Never overwrites: an existing id
        raises sqlalchemy.exc.IntegrityError (from the primary key) at
        the flush below.

        model.validate() already ran at construction; it runs again
        here as a check at the persistence boundary, since a frozen
        dataclass can still be mutated via object.__setattr__.
        """
        model.validate()
        data = model.to_dict()
        row = self._row_type(
            id=model.id,
            family_name=model.family_name,
            version_label=model.version_label,
            derived_from=model.derived_from,
            simulation_job_id=model.simulation_job_id,
            user_edit=model.user_edit,
            data=data,
            **self._extra_columns(model, data),
        )
        session.add(row)
        session.flush()

    def delete(self, session: Session, model_id: IDT) -> None:
        """Remove the row if present. No refcount check — the caller
        must confirm nothing references `model_id` first."""
        row = session.get(self._row_type, model_id)
        if row is not None:
            session.delete(row)

    def _extra_columns(self, model: ModelT, data: dict) -> dict[str, Any]:
        """Values for any columns beyond VersionedRow's. None by
        default."""
        return {}


class PolymorphicVersionedDAO(VersionedDAO[RowT, ModelT, IDT]):
    """A VersionedDAO whose table holds several concrete subclasses,
    discriminated by the `type` column (copied from to_dict()'s
    "type" key)."""

    def _extra_columns(self, model: ModelT, data: dict) -> dict[str, Any]:
        return {"type": data["type"]}
