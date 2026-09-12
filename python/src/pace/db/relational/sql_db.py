"""
pace/db/relational/sql_db.py

SqlDB — owns the actual relational database engine and connection
lifecycle; every repository receives a SqlDB instance rather than
constructing its own connection.

Named distinctly from the higher-level PaceDB composition root
(PaceDB.registry / .gt / .data / etc., per the datastore design) —
this class only knows about the raw SQLAlchemy engine/session, one
layer below Registry and well below PaceDB itself.

v1 targets SQLite (a single file on disk, no server process) — matches
the earlier decision to avoid standing up and operating a Postgres
server for a solo build. The connection string is the only thing that
changes if/when this migrates to Postgres later; SQLAlchemy abstracts
the rest.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from pace.db.relational.base import Base


class SqlDB:
    def __init__(self, db_url: str = "sqlite:///pace.db", echo: bool = False):
        # check_same_thread=False: SQLite's default forbids using a
        # connection from a different thread than it was created on;
        # relevant the moment this is used from an async FastAPI app.
        # Real concurrent-write safety at that point is a separate
        # question from this flag — worth revisiting once the API
        # layer exists, not a v1 blocker.
        connect_args = (
            {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        )
        self._engine = create_engine(db_url, echo=echo, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Create every table that doesn't exist yet.

        Dev/v1 convenience only — this has no concept of schema
        migration (altering an existing table when a model changes).
        Alembic is the standard tool for that; worth adopting before
        this schema needs to evolve against real, populated data rather
        than being freely recreated.
        """
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """One transaction per `with` block: commits on clean exit,
        rolls back on any exception, always closes. Repositories should
        never hold a session open across multiple calls."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
