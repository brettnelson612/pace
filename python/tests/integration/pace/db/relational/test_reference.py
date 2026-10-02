"""
tests/integration/pace/db/relational/test_reference.py

ReferenceDAO tests against a real DB — writes run inside a
RegistryDB.transaction(). The uniqueness constraint specifically can't
be exercised without a DB (it's enforced by SQLite, as a backstop to
reference extraction's own deduplication).
"""

from __future__ import annotations

import pytest
from pace.core.reference_types import ReferenceableType
from pace.db.pace_db import PaceDB
from pace.db.relational.reference import Reference
from sqlalchemy.exc import IntegrityError


def _add(pace_db: PaceDB, references: list[Reference]) -> None:
    with pace_db.registry.transaction() as session:
        pace_db.registry.references.add_many(session, references)


def _edge(source_id: str = "pin-1", target_id: str = "pellet-1") -> Reference:
    return Reference(
        source_type=ReferenceableType.CCOMPONENT,
        source_id=source_id,
        target_type=ReferenceableType.LCOMPONENT,
        target_id=target_id,
    )


def test_add_many_empty_list_is_a_no_op(pace_db: PaceDB):
    _add(pace_db, [])
    assert (
        pace_db.registry.references.count_incoming(
            ReferenceableType.LCOMPONENT, "pellet-1"
        )
        == 0
    )


def test_add_many_then_incoming_and_outgoing(pace_db: PaceDB):
    edge = _edge()
    _add(pace_db, [edge])

    incoming = pace_db.registry.references.incoming(
        ReferenceableType.LCOMPONENT, "pellet-1"
    )
    assert incoming == [(ReferenceableType.CCOMPONENT, "pin-1")]

    outgoing = pace_db.registry.references.outgoing(
        ReferenceableType.CCOMPONENT, "pin-1"
    )
    assert outgoing == [(ReferenceableType.LCOMPONENT, "pellet-1")]


def test_count_incoming(pace_db: PaceDB):
    _add(pace_db, [_edge(source_id="pin-1"), _edge(source_id="pin-2")])
    assert (
        pace_db.registry.references.count_incoming(
            ReferenceableType.LCOMPONENT, "pellet-1"
        )
        == 2
    )


def test_duplicate_edge_rejected_by_unique_constraint(pace_db: PaceDB):
    """The DB-level backstop: an exact duplicate (source, target) edge
    must be rejected, since a caller-side dedup bug is exactly what
    this constraint exists to catch."""
    edge = _edge()
    _add(pace_db, [edge])
    with pytest.raises(IntegrityError):
        _add(pace_db, [edge])


def test_remove_outgoing_clears_only_that_sources_edges(pace_db: PaceDB):
    _add(
        pace_db,
        [
            _edge(source_id="pin-1", target_id="pellet-1"),
            _edge(source_id="pin-1", target_id="pellet-2"),
            _edge(source_id="pin-2", target_id="pellet-1"),
        ],
    )
    with pace_db.registry.transaction() as session:
        pace_db.registry.references.remove_outgoing(
            session, ReferenceableType.CCOMPONENT, "pin-1"
        )

    assert (
        pace_db.registry.references.outgoing(ReferenceableType.CCOMPONENT, "pin-1")
        == []
    )
    # pin-2's edge must be untouched
    assert pace_db.registry.references.outgoing(
        ReferenceableType.CCOMPONENT, "pin-2"
    ) == [(ReferenceableType.LCOMPONENT, "pellet-1")]


def test_outgoing_many_batches_across_heterogeneous_source_types(pace_db: PaceDB):
    """outgoing_many() must handle a frontier mixing different source
    types in one call — this is what makes the hydration walk cheap
    across CComponent AND LComponent frontiers together."""
    _add(
        pace_db,
        [
            _edge(source_id="pin-1", target_id="pellet-1"),
            Reference(
                source_type=ReferenceableType.LCOMPONENT,
                source_id="pellet-1",
                target_type=ReferenceableType.GEOMETRY,
                target_id="geo-1",
            ),
        ],
    )
    edges = pace_db.registry.references.outgoing_many(
        [
            (ReferenceableType.CCOMPONENT, "pin-1"),
            (ReferenceableType.LCOMPONENT, "pellet-1"),
        ]
    )
    assert set(edges) == {
        (ReferenceableType.LCOMPONENT, "pellet-1"),
        (ReferenceableType.GEOMETRY, "geo-1"),
    }


def test_outgoing_many_empty_list_returns_empty(pace_db: PaceDB):
    assert pace_db.registry.references.outgoing_many([]) == []


def test_add_writes_a_single_edge(pace_db: PaceDB):
    with pace_db.registry.transaction() as session:
        pace_db.registry.references.add(session, _edge())
    assert pace_db.registry.references.outgoing(
        ReferenceableType.CCOMPONENT, "pin-1"
    ) == [(ReferenceableType.LCOMPONENT, "pellet-1")]


def test_edges_from_a_failed_transaction_are_not_kept(pace_db: PaceDB):
    with pytest.raises(RuntimeError):
        with pace_db.registry.transaction() as session:
            pace_db.registry.references.add_many(session, [_edge()])
            raise RuntimeError("abort")
    assert (
        pace_db.registry.references.count_incoming(
            ReferenceableType.LCOMPONENT, "pellet-1"
        )
        == 0
    )
