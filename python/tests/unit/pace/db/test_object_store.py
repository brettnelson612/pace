"""
tests/unit/pace/db/test_object_store.py

Covers: ObjectStore put/get/delete/list against a temporary directory —
nested keys, overwriting a key, missing keys, listing by prefix, and
rejecting keys that resolve outside the store root.
"""

from __future__ import annotations

import pytest
from pace.db.object_store.object_store import ObjectStore


@pytest.fixture
def store(tmp_path) -> ObjectStore:
    return ObjectStore(root=tmp_path / "store")


def test_root_is_created(tmp_path):
    ObjectStore(root=tmp_path / "new_root")
    assert (tmp_path / "new_root").is_dir()


def test_put_then_get_nested_key(store: ObjectStore):
    store.put("simulation_jobs/abc/checkpoint-000.npz", b"\x00\x01")
    assert store.get("simulation_jobs/abc/checkpoint-000.npz") == b"\x00\x01"


def test_put_overwrites_an_existing_key(store: ObjectStore):
    store.put("a.bin", b"first")
    store.put("a.bin", b"second")
    assert store.get("a.bin") == b"second"


def test_get_missing_key_returns_none(store: ObjectStore):
    assert store.get("missing.bin") is None


def test_delete_removes_the_key(store: ObjectStore):
    store.put("a.bin", b"data")
    store.delete("a.bin")
    assert store.get("a.bin") is None


def test_delete_missing_key_is_a_no_op(store: ObjectStore):
    store.delete("missing.bin")


def test_list_everything_sorted(store: ObjectStore):
    store.put("b/2.bin", b"")
    store.put("a/1.bin", b"")
    store.put("top.bin", b"")
    assert store.list() == ["a/1.bin", "b/2.bin", "top.bin"]


def test_list_by_directory_prefix(store: ObjectStore):
    store.put("runs/1/a.bin", b"")
    store.put("runs/1/b.bin", b"")
    store.put("runs/2/a.bin", b"")
    assert store.list("runs/1") == ["runs/1/a.bin", "runs/1/b.bin"]


def test_list_with_a_key_as_prefix_returns_that_key(store: ObjectStore):
    store.put("runs/1/a.bin", b"")
    assert store.list("runs/1/a.bin") == ["runs/1/a.bin"]


def test_list_missing_prefix_is_empty(store: ObjectStore):
    assert store.list("nothing/here") == []


@pytest.mark.parametrize("key", ["../escape.bin", "a/../../escape.bin"])
def test_keys_outside_the_root_are_rejected(store: ObjectStore, key):
    with pytest.raises(ValueError):
        store.put(key, b"data")
