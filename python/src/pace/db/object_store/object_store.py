"""
pace/db/object_store/object_store.py

ObjectStore — a thin key/value client for large raw blobs that don't
belong in a relational row (GT run checkpoints, raw solver output).
Deliberately dumb: get/put/delete/list, keyed by an opaque string path,
no querying, no schema. Anything that needs to be queried or joined on
belongs in the relational store instead (see relational/); a repository
composing both (e.g. GTRawRepository) is the hybrid case that decides
what goes where.

v1 backs onto a plain local directory — one file per key, key segments
("gt_runs/abc123/checkpoint-000.npz") map to nested subdirectories.
Matches the same reasoning as SqlDB defaulting to SQLite: no server
process to stand up for a solo build. The only thing that changes if/
when this migrates to S3-compatible storage later is this class's
internals — callers only ever see get/put/delete/list.
"""

from __future__ import annotations

from pathlib import Path


class ObjectStore:
    def __init__(self, root: str | Path = "pace_object_store"):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._path_for(key)
        path.unlink(missing_ok=True)

    def list(self, prefix: str = "") -> list[str]:
        """Every key currently stored under `prefix`, as the same
        forward-slash key strings passed to put() — not filesystem
        paths."""
        base = self._path_for(prefix) if prefix else self._root
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        return sorted(
            str(path.relative_to(self._root).as_posix())
            for path in base.rglob("*")
            if path.is_file()
        )

    def _path_for(self, key: str) -> Path:
        """Resolve a key to an on-disk path, rejecting anything that
        would escape the store root (e.g. `../../etc/passwd`) — keys
        come from application code, not untrusted input, but this is a
        cheap guard against a mistaken key composition."""
        path = (self._root / key).resolve()
        if self._root.resolve() not in path.parents and path != self._root.resolve():
            raise ValueError(f"key {key!r} resolves outside the object store root")
        return path
