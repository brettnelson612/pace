"""
tests/unit/pace/core/test_ids.py

ids.py is purely static typing sugar (typing.NewType over str) — these
tests exist to pin down the runtime behavior the module's own docstring
promises: zero-cost wrapping, and calling on None returning None
unchanged (so from_dict() call sites never need to guard an Optional
field before wrapping it).
"""

from pace.core.ids import (
    CComponentID,
    GeometryID,
    GTRunID,
    LatticeID,
    LComponentID,
    MaterialID,
    ComponentPlacementID,
    ReactorBlueprintID,
)

ALL_ID_TYPES = [
    GeometryID,
    MaterialID,
    LComponentID,
    ComponentPlacementID,
    CComponentID,
    LatticeID,
    ReactorBlueprintID,
    GTRunID,
]


class TestNewTypeIsZeroCost:
    def test_wrapped_value_equals_bare_string(self):
        for id_type in ALL_ID_TYPES:
            assert id_type("abc-1") == "abc-1"

    def test_wrapped_value_is_still_a_plain_str_at_runtime(self):
        # NewType wrapping is purely static (mypy-only) — at runtime,
        # calling one is the identity function
        for id_type in ALL_ID_TYPES:
            assert isinstance(id_type("abc-1"), str)
            assert type(id_type("abc-1")) is str


class TestNewTypeOnNone:
    def test_calling_on_none_returns_none_unchanged(self):
        # from_dict() implementations rely on this: they can wrap an
        # Optional field (e.g. derived_from) without checking for None
        # first
        for id_type in ALL_ID_TYPES:
            assert id_type(None) is None
