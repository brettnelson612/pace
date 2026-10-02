"""
tests/unit/pace/core/test_pace_object.py

PaceObject is deliberately minimal (to_dict/from_dict only), with a
default no-op validate() that __post_init__ wires up automatically.
Exercised here via small local dataclass subclasses, since the
concrete domain classes already cover the "validate() actually does
something" case extensively elsewhere.
"""

from dataclasses import FrozenInstanceError, dataclass

import pytest
from pace.core.pace_object import PaceObject


@dataclass(frozen=True)
class _Minimal(PaceObject):
    """Uses the base class's default no-op validate()."""

    value: int = 0

    def to_dict(self) -> dict:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "_Minimal":
        return cls(value=data["value"])


@dataclass(frozen=True)
class _Validating(PaceObject):
    """Overrides validate() to enforce a non-negative value."""

    value: int = 0

    def to_dict(self) -> dict:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "_Validating":
        return cls(value=data["value"])

    def validate(self) -> None:
        if self.value < 0:
            raise ValueError(f"value must be non-negative, got {self.value}")


class TestPaceObjectABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            PaceObject()  # pyright: ignore[reportAbstractUsage]

    def test_subclass_missing_to_dict_cannot_instantiate(self):
        class MissingToDict(PaceObject):
            @classmethod
            def from_dict(cls, data: dict) -> "MissingToDict":
                return cls()

        with pytest.raises(TypeError):
            MissingToDict()  # pyright: ignore[reportAbstractUsage]

    def test_subclass_missing_from_dict_cannot_instantiate(self):
        class MissingFromDict(PaceObject):
            def to_dict(self) -> dict:
                return {}

        with pytest.raises(TypeError):
            MissingFromDict()  # pyright: ignore[reportAbstractUsage]


class TestDefaultValidateIsNoop:
    def test_no_override_never_raises(self):
        # the base class's validate() is a plain no-op — any value is
        # accepted when a subclass doesn't override it
        _Minimal(value=-999)  # should not raise

    def test_round_trip(self):
        obj = _Minimal(value=5)
        assert _Minimal.from_dict(obj.to_dict()) == obj


class TestPostInitCallsValidate:
    def test_valid_value_does_not_raise(self):
        _Validating(value=1)  # should not raise

    def test_invalid_value_raises_at_construction(self):
        # __post_init__ -> self.validate() runs during __init__, so the
        # invalid value is rejected before the object ever exists
        with pytest.raises(ValueError):
            _Validating(value=-1)

    def test_frozen(self):
        obj = _Validating(value=1)
        with pytest.raises(FrozenInstanceError):
            obj.value = 2  # type: ignore[misc]
