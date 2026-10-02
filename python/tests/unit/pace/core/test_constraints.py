"""
tests/unit/pace/core/test_constraints.py

Exercises validate_fields() directly against ad-hoc dataclasses, since
geometry.py/material.py only ever exercise it indirectly through their
own field metadata.
"""

from dataclasses import dataclass, field

import pytest
from pace.core.constraints import Constraint, validate_fields


@dataclass
class _Positive:
    value: float = field(metadata={"constraint": Constraint.POSITIVE})


@dataclass
class _NonNegative:
    value: float = field(metadata={"constraint": Constraint.NON_NEGATIVE})


@dataclass
class _Ranged:
    value: float = field(metadata={"range": (1, 10)})


@dataclass
class _PositiveAndRanged:
    value: float = field(metadata={"constraint": Constraint.POSITIVE, "range": (1, 10)})


@dataclass
class _OptionalPositive:
    value: float | None = field(
        default=None, metadata={"constraint": Constraint.POSITIVE}
    )


@dataclass
class _NoConstraint:
    value: float = 0.0


class TestPositive:
    def test_positive_value_allowed(self):
        validate_fields(_Positive(value=1.0))  # should not raise

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            validate_fields(_Positive(value=0.0))

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            validate_fields(_Positive(value=-1.0))


class TestNonNegative:
    def test_zero_allowed(self):
        validate_fields(_NonNegative(value=0.0))  # should not raise

    def test_positive_allowed(self):
        validate_fields(_NonNegative(value=1.0))  # should not raise

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            validate_fields(_NonNegative(value=-0.1))


class TestRange:
    def test_within_range_allowed(self):
        validate_fields(_Ranged(value=5))  # should not raise

    def test_at_lower_bound_allowed(self):
        validate_fields(_Ranged(value=1))  # inclusive lower bound

    def test_at_upper_bound_allowed(self):
        validate_fields(_Ranged(value=10))  # inclusive upper bound

    def test_below_range_rejected(self):
        with pytest.raises(ValueError):
            validate_fields(_Ranged(value=0))

    def test_above_range_rejected(self):
        with pytest.raises(ValueError):
            validate_fields(_Ranged(value=11))


class TestCombinedConstraintAndRange:
    def test_valid_value_allowed(self):
        validate_fields(_PositiveAndRanged(value=5))  # should not raise

    def test_positive_but_below_range_rejected(self):
        # 0.5 satisfies POSITIVE but fails the (1, 10) range check —
        # both checks apply independently
        with pytest.raises(ValueError):
            validate_fields(_PositiveAndRanged(value=0.5))

    def test_zero_rejected_by_positive_check_before_range(self):
        with pytest.raises(ValueError):
            validate_fields(_PositiveAndRanged(value=0))


class TestNoneSkipsValidation:
    def test_unset_optional_field_with_constraint_metadata_does_not_raise(self):
        # validate_fields() must skip a None value entirely, not treat
        # it as failing a POSITIVE/range check
        validate_fields(_OptionalPositive(value=None))  # should not raise

    def test_set_optional_field_still_enforces_constraint(self):
        with pytest.raises(ValueError):
            validate_fields(_OptionalPositive(value=-1.0))


class TestNoConstraintMetadata:
    def test_field_with_no_metadata_never_raises(self):
        validate_fields(_NoConstraint(value=-999.0))  # should not raise
