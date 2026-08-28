"""
Declarative field-level validation: the Constraint enum (POSITIVE,
NON_NEGATIVE) and validate_fields(), which walks a dataclass's fields
and checks each against its "constraint"/"range" metadata generically.

Domain-agnostic — used by geometry.py and material.py alike, which is
why this isn't colocated in either. Relational checks between two fields
on the same class (e.g. inner_radius_m < outer_radius_m) aren't
expressible here and stay hand-written per class.
"""

from dataclasses import fields
from enum import Enum


class Constraint(Enum):
    POSITIVE = "positive"
    NON_NEGATIVE = "non_negative"


def validate_fields(obj) -> None:
    # validate each field based on it's established constraints
    for f in fields(obj):
        value = getattr(obj, f.name)

        # specific constraints
        constraint = f.metadata.get("constraint")
        if constraint == Constraint.POSITIVE and value <= 0:
            raise ValueError(f"{f.name} must be positive, got {value}")
        elif constraint == Constraint.NON_NEGATIVE and value < 0:
            raise ValueError(f"{f.name} must be non-negative, got {value}")

        # value range constraints
        bounds = f.metadata.get("range")
        if bounds is not None:
            minimum, maximum = bounds
            if not (minimum <= value <= maximum):
                raise ValueError(
                    f"{f.name} must be in [{minimum}, {maximum}], got {value}"
                )
