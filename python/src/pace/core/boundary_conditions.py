"""
pace/core/boundary_conditions.py
"""

from enum import StrEnum


class NeutronBC(StrEnum):
    """What happens to a neutron reaching a face of the Reactor's bounds.

    Values (OpenMC's boundary types, minus 'transmission', which is the
    interior-surface default and never valid on the outer boundary):
        - vacuum: the neutron leaves and never returns (leakage). Use
              where the real system ends, e.g. the top and bottom of a
              3D pin.
        - reflective: mirror reflection — "my neighbour is my mirror
              image". Reflective on all sides of a pin cell models an
              infinite array (VERA problems 1 and 2).
        - periodic: leaves one face, re-enters the paired opposite face
              — "my neighbour is an identical copy". Must be set on both
              faces of a pair (see PERIODIC_PARTNER).
        - white: returns with a random (cosine-distributed) direction;
              mainly for cylindricalized unit cells.
    """

    VACUUM = "vacuum"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"
    WHITE = "white"


class ThermalBC(StrEnum):
    """Heat-conduction condition on a face of the Reactor's bounds.

    Values:
        - adiabatic: no heat crosses the face. Also the symmetry-plane
              condition, and the default for any face not listed.

    FIXED_TEMPERATURE and CONVECTIVE carry values, so they arrive as
    dataclasses when a model first needs them.
    """

    ADIABATIC = "adiabatic"


class FlowBC(StrEnum):
    """Flow condition on a face of the ReactorBlueprint's bounds.

    Marks a face as the inlet or the outlet. Flow rate, inlet
    temperature and system pressure are operating conditions (see
    operating_conditions.py).
    """

    INLET = "inlet"
    OUTLET = "outlet"
