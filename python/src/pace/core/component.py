"""
Reactor structure objects: LComponent (geometry version + material
version, no position), PComponent (a position + one LComponent or
CComponent; only ever a member of a CComponent), and CComponent (a
collection of >=2 PComponents, no position).

Deliberately has no import dependency on geometry.py or material.py —
LComponent references GeometryVersionID/MaterialVersionID only (bare
IDs, never embedded objects), per PaceObject's bare-IDs-always rule.
"""
