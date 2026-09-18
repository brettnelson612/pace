"""
pace/db/relational/table_names.py

The canonical name of every relational table, defined once here and
imported everywhere a __tablename__ or a ForeignKey string needs it —
including across files. A typo here is a typo everywhere it's used,
rather than two independent string literals silently drifting apart
(e.g. a table renamed in one place but not the other).

No SQLAlchemy import, no dependency on any Row class — a pure leaf
module, safe to import from anywhere in db/relational/ without risk of
a circular import.
"""

GEOMETRIES_TABLE_NAME = "geometries"
MATERIALS_TABLE_NAME = "materials"
LCOMPONENTS_TABLE_NAME = "lcomponents"
CCOMPONENTS_TABLE_NAME = "ccomponents"
REFERENCES_TABLE_NAME = "references"
