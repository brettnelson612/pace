"""
pace/db/relational/base.py

The shared SQLAlchemy declarative base every ORM row model inherits
from. All table metadata (used by SqlDB.create_all()) is collected
here via Base.metadata.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
