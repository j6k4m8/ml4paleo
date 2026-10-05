"""
The SQLAlchemy declarative base shared by every table.
"""

import datetime
import secrets
import time
import uuid

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Stable constraint names, so Alembic migrations can refer to them.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def uuid7() -> uuid.UUID:
    """
    Return a UUID version 7: a millisecond timestamp followed by random bits,
    so new primary keys sort roughly by creation time and index well.
    """
    timestamp_ms = time.time_ns() // 1_000_000
    random_bits = int.from_bytes(secrets.token_bytes(10), "big")
    value = (timestamp_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76  # version
    value |= ((random_bits >> 62) & 0xFFF) << 64
    value |= 0b10 << 62  # RFC 4122 variant
    value |= random_bits & ((1 << 62) - 1)
    return uuid.UUID(int=value)


class TimestampMixin:
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
