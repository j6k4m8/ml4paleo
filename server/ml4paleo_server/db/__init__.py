"""
Database access: the declarative base, the tables, and async sessions.
"""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import Base, uuid7
from .models import (
    JOB_STATUSES,
    AuditEvent,
    AuthToken,
    EmailOutbox,
    Job,
    JobAttempt,
    JobDep,
    Project,
    ProjectMember,
    QuotaRequest,
    RateLimit,
    SiteSetting,
    User,
    UserSession,
    UserUsage,
    Worker,
)


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency that yields one session per request.
    """
    async with request.app.state.sessionmaker() as session:
        yield session


__all__ = [
    "JOB_STATUSES",
    "AuditEvent",
    "AuthToken",
    "Base",
    "EmailOutbox",
    "Job",
    "JobAttempt",
    "JobDep",
    "Project",
    "ProjectMember",
    "QuotaRequest",
    "RateLimit",
    "SiteSetting",
    "User",
    "UserSession",
    "UserUsage",
    "Worker",
    "create_engine",
    "create_sessionmaker",
    "get_session",
    "uuid7",
]
