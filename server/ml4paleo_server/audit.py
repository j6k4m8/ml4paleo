"""
The audit log: a record of who did what to accounts and projects.
"""

import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from .auth.ratelimit import client_key
from .db import AuditEvent


def record(
    db: AsyncSession,
    *,
    actor_id: uuid.UUID | None,
    action: str,
    target_type: str,
    target_id: uuid.UUID | str,
    request: Request | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Add an audit event. It is saved when the caller's transaction commits, so
    an action and its record succeed or fail together.
    """
    db.add(
        AuditEvent(
            actor_user_id=actor_id,
            ip=client_key(request) if request is not None else None,
            action=action,
            target_type=target_type,
            target_id=str(target_id),
            details=details or {},
        )
    )
