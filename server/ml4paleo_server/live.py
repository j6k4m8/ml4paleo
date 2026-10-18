"""Reservations expire without requiring a browser to clean up."""

import datetime

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import ArtifactHead, LiveSession, Project, ProjectMember, User

TTL = datetime.timedelta(seconds=45)


async def active(db: AsyncSession, now: datetime.datetime) -> bool:
    return bool(
        await db.scalar(
            select(
                exists().where(
                    LiveSession.expires_at > now,
                    Project.id == LiveSession.project_id,
                    Project.deleted_at.is_(None),
                    User.id == LiveSession.user_id,
                    User.status == "active",
                    ProjectMember.project_id == LiveSession.project_id,
                    ProjectMember.user_id == LiveSession.user_id,
                    ArtifactHead.project_id == LiveSession.project_id,
                    ArtifactHead.slot == "image",
                    ArtifactHead.artifact_id == LiveSession.image_id,
                )
            )
        )
    )
