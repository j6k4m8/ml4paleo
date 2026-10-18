"""Live browser heartbeats; independent per tab, scoped to its user/project."""

import datetime
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from .. import artifacts, live
from ..auth.deps import CurrentAuth, DbSession
from ..db import LiveSession, User
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/live-sessions", tags=["live"])


class LiveIn(BaseModel):
    image_artifact_id: uuid.UUID


@router.put("/{session_id}", status_code=204)
async def renew(
    session_id: uuid.UUID,
    body: LiveIn,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    image = await artifacts.head(db, project.id, "image")
    if image is None or image.id != body.image_artifact_id:
        raise HTTPException(
            status_code=409, detail="The image changed; reload the annotator."
        )
    now = datetime.datetime.now(datetime.UTC)
    await db.scalar(
        select(User.id).where(User.id == auth.user.id).with_for_update(key_share=True)
    )
    # Bound a user's active leases even if buggy tabs repeatedly make IDs.
    existing = await db.get(LiveSession, session_id)
    if (existing is None or existing.expires_at <= now) and (
        await db.scalar(
            select(func.count())
            .select_from(LiveSession)
            .where(LiveSession.user_id == auth.user.id, LiveSession.expires_at > now)
        )
        or 0
    ) >= 32:
        raise HTTPException(status_code=429, detail="Too many active Live sessions.")
    result = await db.scalar(
        insert(LiveSession)
        .values(
            id=session_id,
            project_id=project.id,
            user_id=auth.user.id,
            image_id=image.id,
            expires_at=now + live.TTL,
        )
        .on_conflict_do_update(
            index_elements=[LiveSession.id],
            set_={"expires_at": now + live.TTL, "image_id": image.id},
            where=(LiveSession.user_id == auth.user.id)
            & (LiveSession.project_id == project.id),
        )
        .returning(LiveSession.id)
    )
    if result is None:
        raise HTTPException(status_code=404, detail="No such Live session.")
    await db.commit()


@router.delete("/{session_id}", status_code=204)
async def release(
    session_id: uuid.UUID, project: MemberProject, auth: CurrentAuth, db: DbSession
) -> None:
    await db.execute(
        delete(LiveSession).where(
            LiveSession.id == session_id,
            LiveSession.project_id == project.id,
            LiveSession.user_id == auth.user.id,
        )
    )
    await db.commit()
