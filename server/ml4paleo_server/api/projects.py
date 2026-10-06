"""
Projects and their collaborators.

Every route under `/api/projects/{project_id}` goes through `MemberProject`,
which answers 404 to anyone who isn't a member, so other people's projects
look the same as projects that don't exist.
"""

import datetime
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select

from .. import audit
from ..auth.deps import CurrentAuth, DbSession
from ..db import Project, ProjectMember, User
from ..pipelines import train

router = APIRouter(prefix="/api/projects", tags=["projects"])


async def member_project(
    project_id: uuid.UUID, auth: CurrentAuth, db: DbSession
) -> Project:
    project = await db.scalar(
        select(Project)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .where(
            Project.id == project_id,
            Project.deleted_at.is_(None),
            ProjectMember.user_id == auth.user.id,
        )
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


MemberProject = Annotated[Project, Depends(member_project)]


class MemberOut(BaseModel):
    user_id: str
    username: str
    is_owner: bool


class ProjectOut(BaseModel):
    id: str
    name: str
    owner: str
    created_at: datetime.datetime
    members: list[MemberOut] | None = None


class ProjectName(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Give the project a name.")
        return value


async def _members(db: DbSession, project: Project) -> list[MemberOut]:
    rows = (
        await db.execute(
            select(User.id, User.username)
            .join(ProjectMember, ProjectMember.user_id == User.id)
            .where(ProjectMember.project_id == project.id)
            .order_by(User.username)
        )
    ).all()
    return [
        MemberOut(
            user_id=str(user_id),
            username=username,
            is_owner=user_id == project.owner_id,
        )
        for user_id, username in rows
    ]


async def _project_out(
    db: DbSession, project: Project, with_members: bool = False
) -> ProjectOut:
    owner = await db.scalar(select(User.username).where(User.id == project.owner_id))
    return ProjectOut(
        id=str(project.id),
        name=project.name,
        owner=owner or "",
        created_at=project.created_at,
        members=await _members(db, project) if with_members else None,
    )


@router.post("", status_code=201)
async def create_project(
    body: ProjectName, request: Request, auth: CurrentAuth, db: DbSession
) -> ProjectOut:
    project = Project(name=body.name, owner_id=auth.user.id)
    db.add(project)
    await db.flush()
    db.add(ProjectMember(project_id=project.id, user_id=auth.user.id))
    audit.record(
        db,
        actor_id=auth.user.id,
        action="project.create",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"name": project.name},
    )
    await db.commit()
    await db.refresh(project)
    return await _project_out(db, project, with_members=True)


@router.get("")
async def list_projects(auth: CurrentAuth, db: DbSession) -> list[ProjectOut]:
    projects = (
        await db.scalars(
            select(Project)
            .join(ProjectMember, ProjectMember.project_id == Project.id)
            .where(ProjectMember.user_id == auth.user.id, Project.deleted_at.is_(None))
            .order_by(Project.created_at.desc())
        )
    ).all()
    return [await _project_out(db, project) for project in projects]


@router.get("/{project_id}")
async def get_project(project: MemberProject, db: DbSession) -> ProjectOut:
    return await _project_out(db, project, with_members=True)


@router.patch("/{project_id}")
async def rename_project(
    body: ProjectName,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> ProjectOut:
    old_name, project.name = project.name, body.name
    audit.record(
        db,
        actor_id=auth.user.id,
        action="project.rename",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"from": old_name, "to": body.name},
    )
    await db.commit()
    return await _project_out(db, project, with_members=True)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project: MemberProject, request: Request, auth: CurrentAuth, db: DbSession
) -> None:
    """
    Only the owner can delete a project. Deletion hides it at once, stops its
    trainings, and gives back its models' slots; its data is removed later by
    storage garbage collection.
    """
    if project.owner_id != auth.user.id:
        raise HTTPException(status_code=403, detail="Only the owner can delete it.")
    await train.stop_project(db, project)
    project.deleted_at = datetime.datetime.now(datetime.UTC)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="project.delete",
        target_type="project",
        target_id=project.id,
        request=request,
    )
    await db.commit()


class MemberIn(BaseModel):
    # Collaborators are added by username only. Looking people up by email
    # would let anyone learn which addresses have accounts.
    username: str = Field(min_length=1, max_length=64)


@router.get("/{project_id}/members")
async def list_members(project: MemberProject, db: DbSession) -> list[MemberOut]:
    return await _members(db, project)


@router.post("/{project_id}/members", status_code=201)
async def add_member(
    body: MemberIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> list[MemberOut]:
    name = body.username.strip().lower()
    user = await db.scalar(
        select(User).where(User.username == name, User.status == "active")
    )
    if user is None:
        raise HTTPException(status_code=404, detail="No one with that username.")
    already = await db.scalar(
        select(func.count())
        .select_from(ProjectMember)
        .where(ProjectMember.project_id == project.id, ProjectMember.user_id == user.id)
    )
    if not already:
        db.add(
            ProjectMember(project_id=project.id, user_id=user.id, added_by=auth.user.id)
        )
        audit.record(
            db,
            actor_id=auth.user.id,
            action="project.member.add",
            target_type="project",
            target_id=project.id,
            request=request,
            details={"user_id": str(user.id), "username": user.username},
        )
        await db.commit()
    return await _members(db, project)


@router.delete("/{project_id}/members/{user_id}", status_code=204)
async def remove_member(
    user_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    """
    Collaborators can remove anyone except the owner (including themselves,
    to leave). The owner can't be removed.
    """
    if user_id == project.owner_id:
        raise HTTPException(status_code=403, detail="The owner can't be removed.")
    membership = await db.get(ProjectMember, (project.id, user_id))
    if membership is None:
        raise HTTPException(status_code=404, detail="That person isn't a member.")
    await db.delete(membership)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="project.member.remove",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"user_id": str(user_id)},
    )
    await db.commit()
