"""
Live updates (server-sent events) are limited per account, since each open
stream queries the database every second.
"""

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from helpers import signup
from ml4paleo_server import streams


def test_streams_count_only_while_they_run(monkeypatch):
    monkeypatch.setattr(streams, "PER_USER", 2)
    user = uuid.uuid4()

    async def events():
        yield "a"
        yield "b"

    async def run():
        first = streams.counted(user, events())
        second = streams.counted(user, events())
        # Made but not started: nothing counted yet.
        streams.check(user)
        assert await anext(first) == "a"
        assert await anext(second) == "a"
        with pytest.raises(HTTPException) as refused:
            streams.check(user)
        assert refused.value.status_code == 429
        # Finishing (or a client leaving) gives the place back.
        assert [e async for e in first] == ["b"]
        streams.check(user)
        await second.aclose()
        assert user not in streams._open

    asyncio.run(run())


def test_too_many_label_streams_are_refused(new_browser, monkeypatch):
    monkeypatch.setattr(streams, "PER_USER", 0)
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    refused = ada.get(f"/api/projects/{project}/labels/events")
    assert refused.status_code == 429
