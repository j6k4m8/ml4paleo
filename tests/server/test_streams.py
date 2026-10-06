"""
Live updates (server-sent events) are limited per account, since each open
stream queries the database every second.
"""

import asyncio
import gc
import uuid

import pytest
from fastapi import HTTPException
from helpers import signup
from ml4paleo_server import streams


def test_places_are_taken_at_once_and_given_back_on_every_ending(monkeypatch):
    monkeypatch.setattr(streams, "PER_USER", 2)
    user = uuid.uuid4()

    async def events():
        yield "a"
        yield "b"

    async def run():
        first = streams.response(streams.reserve(user), events())
        second = streams.response(streams.reserve(user), events())
        # Both places are taken before either stream starts, so a third
        # request arriving meanwhile is refused.
        with pytest.raises(HTTPException) as refused:
            streams.reserve(user)
        assert refused.value.status_code == 429
        # A stream that runs to its end gives its place back...
        assert [e async for e in first.body_iterator] == ["a", "b"]
        streams.reserve(user).release()
        # ...as does one dropped before it ever started (a client gone
        # before the response began).
        del second
        gc.collect()
        assert user not in streams._open
        # And giving a place back twice counts once.
        slot = streams.reserve(user)
        slot.release()
        slot.release()
        assert user not in streams._open

    asyncio.run(run())


def test_too_many_label_streams_are_refused(new_browser, monkeypatch):
    monkeypatch.setattr(streams, "PER_USER", 0)
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    refused = ada.get(f"/api/projects/{project}/labels/events")
    assert refused.status_code == 429
