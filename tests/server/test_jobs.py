"""
The job queue: claim order and matching, pipelines, leases, retries,
cancellation, and the reaper.
"""

import asyncio
import datetime
import hashlib

import pytest
from helpers import run_db
from ml4paleo_server import jobs
from ml4paleo_server.db import (
    Job,
    JobAttempt,
    Worker,
    create_engine,
    create_sessionmaker,
)
from sqlalchemy import select, update

from ml4paleo.protocol import Tier, WorkerCaps

CPU = WorkerCaps(version="test", kinds=["noop", "train"])
PAST = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)


async def make_worker(db, name="worker") -> Worker:
    worker = Worker(
        name=name,
        pool="local",
        token_hash=hashlib.sha256(name.encode()).hexdigest(),
        caps={},
    )
    db.add(worker)
    await db.flush()
    return worker


async def claim_kinds(db, worker, caps=CPU) -> list[str]:
    """
    Claim jobs until none are left, and return their names in claim order.
    """
    names = []
    while (claimed := await jobs.claim(db, worker, caps)) is not None:
        names.append(claimed.job.payload["name"])
    return names


async def expire(db, job: Job) -> None:
    await db.execute(update(Job).where(Job.id == job.id).values(lease_expires_at=PAST))


async def make_due(db, job: Job) -> None:
    await db.execute(update(Job).where(Job.id == job.id).values(not_before=PAST))


async def status_of(db, job: Job) -> str:
    await db.refresh(job)
    return job.status


def test_claims_follow_tier_then_pipeline_order(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        first = await jobs.enqueue(db, "noop", {"name": "first"})
        await jobs.enqueue(db, "noop", {"name": "idle"}, tier=Tier.BACKGROUND)
        await jobs.enqueue(db, "noop", {"name": "urgent"}, tier=Tier.INTERACTIVE)
        await jobs.enqueue(db, "noop", {"name": "second"})
        # Added last, but to the first pipeline, so it keeps that place.
        await jobs.enqueue(db, "noop", {"name": "first-more"}, pipeline=first)
        return await claim_kinds(db, worker)

    assert run_db(migrated_database_url, scenario) == [
        "urgent",
        "first",
        "first-more",
        "second",
        "idle",
    ]


def test_claims_match_kinds_labels_and_gpu_memory(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        await jobs.enqueue(
            db,
            "train",
            {"name": "big-gpu"},
            required_labels=["gpu"],
            min_vram_gb=16,
        )
        await jobs.enqueue(db, "mesh", {"name": "other-kind"})
        small_gpu = CPU.model_copy(update={"labels": ["gpu"], "vram_gb": 8})
        big_gpu = CPU.model_copy(update={"labels": ["gpu", "x"], "vram_gb": 24})
        return (
            await claim_kinds(db, worker),
            await claim_kinds(db, worker, small_gpu),
            await claim_kinds(db, worker, big_gpu),
        )

    assert run_db(migrated_database_url, scenario) == ([], [], ["big-gpu"])


def test_dependencies_hold_jobs_until_they_succeed(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        probe = await jobs.enqueue(db, "noop", {"name": "probe"})
        slabs = [
            await jobs.enqueue(
                db, "noop", {"name": f"slab{i}"}, pipeline=probe, depends_on=[probe]
            )
            for i in range(2)
        ]
        finalize = await jobs.enqueue(
            db, "noop", {"name": "finalize"}, pipeline=probe, depends_on=slabs
        )
        order = []
        while (claimed := await jobs.claim(db, worker, CPU)) is not None:
            order.append(claimed.job.payload["name"])
            # Only one job is ever ready at a time here, except the slabs.
            await jobs.complete(db, claimed.job.id, worker, claimed.lease_token, {})
        status = await jobs.pipeline_status(db, probe.root_id)
        return order, await status_of(db, finalize), status.status, status.progress

    order, finalize, status, progress = run_db(migrated_database_url, scenario)
    assert order == ["probe", "slab0", "slab1", "finalize"]
    assert (finalize, status, progress) == ("succeeded", "succeeded", 1.0)


def test_jobs_only_depend_on_their_own_pipeline(migrated_database_url):
    async def scenario(db):
        a = await jobs.enqueue(db, "noop", {})
        b = await jobs.enqueue(db, "noop", {})
        with pytest.raises(ValueError):
            await jobs.enqueue(db, "noop", {}, pipeline=b, depends_on=[a])
        with pytest.raises(ValueError):
            await jobs.enqueue(db, "noop", {}, depends_on=[a])

    run_db(migrated_database_url, scenario)


def test_idempotency_keys_return_the_existing_job(migrated_database_url):
    async def scenario(db):
        first = await jobs.enqueue(db, "noop", {}, idempotency_key="export:abc")
        second = await jobs.enqueue(db, "noop", {}, idempotency_key="export:abc")
        return first.id == second.id

    assert run_db(migrated_database_url, scenario)


def test_parents_finishing_together_still_release_the_child(migrated_database_url):
    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)
        try:
            async with sessionmaker() as db:
                worker = await make_worker(db)
                root = await jobs.enqueue(db, "noop", {"name": "root"})
                parents = [
                    await jobs.enqueue(db, "noop", {}, pipeline=root, depends_on=[root])
                    for _ in range(2)
                ]
                child = await jobs.enqueue(
                    db, "noop", {}, pipeline=root, depends_on=parents
                )
                first = await jobs.claim(db, worker, CPU)
                await jobs.complete(db, root.id, worker, first.lease_token, {})
                leases = [await jobs.claim(db, worker, CPU) for _ in parents]
                await db.commit()

            async def finish(claimed):
                async with sessionmaker() as db:
                    await jobs.complete(
                        db, claimed.job.id, worker, claimed.lease_token, {}
                    )
                    # Hold the transaction open so the two completions overlap.
                    await asyncio.sleep(0.3)
                    await db.commit()

            await asyncio.gather(*(finish(lease) for lease in leases))
            async with sessionmaker() as db:
                return await db.scalar(select(Job.status).where(Job.id == child.id))
        finally:
            await engine.dispose()

    assert asyncio.run(race()) == "queued"


def test_a_lost_lease_cannot_report(migrated_database_url):
    async def scenario(db):
        slow, fast = await make_worker(db, "slow"), await make_worker(db, "fast")
        job = await jobs.enqueue(db, "noop", {})
        first = await jobs.claim(db, slow, CPU)
        await expire(db, job)
        assert await jobs.reap(db) == 1
        await make_due(db, job)
        second = await jobs.claim(db, fast, CPU)
        assert second is not None and second.job.attempts == 2
        for report in (
            jobs.heartbeat(db, job.id, slow, first.lease_token),
            jobs.complete(db, job.id, slow, first.lease_token, {"from": "slow"}),
            jobs.fail(db, job.id, slow, first.lease_token, "late"),
        ):
            with pytest.raises(jobs.LeaseLost):
                await report
        await jobs.complete(db, job.id, fast, second.lease_token, {"from": "fast"})
        # Repeating a completion is harmless; it keeps the first result.
        await jobs.complete(db, job.id, fast, second.lease_token, {"from": "again"})
        await db.refresh(job)
        outcomes = (
            await db.scalars(
                select(JobAttempt.outcome)
                .where(JobAttempt.job_id == job.id)
                .order_by(JobAttempt.attempt)
            )
        ).all()
        return job.status, job.result, outcomes

    assert run_db(migrated_database_url, scenario) == (
        "succeeded",
        {"from": "fast"},
        ["expired", "succeeded"],
    )


def test_failures_back_off_then_fail_the_pipeline(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        job = await jobs.enqueue(db, "noop", {}, max_attempts=2)
        waiting = await jobs.enqueue(db, "noop", {}, pipeline=job, depends_on=[job])
        claimed = await jobs.claim(db, worker, CPU)
        await jobs.fail(db, job.id, worker, claimed.lease_token, "flaky disk")
        backing_off = await jobs.claim(db, worker, CPU)
        await make_due(db, job)
        claimed = await jobs.claim(db, worker, CPU)
        await jobs.fail(db, job.id, worker, claimed.lease_token, "flaky disk again")
        status = await jobs.pipeline_status(db, job.root_id)
        return (
            backing_off,
            await status_of(db, job),
            job.error,
            await status_of(db, waiting),
            status.status,
        )

    assert run_db(migrated_database_url, scenario) == (
        None,
        "failed",
        "flaky disk again",
        "cancelled",
        "failed",
    )


@pytest.mark.parametrize(
    ("retryable", "expected"), [(False, "failed"), (True, "queued")]
)
def test_a_pipeline_can_refuse_a_success(migrated_database_url, retryable, expected):
    async def scenario(db):
        worker = await make_worker(db)
        job = await jobs.enqueue(db, "noop", {})
        claimed = await jobs.claim(db, worker, CPU)

        async def next_steps(done):
            await jobs.enqueue(db, "noop", {}, pipeline=done, depends_on=[done])
            raise jobs.Rejected("It won't fit.", retryable=retryable)

        with pytest.raises(jobs.Rejected):
            await jobs.complete(
                db, job.id, worker, claimed.lease_token, {"ok": 1}, after=next_steps
            )
        added = await db.scalars(select(Job.id).where(Job.id != job.id))
        outcomes = await db.scalars(
            select(JobAttempt.outcome).where(JobAttempt.job_id == job.id)
        )
        return (
            await status_of(db, job),
            job.result,
            job.error,
            added.all(),
            outcomes.all(),
        )

    # The success and the jobs it added are undone; the attempt failed.
    assert run_db(migrated_database_url, scenario) == (
        expected,
        None,
        "It won't fit.",
        [],
        ["failed"],
    )


def test_permanent_failures_are_not_retried(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        job = await jobs.enqueue(db, "noop", {})
        claimed = await jobs.claim(db, worker, CPU)
        await jobs.fail(
            db, job.id, worker, claimed.lease_token, "not an image", retryable=False
        )
        return await status_of(db, job), job.attempts

    assert run_db(migrated_database_url, scenario) == ("failed", 1)


def test_cancelling_a_pipeline(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        running = await jobs.enqueue(db, "noop", {})
        waiting = await jobs.enqueue(
            db, "noop", {}, pipeline=running, depends_on=[running]
        )
        claimed = await jobs.claim(db, worker, CPU)
        await jobs.cancel_pipeline(db, running.root_id)
        beat = await jobs.heartbeat(db, running.id, worker, claimed.lease_token)
        with pytest.raises(jobs.JobCancelled):
            await jobs.complete(db, running.id, worker, claimed.lease_token, {})
        status = await jobs.pipeline_status(db, running.root_id)
        return (
            beat.cancel_requested,
            await status_of(db, running),
            await status_of(db, waiting),
            status.status,
        )

    assert run_db(migrated_database_url, scenario) == (
        True,
        "cancelled",
        "cancelled",
        "cancelled",
    )


def test_released_jobs_go_back_without_using_an_attempt(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        job = await jobs.enqueue(db, "noop", {"name": "job"})
        claimed = await jobs.claim(db, worker, CPU)
        await jobs.release(db, job.id, worker, claimed.lease_token)
        again = await jobs.claim(db, worker, CPU)
        return await status_of(db, job), again is not None, job.attempts

    assert run_db(migrated_database_url, scenario) == ("leased", True, 1)


def test_the_reaper_gives_up_after_max_attempts(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        job = await jobs.enqueue(db, "noop", {}, max_attempts=3)
        for _ in range(3):
            await make_due(db, job)
            assert await jobs.claim(db, worker, CPU) is not None
            await expire(db, job)
            await jobs.reap(db)
        return await status_of(db, job), job.attempts, job.error

    status, attempts, error = run_db(migrated_database_url, scenario)
    assert (status, attempts) == ("failed", 3)
    assert "stopped responding" in error


def test_the_reaper_releases_jobs_left_blocked(migrated_database_url):
    async def scenario(db):
        root = await jobs.enqueue(db, "noop", {})
        child = await jobs.enqueue(db, "noop", {}, pipeline=root, depends_on=[root])
        # As if the parent's completion never queued its child.
        await db.execute(
            update(Job).where(Job.id == root.id).values(status="succeeded")
        )
        await jobs.reap(db)
        return await status_of(db, child)

    assert run_db(migrated_database_url, scenario) == "queued"


def test_pipeline_progress_is_weighted(migrated_database_url):
    async def scenario(db):
        worker = await make_worker(db)
        root = await jobs.enqueue(db, "noop", {"name": "small"}, weight=1)
        await jobs.enqueue(db, "noop", {"name": "big"}, pipeline=root, weight=3)
        small = await jobs.claim(db, worker, CPU)
        big = await jobs.claim(db, worker, CPU)
        await jobs.complete(db, small.job.id, worker, small.lease_token, {})
        await jobs.heartbeat(db, big.job.id, worker, big.lease_token, progress=0.5)
        status = await jobs.pipeline_status(db, root.root_id)
        return status.status, status.progress, status.jobs

    assert run_db(migrated_database_url, scenario) == ("running", 0.625, 2)


def test_the_admin_status_filter_matches_the_job_statuses():
    from typing import get_args

    from ml4paleo_server.api.admin_jobs import JobStatus
    from ml4paleo_server.db import JOB_STATUSES

    assert get_args(JobStatus) == JOB_STATUSES


async def _two_sessions(database_url, scenario):
    """
    Run `scenario(sessionmaker)` with its own engine, for tests that interleave
    transactions by hand.
    """
    engine = create_engine(database_url)
    try:
        return await scenario(create_sessionmaker(engine))
    finally:
        await engine.dispose()


def test_cancelling_catches_a_job_being_requeued(migrated_database_url):
    async def scenario(sessionmaker):
        async with sessionmaker() as db:
            worker = await make_worker(db)
            root = await jobs.enqueue(db, "noop", {"name": "root"})
            job = await jobs.enqueue(db, "noop", {"name": "job"}, pipeline=root)
            root_lease = await jobs.claim(db, worker, CPU)
            job_lease = await jobs.claim(db, worker, CPU)
            await db.commit()
        async with sessionmaker() as failing, sessionmaker() as cancelling:
            # The worker's retryable failure holds the job's row...
            await jobs.fail(failing, job.id, worker, job_lease.lease_token, "flaky")
            # ...while the pipeline is cancelled, which must not wait for it.
            await asyncio.wait_for(jobs.cancel_pipeline(cancelling, root.id), 5)
            await cancelling.commit()
            await failing.commit()
        async with sessionmaker() as db:
            await db.execute(update(Job).values(not_before=PAST))
            requeued = await db.scalar(select(Job.status).where(Job.id == job.id))
            claimed = await jobs.claim(db, worker, CPU)
            await jobs.sweep(db)
            beat = await jobs.heartbeat(db, root.id, worker, root_lease.lease_token)
            final = await db.scalar(select(Job.status).where(Job.id == job.id))
            return requeued, claimed, final, beat.cancel_requested

    result = asyncio.run(_two_sessions(migrated_database_url, scenario))
    assert result == ("queued", None, "cancelled", True)


@pytest.mark.parametrize(
    ("outcome", "expected"), [("fails", "error"), ("succeeds", "queued")]
)
def test_enqueue_rechecks_dependencies(migrated_database_url, outcome, expected):
    async def scenario(sessionmaker):
        async with sessionmaker() as db:
            worker = await make_worker(db)
            root = await jobs.enqueue(db, "noop", {})
            lease = await jobs.claim(db, worker, CPU)
            await db.commit()
        async with sessionmaker() as stale:
            # Loaded while the dependency is still running...
            dependency = await stale.get(Job, root.id)
            async with sessionmaker() as db:
                if outcome == "fails":
                    await jobs.fail(
                        db, root.id, worker, lease.lease_token, "bad", retryable=False
                    )
                else:
                    await jobs.complete(db, root.id, worker, lease.lease_token, {})
                await db.commit()
            # ...but it finished before the new job was added.
            try:
                job = await jobs.enqueue(
                    stale, "noop", {}, pipeline=dependency, depends_on=[dependency]
                )
            except ValueError:
                return "error"
            return job.status

    assert asyncio.run(_two_sessions(migrated_database_url, scenario)) == expected


def test_concurrent_enqueues_with_one_key_make_one_job(migrated_database_url):
    async def scenario(sessionmaker):
        async def add(delay):
            async with sessionmaker() as db:
                job = await jobs.enqueue(db, "noop", {}, idempotency_key="export:1")
                await asyncio.sleep(delay)
                await db.commit()
                return job.id

        first, second = await asyncio.gather(add(0.3), add(0))
        async with sessionmaker() as db:
            count = len((await db.scalars(select(Job.id))).all())
        return first == second, count

    assert asyncio.run(_two_sessions(migrated_database_url, scenario)) == (True, 1)


def test_cancelling_does_not_deadlock_with_a_completion(migrated_database_url):
    async def scenario(sessionmaker):
        async with sessionmaker() as db:
            worker = await make_worker(db)
            root = await jobs.enqueue(db, "noop", {})
            lease = await jobs.claim(db, worker, CPU)
            await jobs.complete(db, root.id, worker, lease.lease_token, {})
            middle = await jobs.enqueue(
                db, "noop", {}, pipeline=root, depends_on=[root]
            )
            last = await jobs.enqueue(
                db, "noop", {}, pipeline=root, depends_on=[middle]
            )
            lease = await jobs.claim(db, worker, CPU)
            await db.commit()
        async with sessionmaker() as completing, sessionmaker() as cancelling:
            # The completion holds the job and its waiting child...
            await jobs.complete(completing, middle.id, worker, lease.lease_token, {})
            # ...and cancelling the pipeline must not wait for either.
            await asyncio.wait_for(jobs.cancel_pipeline(cancelling, root.id), 5)
            await cancelling.commit()
            await completing.commit()
        async with sessionmaker() as db:
            claimed = await jobs.claim(db, worker, CPU)
            await jobs.sweep(db)
            status = await db.scalar(select(Job.status).where(Job.id == last.id))
            return claimed, status

    result = asyncio.run(_two_sessions(migrated_database_url, scenario))
    assert result == (None, "cancelled")


def test_sibling_failures_at_once_do_not_deadlock(migrated_database_url):
    async def scenario(sessionmaker):
        async with sessionmaker() as db:
            worker = await make_worker(db)
            root = await jobs.enqueue(db, "noop", {})
            lease = await jobs.claim(db, worker, CPU)
            await jobs.complete(db, root.id, worker, lease.lease_token, {})
            siblings = [
                await jobs.enqueue(db, "noop", {}, pipeline=root, depends_on=[root])
                for _ in range(2)
            ]
            leases = [await jobs.claim(db, worker, CPU) for _ in siblings]
            await db.commit()

        async def fail(claimed):
            async with sessionmaker() as db:
                await jobs.fail(
                    db, claimed.job.id, worker, claimed.lease_token, "bad input", False
                )
                await asyncio.sleep(0.2)
                await db.commit()

        await asyncio.wait_for(asyncio.gather(*(fail(lease) for lease in leases)), 10)
        async with sessionmaker() as db:
            return sorted(
                (
                    await db.scalars(
                        select(Job.status).where(Job.id.in_([s.id for s in siblings]))
                    )
                ).all()
            )

    # Whichever failure commits first cancels the pipeline, so the other one
    # may end up cancelled rather than failed; neither waits on the other.
    result = asyncio.run(_two_sessions(migrated_database_url, scenario))
    assert "failed" in result and set(result) <= {"failed", "cancelled"}
