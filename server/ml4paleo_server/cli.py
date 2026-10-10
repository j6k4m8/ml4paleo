"""
Command-line entry point for `ml4paleo-server`.
"""

import argparse
import asyncio

from ml4paleo_server import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml4paleo-server")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="Run the API server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--workers", type=int, help="Worker processes (default: M4P_API_WORKERS)"
    )

    commands.add_parser(
        "migrate",
        help="Upgrade the database to the latest schema, and create the first "
        "admin account if there is none.",
    )
    reset = commands.add_parser(
        "reset-password",
        help="Give an account a random password that must be changed at login.",
    )
    reset.add_argument("username")
    reset_two_factor = commands.add_parser(
        "reset-two-factor",
        help="Turn off an account's two-factor sign-in (for example after the "
        "server secret key changed). Admins must set it up again at next login.",
    )
    reset_two_factor.add_argument("username")
    enable = commands.add_parser(
        "enable-user",
        help="Let a disabled account sign in again (for example the last admin, "
        "disabled by mistake).",
    )
    enable.add_argument("username")
    set_email = commands.add_parser(
        "set-email",
        help="Give an account an email address, counted as confirmed (for "
        "example so the admin gets requests for more storage by email, or to "
        "lift someone's starter limits when email isn't set up).",
    )
    set_email.add_argument("username")
    set_email.add_argument("email")
    commands.add_parser(
        "housekeeper", help="Run background upkeep (sending queued email)."
    )
    commands.add_parser(
        "check-migrations",
        help="Fail if the models have changes that no migration covers.",
    )
    check_workers = commands.add_parser(
        "check-workers",
        help="Queue a diagnostic job (which also writes and reads a scratch "
        "object through the worker's storage access) and wait for a worker to "
        "run it.",
    )
    check_workers.add_argument(
        "--timeout", type=float, default=60, help="Seconds to wait (default: 60)."
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from ml4paleo_server.settings import Settings
        from ml4paleo_server.viewer import require_neuroglancer

        settings = Settings()
        # Refuse an incomplete install before starting a worker supervisor
        # that would otherwise keep restarting failed API processes.
        require_neuroglancer(settings)
        uvicorn.run(
            "ml4paleo_server.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers or settings.api_workers,
            proxy_headers=True,
            forwarded_allow_ips=settings.forwarded_allow_ips,
        )
    elif args.command == "migrate":
        from ml4paleo_server import migrations
        from ml4paleo_server.settings import Settings

        migrations.upgrade(Settings().database_url.get_secret_value())
        asyncio.run(_ensure_local_worker())
        password = asyncio.run(_ensure_admin())
        if password is not None:
            print(
                "\n"
                "Created the admin account.\n"
                f"  username: admin\n  password: {password}\n"
                "Sign in, then change the password and set up two-factor sign-in. "
                "This password is not shown again.\n",
                flush=True,
            )
    elif args.command == "check-migrations":
        from ml4paleo_server import migrations
        from ml4paleo_server.settings import Settings

        migrations.check(Settings().database_url.get_secret_value())
    elif args.command == "reset-two-factor":
        asyncio.run(_reset_two_factor(args.username))
        print(f"Turned off two-factor sign-in for {args.username}.", flush=True)
    elif args.command == "enable-user":
        asyncio.run(_enable_user(args.username))
        print(f"{args.username} can sign in again.", flush=True)
    elif args.command == "set-email":
        asyncio.run(_set_email(args.username, args.email))
        print(f"{args.username}'s email is now {args.email}.", flush=True)
    elif args.command == "reset-password":
        password = asyncio.run(_reset_password(args.username))
        print(f"New password for {args.username}: {password}", flush=True)
    elif args.command == "check-workers":
        return asyncio.run(_check_workers(args.timeout))
    elif args.command == "housekeeper":
        from ml4paleo_server.housekeeper import run_forever

        asyncio.run(run_forever())
    else:
        parser.print_help()
    return 0


async def _ensure_admin() -> str | None:
    from ml4paleo_server.auth import ensure_admin
    from ml4paleo_server.db import create_engine, create_sessionmaker
    from ml4paleo_server.settings import Settings

    settings = Settings()
    initial = settings.initial_admin_password
    engine = create_engine(settings.database_url.get_secret_value())
    try:
        async with create_sessionmaker(engine)() as db:
            return await ensure_admin(
                db, initial.get_secret_value() if initial else None
            )
    finally:
        await engine.dispose()


async def _ensure_local_worker() -> None:
    from ml4paleo_server.db import create_engine, create_sessionmaker
    from ml4paleo_server.jobs.workers import ensure_local_worker
    from ml4paleo_server.settings import Settings

    settings = Settings()
    if settings.local_worker_token is None:
        return
    engine = create_engine(settings.database_url.get_secret_value())
    try:
        async with create_sessionmaker(engine)() as db:
            await ensure_local_worker(
                db, settings.local_worker_token.get_secret_value()
            )
    finally:
        await engine.dispose()


async def _check_workers(timeout: float) -> int:
    import time

    from ml4paleo.protocol import Tier
    from ml4paleo_server import jobs
    from ml4paleo_server.api.admin_jobs import check_grant
    from ml4paleo_server.db import Job, Worker, create_engine, create_sessionmaker
    from ml4paleo_server.settings import Settings

    engine = create_engine(Settings().database_url.get_secret_value())
    sessionmaker = create_sessionmaker(engine)
    try:
        async with sessionmaker() as db:
            job = await jobs.enqueue(
                db,
                "noop",
                {"seconds": 0, "check_storage": True},
                tier=Tier.INTERACTIVE,
                max_attempts=1,
                grants=[check_grant()],
            )
            await db.commit()
        started = time.monotonic()
        while True:
            async with sessionmaker() as db:
                current = await db.get(Job, job.id)
                assert current is not None
                worker = (
                    await db.get(Worker, current.lease_worker_id)
                    if current.lease_worker_id
                    else None
                )
                if current.status in jobs.queue.FINISHED or (
                    time.monotonic() - started > timeout
                ):
                    break
            await asyncio.sleep(1)
        elapsed = time.monotonic() - started
        if current.status == "succeeded":
            name = worker.name if worker else "a worker"
            print(
                f"Worker {name!r} ran the check job, including a storage round "
                f"trip, in {elapsed:.1f} s.",
                flush=True,
            )
            return 0
        if current.status not in jobs.queue.FINISHED:
            async with sessionmaker() as db:
                await jobs.cancel_pipeline(db, job.id)
                await db.commit()
            print(
                f"No worker ran the check job within {timeout:.0f} s "
                f"(it was {current.status}).",
                flush=True,
            )
            return 1
        print(f"The check job {current.status}: {current.error}", flush=True)
        return 1
    finally:
        await engine.dispose()


async def _reset_two_factor(username: str) -> None:
    from sqlalchemy import select

    from ml4paleo_server.auth.sessions import delete_user_sessions
    from ml4paleo_server.db import User, create_engine, create_sessionmaker
    from ml4paleo_server.settings import Settings

    engine = create_engine(Settings().database_url.get_secret_value())
    try:
        async with create_sessionmaker(engine)() as db:
            user = await db.scalar(
                select(User).where(User.username == username.lower())
            )
            if user is None:
                raise SystemExit(f"No user named {username!r}.")
            user.totp_secret_enc = None
            user.totp_pending_enc = None
            user.totp_last_step = None
            await delete_user_sessions(db, user.id)
            await db.commit()
    finally:
        await engine.dispose()


async def _reset_password(username: str) -> str:
    from sqlalchemy import select

    from ml4paleo_server.auth import expire_reset_tokens, random_password
    from ml4paleo_server.auth.passwords import hash_password
    from ml4paleo_server.auth.sessions import delete_user_sessions
    from ml4paleo_server.db import User, create_engine, create_sessionmaker
    from ml4paleo_server.settings import Settings

    engine = create_engine(Settings().database_url.get_secret_value())
    try:
        async with create_sessionmaker(engine)() as db:
            user = await db.scalar(
                select(User).where(User.username == username.lower())
            )
            if user is None:
                raise SystemExit(f"No user named {username!r}.")
            password = random_password()
            user.password_hash = await hash_password(password)
            user.must_change_password = True
            await delete_user_sessions(db, user.id)
            await expire_reset_tokens(db, user.id)
            await db.commit()
            return password
    finally:
        await engine.dispose()


async def _enable_user(username: str) -> None:
    from sqlalchemy import select

    from ml4paleo_server.db import User, create_engine, create_sessionmaker
    from ml4paleo_server.settings import Settings

    engine = create_engine(Settings().database_url.get_secret_value())
    try:
        async with create_sessionmaker(engine)() as db:
            user = await db.scalar(
                select(User).where(User.username == username.lower())
            )
            if user is None:
                raise SystemExit(f"No user named {username!r}.")
            user.status = "active"
            await db.commit()
    finally:
        await engine.dispose()


async def _set_email(username: str, email: str) -> None:
    import datetime

    from pydantic import TypeAdapter, ValidationError
    from sqlalchemy import select

    from ml4paleo_server.api.auth import Email
    from ml4paleo_server.db import User, create_engine, create_sessionmaker
    from ml4paleo_server.settings import Settings

    try:
        address = TypeAdapter(Email).validate_python(email)
    except ValidationError:
        raise SystemExit(f"{email!r} isn't an email address.") from None
    engine = create_engine(Settings().database_url.get_secret_value())
    try:
        async with create_sessionmaker(engine)() as db:
            user = await db.scalar(
                select(User).where(User.username == username.lower())
            )
            if user is None:
                raise SystemExit(f"No user named {username!r}.")
            taken = await db.scalar(
                select(User.id).where(User.email == address, User.id != user.id)
            )
            if taken is not None:
                raise SystemExit("Another account has that email address.")
            user.email = address
            # Whoever runs this on the server vouches for the address.
            user.email_verified_at = datetime.datetime.now(datetime.UTC)
            await db.commit()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
