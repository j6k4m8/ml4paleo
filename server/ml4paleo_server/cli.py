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
    commands.add_parser(
        "housekeeper", help="Run background upkeep (sending queued email)."
    )
    commands.add_parser(
        "check-migrations",
        help="Fail if the models have changes that no migration covers.",
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from ml4paleo_server.settings import Settings

        settings = Settings()
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
    elif args.command == "reset-password":
        password = asyncio.run(_reset_password(args.username))
        print(f"New password for {args.username}: {password}", flush=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
