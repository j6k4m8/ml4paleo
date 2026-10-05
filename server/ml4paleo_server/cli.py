"""
Command-line entry point for `ml4paleo-server`.
"""

import argparse

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

    commands.add_parser("migrate", help="Upgrade the database to the latest schema.")
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
    elif args.command in ("migrate", "check-migrations"):
        from ml4paleo_server import migrations
        from ml4paleo_server.settings import Settings

        database_url = Settings().database_url.get_secret_value()
        if args.command == "migrate":
            migrations.upgrade(database_url)
        else:
            migrations.check(database_url)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
