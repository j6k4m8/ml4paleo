"""
Database migrations, managed with Alembic.

Use the `ml4paleo-server migrate` command rather than the `alembic` CLI; it
builds the Alembic configuration in code, so no `alembic.ini` is needed.
"""

import pathlib

from alembic import command
from alembic.config import Config

SCRIPT_LOCATION = pathlib.Path(__file__).parent


def alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(SCRIPT_LOCATION))
    # ConfigParser treats "%" as interpolation, so escape it in passwords.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def upgrade(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str, revision: str) -> None:
    command.downgrade(alembic_config(database_url), revision)


def check(database_url: str) -> None:
    """
    Raise if the models have changes that no migration covers.
    """
    command.check(alembic_config(database_url))
