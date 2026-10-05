"""
Server configuration, read from environment variables.

Every setting has the prefix `M4P_`, and nested settings use `__` (for
example `M4P_STORAGE__URL`). Any setting can instead be read from a file by
setting the same name with a `_FILE` suffix (for example
`M4P_SECRET_KEY_FILE=/run/secrets/m4p_secret_key`), which is how Docker and
Kubernetes secrets are usually mounted. A direct value wins over a `_FILE`.
"""

import os
import pathlib
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ENV_PREFIX = "M4P_"
NESTED_DELIMITER = "__"


class StorageSettings(BaseModel):
    """
    Where project data lives. `url` is a storage URL (see
    `ml4paleo.storage.StorageGrant`), for example `s3://ml4paleo` with
    `endpoint` pointing at SeaweedFS on a single box.
    """

    url: str = "file:///var/lib/ml4paleo/data"
    endpoint: str | None = None
    # The S3 endpoint as browsers reach it (for presigned uploads), if it
    # differs from `endpoint`.
    public_endpoint: str | None = None
    region: str | None = None
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=NESTED_DELIMITER,
        extra="ignore",
    )

    # The URL people use to reach the app, for example https://ml4paleo.org.
    public_url: str = "http://localhost:8000"
    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://ml4paleo:ml4paleo@localhost:5432/ml4paleo"
    )
    # Signs session and CSRF tokens. Must be long and random in production.
    secret_key: SecretStr = Field(default=SecretStr(""), repr=False)
    storage: StorageSettings = StorageSettings()
    # The built web app (`web/build`). When missing, the API still runs and
    # serves a placeholder page.
    web_dir: pathlib.Path | None = None
    # Number of API worker processes.
    api_workers: int = Field(default=4, ge=1)
    # Addresses of reverse proxies whose X-Forwarded-For header is trusted
    # (comma-separated IPs or networks, or "*"). Client IPs feed rate limits,
    # so only trust proxies that overwrite the header.
    forwarded_allow_ips: str = "127.0.0.1"
    # Plain HTTP sends passwords and session cookies in the clear, so the
    # server refuses an http:// public URL unless it is a local address or
    # this is set.
    allow_insecure_http: bool = False

    @field_validator("public_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_https(self) -> bool:
        return self.public_url.startswith("https://")

    @property
    def is_local(self) -> bool:
        host = urlsplit(self.public_url).hostname or ""
        return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _FileEnvSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


class _FileEnvSource(PydanticBaseSettingsSource):
    """
    Read `M4P_<NAME>_FILE` variables: each one names a file whose contents
    (with surrounding whitespace removed) become the value of `M4P_<NAME>`.
    Empty files are ignored.
    """

    def get_field_value(self, field, field_name):  # pragma: no cover - unused
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, path in os.environ.items():
            if not (key.startswith(ENV_PREFIX) and key.endswith("_FILE")):
                continue
            name = key[len(ENV_PREFIX) : -len("_FILE")].lower()
            content = pathlib.Path(path).read_text().strip()
            if not content:
                # An empty file means the setting is unset.
                continue
            target = values
            *parents, leaf = name.split(NESTED_DELIMITER)
            for parent in parents:
                target = target.setdefault(parent, {})
            target[leaf] = content
        return values
