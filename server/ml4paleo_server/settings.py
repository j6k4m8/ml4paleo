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
from typing import Any, Literal
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
    # How workers reach project storage. "proxy" (the default): through the
    # API, limited to each job's own files for as long as it holds the job.
    # "direct": workers on this machine get the server's own credentials,
    # which reach every project; use it only when the local workers are as
    # trusted as the server and can reach the storage themselves (for example
    # a shared disk). Remote and burst workers always use the proxy.
    worker_access: Literal["proxy", "direct"] = "proxy"
    # How long garbage collection keeps replaced and failed artifacts.
    keep_superseded_days: float = Field(default=7, ge=0)
    keep_failed_hours: float = Field(default=48, ge=0)


class AuthSettings(BaseModel):
    # "open": anyone can sign up. "invite": only people with an invite link.
    # Admins can change this at runtime; this is the starting value.
    signup_mode: Literal["open", "invite"] = "open"
    password_min_length: int = Field(default=12, ge=8)
    session_idle_days: float = 7
    session_max_days: float = 30


class SmtpSettings(BaseModel):
    """
    Outgoing email. Email is off when `host` is unset: signups are not
    verified, and admins reset passwords from the command line.
    """

    host: str | None = None
    port: int = 587
    username: str | None = None
    password: SecretStr | None = None
    from_address: str = "ml4paleo <no-reply@localhost>"
    # "starttls" upgrades a plain connection; "tls" connects with TLS.
    security: Literal["starttls", "tls", "none"] = "starttls"

    @property
    def enabled(self) -> bool:
        return bool(self.host)


class QuotaSettings(BaseModel):
    """
    Default per-user limits for this deploy. Each is independent, and unset
    (None) means unlimited. Admins can override any of them per user.

    People see two plain limits: how much they can store, and how many trained
    models they can keep. Compute-time limits exist for deploys that need
    them, but are off by default; the queue and the available machines limit
    how much runs at once.
    """

    storage_gb: float | None = Field(default=10, ge=0)
    trained_models: int | None = Field(default=20, ge=0)
    cpu_hours_per_day: float | None = Field(default=None, ge=0)
    gpu_hours_per_day: float | None = Field(default=None, ge=0)


class V1Settings(BaseModel):
    """
    Importing jobs from the ml4paleo v1 app this one replaced. People claim
    a job by visiting its old link (`/job/<id>`) signed in.
    """

    # A folder with v1's jobs.json (the API reads nothing else of v1's),
    # mounted read-only; unset when there are no v1 jobs to import. A worker
    # started with --v1-volume does the importing.
    volume_path: pathlib.Path | None = None
    # Claims each account, and each address, may try per hour. v1 job ids are
    # only six hex digits, so this keeps people from guessing other people's
    # jobs.
    claims_per_hour: int = Field(default=30, ge=1)
    # Of those, how many may be ids that aren't v1 jobs (people mistype a
    # few; guessing misses nearly every time).
    misses_per_hour: int = Field(default=5, ge=1)
    # Misses from everyone per hour. Past this, nobody can claim until the
    # hour is up; reaching it takes dozens of addresses.
    site_misses_per_hour: int = Field(default=200, ge=1)


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
    auth: AuthSettings = AuthSettings()
    smtp: SmtpSettings = SmtpSettings()
    quota: QuotaSettings = QuotaSettings()
    v1: V1Settings = V1Settings()
    # The first admin account's password (usually M4P_INITIAL_ADMIN_PASSWORD_FILE).
    # Without it, `migrate` generates one and prints it once.
    initial_admin_password: SecretStr | None = None
    # The token that workers on this machine share (usually
    # M4P_LOCAL_WORKER_TOKEN_FILE). `migrate` registers it as the "local" worker.
    local_worker_token: SecretStr | None = None
    # The built web app (`web/build`). When missing, the API still runs and
    # serves a placeholder page.
    web_dir: pathlib.Path | None = None
    # A Neuroglancer build to serve at /neuroglancer/ (the server image has
    # one). Without it, there is no Neuroglancer link.
    neuroglancer_dir: pathlib.Path | None = None
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
