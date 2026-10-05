"""
The server's own access to project storage.
"""

from ml4paleo.storage import StorageGrant

from .settings import Settings


def project_storage(settings: Settings) -> StorageGrant:
    """
    A read-write grant for the root of project storage, with the server's
    own credentials. Never hand this grant to a worker or a browser.
    """
    storage = settings.storage
    credentials = {}
    if storage.access_key_id is not None:
        credentials["access_key_id"] = storage.access_key_id.get_secret_value()
    if storage.secret_access_key is not None:
        credentials["secret_access_key"] = storage.secret_access_key.get_secret_value()
    return StorageGrant(
        url=storage.url,
        access="rw",
        credentials=credentials,  # type: ignore[arg-type]
        endpoint=storage.endpoint,
        region=storage.region,
    )
