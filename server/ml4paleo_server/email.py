"""
Outgoing email.

Requests never talk to the mail server: they add a row to `email_outbox`, and
the housekeeper sends queued mail in the background (`send_pending`). A slow
or broken mail server therefore never slows down or breaks a request.

Messages carry single-use links. Message text is sealed with the server secret
key (`ml4paleo_server.sealing`), so a database dump or backup taken while mail
is queued holds no usable links. Each row is deleted as soon as its message is
sent, and the housekeeper deletes rows that failed for good after a week.
"""

import datetime
import email.message
import logging
import smtplib
import ssl
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from . import sealing
from .db import EmailOutbox
from .settings import Settings, SmtpSettings

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
# Retry after 1, 2, 4, 8, then 16 minutes (about half an hour in all).
FIRST_RETRY = datetime.timedelta(minutes=1)

Sender = Callable[[SmtpSettings, email.message.EmailMessage], None]


SEAL_PURPOSE = "email"


def queue_email(
    db: AsyncSession, settings: Settings, to_address: str, subject: str, body: str
) -> None:
    """
    Queue an email. It is sent after the caller's transaction commits.

    Subjects are stored as they are, so keep anything secret in `body`.
    """
    sealed = sealing.seal(
        settings.secret_key.get_secret_value(), SEAL_PURPOSE, body, to_address
    )
    db.add(EmailOutbox(to_address=to_address, subject=subject, body_sealed=sealed))


def send_with_smtp(smtp: SmtpSettings, message: email.message.EmailMessage) -> None:
    assert smtp.host is not None
    # Verify the mail server's certificate; smtplib doesn't by default.
    context = ssl.create_default_context()
    if smtp.security == "tls":
        server = smtplib.SMTP_SSL(smtp.host, smtp.port, timeout=30, context=context)
    else:
        server = smtplib.SMTP(smtp.host, smtp.port, timeout=30)
    with server:
        if smtp.security == "starttls":
            server.starttls(context=context)
        if smtp.username and smtp.password:
            server.login(smtp.username, smtp.password.get_secret_value())
        server.send_message(message)


async def send_pending(
    sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
    send: Sender = send_with_smtp,
    batch_size: int = 20,
) -> int:
    """
    Send up to `batch_size` queued emails and return how many were sent.

    Each message is claimed, sent, and recorded in its own transaction, so a
    crash re-sends at most the one message in flight. Failed sends are retried
    on later calls, up to `MAX_ATTEMPTS` times. Messages sealed with an older
    secret key can't be read, so they fail without a send.
    """
    smtp = settings.smtp
    if not smtp.enabled:
        return 0
    sent = 0
    for _ in range(batch_size):
        async with sessionmaker() as db:
            item = await db.scalar(
                select(EmailOutbox)
                .where(
                    EmailOutbox.status == "queued",
                    EmailOutbox.next_attempt_at <= datetime.datetime.now(datetime.UTC),
                )
                .order_by(EmailOutbox.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if item is None:
                break
            try:
                body = sealing.unseal(
                    settings.secret_key.get_secret_value(),
                    SEAL_PURPOSE,
                    item.body_sealed,
                    item.to_address,
                )
            except sealing.CannotUnseal:
                log.warning("Email %s was sealed with another secret key", item.id)
                item.status = "failed"
                item.last_error = "sealed with a different server secret key"
                await db.commit()
                continue
            message = email.message.EmailMessage()
            message["From"] = smtp.from_address
            message["To"] = item.to_address
            message["Subject"] = item.subject
            message.set_content(body)
            item.attempts += 1
            try:
                await run_in_threadpool(send, smtp, message)
            except Exception as exc:  # noqa: BLE001 - any send failure is retried
                log.warning("Sending email %s failed: %s", item.id, exc)
                item.last_error = str(exc)[:2000]
                item.next_attempt_at = datetime.datetime.now(datetime.UTC) + (
                    FIRST_RETRY * 2 ** (item.attempts - 1)
                )
                if item.attempts >= MAX_ATTEMPTS:
                    item.status = "failed"
                await db.commit()
                if item.status == "queued":
                    # Leave the rest for the next pass rather than hammering a
                    # mail server that is down.
                    break
            else:
                await db.delete(item)
                await db.commit()
                sent += 1
    return sent
