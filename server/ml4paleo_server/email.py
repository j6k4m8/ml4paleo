"""
Outgoing email.

Requests never talk to the mail server: they add a row to `email_outbox`, and
the housekeeper sends queued mail in the background (`send_pending`). A slow
or broken mail server therefore never slows down or breaks a request.
"""

import datetime
import email.message
import logging
import smtplib
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from .db import EmailOutbox
from .settings import SmtpSettings

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5

Sender = Callable[[SmtpSettings, email.message.EmailMessage], None]


def queue_email(db: AsyncSession, to_address: str, subject: str, body: str) -> None:
    """
    Queue an email. It is sent after the caller's transaction commits.
    """
    db.add(EmailOutbox(to_address=to_address, subject=subject, body=body))


def send_with_smtp(smtp: SmtpSettings, message: email.message.EmailMessage) -> None:
    assert smtp.host is not None
    connect = smtplib.SMTP_SSL if smtp.security == "tls" else smtplib.SMTP
    with connect(smtp.host, smtp.port, timeout=30) as server:
        if smtp.security == "starttls":
            server.starttls()
        if smtp.username and smtp.password:
            server.login(smtp.username, smtp.password.get_secret_value())
        server.send_message(message)


async def send_pending(
    sessionmaker: async_sessionmaker[AsyncSession],
    smtp: SmtpSettings,
    send: Sender = send_with_smtp,
    batch_size: int = 20,
) -> int:
    """
    Send up to `batch_size` queued emails and return how many were sent.
    Failed sends are retried on later calls, up to `MAX_ATTEMPTS` times.
    """
    if not smtp.enabled:
        return 0
    sent = 0
    async with sessionmaker() as db:
        queued = (
            await db.scalars(
                select(EmailOutbox)
                .where(EmailOutbox.status == "queued")
                .order_by(EmailOutbox.created_at)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
        ).all()
        for item in queued:
            message = email.message.EmailMessage()
            message["From"] = smtp.from_address
            message["To"] = item.to_address
            message["Subject"] = item.subject
            message.set_content(item.body)
            item.attempts += 1
            try:
                await run_in_threadpool(send, smtp, message)
            except Exception as exc:  # noqa: BLE001 - any send failure is retried
                log.warning("Sending email %s failed: %s", item.id, exc)
                item.last_error = str(exc)[:2000]
                if item.attempts >= MAX_ATTEMPTS:
                    item.status = "failed"
            else:
                item.status = "sent"
                item.sent_at = datetime.datetime.now(datetime.UTC)
                sent += 1
        await db.commit()
    return sent
