"""
Sign-up, sign-in, sessions, password changes and resets, email
verification, and two-factor setup.
"""

import datetime
import re
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import AfterValidator, BaseModel, Field
from sqlalchemy import or_, select, update

from ..auth import expire_reset_tokens, passwords, ratelimit, totp
from ..auth.deps import DbSession, EngineDep, OptionalAuth, SettingsDep, SetupAuth
from ..auth.ratelimit import client_key
from ..auth.sessions import (
    clear_session_cookie,
    create_session,
    delete_session,
    delete_user_sessions,
    set_session_cookie,
)
from ..auth.tokens import csrf_token, new_token, token_hash
from ..db import AuthToken, User
from ..email import queue_email
from ..settings import Settings
from ..site_settings import get_signup_mode

router = APIRouter(prefix="/api/auth", tags=["auth"])

USERNAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{2,31}")
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
VERIFY_TOKEN_LIFETIME = datetime.timedelta(days=3)
RESET_TOKEN_LIFETIME = datetime.timedelta(hours=1)
MINUTE = datetime.timedelta(minutes=1)
QUARTER_HOUR = datetime.timedelta(minutes=15)
HOUR = datetime.timedelta(hours=1)
# Names people could mistake for the site's own accounts.
RESERVED_USERNAMES = frozenset(
    {"admin", "administrator", "root", "system", "support", "ml4paleo", "staff"}
)


def _is_reserved(username: str) -> bool:
    """
    True for reserved names and look-alikes such as "admin1" or "root_".
    """
    core = re.sub(r"[._-]", "", username).rstrip("0123456789")
    return core in RESERVED_USERNAMES


def _username(value: str) -> str:
    value = value.strip().lower()
    if not USERNAME_PATTERN.fullmatch(value):
        raise ValueError(
            "Usernames are 3-32 characters: lowercase letters, digits, '.', '_', "
            "or '-', starting with a letter or digit."
        )
    return value


def _email(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    value = value.strip().lower()
    if len(value) > 254 or not EMAIL_PATTERN.fullmatch(value):
        raise ValueError("Enter a valid email address.")
    return value


Username = Annotated[str, AfterValidator(_username)]
Email = Annotated[str | None, AfterValidator(_email)]
Password = Annotated[str, Field(max_length=passwords.MAX_PASSWORD_LENGTH)]


class UserOut(BaseModel):
    id: str
    username: str
    email: str | None
    email_verified: bool
    is_admin: bool
    status: str
    must_change_password: bool
    two_factor_enabled: bool

    @classmethod
    def of(cls, user: User) -> "UserOut":
        return cls(
            id=str(user.id),
            username=user.username,
            email=user.email,
            email_verified=user.email_verified_at is not None,
            is_admin=user.is_admin,
            status=user.status,
            must_change_password=user.must_change_password,
            two_factor_enabled=user.totp_secret_enc is not None,
        )


class SessionOut(BaseModel):
    user: UserOut
    csrf_token: str
    # Steps the user must finish before using the app.
    required_steps: list[str]


def _session_out(settings: Settings, user: User, token: str) -> SessionOut:
    steps = []
    if user.must_change_password:
        steps.append("change_password")
    if user.is_admin and user.totp_secret_enc is None:
        steps.append("set_up_two_factor")
    if user.status == "unverified":
        steps.append("verify_email")
    return SessionOut(
        user=UserOut.of(user),
        csrf_token=csrf_token(settings.secret_key.get_secret_value(), token),
        required_steps=steps,
    )


def _check_password(settings: Settings, password: str, user: User) -> None:
    problems = passwords.password_problems(
        password,
        min_length=settings.auth.password_min_length,
        username=user.username,
        email=user.email,
    )
    if problems:
        raise HTTPException(status_code=422, detail=" ".join(problems))


async def _issue_token(
    db: DbSession, kind: str, lifetime: datetime.timedelta, **fields
) -> str:
    token = new_token()
    db.add(
        AuthToken(
            token_hash=token_hash(token),
            kind=kind,
            expires_at=datetime.datetime.now(datetime.UTC) + lifetime,
            **fields,
        )
    )
    return token


async def _use_token(db: DbSession, kind: str, token: str) -> AuthToken:
    """
    Mark a single-use token as used and return it, or raise 400 if it is
    unknown, used, or expired.
    """
    now = datetime.datetime.now(datetime.UTC)
    used = await db.scalar(
        update(AuthToken)
        .where(
            AuthToken.token_hash == token_hash(token),
            AuthToken.kind == kind,
            AuthToken.used_at.is_(None),
            AuthToken.expires_at > now,
        )
        .values(used_at=now)
        .returning(AuthToken)
    )
    if used is None:
        raise HTTPException(
            status_code=400, detail="This link is invalid or has expired."
        )
    return used


def _queue_verification(db: DbSession, settings: Settings, user: User, token: str):
    queue_email(
        db,
        settings,
        user.email or "",
        "Confirm your ml4paleo email address",
        f"Hi {user.username},\n\nConfirm your email address by opening this link:\n\n"
        f"{settings.public_url}/verify-email?token={token}\n\n"
        "If you didn't sign up for ml4paleo, you can ignore this message.\n",
    )


class ConfigOut(BaseModel):
    signup_mode: str
    email_enabled: bool
    password_min_length: int


@router.get("/config")
async def auth_config(db: DbSession, settings: SettingsDep) -> ConfigOut:
    return ConfigOut(
        signup_mode=await get_signup_mode(db, settings),
        email_enabled=settings.smtp.enabled,
        password_min_length=settings.auth.password_min_length,
    )


class SignupIn(BaseModel):
    username: Username
    password: Password
    email: Email = None
    invite: str | None = None


@router.post("/signup", status_code=201)
async def signup(
    body: SignupIn,
    request: Request,
    response: Response,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> SessionOut:
    await ratelimit.hit(
        engine, f"signup:ip:{client_key(request)}", limit=5, window=HOUR
    )
    if _is_reserved(body.username):
        raise HTTPException(status_code=422, detail="That username is reserved.")
    invite = None
    if await get_signup_mode(db, settings) == "invite":
        if not body.invite:
            raise HTTPException(status_code=403, detail="Sign-up needs an invite link.")
        invite = await _use_token(db, "invite", body.invite)
    if settings.smtp.enabled and body.email is None:
        raise HTTPException(status_code=422, detail="Enter an email address.")
    taken = await db.scalar(
        select(User.id).where(
            or_(User.username == body.username, User.email == body.email)
            if body.email
            else User.username == body.username
        )
    )
    if taken is not None:
        raise HTTPException(status_code=409, detail="That username or email is taken.")

    user = User(username=body.username, email=body.email)
    _check_password(settings, body.password, user)
    user.password_hash = await passwords.hash_password(body.password)
    # An address counts as verified only when ml4paleo has emailed it: a
    # verification link, or an admin's invite addressed to it. Without SMTP,
    # addresses stay unverified, so password resets never go to them.
    invited_email = invite is not None and invite.email and invite.email == body.email
    if invited_email:
        user.email_verified_at = datetime.datetime.now(datetime.UTC)
    elif settings.smtp.enabled:
        user.status = "unverified"
    db.add(user)
    await db.flush()
    if user.status == "unverified":
        token = await _issue_token(
            db, "verify", VERIFY_TOKEN_LIFETIME, user_id=user.id, email=user.email
        )
        _queue_verification(db, settings, user, token)
    session_token = await create_session(db, settings, user, request)
    await db.commit()
    set_session_cookie(response, settings, session_token)
    return _session_out(settings, user, session_token)


class LoginIn(BaseModel):
    username: str = Field(max_length=254)
    password: Password
    totp_code: str | None = Field(default=None, max_length=12)


@router.post("/login")
async def login(
    body: LoginIn,
    request: Request,
    response: Response,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> SessionOut:
    name = body.username.strip().lower()
    client = client_key(request)
    await ratelimit.hit(engine, f"login:ip:{client}", limit=30, window=MINUTE)
    user = await db.scalar(
        select(User).where(or_(User.username == name, User.email == name))
    )
    # Limit guesses per account (whichever name it was given by) and client,
    # so one attacker can't lock the owner out from elsewhere. A much looser
    # cap per account slows down guessing from many addresses at once.
    account = f"user:{user.id}" if user else f"name:{name}"
    await ratelimit.hit(engine, f"login:{account}:{client}", limit=5, window=MINUTE)
    await ratelimit.hit(engine, f"login:{account}:{client}:h", limit=30, window=HOUR)
    await ratelimit.hit(engine, f"login:{account}", limit=300, window=HOUR)
    valid = await passwords.verify_password(
        body.password, user.password_hash if user else None
    )
    if user is None or not valid or user.status == "disabled":
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    if user.totp_secret_enc is not None:
        if not body.totp_code:
            raise HTTPException(status_code=401, detail="totp_required")
        # Count wrong codes only, so signing in normally never locks anyone out.
        failures = f"totp-failures:user:{user.id}"
        await ratelimit.peek(engine, failures, limit=5, window=QUARTER_HOUR)
        try:
            secret = totp.decrypt(
                settings.secret_key.get_secret_value(), user.totp_secret_enc
            )
        except totp.UnreadableSecret:
            raise HTTPException(
                status_code=401,
                detail="Two-factor sign-in can't be checked for this account. "
                "Ask an administrator to reset it.",
            ) from None
        step = totp.verify(secret, body.totp_code, after_step=user.totp_last_step)
        # Record the code's step only if no concurrent sign-in used it first.
        claimed = (
            step is not None
            and await db.scalar(
                update(User)
                .where(
                    User.id == user.id,
                    or_(User.totp_last_step.is_(None), User.totp_last_step < step),
                )
                .values(totp_last_step=step)
                .returning(User.id)
            )
            is not None
        )
        if not claimed:
            await ratelimit.hit(engine, failures, limit=5, window=QUARTER_HOUR)
            raise HTTPException(status_code=401, detail="Wrong two-factor code.")
    session_token = await create_session(db, settings, user, request)
    await db.commit()
    set_session_cookie(response, settings, session_token)
    return _session_out(settings, user, session_token)


@router.post("/logout", status_code=204)
async def logout(
    auth: OptionalAuth, response: Response, db: DbSession, settings: SettingsDep
) -> None:
    if auth is not None:
        await delete_session(db, auth.token)
        await db.commit()
    clear_session_cookie(response, settings)
    # Project data the browser cached (viewers cache image chunks) shouldn't
    # outlive the session, for example on a shared lab computer.
    response.headers["Clear-Site-Data"] = '"cache"'


@router.get("/session")
async def current_session(auth: SetupAuth, settings: SettingsDep) -> SessionOut:
    return _session_out(settings, auth.user, auth.token)


class PasswordChangeIn(BaseModel):
    current_password: Password
    new_password: Password


@router.post("/password", status_code=204)
async def change_password(
    body: PasswordChangeIn,
    auth: SetupAuth,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> None:
    await ratelimit.hit(engine, f"password:user:{auth.user.id}", limit=5, window=MINUTE)
    if not await passwords.verify_password(
        body.current_password, auth.user.password_hash
    ):
        raise HTTPException(status_code=403, detail="Your current password is wrong.")
    if body.new_password == body.current_password:
        raise HTTPException(status_code=422, detail="Choose a new password.")
    _check_password(settings, body.new_password, auth.user)
    auth.user.password_hash = await passwords.hash_password(body.new_password)
    auth.user.must_change_password = False
    await delete_user_sessions(db, auth.user.id, keep_token=auth.token)
    await expire_reset_tokens(db, auth.user.id)
    await db.commit()


class TotpSetupOut(BaseModel):
    secret: str
    otpauth_uri: str


@router.post("/totp/setup")
async def start_totp_setup(
    auth: SetupAuth, db: DbSession, settings: SettingsDep
) -> TotpSetupOut:
    if auth.user.totp_secret_enc is not None:
        raise HTTPException(status_code=409, detail="Two-factor sign-in is already on.")
    secret = totp.new_secret()
    auth.user.totp_pending_enc = totp.encrypt(
        settings.secret_key.get_secret_value(), secret
    )
    await db.commit()
    return TotpSetupOut(
        secret=secret, otpauth_uri=totp.provisioning_uri(secret, auth.user.username)
    )


class TotpConfirmIn(BaseModel):
    code: str = Field(max_length=12)


@router.post("/totp/confirm", status_code=204)
async def confirm_totp_setup(
    body: TotpConfirmIn,
    auth: SetupAuth,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> None:
    await ratelimit.hit(engine, f"totp:user:{auth.user.id}", limit=5, window=MINUTE)
    if auth.user.totp_pending_enc is None:
        raise HTTPException(status_code=409, detail="Start two-factor setup first.")
    key = settings.secret_key.get_secret_value()
    try:
        pending = totp.decrypt(key, auth.user.totp_pending_enc)
    except totp.UnreadableSecret:
        auth.user.totp_pending_enc = None
        await db.commit()
        raise HTTPException(
            status_code=409, detail="Start two-factor setup again."
        ) from None
    step = totp.verify(pending, body.code)
    if step is None:
        raise HTTPException(
            status_code=422, detail="That code is wrong. Try the next one."
        )
    auth.user.totp_secret_enc = auth.user.totp_pending_enc
    auth.user.totp_pending_enc = None
    auth.user.totp_last_step = step
    await db.commit()


class TokenIn(BaseModel):
    token: str = Field(max_length=128)


@router.post("/verify-email", status_code=204)
async def verify_email(body: TokenIn, db: DbSession) -> None:
    used = await _use_token(db, "verify", body.token)
    user = await db.get(User, used.user_id)
    if user is None or user.email != used.email:
        raise HTTPException(
            status_code=400, detail="This link is invalid or has expired."
        )
    user.email_verified_at = datetime.datetime.now(datetime.UTC)
    if user.status == "unverified":
        user.status = "active"
    await db.commit()


class ResetRequestIn(BaseModel):
    email: Email


@router.post("/password-reset/request", status_code=202)
async def request_password_reset(
    body: ResetRequestIn,
    request: Request,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> None:
    """
    Always answers 202, so the response doesn't reveal which emails have
    accounts.
    """
    if not settings.smtp.enabled or body.email is None:
        return
    await ratelimit.hit(
        engine, f"reset:ip:{client_key(request)}", limit=10, window=HOUR
    )
    await ratelimit.hit(engine, f"reset:email:{body.email}", limit=3, window=HOUR)
    user = await db.scalar(select(User).where(User.email == body.email))
    if user is None or user.email_verified_at is None or user.status == "disabled":
        return
    token = await _issue_token(
        db, "reset", RESET_TOKEN_LIFETIME, user_id=user.id, email=user.email
    )
    queue_email(
        db,
        settings,
        body.email,
        "Reset your ml4paleo password",
        f"Hi {user.username},\n\nChoose a new password by opening this link "
        f"within an hour:\n\n{settings.public_url}/reset-password?token={token}\n\n"
        "If you didn't ask to reset your password, you can ignore this message.\n",
    )
    await db.commit()


class ResetConfirmIn(BaseModel):
    token: str = Field(max_length=128)
    new_password: Password


@router.post("/password-reset/confirm", status_code=204)
async def confirm_password_reset(
    body: ResetConfirmIn, db: DbSession, settings: SettingsDep
) -> None:
    used = await _use_token(db, "reset", body.token)
    user = await db.get(User, used.user_id)
    if user is None or user.email != used.email or user.status == "disabled":
        raise HTTPException(
            status_code=400, detail="This link is invalid or has expired."
        )
    _check_password(settings, body.new_password, user)
    user.password_hash = await passwords.hash_password(body.new_password)
    user.must_change_password = False
    await delete_user_sessions(db, user.id)
    await expire_reset_tokens(db, user.id)
    await db.commit()


@router.post("/verify-email/resend", status_code=202)
async def resend_verification(
    auth: SetupAuth, db: DbSession, engine: EngineDep, settings: SettingsDep
) -> None:
    if auth.user.status != "unverified" or not auth.user.email:
        return
    await ratelimit.hit(engine, f"verify:user:{auth.user.id}", limit=3, window=HOUR)
    token = await _issue_token(
        db, "verify", VERIFY_TOKEN_LIFETIME, user_id=auth.user.id, email=auth.user.email
    )
    _queue_verification(db, settings, auth.user, token)
    await db.commit()
