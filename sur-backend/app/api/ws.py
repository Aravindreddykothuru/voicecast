"""WebSocket gateway: one channel per project, /ws/projects/{id}.

Subscribes to the project's Redis pub/sub channel (see app/pipeline/events.py)
and forwards every message verbatim to the connected client(s) as JSON.
Multiple browser tabs/viewers on the same project each get their own
subscriber -- Redis pub/sub fans out to all of them for free.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.config import get_settings
from app.core.security import decode_access_token
from app.db import SessionLocal
from app.models.base import is_uuid
from app.models.project import Project
from app.models.user import User
from app.pipeline.events import channel_name

logger = logging.getLogger(__name__)
router = APIRouter()

# 1008 = policy violation. Closing with a reason rather than accepting and
# going silent, so a client can tell "not yours" from "no events yet".
WS_POLICY_VIOLATION = 1008


BEARER_SUBPROTOCOL = "bearer.token"


def token_from_subprotocol(header: str | None) -> str | None:
    """The session token out of Sec-WebSocket-Protocol, or None.

    A browser cannot set arbitrary headers on a WebSocket handshake, which is
    why this used to travel as ?token=. It does not have to: the second
    argument to `new WebSocket(url, protocols)` becomes
    Sec-WebSocket-Protocol, and that is a header. The difference matters --
    a query string is written to the access log in plaintext, and uvicorn
    duly logged every session JWT this app has ever opened a socket with:

        WebSocket /ws/projects/<id>?token=eyJhbGciOi... [accepted]

    Tokens in URLs also reach proxy logs, browser history and Referer
    headers. The offer is "bearer.token, <the token>".
    """
    if not header:
        return None
    parts = [p.strip() for p in header.split(",")]
    if len(parts) >= 2 and parts[0] == BEARER_SUBPROTOCOL and parts[1]:
        return parts[1]
    return None


def _caller_owns_project(project_id: str, token: str | None, user_email: str | None) -> bool:
    """Same two-mode identity as get_current_user: a bearer token wins, and
    the X-User-Email dev stub is the fallback. The token arrives as a
    WebSocket subprotocol (see token_from_subprotocol); user_email is a
    non-secret dev stub and stays a query parameter."""
    if not is_uuid(project_id):
        return False

    db = SessionLocal()
    try:
        user = None
        if token:
            user_id = decode_access_token(token)
            if user_id:
                user = db.get(User, user_id)
        elif user_email and get_settings().dev_email_auth_enabled:
            # Same stub as get_current_user, and gated the same way: in
            # production an email query parameter is not an identity, it is
            # just a string the caller chose.
            user = db.execute(
                select(User).where(User.email == user_email.strip().lower())
            ).scalar_one_or_none()

        if user is None:
            return False
        project = db.get(Project, project_id)
        return project is not None and project.user_id == user.id
    finally:
        db.close()


@router.websocket("/ws/projects/{project_id}")
async def project_events(
    websocket: WebSocket,
    project_id: str,
    user_email: str | None = None,
):
    # The token is a subprotocol, never a query parameter: the query string
    # is logged in plaintext by uvicorn and every proxy in between.
    offered = websocket.headers.get("sec-websocket-protocol")
    token = token_from_subprotocol(offered)

    # Authorize BEFORE accepting. This channel carries transcript text,
    # detected language and error messages; without this check anyone who
    # knew or guessed a project id could stream another user's run.
    if not _caller_owns_project(project_id, token, user_email):
        await websocket.close(code=WS_POLICY_VIOLATION, reason="Not authorized for this project")
        return

    # A client that offered a subprotocol must be told which one was chosen,
    # or the browser fails the handshake after the server has accepted it.
    await websocket.accept(subprotocol=BEARER_SUBPROTOCOL if token else None)
    settings = get_settings()

    try:
        redis_client = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        pubsub = redis_client.pubsub()
        await asyncio.wait_for(pubsub.subscribe(channel_name(project_id)), timeout=2.0)
    except Exception as e:
        logger.warning("Redis unavailable for project %s websocket: %s; running in keepalive mode", project_id, e)
        # Send initial connected message to frontend
        await websocket.send_text('{"type":"connected","status":"online","redis":"offline"}')
        try:
            while True:
                await asyncio.sleep(20)
                await websocket.send_text('{"type":"ping"}')
        except (WebSocketDisconnect, asyncio.CancelledError):
            logger.info("client disconnected from project %s events (keepalive)", project_id)
        return

    async def forward() -> None:
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                data = message["data"]
                text = data.decode() if isinstance(data, bytes) else data
                await websocket.send_text(text)
        except Exception:
            pass

    forward_task = asyncio.create_task(forward())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("client disconnected from project %s events", project_id)
    finally:
        forward_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await forward_task
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(channel_name(project_id))
            await pubsub.close()
            await redis_client.close()

