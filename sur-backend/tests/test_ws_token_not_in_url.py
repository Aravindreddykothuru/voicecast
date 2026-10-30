"""The project WebSocket must take its token from the handshake, not the URL.

A query string is written to the access log in plaintext. uvicorn logged
every session JWT this app ever opened a socket with:

    WebSocket /ws/projects/<id>?token=eyJhbGciOi... [accepted]

Tokens in URLs also reach proxy logs, browser history and Referer headers.
`new WebSocket(url, protocols)` sets Sec-WebSocket-Protocol, which is a
header, so the token travels there instead.
"""
from __future__ import annotations

import pytest

from app.api.ws import BEARER_SUBPROTOCOL, project_events, token_from_subprotocol


def test_token_is_read_from_the_subprotocol_offer():
    assert token_from_subprotocol(f"{BEARER_SUBPROTOCOL}, abc.def.ghi") == "abc.def.ghi"
    # Whitespace around the comma is allowed by the header grammar.
    assert token_from_subprotocol(f"{BEARER_SUBPROTOCOL},abc.def.ghi") == "abc.def.ghi"


@pytest.mark.parametrize("header", [None, "", "bearer.token", "something-else, abc", f"{BEARER_SUBPROTOCOL}, "])
def test_anything_that_is_not_a_real_offer_yields_no_token(header):
    assert token_from_subprotocol(header) is None


def test_the_endpoint_no_longer_accepts_a_token_query_parameter():
    """The signature is the contract: FastAPI binds query parameters by name,
    so a `token` parameter here is the same as advertising ?token=."""
    import inspect

    params = inspect.signature(project_events).parameters
    assert "token" not in params, (
        "project_events still takes a `token` query parameter; that is the thing that "
        "put session JWTs into the access log"
    )
    assert "user_email" in params, "the non-secret dev stub should still be accepted"


def test_a_crafted_token_cannot_smuggle_a_second_value():
    # Only the first two comma-separated entries are meaningful, and the
    # first must be exactly the marker.
    assert token_from_subprotocol("bearer.token, a, b") == "a"
    assert token_from_subprotocol("x, bearer.token, a") is None
