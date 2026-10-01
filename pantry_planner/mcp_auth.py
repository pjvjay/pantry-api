"""
Bearer auth for the /mcp HTTP transport.

The MCP SDK will only install its own bearer middleware alongside
`AuthSettings(issuer_url=..., resource_server_url=...)`, which advertises
an OAuth authorization server this deployment does not run. So the same
SDK pieces — `BearerAuthBackend`, `AuthContextMiddleware`,
`get_access_token()` — are composed by hand around the Starlette app
`mcp_server.http_app()` builds, plus one gate of our own.

Policy (read by every request, never cached at app build, so tests and
operators can change MCP_AUTH_TOKENS without rebuilding the app):

  * tokens configured  → every `http.` request to /mcp needs a valid
                         `Authorization: Bearer <secret>`; anything else is
                         a 401 with a WWW-Authenticate challenge.
  * no tokens          → the endpoint is anonymous, as before; tools that
                         write refuse (mcp_server._require_write) so an
                         unauthenticated internet cannot fill the review
                         queue.
  * stdio              → never passes through here; the operator's own
                         process is trusted.

A token's label becomes the AccessToken.client_id, which the write tools
record as the submitter / reviewer. The secret is never logged or
returned: the only place it exists after parsing is settings().
"""
from __future__ import annotations

import hmac
import json

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser, BearerAuthBackend
from mcp.server.auth.provider import AccessToken, TokenVerifier
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# One static token grants everything; there is no finer-grained principal.
SCOPES = ["read", "plan", "write"]
REALM = "pantry-mcp"


class StaticTokenVerifier(TokenVerifier):
    """Match a presented secret against the configured ones.

    Every configured secret is compared with hmac.compare_digest — never
    `==`, and the loop never breaks on a match or skips on a length
    mismatch, so the time taken does not say which (or whether a) secret
    was close. Reads settings() per call so the token set is live.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        from .config import settings

        presented = token.encode("utf-8")
        matched: str | None = None
        for label, secret in settings().mcp_auth_tokens:
            if hmac.compare_digest(presented, secret.encode("utf-8")) and matched is None:
                matched = label
        if matched is None:
            return None
        return AccessToken(token=token, client_id=matched, scopes=list(SCOPES))


class RequireBearerWhenConfigured:
    """Pure-ASGI gate: 401 any `http.` request that did not authenticate
    while tokens are configured. Lifespan and other scope types pass
    through untouched. A missing or garbled Authorization header reaches
    this gate as "no user" (the SDK backend returns None for anything that
    is not `Bearer <token>`), so it is a 401, never a 500."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from .config import settings

        if settings().mcp_auth_tokens and not isinstance(
                scope.get("user"), AuthenticatedUser):
            await _send_401(send)
            return
        await self.app(scope, receive, send)


async def _send_401(send: Send) -> None:
    body = json.dumps({
        "error": "invalid_token",
        "error_description": (
            "Authentication required: send Authorization: Bearer <token>"),
    }).encode()
    challenge = f'Bearer realm="{REALM}", error="invalid_token"'
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"www-authenticate", challenge.encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})


def install(app):
    """Wrap a Starlette app so requests pass, outermost first, through
    AuthenticationMiddleware(BearerAuthBackend) → AuthContextMiddleware →
    RequireBearerWhenConfigured → the app's routes.

    Starlette's add_middleware inserts at the OUTSIDE, so the layers are
    added innermost first. Adding rather than wrapping keeps the result a
    Starlette app (routes stay inspectable; api.py mounts it as before).
    """
    app.add_middleware(RequireBearerWhenConfigured)
    app.add_middleware(AuthContextMiddleware)
    app.add_middleware(AuthenticationMiddleware,
                       backend=BearerAuthBackend(StaticTokenVerifier()))
    return app
