"""Authentication utilities for bot builder service."""

from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import jwt
import structlog
from jwt import PyJWKClient
from pydantic import BaseModel
from sanic import response
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.builder.config import (
    AUTH0_CLIENT_ID,
    AUTH0_ISSUER,
    AUTH_REQUIRED_AFTER_MINUTES,
    JWKS_URL,
)
from rasa.builder.models import ApiErrorResponse
from rasa.builder.project_info import ProjectInfo

HEADER_USER_ID = "X-User-Id"

structlogger = structlog.get_logger()

RouteHandler = Callable[..., Awaitable[HTTPResponse]]
ProtectedDecorator = Callable[[RouteHandler], RouteHandler]


def verify_auth0_token(token: str) -> Dict[str, Any]:
    """Verify JWT token from Auth0.

    Args:
        token: JWT token string

    Returns:
        Dict containing the token payload

    Raises:
        Exception: If token verification fails
    """
    jwk_client = PyJWKClient(JWKS_URL)
    signing_key = jwk_client.get_signing_key_from_jwt(token)

    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=AUTH0_CLIENT_ID,
        issuer=AUTH0_ISSUER,
    )
    return payload


class Auth0TokenVerificationResult(BaseModel):
    payload: Optional[Dict[str, Any]]
    error_message: Optional[str]


def extract_and_verify_auth0_token(
    auth_header: str,
) -> Auth0TokenVerificationResult:
    """Extract and verify JWT token from Authorization header.

    Args:
        auth_header: Authorization header value

    Returns:
        Auth0TokenVerificationResult: Contains payload and error_message.
    """
    # Check Authorization header format
    if not auth_header.startswith("Bearer "):
        return Auth0TokenVerificationResult(
            payload=None, error_message="Missing or invalid Authorization header"
        )

    # Extract token
    token = auth_header.split(" ")[1]

    # Verify token
    try:
        payload = verify_auth0_token(token)
        return Auth0TokenVerificationResult(payload=payload, error_message=None)
    except Exception as e:
        return Auth0TokenVerificationResult(
            payload=None, error_message=f"Invalid token: {e!s}"
        )


def is_auth_required_now(
    *, always_required: bool = False, project_info: Optional[ProjectInfo] = None
) -> bool:
    """Return whether authentication is required right now based on config.

    - If always_required is True, returns True
    - Else, returns True iff current UTC time >= first_used + configured delta
    """
    if always_required:
        return True
    now = datetime.now(timezone.utc)

    if project_info is None:
        return False

    first_used = project_info.first_used_dt()
    if first_used is None:
        return False

    delta = timedelta(minutes=AUTH_REQUIRED_AFTER_MINUTES)
    auth_required_start = (first_used + delta).astimezone(timezone.utc)
    return now >= auth_required_start


def _check_and_attach_auth(request: Any) -> Tuple[bool, Optional[str]]:
    """Validate Authorization header. On success, store payload on request.ctx."""
    auth_header = request.headers.get("Authorization", "")
    verification_result = extract_and_verify_auth0_token(auth_header)
    if verification_result.error_message is not None:
        return False, verification_result.error_message

    # Attach payload for downstream consumers
    try:
        request.ctx.auth_payload = verification_result.payload
    except Exception:
        # If ctx is not available for some reason, ignore silently
        pass
    return True, None


def protected(*, always_required: bool = False) -> ProtectedDecorator:
    """Decorator to protect routes with conditional authentication.

    Authentication rules:
    - If always_required=True: token is always required
    - Else: token is required only after RASA_AUTH_AFTER timestamp

    On failure returns a 401 with a JSON error body.
    """

    def decorator(f: RouteHandler) -> RouteHandler:
        @wraps(f)
        async def decorated_function(
            request: Request, *args: Any, **kwargs: Any
        ) -> HTTPResponse:
            # Global bypass: allow turning off auth entirely
            if not getattr(request.app.config, "USE_AUTHENTICATION", True):
                return await f(request, *args, **kwargs)

            project_info: Optional[ProjectInfo] = None
            try:
                project_info = request.app.ctx.project_generator.project_info
            except Exception:
                project_info = None

            if is_auth_required_now(
                always_required=always_required, project_info=project_info
            ):
                ok, err = _check_and_attach_auth(request)
                if not ok:
                    structlogger.error(
                        "builder.auth.token_verification_failed", error=err
                    )
                    return response.json(
                        ApiErrorResponse(
                            error=err,
                            details={"expected": "Bearer <valid_token>"},
                        ).model_dump(),
                        status=401,
                    )
            return await f(request, *args, **kwargs)

        return decorated_function

    return decorator
