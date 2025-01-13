import os
from functools import cache
from http import HTTPStatus
from typing import Any, Dict, Optional

import jwt
import requests
import structlog

from rasa.shared.exceptions import RasaException

structlogger = structlog.get_logger()


AUTH_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8081/auth")

AUTH_REALM = os.getenv("KEYCLOAK_REALM", "rasa-studio")


class UserToServiceAuthenticationError(RasaException):
    """Raised when the user authentication fails."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


@cache
def get_public_key_from_keycloak() -> Optional[str]:
    """Fetch the public key from the keycloak server."""
    realm_url = f"{AUTH_URL}/realms/{AUTH_REALM}"

    try:
        response = requests.get(realm_url)
    except requests.RequestException as error:
        structlogger.error("model_api.auth.keycloak_request_failed", error=str(error))
        return None

    if response.status_code != HTTPStatus.OK:
        structlogger.error(
            "model_api.auth.keycloak_public_key_fetch_failed",
            status_code=response.status_code,
            response=response.text,
        )
        return None

    public_key = response.json().get("public_key")

    if public_key is None:
        structlogger.error(
            "model_runner.keycloak_public_key_not_found",
            response=response.text,
        )
        return None

    public_key = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"
    return public_key


def authenticate_user_to_service(token: str) -> Dict[str, Any]:
    """Authenticate the user to the model service."""
    if not token:
        structlogger.debug("model_api.auth.no_token_provided")
        raise UserToServiceAuthenticationError("No token provided.")

    public_key = get_public_key_from_keycloak()

    if public_key is None:
        raise UserToServiceAuthenticationError(
            "Failed to fetch public key from keycloak."
        )

    try:
        return jwt.decode(
            token,
            public_key,
            algorithms=["RS256", "HS256", "HS512", "ES256"],
            audience="account",
        )
    except jwt.InvalidKeyError as error:
        structlogger.info("model_api.auth.invalid_jwt_key", error=str(error))
        raise UserToServiceAuthenticationError("Invalid JWT key.")
    except jwt.InvalidTokenError as error:
        structlogger.info("model_api.auth.invalid_jwt_token", error=str(error))
        raise UserToServiceAuthenticationError("Invalid JWT token.") from error
