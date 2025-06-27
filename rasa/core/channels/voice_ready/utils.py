from dataclasses import dataclass
from typing import Optional

import structlog

from rasa.shared.exceptions import InvalidConfigException

structlogger = structlog.get_logger()


def validate_username_password_credentials(
    username: Optional[str], password: Optional[str], channel_name: str
) -> None:
    """Validate that username and password are both provided or both None.

    Args:
        username: The username credential, or None
        password: The password credential, or None
        channel_name: The name of the channel for error message

    Raises:
        InvalidConfigException: If only one of username/password is provided
    """
    if (not username) != (not password):
        raise InvalidConfigException(
            f"In {channel_name} channel, either both username and password "
            "or neither should be provided."
        )


def validate_voice_license_scope() -> None:
    from rasa.utils.licensing import (
        PRODUCT_AREA,
        VOICE_SCOPE,
        validate_license_from_env,
    )

    """Validate that the correct license scope is present."""
    structlogger.info(
        f"Validating current Rasa Pro license scope which must include "
        f"the '{VOICE_SCOPE}' scope to use the voice channel."
    )

    voice_product_scope = PRODUCT_AREA + " " + VOICE_SCOPE
    validate_license_from_env(product_area=voice_product_scope)


@dataclass
class CallParameters:
    """Standardized call parameters for voice channels."""

    call_id: str
    user_phone: str
    bot_phone: Optional[str] = None
    user_name: Optional[str] = None
    user_host: Optional[str] = None
    bot_host: Optional[str] = None
    direction: Optional[str] = None
    stream_id: Optional[str] = None
