import structlog
from dataclasses import dataclass
from typing import Optional


structlogger = structlog.get_logger()


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
    bot_phone: str
    user_name: Optional[str] = None
    user_host: Optional[str] = None
    bot_host: Optional[str] = None
    direction: Optional[str] = None
    stream_id: Optional[str] = None
