import structlog

from rasa.utils.licensing import (
    PRODUCT_AREA,
    VOICE_SCOPE,
    validate_license_from_env,
)

structlogger = structlog.get_logger()


def validate_voice_license_scope() -> None:
    """Validate that the correct license scope is present."""
    structlogger.info(
        f"Validating current Rasa Pro license scope which must include "
        f"the '{VOICE_SCOPE}' scope to use the voice channel."
    )

    voice_product_scope = PRODUCT_AREA + " " + VOICE_SCOPE
    validate_license_from_env(product_area=voice_product_scope)
