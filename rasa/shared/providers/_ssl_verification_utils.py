import os
import litellm
from rasa.shared.constants import (
    RASA_CA_BUNDLE_ENV_VAR,
    REQUESTS_CA_BUNDLE_ENV_VAR,
    RASA_SSL_CERTIFICATE_ENV_VAR,
    LITELLM_SSL_VERIFY_ENV_VAR,
    LITELLM_SSL_CERTIFICATE_ENV_VAR,
)

import structlog

structlogger = structlog.get_logger()


def ensure_ssl_certificates_for_litellm() -> None:
    """
    Ensure SSL certificates configuration for LiteLLM based on environment
    variables.

    Environment variable priority (ssl verify):
    1. `RASA_CA_BUNDLE`: Preferred for SSL verification.
    2. `REQUESTS_CA_BUNDLE`: Deprecated; use `RASA_CA_BUNDLE_ENV_VAR` instead.
    3. `SSL_VERIFY`: Fallback for SSL verification.

    Environment variable priority (ssl certificate):
    1. `RASA_SSL_CERTIFICATE`: Preferred for client certificate.
    2. `SSL_CERTIFICATE`: Fallback for client certificate.
    """
    ssl_verify = (
        os.environ.get(RASA_CA_BUNDLE_ENV_VAR)
        # Deprecated
        or os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR)
        # From LiteLLM, use as a fallback
        or os.environ.get(LITELLM_SSL_VERIFY_ENV_VAR)
        or None
    )

    ssl_certificate = (
        os.environ.get(RASA_SSL_CERTIFICATE_ENV_VAR)
        # From LiteLLM, use as a fallback
        or os.environ.get(LITELLM_SSL_CERTIFICATE_ENV_VAR)
        or None
    )

    structlogger.debug(
        "ensure_ssl_certificates_for_litellm",
        ssl_verify=ssl_verify,
        ssl_certificate=ssl_certificate,
    )

    if ssl_verify is not None:
        litellm.ssl_verify = ssl_verify  # type: ignore
    if ssl_certificate is not None:
        litellm.ssl_certificate = ssl_certificate
