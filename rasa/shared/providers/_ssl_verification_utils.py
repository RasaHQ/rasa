import os
from typing import Optional, Union

import httpx
import litellm
from rasa.shared.constants import (
    RASA_CA_BUNDLE_ENV_VAR,
    REQUESTS_CA_BUNDLE_ENV_VAR,
    RASA_SSL_CERTIFICATE_ENV_VAR,
    LITELLM_SSL_VERIFY_ENV_VAR,
    LITELLM_SSL_CERTIFICATE_ENV_VAR,
)

import structlog

from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()


def ensure_ssl_certificates_for_litellm_non_openai_based_clients() -> None:
    """
    Ensure SSL certificates configuration for LiteLLM based on environment
    variables for clients that are not utilizing OpenAI's clients from
    `openai` library.
    """
    ssl_verify = _get_ssl_verify()
    ssl_certificate = _get_ssl_cert()

    structlogger.debug(
        "ensure_ssl_certificates_for_litellm_non_openai_based_clients",
        ssl_verify=ssl_verify,
        ssl_certificate=ssl_certificate,
    )

    if ssl_verify is not None:
        litellm.ssl_verify = ssl_verify
    if ssl_certificate is not None:
        litellm.ssl_certificate = ssl_certificate


def ensure_ssl_certificates_for_litellm_openai_based_clients() -> None:
    """
    Ensure SSL certificates configuration for LiteLLM based on environment
    variables for clients that are utilizing OpenAI's clients from
    `openai` library.

    The ssl configuration is ensured by setting `litellm.client_session` and
    `litellm.aclient_session` if not previously set.
    """
    client_args = {}

    ssl_verify = _get_ssl_verify()
    ssl_certificate = _get_ssl_cert()

    structlogger.debug(
        "ensure_ssl_certificates_for_litellm_openai_based_clients",
        ssl_verify=ssl_verify,
        ssl_certificate=ssl_certificate,
    )

    if ssl_verify is not None:
        client_args["verify"] = ssl_verify
    if ssl_certificate is not None:
        client_args["cert"] = ssl_certificate

    if client_args and not isinstance(litellm.aclient_session, httpx.AsyncClient):
        litellm.aclient_session = httpx.AsyncClient(**client_args)
    if client_args and not isinstance(litellm.client_session, httpx.Client):
        litellm.client_session = httpx.Client(**client_args)


def _get_ssl_verify() -> Optional[Union[bool, str]]:
    """
    Environment variable priority (ssl verify):
    1. `RASA_CA_BUNDLE`: Preferred for SSL verification.
    2. `REQUESTS_CA_BUNDLE`: Deprecated; use `RASA_CA_BUNDLE_ENV_VAR` instead.
    3. `SSL_VERIFY`: Fallback for SSL verification.

    Returns:
        Path to a self-signed SSL certificate or None if no SSL certificate is found.
    """
    if os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR) and os.environ.get(
        RASA_CA_BUNDLE_ENV_VAR
    ):
        raise_deprecation_warning(
            "Both REQUESTS_CA_BUNDLE and RASA_CA_BUNDLE environment variables are set. "
            "RASA_CA_BUNDLE will be used as the SSL verification path.\n"
            "Support of the REQUESTS_CA_BUNDLE environment variable is deprecated and "
            "will be removed in Rasa Pro 4.0.0. Please set the RASA_CA_BUNDLE "
            "environment variable instead."
        )
    elif os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR):
        raise_deprecation_warning(
            "Support of the REQUESTS_CA_BUNDLE environment variable is deprecated and "
            "will be removed in Rasa Pro 4.0.0. Please set the RASA_CA_BUNDLE "
            "environment variable instead."
        )

    return (
        os.environ.get(RASA_CA_BUNDLE_ENV_VAR)
        # Deprecated
        or os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR)
        # From LiteLLM, use as a fallback
        or os.environ.get(LITELLM_SSL_VERIFY_ENV_VAR)
        or None
    )


def _get_ssl_cert() -> Optional[str]:
    """
    Environment variable priority (ssl certificate):
    1. `RASA_SSL_CERTIFICATE`: Preferred for client certificate.
    2. `SSL_CERTIFICATE`: Fallback for client certificate.

    Returns:
        Path to a SSL certificate or None if no SSL certificate is found.
    """
    return (
        os.environ.get(RASA_SSL_CERTIFICATE_ENV_VAR)
        # From LiteLLM, use as a fallback
        or os.environ.get(LITELLM_SSL_CERTIFICATE_ENV_VAR)
        or None
    )
