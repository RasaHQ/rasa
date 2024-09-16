from typing import Optional
from unittest.mock import patch

import httpx
import pytest
import litellm

from rasa.shared.constants import (
    RASA_CA_BUNDLE_ENV_VAR,
    REQUESTS_CA_BUNDLE_ENV_VAR,
    RASA_SSL_CERTIFICATE_ENV_VAR,
    LITELLM_SSL_VERIFY_ENV_VAR,
    LITELLM_SSL_CERTIFICATE_ENV_VAR,
)

import structlog

from rasa.shared.providers._ssl_verification_utils import (
    ensure_ssl_certificates_for_litellm_non_openai_based_clients,
    ensure_ssl_certificates_for_litellm_openai_based_clients,
    _get_ssl_verify,
)
from pytest import MonkeyPatch

structlogger = structlog.get_logger()


class MockHttpxAsyncClient(httpx.AsyncClient):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs


class MockHttpxClient(httpx.AsyncClient):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs


@pytest.fixture
def reset_litellm_sessions():
    # Setup: Reset the global settings before each test
    litellm.aclient_session = None
    litellm.client_session = None

    yield

    # Teardown: Reset the global settings after each test
    litellm.aclient_session = None
    litellm.client_session = None


@pytest.mark.parametrize(
    "env_var," "certificate_path",
    [
        # Certificate path not set
        (RASA_CA_BUNDLE_ENV_VAR, None),
        (REQUESTS_CA_BUNDLE_ENV_VAR, None),
        (LITELLM_SSL_VERIFY_ENV_VAR, None),
        # Certificate path set
        (RASA_CA_BUNDLE_ENV_VAR, "path/to/certificate.pem"),
        (REQUESTS_CA_BUNDLE_ENV_VAR, "path/to/certificate.pem"),
        (LITELLM_SSL_VERIFY_ENV_VAR, "path/to/certificate.pem"),
    ],
)
def test_ensure_ssl_certificates_for_litellm_non_openai_based_clients_sets_verify(
    monkeypatch: MonkeyPatch,
    env_var: Optional[str],
    certificate_path: Optional[str],
    reset_litellm_sessions,
) -> None:
    # Given
    if certificate_path is not None:
        monkeypatch.setenv(env_var, certificate_path)
    # When
    ensure_ssl_certificates_for_litellm_non_openai_based_clients()
    # Then
    if certificate_path is not None:
        assert litellm.ssl_verify == certificate_path
    else:
        # The default value is `True`
        assert litellm.ssl_verify


@pytest.mark.parametrize(
    "env_var," "certificate_path",
    [
        # Certificate path not set
        (RASA_SSL_CERTIFICATE_ENV_VAR, None),
        (LITELLM_SSL_CERTIFICATE_ENV_VAR, None),
        # Certificate path set
        (RASA_SSL_CERTIFICATE_ENV_VAR, "path/to/certificate.pem"),
        (LITELLM_SSL_CERTIFICATE_ENV_VAR, "path/to/certificate.pem"),
    ],
)
def test_ensure_ssl_certificates_for_litellm_non_openai_based_clients_sets_cert(
    monkeypatch: MonkeyPatch,
    env_var: Optional[str],
    certificate_path: Optional[str],
    reset_litellm_sessions,
) -> None:
    # Given
    if certificate_path is not None:
        monkeypatch.setenv(env_var, certificate_path)
    # When
    ensure_ssl_certificates_for_litellm_non_openai_based_clients()
    # Then
    assert litellm.ssl_certificate == certificate_path


@pytest.mark.parametrize(
    "env_var," "certificate_path",
    [
        # Certificate path not set
        (RASA_CA_BUNDLE_ENV_VAR, None),
        (REQUESTS_CA_BUNDLE_ENV_VAR, None),
        (LITELLM_SSL_VERIFY_ENV_VAR, None),
        # Certificate path set
        (RASA_CA_BUNDLE_ENV_VAR, "path/to/certificate.pem"),
        (REQUESTS_CA_BUNDLE_ENV_VAR, "path/to/certificate.pem"),
        (LITELLM_SSL_VERIFY_ENV_VAR, "path/to/certificate.pem"),
    ],
)
@patch("httpx.AsyncClient", MockHttpxAsyncClient)
@patch("httpx.Client", MockHttpxClient)
def test_ensure_ssl_certificates_for_litellm_openai_based_clients_sets_verify(
    env_var: Optional[str],
    certificate_path: Optional[str],
    monkeypatch: MonkeyPatch,
    reset_litellm_sessions,
) -> None:
    # Given
    monkeypatch.delenv(env_var, raising=False)
    if certificate_path is not None:
        monkeypatch.setenv(env_var, str(certificate_path))

    # When
    ensure_ssl_certificates_for_litellm_openai_based_clients()

    # Then
    if certificate_path is not None:
        litellm.aclient_session.init_kwargs = {"verify": certificate_path}
        litellm.client_session.init_kwargs = {"verify": certificate_path}
    else:
        assert litellm.aclient_session is None
        assert litellm.client_session is None


@pytest.mark.parametrize(
    "env_var," "certificate_path",
    [
        # Certificate path not set
        (RASA_SSL_CERTIFICATE_ENV_VAR, None),
        (LITELLM_SSL_CERTIFICATE_ENV_VAR, None),
        # Certificate path set
        (RASA_SSL_CERTIFICATE_ENV_VAR, "path/to/certificate.pem"),
        (LITELLM_SSL_CERTIFICATE_ENV_VAR, "path/to/certificate.pem"),
    ],
)
@patch("httpx.AsyncClient", MockHttpxAsyncClient)
@patch("httpx.Client", MockHttpxClient)
def test_ensure_ssl_certificates_for_litellm_openai_based_clients_sets_cert(
    env_var: Optional[str],
    certificate_path: Optional[str],
    monkeypatch: MonkeyPatch,
    reset_litellm_sessions,
) -> None:
    # Given
    monkeypatch.delenv(env_var, raising=False)
    if certificate_path is not None:
        monkeypatch.setenv(env_var, str(certificate_path))

    # When
    ensure_ssl_certificates_for_litellm_openai_based_clients()

    # Then
    if certificate_path is not None:
        litellm.aclient_session.init_kwargs = {"cert": certificate_path}
        litellm.client_session.init_kwargs = {"cert": certificate_path}
    else:
        assert litellm.aclient_session is None
        assert litellm.client_session is None


@pytest.mark.parametrize(
    "env_var," "certificate_path",
    [
        (REQUESTS_CA_BUNDLE_ENV_VAR, "path/to/certificate.pem"),
    ],
)
def test_ensure_deprecation_warning_is_thrown(
    env_var: Optional[str],
    certificate_path: Optional[str],
    monkeypatch: MonkeyPatch,
    reset_litellm_sessions,
):
    # Given
    monkeypatch.delenv(env_var, raising=False)
    if certificate_path is not None:
        monkeypatch.setenv(env_var, str(certificate_path))

    with pytest.warns(FutureWarning) as record:
        assert _get_ssl_verify() == "path/to/certificate.pem"

    assert len(record) == 1
    assert (
        record[0].message.args[0] == "Support of the REQUESTS_CA_BUNDLE "
        "environment variable is deprecated and will be removed in Rasa Pro 4.0.0. "
        "Please set the RASA_CA_BUNDLE environment variable instead."
    )
    assert isinstance(record[0].message, FutureWarning)


@pytest.mark.parametrize(
    "env_vars_and_paths",
    [
        {
            REQUESTS_CA_BUNDLE_ENV_VAR: "path/to/certificate1.pem",
            RASA_CA_BUNDLE_ENV_VAR: "path/to/certificate2.pem",
        },
    ],
)
def test_ensure_correct_env_var_is_used(
    env_vars_and_paths: dict,
    monkeypatch: MonkeyPatch,
    reset_litellm_sessions,
):
    # Given
    for env_var, certificate_path in env_vars_and_paths.items():
        monkeypatch.delenv(env_var, raising=False)
        if certificate_path is not None:
            monkeypatch.setenv(env_var, str(certificate_path))

    with pytest.warns(FutureWarning) as record:
        assert _get_ssl_verify() == "path/to/certificate2.pem"

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "Both REQUESTS_CA_BUNDLE and RASA_CA_BUNDLE environment variables are set. "
        "RASA_CA_BUNDLE will be used as the SSL verification path.\n"
        "Support of the REQUESTS_CA_BUNDLE environment variable is deprecated and "
        "will be removed in Rasa Pro 4.0.0. Please set the RASA_CA_BUNDLE "
        "environment variable instead."
    )
    assert isinstance(record[0].message, FutureWarning)
