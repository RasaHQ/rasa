import json
import time
from http import HTTPStatus
from unittest import mock
from unittest.mock import Mock

import freezegun
import jwt
import pytest
from _pytest.logging import LogCaptureFixture
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm

from rasa.core.channels import BotFrameworkInput

DAY_IN_SECONDS = 60 * 60 * 24

MS_OPENID_CONFIG_RESPONSE = {
    "issuer": "https://api.botframework.com",
    "authorization_endpoint": "https://invalid.botframework.com",
    "jwks_uri": "https://login.botframework.com/v1/.well-known/keys",
    "id_token_signing_alg_values_supported": ["RS256"],
    "token_endpoint_auth_methods_supported": ["private_key_jwt"],
}


@pytest.fixture(scope="function")
def rsa_private_key() -> RSAPrivateKey:
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


@pytest.fixture(scope="function")
@mock.patch("requests.get")
def bot_framework_input(mock_requests: Mock, rsa_private_key: RSAPrivateKey):
    jwk = json.loads(RSAAlgorithm.to_jwk(rsa_private_key.public_key()))
    jwk["kid"] = "key_id"

    openid_config_mock_response = mock.Mock()
    openid_config_mock_response.raise_for_status = mock.Mock()
    openid_config_mock_response.json.return_value = MS_OPENID_CONFIG_RESPONSE

    openid_keys_mock_response = mock.Mock()
    openid_keys_mock_response.raise_for_status = mock.Mock()
    openid_keys_mock_response.json.return_value = {"keys": [jwk]}

    mock_requests.side_effect = [openid_config_mock_response, openid_keys_mock_response]

    return BotFrameworkInput("app_id", "app_password")


def test_successful_jwt_signature_verification(
    bot_framework_input: BotFrameworkInput,
    rsa_private_key: RSAPrivateKey,
):
    encoded = jwt.encode(
        {
            "serviceurl": "https://webchat.botframework.com/",
            "nbf": int(time.time()),
            "exp": int(time.time()) + DAY_IN_SECONDS,
            "iss": "https://api.botframework.com",
            "aud": "app_id",
        },
        rsa_private_key,
        algorithm="RS256",
        headers={"kid": "key_id", "alg": "RS256"},
    )

    with pytest.warns(None):
        resp = bot_framework_input._validate_auth(f"Bearer {encoded}")
        assert resp is None


@mock.patch("requests.get")
def test_jwk_is_updated_daily(mock_requests: Mock):
    with freezegun.freeze_time("2012-01-14 08:00:00"):
        bot_framework_input = BotFrameworkInput("app_id", "app_password")
        # Two calls at the beginning - on to retrieve open ID metadata,
        # the second one to actually get the keys.
        assert mock_requests.call_count == 2

        bot_framework_input._validate_auth("Bearer token_invalid")
        assert mock_requests.call_count == 2

    with freezegun.freeze_time("2012-01-14 11:00:00"):
        bot_framework_input._validate_auth("Bearer token_invalid")
        assert mock_requests.call_count == 2

    with freezegun.freeze_time("2012-01-15 11:00:00"):
        bot_framework_input._validate_auth("Bearer token_invalid")
        assert mock_requests.call_count == 4


def test_validate_auth_returns_unauthorized_for_invalid_jwt_token(
    bot_framework_input: BotFrameworkInput,
    rsa_private_key: RSAPrivateKey,
    caplog: LogCaptureFixture,
):
    encoded = jwt.encode(
        {
            "serviceurl": "https://webchat.botframework.com/",
            "nbf": int(time.time()) + DAY_IN_SECONDS,
            "exp": int(time.time()) + 2 * DAY_IN_SECONDS,
            "iss": "https://api.botframework.com",
            "aud": "app_id",
        },
        rsa_private_key,
        algorithm="RS256",
        headers={"kid": "key_id", "alg": "RS256"},
    )

    resp = bot_framework_input._validate_auth(f"Bearer {encoded}")

    assert resp is not None
    assert resp.status == HTTPStatus.UNAUTHORIZED
    assert resp.body.decode() == "Could not validate JWT token."
    assert [msg for msg in caplog.messages] == [
        "Bot framework JWT token could not be verified.",
        "The token is not yet valid (nbf)",
    ]


def test_validate_auth_returns_unauthorized_for_absent_header(
    bot_framework_input: BotFrameworkInput,
    caplog: LogCaptureFixture,
):
    resp = bot_framework_input._validate_auth(None)
    assert resp is not None
    assert resp.status == HTTPStatus.UNAUTHORIZED
    assert resp.body.decode() == "No authorization header provided."


def test_validate_auth_returns_unauthorized_for_non_bearer_header(
    bot_framework_input: BotFrameworkInput,
):
    resp = bot_framework_input._validate_auth("Basic dXNlcm5hbWU6cGFzc3dvcmQ=")
    assert resp is not None
    assert resp.status == HTTPStatus.UNAUTHORIZED
    assert resp.body.decode() == "No Bearer token provided in Authorization header."


def test_validate_auth_returns_bad_request_for_invalid_header(
    bot_framework_input: BotFrameworkInput,
):
    resp = bot_framework_input._validate_auth("Bearer invalid")
    assert resp is not None
    assert resp.status == HTTPStatus.UNAUTHORIZED
    assert resp.body.decode() == "Could not validate JWT token."
