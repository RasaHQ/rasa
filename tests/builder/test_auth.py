from unittest.mock import MagicMock, patch

import pytest
from sanic import Blueprint, Sanic, response

from rasa.builder import auth
from rasa.builder.auth import Auth0TokenVerificationResult, protected
from rasa.builder.service import bp as builder_bp


def _create_app_with_route(always_required: bool = False) -> Sanic:
    app = Sanic("auth_test_app")
    bp = Blueprint("test_bp", url_prefix="/test")

    @bp.route("/protected", methods=["GET"])  # type: ignore[misc]
    @protected(always_required=always_required)
    async def protected_route(_request):  # type: ignore[no-untyped-def]
        return response.json({"ok": True})

    app.blueprint(bp)
    return app


@pytest.mark.asyncio
async def test_protected_skips_when_auth_disabled(monkeypatch) -> None:
    app = _create_app_with_route(always_required=False)
    app.config.USE_AUTHENTICATION = False

    async with app.asgi_client as client:
        _, resp = await client.get("/test/protected")

    assert resp.status == 200


@pytest.mark.asyncio
async def test_protected_requires_auth_when_required(monkeypatch) -> None:
    app = _create_app_with_route(always_required=False)
    app.config.USE_AUTHENTICATION = True

    # Force auth to be required now
    monkeypatch.setattr("rasa.builder.auth.is_auth_required_now", lambda *_, **__: True)

    async with app.asgi_client as client:
        _, resp = await client.get("/test/protected")

    assert resp.status == 401
    body = resp.json
    assert body["details"]["expected"] == "Bearer <valid_token>"


@pytest.mark.asyncio
async def test_protected_allows_with_valid_bearer(monkeypatch) -> None:
    app = _create_app_with_route(always_required=False)
    app.config.USE_AUTHENTICATION = True

    # Force auth to be required now
    monkeypatch.setattr("rasa.builder.auth.is_auth_required_now", lambda *_, **__: True)

    # Pretend token is valid
    def _ok_verification(_hdr: str) -> Auth0TokenVerificationResult:
        return Auth0TokenVerificationResult(payload={"sub": "user"}, error_message=None)

    monkeypatch.setattr(
        "rasa.builder.auth.extract_and_verify_auth0_token", _ok_verification
    )

    async with app.asgi_client as client:
        _, resp = await client.get(
            "/test/protected", headers={"Authorization": "Bearer abc"}
        )

    assert resp.status == 200


@pytest.mark.asyncio
async def test_download_requires_authorization_header(monkeypatch) -> None:
    # Reuse the real builder blueprint to hit the /api/download route
    app = Sanic("download_auth_test")
    app.blueprint(builder_bp)
    app.ctx.input_channel = type("X", (), {})()  # minimal ctx for blueprint usage

    async with app.asgi_client as client:
        _, resp = await client.get("/api/download")

    assert resp.status == 401


DUMMY_TOKEN = "dummy.jwt.token"
DUMMY_PAYLOAD = {"sub": "user123", "aud": "client_id", "iss": "issuer"}


@patch("rasa.builder.auth.PyJWKClient")
@patch("rasa.builder.auth.jwt.decode")
def test_verify_auth0_token_success(
    mock_decode: MagicMock, mock_jwk_client: MagicMock
) -> None:
    mock_jwk = MagicMock()
    mock_jwk.key = "key"
    mock_jwk_client.return_value.get_signing_key_from_jwt.return_value = mock_jwk
    mock_decode.return_value = DUMMY_PAYLOAD
    result = auth.verify_auth0_token(DUMMY_TOKEN)
    assert result == DUMMY_PAYLOAD
    mock_jwk_client.assert_called_once()
    mock_decode.assert_called_once()


@patch("rasa.builder.auth.verify_auth0_token")
def test_extract_and_verify_auth0_token_success(mock_verify: MagicMock) -> None:
    mock_verify.return_value = DUMMY_PAYLOAD
    header = "Bearer testtoken"
    result = auth.extract_and_verify_auth0_token(header)
    assert result.payload == DUMMY_PAYLOAD
    assert result.error_message is None


@patch("rasa.builder.auth.verify_auth0_token", side_effect=Exception("fail"))
def test_extract_and_verify_auth0_token_invalid_token(mock_verify: MagicMock) -> None:
    header = "Bearer testtoken"
    result = auth.extract_and_verify_auth0_token(header)
    assert result.payload is None
    assert result.error_message.startswith("Invalid token:")


def test_extract_and_verify_auth0_token_invalid_header() -> None:
    header = "NotBearer sometoken"
    result = auth.extract_and_verify_auth0_token(header)
    assert result.payload is None
    assert result.error_message == "Missing or invalid Authorization header"


def test_auth0tokenverificationresult_model() -> None:
    # Just test pydantic model works
    r = auth.Auth0TokenVerificationResult(payload={"foo": 1}, error_message=None)
    assert r.payload == {"foo": 1}
    assert r.error_message is None
