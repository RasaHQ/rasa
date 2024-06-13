import ssl
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from aiohttp import ClientSession
from pytest import MonkeyPatch

from rasa.shared.providers.openai.session_handler import OpenAISessionHandler


@pytest.fixture
def valid_certificate_path(tmpdir: Path) -> Path:
    valid_certificate = tmpdir / "valid_certificate.pem"
    with open(valid_certificate, "w") as file:
        file.write("Test certificate")
    return valid_certificate


@pytest.fixture
def mock_create_session(monkeypatch: MonkeyPatch):
    _mock_create_session = MagicMock()
    monkeypatch.setattr(
        "rasa.shared.providers.openai.session_handler.OpenAISessionHandler._create_session",
        _mock_create_session,
    )
    return _mock_create_session


@pytest.fixture
def mock_openai_aiosession(monkeypatch: MonkeyPatch):
    _mock_openai_aiosession = MagicMock(spec=ContextVar)
    monkeypatch.setattr("openai.aiosession", _mock_openai_aiosession)
    return _mock_openai_aiosession


@pytest.mark.asyncio
async def test_enter_context_creates_session(
    mock_openai_aiosession: MagicMock,
    mock_create_session: MagicMock,
    valid_certificate_path: Path,
) -> None:
    """
    Test the case when the openai.aiosession is having no session set.
    In this case we expect OpenAISessionHandler to:
    1. create new session
    2. set the session on the openai.aiosession context var
    """
    # Given
    mock_openai_aiosession.get.return_value = None
    mock_session = MagicMock(spec=ClientSession)
    mock_create_session.return_value = mock_session
    handler = OpenAISessionHandler(certificate_path=str(valid_certificate_path))

    # When
    session = await handler.__aenter__()

    # Then
    mock_create_session.assert_called_once()
    assert session is not None
    assert session == mock_session


@pytest.mark.asyncio
async def test_enter_context_uses_already_created_session_from_context_var(
    mock_openai_aiosession: MagicMock,
    mock_create_session: MagicMock,
    valid_certificate_path: Path,
) -> None:
    """
    Test the case when the openai.aiosession already has a session set.
    In this case we expect OpenAISessionHandler to:
    1. not create new session
    2. use existing session from the openai.aiosession context var
    """
    # Given
    mock_session = MagicMock(spec=ClientSession)
    mock_openai_aiosession.get.return_value = mock_session
    handler = OpenAISessionHandler(certificate_path=str(valid_certificate_path))

    # When
    session = await handler.__aenter__()

    # Then
    mock_create_session.assert_not_called()
    mock_openai_aiosession.set.assert_not_called()
    assert session is not None
    assert session == mock_session


@pytest.mark.asyncio
async def test_exit_context_closes_session_and_clears_context_var(
    mock_openai_aiosession: MagicMock, valid_certificate_path: Path
) -> None:
    # Given
    mock_session = MagicMock(spec=ClientSession)
    handler = OpenAISessionHandler(certificate_path=str(valid_certificate_path))
    mock_openai_aiosession.get.return_value = mock_session

    # When
    await handler.__aexit__(None, None, None)

    # Then
    # check that the session.close is called and that contextvar is set to None
    mock_session.close.assert_called_once()
    mock_openai_aiosession.set.assert_called_once_with(None)


@pytest.mark.asyncio
@patch("ssl.create_default_context")
async def test_create_session_with_valid_ssl_context(
    mock_ssl_create_default_context: MagicMock, valid_certificate_path: Path
):
    # Given
    handler = OpenAISessionHandler(certificate_path=str(valid_certificate_path))
    mock_ssl_create_default_context.return_value = ssl.SSLContext()

    # When
    session = handler._create_session()

    # Then
    assert isinstance(session, ClientSession)
    mock_ssl_create_default_context.assert_called_with(
        purpose=ssl.Purpose.SERVER_AUTH, cafile=str(valid_certificate_path)
    )


@pytest.mark.asyncio
async def test_invalid_certificate_path_raises_value_error():
    invalid_path = "invalid_certificate.pem"
    handler = OpenAISessionHandler(certificate_path=invalid_path)
    with pytest.raises(ValueError):
        handler._create_ssl_context()


@pytest.mark.asyncio
@patch("ssl.create_default_context", side_effect=ssl.SSLError())
async def test_exception_in_ssl_context_creation(valid_certificate_path):
    handler = OpenAISessionHandler(certificate_path=str(valid_certificate_path))
    with pytest.raises(Exception):
        handler._create_ssl_context()
