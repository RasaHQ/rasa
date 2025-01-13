from pathlib import Path
from time import time

import jwt
import pytest
from pytest import MonkeyPatch


@pytest.fixture
def valid_jwt_token(test_private_key: str) -> str:
    return jwt.encode(
        {"aud": "account", "sub": "test_user", "exp": time() + 3600},
        test_private_key,
        algorithm="RS256",
    )


@pytest.fixture
def jwt_token_with_invalid_audience(test_private_key: str) -> str:
    return jwt.encode(
        {"aud": "invalid_audience", "sub": "test_user", "exp": time() + 3600},
        test_private_key,
        algorithm="RS256",
    )


@pytest.fixture(autouse=True)
def server_base_directory(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "RASA_MODEL_SERVER_BASE_DIRECTORY", str(tmp_path / "working-data")
    )
