from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_validate_voice_license_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the validate_voice_license_scope function."""
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.voice_channel.validate_voice_license_scope",
        MagicMock(),
    )
