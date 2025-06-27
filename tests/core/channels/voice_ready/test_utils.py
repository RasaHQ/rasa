import pytest

from rasa.core.channels.voice_ready.utils import validate_username_password_credentials
from rasa.shared.exceptions import InvalidConfigException


@pytest.mark.parametrize(
    "username,password,channel_name",
    [
        ("test_user", "test_pass", "test_channel"),
        ("admin", "secret123", "voice_channel"),
        (None, None, "another_channel"),
    ],
)
def test_validate_username_password_credentials_valid_cases(
    username, password, channel_name
):
    """Test that no exception is raised for valid username/password combinations."""
    validate_username_password_credentials(
        username=username, password=password, channel_name=channel_name
    )


@pytest.mark.parametrize(
    "username,password,channel_name",
    [
        ("test_user", None, "test_channel"),
        (None, "test_pass", "test_channel"),
        ("admin", None, "voice_channel"),
        (None, "secret123", "voice_channel"),
    ],
)
def test_validate_username_password_credentials_invalid_cases(
    username, password, channel_name
):
    """Test that exception is raised for invalid username/password combinations."""
    with pytest.raises(InvalidConfigException) as exc_info:
        validate_username_password_credentials(
            username=username, password=password, channel_name=channel_name
        )

    assert channel_name in str(exc_info.value)
    assert "either both username and password or neither should be provided" in str(
        exc_info.value
    )
