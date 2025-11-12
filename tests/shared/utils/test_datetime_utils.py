"""Unit tests for datetime_utils."""

import pytest
import structlog

from rasa.exceptions import ValidationError
from rasa.shared.utils.datetime_utils import (
    get_current_datetime,
    validate_datetime_configuration,
)


@pytest.mark.parametrize(
    "include_date_time, timezone, is_custom_timezone_provided, should_raise,"
    "should_warn",
    [
        # Valid timezone when include_date_time is True
        (True, "UTC", True, False, False),
        (True, "America/New_York", True, False, False),
        (True, "Europe/London", True, False, False),
        (True, "Asia/Tokyo", True, False, False),
        # Invalid timezone when include_date_time is True - should raise
        (True, "Invalid/Timezone", True, True, False),
        (True, "", True, True, False),
        # include_date_time is False with custom timezone - should warn
        (False, "UTC", True, False, True),
        (False, "America/New_York", True, False, True),
        # include_date_time is False without custom timezone - no warning
        (False, "UTC", False, False, False),
    ],
)
def test_validate_datetime_configuration(
    include_date_time: bool,
    timezone: str,
    is_custom_timezone_provided: bool,
    should_raise: bool,
    should_warn: bool,
) -> None:
    """Test validate_datetime_configuration with various scenarios."""
    component_name = "TestComponent"

    if should_raise:
        # Should raise ValidationError for invalid timezone when
        # include_date_time is True
        with pytest.raises(ValidationError) as exc_info:
            validate_datetime_configuration(
                include_date_time=include_date_time,
                timezone=timezone,
                is_custom_timezone_provided=is_custom_timezone_provided,
                component_name=component_name,
            )

        assert (
            exc_info.value.code
            == "datetime_utils.validate_datetime_configuration.invalid_timezone"
        )
        assert component_name in exc_info.value.info
        assert "Invalid timezone configuration" in exc_info.value.info
    elif should_warn:
        # Should warn when timezone is provided but include_date_time is False
        with structlog.testing.capture_logs() as caplog:
            validate_datetime_configuration(
                include_date_time=include_date_time,
                timezone=timezone,
                is_custom_timezone_provided=is_custom_timezone_provided,
                component_name=component_name,
            )

        # Check that a warning was logged
        warning_logs = [
            log
            for log in caplog
            if log["event"]
            == "datetime_utils.validate_datetime_configuration.timezone_not_allowed"
        ]
        assert len(warning_logs) == 1
        assert warning_logs[0]["component"] == component_name
        assert warning_logs[0]["timezone"] == timezone
        assert "not allowed" in warning_logs[0]["event_info"]
    else:
        # Should pass without raising or warning
        validate_datetime_configuration(
            include_date_time=include_date_time,
            timezone=timezone,
            is_custom_timezone_provided=is_custom_timezone_provided,
            component_name=component_name,
        )


@pytest.mark.parametrize(
    "timezone",
    [
        "UTC",
        "America/New_York",
        "Europe/London",
    ],
)
def test_get_current_datetime_correct_timezone(timezone: str) -> None:
    """Test that get_current_datetime uses the correct timezone."""
    from zoneinfo import ZoneInfo

    result = get_current_datetime(timezone)

    # Verify the timezone is correct by comparing ZoneInfo objects
    assert result.tzinfo == ZoneInfo(timezone)

    # For UTC, tzname() should always return "UTC"
    if timezone == "UTC":
        assert result.tzname() == "UTC"


def test_get_current_datetime_invalid_timezone_raises() -> None:
    """Test that get_current_datetime raises for invalid timezone."""
    # Note: This function assumes timezone is validated, but we test the behavior
    # when an invalid timezone is passed
    with pytest.raises((KeyError, ValueError)):
        get_current_datetime("Invalid/Timezone")
