"""Utility functions for datetime handling in prompt templates."""

from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import structlog

from rasa.exceptions import ValidationError

structlogger = structlog.get_logger()


def validate_datetime_configuration(
    include_date_time: bool,
    timezone: str,
    is_custom_timezone_provided: bool,
    component_name: str,
    **validation_error_kwargs: Any,
) -> None:
    """Validate datetime configuration for a component.

    This function validates both include_date_time and timezone.
    If include_date_time is True, timezone must be valid.

    Args:
        include_date_time: Whether datetime is enabled.
        timezone: Timezone string to validate.
        is_custom_timezone_provided: Whether a custom timezone is provided.
        component_name: Name of the component/agent for error messages.
        **validation_error_kwargs: Additional keyword arguments to pass
        to ValidationError (e.g., agent_name for agent-specific errors).

    Raises:
        ValidationError: If timezone is invalid.
    """
    if include_date_time:
        # Validate the timezone if include_date_time is True
        try:
            ZoneInfo(timezone)
        except Exception as e:
            event_info = (
                f"Invalid timezone configuration for `{component_name}`: {e!s}. "
                f"Please provide a valid IANA timezone name (e.g., 'UTC', "
                f"'America/New_York', 'Europe/London'). Refer to "
                f"`https://en.wikipedia.org/wiki/List_of_tz_database_time_zones` "
                f"for a list of valid timezone names."
            )
            structlogger.error(
                "datetime_utils.validate_datetime_configuration.invalid_timezone",
                component=component_name,
                timezone=timezone,
                event_info=event_info,
                **validation_error_kwargs,
            )
            raise ValidationError(
                code="datetime_utils.validate_datetime_configuration.invalid_timezone",
                event_info=event_info,
            ) from e

    elif not include_date_time and is_custom_timezone_provided:
        # Warn if timezone is provided when include_date_time is False
        structlogger.warning(
            "datetime_utils.validate_datetime_configuration.timezone_not_allowed",
            component=component_name,
            timezone=timezone,
            event_info=(
                f"Timezone configuration for `{component_name}` is not allowed "
                f"when `include_date_time` is False."
            ),
            **validation_error_kwargs,
        )


def get_current_datetime(timezone: str) -> datetime:
    """Get the current datetime with timezone support.

    Args:
        timezone: Timezone string (e.g., "UTC", "America/New_York").
                  Must be a valid IANA timezone name.

    Returns:
        A datetime object with timezone info.

    Note:
        Timezone validation should be done at component initialization.
        This function assumes the timezone is already validated.
    """
    tz = ZoneInfo(timezone)
    return datetime.now(tz)


def resolve_datetime(mocked_datetime_value: Optional[str], timezone: str) -> datetime:
    """Resolve datetime for template rendering.

    Returns mocked_datetime value if provided (for e2e tests),
    otherwise returns the current datetime.

    Args:
        mocked_datetime_value: The value from the mocked_datetime slot.
            Expected to be an ISO 8601 format string (e.g., '2024-01-15T14:30:45+00:00')
        timezone: Timezone string (e.g., "UTC", "America/New_York").
                  Must be a valid IANA timezone name.

    Returns:
        A timezone-aware datetime object.
    """
    if mocked_datetime_value is not None:
        return datetime.fromisoformat(mocked_datetime_value)
    return get_current_datetime(timezone=timezone)
