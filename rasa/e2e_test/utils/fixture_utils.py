"""Validate and convert fixtures (e.g. mocked_datetime to ISO) at load time."""

import dataclasses
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from rasa.e2e_test.e2e_test_case import TestCaseFixtures
from rasa.exceptions import ValidationError
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_test_case import (
        DialogueUnderstandingTestCase,
        Fixture,
        TestCase,
        TestCaseFixtures,
    )


def _get_validated_mocked_datetime(mocked_datetime_value: Any) -> Optional[str]:
    """Validates and converts mocked_datetime to ISO 8601 format.

    Args:
        mocked_datetime_value: The value of the mocked_datetime slot.
            Expected to be a string from YAML fixtures, or None.

    Returns:
        An ISO 8601 format string (timezone-aware) or None.

    Raises:
        ValidationError: If the mocked_datetime value cannot be converted
            to a datetime object.
    """
    if mocked_datetime_value is None:
        return None

    if not isinstance(mocked_datetime_value, str):
        raise ValidationError(
            code="e2e_test_runner.validate_mocked_datetime.invalid_value_type",
            event_info="Unable to convert to a valid datetime.",
        )

    valid_datetime_formats = {
        "%Y-%m-%dT%H:%M:%S%z": True,
        "%Y-%m-%d %H:%M:%S": False,
        "%Y-%m-%dT%H:%M:%S": False,
        "%Y-%m-%d": False,
    }

    for datetime_format, timezone_aware in valid_datetime_formats.items():
        try:
            parsed = datetime.strptime(mocked_datetime_value, datetime_format)
            if not timezone_aware:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            continue

    raise ValidationError(
        code="e2e_test_runner.validate_mocked_datetime.invalid_value_format",
        event_info="Unable to convert to a valid datetime.",
    )


def validate_and_convert_fixtures(fixtures: List["Fixture"]) -> List["Fixture"]:
    """Validate and convert all fixtures (e.g. mocked_datetime to ISO).

    Ensures we fail fast on validation errors. Collects all validation errors
    and reports them in a single ValidationError. Can be called at load time
    so the test suite is guaranteed valid for run_tests.

    Args:
        fixtures: List of all fixtures to validate and convert.

    Returns:
        List of fixtures with converted mocked_datetime values.

    Raises:
        ValidationError: If any fixture has invalid mocked_datetime values.
    """
    validation_errors: List[Tuple[str, str]] = []
    converted_fixtures = []

    for fixture in fixtures:
        converted_slots = fixture.slots_set.copy()
        for slot_name, slot_value in fixture.slots_set.items():
            if slot_name == MOCKED_DATETIME_SLOT:
                try:
                    converted_value = _get_validated_mocked_datetime(slot_value)
                    converted_slots[slot_name] = converted_value
                except ValidationError:
                    validation_errors.append((fixture.name, str(slot_value)))

        converted_fixture = dataclasses.replace(fixture, slots_set=converted_slots)
        converted_fixtures.append(converted_fixture)

    if validation_errors:
        valid_formats = [
            "YYYY-MM-DDTHH:MM:SS±HH:MM  e.g. '2024-01-15T14:30:00+05:30'",
            "YYYY-MM-DDTHH:MM:SS±HHMM   e.g. '2024-01-15T14:30:00+0530'",
            "YYYY-MM-DD HH:MM:SS        e.g. '2024-01-15 14:30:00'",
            "YYYY-MM-DDTHH:MM:SS        e.g. '2024-01-15T14:30:00'",
            "YYYY-MM-DD                 e.g. '2024-01-15'",
        ]
        error_details = "\n".join(
            f"{i + 1}. Fixture - `{name}`, mocked_datetime value: `{val}`"
            for i, (name, val) in enumerate(validation_errors)
        )
        raise ValidationError(
            code="e2e_test_runner.validate_mocked_datetime.invalid_value_format",
            event_info=(
                "Unable to convert to a valid datetime. Invalid `mocked_datetime` "
                "value present in the following fixtures."
                + f"\n\n{error_details}\n\n"
                + "Accepted formats include:\n"
                + "\n".join(f"  * {fmt}" for fmt in valid_formats)
            ),
        )

    return converted_fixtures


def collect_fixtures_for_test_case(
    test_case: Union["TestCase", "DialogueUnderstandingTestCase"],
    fixtures: List["Fixture"],
) -> List["Fixture"]:
    """Return only the fixtures used by this test case.

    If the test case has no fixture_names, returns an empty list.
    Otherwise returns fixtures for those names, skipping any
    that are not in fixtures (validation will report missing ones).

    Args:
        test_case: The test case.
        fixtures: List of fixtures for the file.

    Returns:
        List of Fixture objects used by the test case.
    """
    if not test_case.fixture_names:
        return []
    return [fixture for fixture in fixtures if fixture.name in test_case.fixture_names]


def extract_test_case_fixtures(
    test_cases: List[Union["TestCase", "DialogueUnderstandingTestCase"]],
    fixtures: Optional[List["Fixture"]],
) -> List["TestCaseFixtures"]:
    """Build TestCaseFixtures for each test case from resolved fixtures.

    Args:
        test_cases: Test cases from the file.
        fixtures: List of fixtures for the file, or None.

    Returns:
        List of TestCaseFixtures (one per test case, only fixtures used by each).
    """
    if not fixtures:
        return []

    fixtures_per_test: List[TestCaseFixtures] = []
    for test_case in test_cases:
        if used_fixtures := collect_fixtures_for_test_case(test_case, fixtures):
            fixtures_per_test.append(
                TestCaseFixtures(
                    test_case_name=test_case.name,
                    file=test_case.file,
                    fixtures=used_fixtures,
                )
            )
    return fixtures_per_test


def get_fixtures_for_test_case(
    test_case: Union["TestCase", "DialogueUnderstandingTestCase"],
    fixtures_per_test: List["TestCaseFixtures"],
) -> List["Fixture"]:
    """Finds the fixtures for a test case in the fixtures_per_test.

    Args:
        test_case: The test case to find the fixtures for.
        fixtures_per_test: The fixtures per test case.

    Returns:
        The fixtures for the test case.
    """
    for test_case_fixture in fixtures_per_test:
        if (
            test_case_fixture.test_case_name == test_case.name
            and test_case_fixture.file == test_case.file
        ):
            return test_case_fixture.fixtures
    return []
