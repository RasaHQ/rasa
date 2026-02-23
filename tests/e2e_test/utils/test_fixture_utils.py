"""Unit tests for rasa.e2e_test.utils.fixture_utils."""

import pytest

from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.e2e_test.e2e_test_case import Fixture, TestCase, TestCaseFixtures
from rasa.e2e_test.utils.fixture_utils import (
    get_fixtures_for_test_case,
    validate_and_convert_fixtures,
)
from rasa.exceptions import ValidationError
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT


@pytest.mark.parametrize(
    "test_case, fixtures_per_test, expected_fixture_names",
    [
        # e2e: match by test case name + file (same value in both)
        (
            TestCase(
                name="my_test",
                steps=[],
                file="/path/to/foo.yml",
            ),
            [
                TestCaseFixtures(
                    test_case_name="my_test",
                    file="/path/to/foo.yml",
                    fixtures=[Fixture("premium", {"membership_type": "premium"})],
                )
            ],
            ["premium"],
        ),
        # e2e: no file on test case -> use "" to match
        (
            TestCase(name="test_hi", steps=[], file=""),
            [
                TestCaseFixtures(
                    test_case_name="test_hi",
                    file="",
                    fixtures=[Fixture("premium", {"premium": True})],
                )
            ],
            ["premium"],
        ),
        # e2e: two entries, first matches
        (
            TestCase(name="test_a", steps=[], file="/dir/test_a.yml"),
            [
                TestCaseFixtures(
                    test_case_name="test_a",
                    file="/dir/test_a.yml",
                    fixtures=[Fixture("f1", {})],
                ),
                TestCaseFixtures(
                    test_case_name="test_b",
                    file="test_b.yml",
                    fixtures=[Fixture("f2", {})],
                ),
            ],
            ["f1"],
        ),
        # DU: same semantics as e2e (matching file in both)
        (
            DialogueUnderstandingTestCase(
                name="cancellation respects scope",
                steps=[DialogueUnderstandingTestStep(actor="user", text="hi")],
                file="/abs/path/to/valid_test_case.yml",
            ),
            [
                TestCaseFixtures(
                    test_case_name="cancellation respects scope",
                    file="/abs/path/to/valid_test_case.yml",
                    fixtures=[
                        Fixture("premium", {"membership_type": "premium"}),
                        Fixture("standard", {"membership_type": "standard"}),
                    ],
                )
            ],
            ["premium", "standard"],
        ),
        # Same file, different test case names (matching file in both)
        (
            TestCase(
                name="test_compact_llm_command_generator_with_mocked_datetime_jan_15",
                steps=[],
                file="/Users/varun/dev/project/e2e/train/datetime_mocking/conftest.yml",
            ),
            [
                TestCaseFixtures(
                    test_case_name="test_compact_llm_command_generator_with_mocked_datetime_jan_15",
                    file="/Users/varun/dev/project/e2e/train/datetime_mocking/conftest.yml",
                    fixtures=[
                        Fixture(
                            "mocked_datetime_jan_15_2024",
                            {"mocked_datetime": "2024-01-15"},
                        )
                    ],
                )
            ],
            ["mocked_datetime_jan_15_2024"],
        ),
        # No match: empty fixtures_per_test
        (
            TestCase(name="orphan", steps=[], file="/some/file.yml"),
            [],
            [],
        ),
        # No match: no matching test_case_name
        (
            TestCase(name="other_test", steps=[], file="/path/other.yml"),
            [
                TestCaseFixtures(
                    test_case_name="my_test",
                    file="foo.yml",
                    fixtures=[Fixture("premium", {})],
                )
            ],
            [],
        ),
        # No match: file mismatch
        (
            TestCase(name="my_test", steps=[], file="/path/to/foo.yml"),
            [
                TestCaseFixtures(
                    test_case_name="my_test",
                    file="/path/to/bar.yml",
                    fixtures=[Fixture("premium", {})],
                )
            ],
            [],
        ),
    ],
)
def test_get_fixtures_for_test_case(
    test_case: TestCase,
    fixtures_per_test: list,
    expected_fixture_names: list,
) -> None:
    """Parameterized tests for get_fixtures_for_test_case."""
    result = get_fixtures_for_test_case(test_case, fixtures_per_test)
    assert [f.name for f in result] == expected_fixture_names


def test_get_fixtures_for_test_case_returns_same_fixture_objects() -> None:
    """Returned list should be the same fixture instances from fixtures_per_test."""
    premium = Fixture("premium", {"membership_type": "premium"})
    fixtures_per_test = [
        TestCaseFixtures(
            test_case_name="my_test",
            file="/dir/test.yml",
            fixtures=[premium],
        )
    ]
    test_case = TestCase(name="my_test", steps=[], file="/dir/test.yml")
    result = get_fixtures_for_test_case(test_case, fixtures_per_test)
    assert result == [premium]
    assert result[0] is premium


def test_validate_and_convert_fixtures_invalid_datetime_raises() -> None:
    """Invalid datetime mocking in a fixture raises ValidationError."""
    fixtures = [
        Fixture("bad_datetime", {MOCKED_DATETIME_SLOT: "not-a-valid-date"}),
    ]
    with pytest.raises(ValidationError) as exc_info:
        validate_and_convert_fixtures(fixtures)
    assert (
        exc_info.value.code
        == "e2e_test_runner.validate_mocked_datetime.invalid_value_format"
    )
    assert "mocked_datetime" in exc_info.value.info
    assert "bad_datetime" in exc_info.value.info


def test_validate_and_convert_fixtures_valid_datetime_converted_to_iso() -> None:
    """Valid datetime mocking in fixtures is converted to ISO format."""
    fixtures = [
        Fixture("mocked_jan_15", {MOCKED_DATETIME_SLOT: "2024-01-15"}),
    ]
    result = validate_and_convert_fixtures(fixtures)
    assert len(result) == 1
    assert result[0].name == "mocked_jan_15"
    assert result[0].slots_set[MOCKED_DATETIME_SLOT] == "2024-01-15T00:00:00+00:00"
