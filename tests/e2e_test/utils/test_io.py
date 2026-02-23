import os
import textwrap
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from matplotlib import pyplot as plt
from pytest import TempPathFactory
from structlog.testing import capture_logs

from rasa.e2e_test.assertions import AssertionFailure, FlowStartedAssertion
from rasa.e2e_test.constants import (
    KEY_BOT_INPUT,
    KEY_FIXTURES,
    KEY_TEST_CASE,
    KEY_TEST_CASES,
    KEY_USER_INPUT,
    STATUS_PASSED,
)
from rasa.e2e_test.e2e_test_case import (
    Fixture,
    Metadata,
    TestCase,
    TestCaseFixtures,
    TestStep,
    TestSuite,
)
from rasa.e2e_test.e2e_test_coverage_report import (
    COVERAGE_COL_NAME,
    FLOW_NAME_COL_NAME,
    LINE_NUMBERS_COL_NAME,
    MISSING_STEPS_COL_NAME,
    NUM_STEPS_COL_NAME,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.e2e_test.utils.io import (
    _merge_fixture_chain,
    _save_coverage_report,
    _save_tested_commands_histogram,
    color_difference,
    extract_fixtures_from_content,
    extract_test_case_from_path,
    get_conftest_files_for_directory,
    is_test_case_file,
    print_failed_case,
    print_test_result,
    read_test_cases,
    save_test_cases_to_yaml,
    split_into_passed_failed,
    transform_results_output_to_yaml,
    write_failed_tests_to_file,
    write_test_results_to_file,
)
from rasa.shared.exceptions import DuplicateFixtureException, RasaException


def _fixtures_per_test_from_list(
    fixtures: List[Fixture],
    test_case_name: str = "",
    file_name: str = "",
) -> List[TestCaseFixtures]:
    """Build fixtures_per_test from a list of fixtures (for tests)."""
    return [
        TestCaseFixtures(
            test_case_name=test_case_name, file=file_name, fixtures=fixtures
        )
    ]


def _get_all_fixtures_from_resolved(
    fixtures_per_test: List[TestCaseFixtures],
) -> List[Fixture]:
    """Flatten fixtures_per_test to unique list of fixtures (for tests)."""
    seen: set = set()
    out: List[Fixture] = []
    for tc in fixtures_per_test:
        for f in tc.fixtures:
            if id(f) not in seen:
                seen.add(id(f))
                out.append(f)
    return out


def test_get_test_results_summary() -> None:
    results = [
        TestResult(TestCase("test_booking", []), True, []),
        TestResult(TestCase("test_mood_great", []), False, ["diff a"]),
        TestResult(TestCase("test_mood_sad", []), False, ["diff b"]),
    ]

    passed, failed = split_into_passed_failed(results)
    assert passed == results[:1]
    assert failed == results[1:]


def test_find_test_case_line_number(e2e_input_folder: Path) -> None:
    test_suite = read_test_cases(str(e2e_input_folder / "e2e_test_cases.yml"))
    input_test_cases = test_suite.test_cases

    assert len(input_test_cases) == 2
    assert input_test_cases[0].name == "test_booking"
    assert input_test_cases[0].line == 2

    assert input_test_cases[1].name == "test_mood_great"
    assert input_test_cases[1].line == 11


def test_get_conftest_files_for_directory_empty(tmp_path: Path) -> None:
    """No conftest files in directory returns empty list."""
    assert get_conftest_files_for_directory(str(tmp_path)) == []


def test_get_conftest_files_for_directory_single_conftest_yml(tmp_path: Path) -> None:
    """Single conftest.yml in directory is returned (root first order)."""
    conftest = tmp_path / "conftest.yml"
    conftest.write_text("fixtures: []")
    result = get_conftest_files_for_directory(str(tmp_path))
    assert len(result) == 1
    assert result[0] == str(conftest)


def test_get_conftest_files_for_directory_prefers_yml_over_yaml(
    tmp_path: Path,
) -> None:
    """When both exist, conftest.yml is chosen (CONFTEST_FILENAMES order)."""
    (tmp_path / "conftest.yml").write_text("fixtures: []")
    (tmp_path / "conftest.yaml").write_text("fixtures: []")
    result = get_conftest_files_for_directory(str(tmp_path))
    assert len(result) == 1
    assert result[0].endswith("conftest.yml")


def test_get_conftest_files_for_directory_nested_root_first(tmp_path: Path) -> None:
    """Conftest files from parent to child: root first, then leaf."""
    root_conftest = tmp_path / "conftest.yml"
    root_conftest.write_text("fixtures: []")
    child = tmp_path / "sub"
    child.mkdir()
    child_conftest = child / "conftest.yml"
    child_conftest.write_text("fixtures: []")

    result = get_conftest_files_for_directory(str(child))
    assert len(result) == 2
    assert result[0] == str(root_conftest)
    assert result[1] == str(child_conftest)


def test_get_conftest_files_for_directory_given_file_path(tmp_path: Path) -> None:
    """When given a file path, uses its directory to search for conftest."""
    sub = tmp_path / "sub"
    sub.mkdir()
    conftest = sub / "conftest.yml"
    conftest.write_text("fixtures: []")
    some_file = sub / "tests.yml"
    some_file.write_text("test_cases: []")

    result = get_conftest_files_for_directory(str(some_file))
    assert len(result) == 1
    assert result[0] == str(conftest)


def test_extract_fixtures_from_content_duplicate_names_raises() -> None:
    """Duplicate fixture names in same content raise DuplicateFixtureException."""
    content = {
        KEY_FIXTURES: [
            {"premium": [{"membership_type": "premium"}]},
            {"premium": [{"membership_type": "gold"}]},
        ]
    }
    with pytest.raises(DuplicateFixtureException) as exc_info:
        extract_fixtures_from_content(content, source_file="/path/conftest.yml")
    assert "Duplicate fixture" in str(exc_info.value)
    assert "premium" in str(exc_info.value)


def test_merge_fixture_chain_child_overrides_parent() -> None:
    """Later fixture dicts override earlier (child conftest overrides parent)."""
    parent_fixtures = {
        "premium": Fixture("premium", {"membership_type": "parent_value"})
    }
    child_fixtures = {"premium": Fixture("premium", {"membership_type": "child_value"})}
    merged = _merge_fixture_chain([parent_fixtures, child_fixtures])
    assert merged["premium"].slots_set["membership_type"] == "child_value"


@pytest.mark.parametrize(
    "test_case_file,expected",
    [
        ("e2e_one_test.yml", True),
        ("e2e_test_cases.yml", True),
        ("e2e_test_cases.yml", True),
        ("e2e_empty.yml", True),
        ("not_a_test_file.yml", False),
        ("foo.bar", False),
    ],
)
def test_is_test_case_file(
    e2e_input_folder: Path, test_case_file: str, expected: bool
) -> None:
    assert is_test_case_file(str(e2e_input_folder / test_case_file)) == expected


def test_find_test_cases_handles_empty(e2e_input_folder: Path) -> None:
    test_suite = read_test_cases(str(e2e_input_folder / "e2e_empty.yml"))

    assert len(test_suite.test_cases) == 0


def test_find_test_sets_file(e2e_input_folder: Path) -> None:
    test_suite = read_test_cases(str(e2e_input_folder))
    input_test_cases = test_suite.test_cases

    assert len(input_test_cases) == 9
    assert input_test_cases[0].file == str(e2e_input_folder / "e2e_one_test.yml")
    assert input_test_cases[1].file == str(
        e2e_input_folder / "e2e_one_test_with_fixtures.yml"
    )
    assert input_test_cases[2].file == str(
        e2e_input_folder / "e2e_test_case_with_slot_was_set.yml"
    )
    assert input_test_cases[3].file == str(e2e_input_folder / "e2e_test_cases.yml")
    assert input_test_cases[4].file == str(e2e_input_folder / "e2e_test_cases.yml")
    assert input_test_cases[5].file == str(
        e2e_input_folder / "e2e_test_cases_with_fixtures.yml"
    )
    assert input_test_cases[6].file == str(
        e2e_input_folder / "e2e_test_cases_with_fixtures.yml"
    )
    assert input_test_cases[7].file == str(
        e2e_input_folder / "e2e_test_cases_with_metadata.yml"
    )
    assert input_test_cases[8].file == str(
        e2e_input_folder / "e2e_test_cases_with_metadata.yml"
    )


def test_color_difference_empty() -> None:
    assert list(color_difference([])) == []


def test_color_difference() -> None:
    assert list(color_difference(["+ Hello", "- World", "^  ++", "?   ~"])) == [
        "[green3]+ Hello[/green3]",
        "[red3]- World[/red3]",
        "[blue3]^  ++[/blue3]",
        "[grey37]?   ~[/grey37]",
    ]


@pytest.mark.parametrize(
    "input_tests_path, expected_results",
    [
        # Paths to files without global fixtures key
        ("data/end_to_end_testing_input_files/e2e_one_test.yml", []),
        ("data/end_to_end_testing_input_files/e2e_test_cases.yml", []),
        # File with fixtures (only fixtures used by test cases in fixtures_per_test)
        (
            "data/end_to_end_testing_input_files/e2e_test_cases_with_fixtures.yml",
            [
                Fixture(
                    name="standard_membership",
                    slots_set={"membership_type": "standard"},
                ),
            ],
        ),
        (
            "data/end_to_end_testing_input_files/e2e_one_test_with_fixtures.yml",
            [
                Fixture(name="premium", slots_set={"membership_type": "premium"}),
            ],
        ),
    ],
)
def test_read_fixtures(input_tests_path: str, expected_results: List[Fixture]) -> None:
    adjusted_path = Path(__file__).parent.parent.parent.parent / input_tests_path
    test_suite = read_test_cases(str(adjusted_path))
    assert sorted(
        _get_all_fixtures_from_resolved(test_suite.fixtures_per_test),
        key=lambda f: f.name,
    ) == sorted(expected_results, key=lambda f: f.name)


def test_duplicate_fixture_in_same_file_raises_error(
    tmp_path: Path,
) -> None:
    """Test that duplicate fixture names in the same file raise an error."""
    test_file = tmp_path / "test_with_duplicate_fixtures.yml"
    test_file.write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: premium
              - premium:
                  - membership_type: gold

            test_cases:
              - test_case: "test_booking"
                steps:
                  - user: "Hi!"
                  - bot: "Hello!"
        """)
    )

    with pytest.raises(RasaException) as exc_info:
        read_test_cases(str(test_file))

    assert "Duplicate fixture 'premium'" in str(exc_info.value)
    assert str(test_file) in str(exc_info.value)


def test_duplicate_fixture_names_across_files_allowed_local_override(
    tmp_path: Path,
) -> None:
    """With conftest, same fixture name in different test file is allowed, local wins"""
    file1 = tmp_path / "test_file_1.yml"
    file1.write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: premium
              - standard:
                  - membership_type: standard

            test_cases:
              - test_case: "test_booking_1"
                fixtures: [premium]
                steps:
                  - user: "Hi!"
                  - bot: "Hello!"
        """)
    )

    file2 = tmp_path / "test_file_2.yml"
    file2.write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: gold
              - gold:
                  - membership_type: gold

            test_cases:
              - test_case: "test_booking_2"
                fixtures: [premium]
                steps:
                  - user: "Hi!"
                  - bot: "Hello!"
        """)
    )

    suite = read_test_cases(str(tmp_path))
    assert len(suite.test_cases) == 2
    tc1 = next(
        (
            t
            for t in suite.fixtures_per_test
            if t.test_case_name == "test_booking_1"
            and Path(t.file).name == "test_file_1.yml"
        ),
        None,
    )
    tc2 = next(
        (
            t
            for t in suite.fixtures_per_test
            if t.test_case_name == "test_booking_2"
            and Path(t.file).name == "test_file_2.yml"
        ),
        None,
    )
    assert tc1 is not None and tc2 is not None
    # File1's premium has membership_type: premium
    premium1 = next(f for f in tc1.fixtures if f.name == "premium")
    assert premium1.slots_set == {"membership_type": "premium"}
    # File2's premium has membership_type: gold (local override)
    premium2 = next(f for f in tc2.fixtures if f.name == "premium")
    assert premium2.slots_set == {"membership_type": "gold"}


def test_conftest_fixtures_loaded_from_parent_dir(tmp_path: Path) -> None:
    """Fixtures from conftest.yml in parent dir are available to test files."""
    (tmp_path / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - shared:
                  - logged_in: true
        """)
    )
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "test_a.yml").write_text(
        textwrap.dedent("""
            test_cases:
              - test_case: "test_one"
                fixtures: [shared]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    suite = read_test_cases(str(sub))
    assert len(suite.test_cases) == 1
    tc = next(
        (
            t
            for t in suite.fixtures_per_test
            if t.test_case_name == "test_one" and Path(t.file).name == "test_a.yml"
        ),
        None,
    )
    assert tc is not None
    shared = next(f for f in tc.fixtures if f.name == "shared")
    assert shared.slots_set == {"logged_in": True}


def test_conftest_fixture_overridden_by_test_file(tmp_path: Path) -> None:
    """When test file defines same fixture as conftest, local definition wins."""
    (tmp_path / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: global_premium
        """)
    )
    (tmp_path / "test_local.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: local_premium
            test_cases:
              - test_case: "test_one"
                fixtures: [premium]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    suite = read_test_cases(str(tmp_path))
    tc = next(
        (
            t
            for t in suite.fixtures_per_test
            if t.test_case_name == "test_one" and Path(t.file).name == "test_local.yml"
        ),
        None,
    )
    assert tc is not None
    premium = next(f for f in tc.fixtures if f.name == "premium")
    assert premium.slots_set == {"membership_type": "local_premium"}


def test_child_conftest_overrides_parent_conftest(tmp_path: Path) -> None:
    """Fixtures defined in a child conftest override those from the parent conftest."""
    (tmp_path / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: parent_value
        """)
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: child_value
        """)
    )
    (sub / "test_a.yml").write_text(
        textwrap.dedent("""
            test_cases:
              - test_case: "test_one"
                fixtures: [premium]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    suite = read_test_cases(str(sub))
    tc = next(
        (
            t
            for t in suite.fixtures_per_test
            if t.test_case_name == "test_one" and Path(t.file).name == "test_a.yml"
        ),
        None,
    )
    assert tc is not None
    premium = next(f for f in tc.fixtures if f.name == "premium")
    assert premium.slots_set == {"membership_type": "child_value"}


def test_duplicate_fixture_in_conftest_raises_error(tmp_path: Path) -> None:
    """Duplicate fixtures within a conftest YAML file raise a validation error."""
    (tmp_path / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - premium:
                  - membership_type: premium
              - premium:
                  - membership_type: gold
        """)
    )
    (tmp_path / "test_a.yml").write_text(
        textwrap.dedent("""
            test_cases:
              - test_case: "test_one"
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    with pytest.raises(RasaException) as exc_info:
        read_test_cases(str(tmp_path))
    assert "Duplicate fixture" in str(exc_info.value)
    assert "conftest" in str(exc_info.value).lower() or "premium" in str(exc_info.value)


def test_path_to_specific_file_resolves_parent_conftest(tmp_path: Path) -> None:
    """When E2E path is a specific file, parent conftest files are resolved."""
    (tmp_path / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - shared:
                  - logged_in: true
        """)
    )
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "test_a.yml").write_text(
        textwrap.dedent("""
            test_cases:
              - test_case: "test_one"
                fixtures: [shared]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    # Path to specific file (not directory) still resolves parent conftest
    suite = read_test_cases(str(sub / "test_a.yml"))
    assert len(suite.test_cases) == 1
    tc = next(
        (
            t
            for t in suite.fixtures_per_test
            if t.test_case_name == "test_one" and Path(t.file).name == "test_a.yml"
        ),
        None,
    )
    assert tc is not None
    shared = next(f for f in tc.fixtures if f.name == "shared")
    assert shared.slots_set == {"logged_in": True}


def test_path_to_specific_test_case_resolves_parent_conftest(tmp_path: Path) -> None:
    """Path file::test_case_name still resolves parent conftest files correctly."""
    (tmp_path / "conftest.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - shared:
                  - logged_in: true
        """)
    )
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "test_a.yml").write_text(
        textwrap.dedent("""
            test_cases:
              - test_case: "test_one"
                fixtures: [shared]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
              - test_case: "test_two"
                steps:
                  - user: "Bye"
                  - bot: "Bye"
        """)
    )
    path_with_case = str(sub / "test_a.yml") + "::test_one"
    suite = read_test_cases(path_with_case)
    assert len(suite.test_cases) == 1
    assert suite.test_cases[0].name == "test_one"
    tc = next(
        (t for t in suite.fixtures_per_test if t.test_case_name == "test_one"),
        None,
    )
    assert tc is not None
    shared = next(f for f in tc.fixtures if f.name == "shared")
    assert shared.slots_set == {"logged_in": True}


def test_invalid_datetime_mocking_raises_error(tmp_path: Path) -> None:
    """Invalid datetime mocking in a fixture raises an error on load."""
    (tmp_path / "test_invalid_datetime.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - bad_datetime:
                  - mocked_datetime: "not-a-valid-date"
            test_cases:
              - test_case: "test_one"
                fixtures: [bad_datetime]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    with pytest.raises(RasaException) as exc_info:
        read_test_cases(str(tmp_path))
    assert "mocked_datetime" in str(exc_info.value) or "Unable to convert" in str(
        exc_info.value
    )


def test_valid_datetime_mocking_converted_to_iso(tmp_path: Path) -> None:
    """Valid datetime mocking in fixtures is converted to ISO and available to tests."""
    (tmp_path / "test_valid_datetime.yml").write_text(
        textwrap.dedent("""
            fixtures:
              - mocked_jan_15:
                  - mocked_datetime: "2024-01-15"
            test_cases:
              - test_case: "test_one"
                fixtures: [mocked_jan_15]
                steps:
                  - user: "Hi"
                  - bot: "Hello"
        """)
    )
    suite = read_test_cases(str(tmp_path))
    tc = next(
        (t for t in suite.fixtures_per_test if t.test_case_name == "test_one"),
        None,
    )
    assert tc is not None
    fixture = next(f for f in tc.fixtures if f.name == "mocked_jan_15")
    assert fixture.slots_set.get("mocked_datetime") == "2024-01-15T00:00:00+00:00"


@pytest.mark.parametrize(
    "input_tests_path, expected_results",
    [
        # Paths to files without global metadata key
        ("data/end_to_end_testing_input_files/e2e_one_test.yml", []),
        ("data/end_to_end_testing_input_files/e2e_test_cases.yml", []),
        # Path to file with global metadata key
        (
            "data/end_to_end_testing_input_files/e2e_test_cases_with_metadata.yml",
            [
                Metadata(
                    name="user_info",
                    metadata={"language": "English", "location": "Europe"},
                ),
                Metadata(name="device_info", metadata={"os": "linux"}),
            ],
        ),
        # Path to directory with files with global metadata key
        (
            "data/end_to_end_testing_input_files",
            [
                Metadata(
                    name="user_info",
                    metadata={"language": "English", "location": "Europe"},
                ),
                Metadata(name="device_info", metadata={"os": "linux"}),
            ],
        ),
    ],
)
def test_read_metadata(input_tests_path: str, expected_results: List[Metadata]) -> None:
    adjusted_path = Path(__file__).parent.parent.parent.parent / input_tests_path
    test_suite = read_test_cases(str(adjusted_path))
    assert test_suite.metadata == expected_results


def test_transform_results_output_to_yaml() -> None:
    """Test that the function transforms the results output to yaml correctly.

    It should add new lines to the strings that starts with `- name`.
    It should strip out white spaces from strings that starts with `\n`.
    It should also filter out all comments from the result output.
    """
    yaml_string = textwrap.dedent(
        """
    test_results:
    - name: happy path
      pass_status: true
      expected_steps:
       - user: "Hi!"
       - bot: "Hey! How are you?"
       - user: "I am feeling amazing."
       - bot: "Great, carry on!"
       #  - test_case: commented_out_test_case
       #    steps:
       #      - user: "Hi!"
       #      - bot: "Hey! How are you?"

      difference: []
    - name: sad path with utter template (failing status)
      pass_status: false
      expected_steps:
       - user: "Hey"
       - utter: utter_greet
       - user: "I feel overwhelmed."
       - utter: utter_cheer_up
       - utter: utter_did_that_help

      difference:
       - '  user: Hey'
       - '  bot: Hey! How are you?'
       - '  user: I feel overwhelmed.'
       - '- bot: Great, carry on!'
       - '- * No Response *'
       - '+ bot: utter_cheer_up'
       - '+ bot: utter_did_that_help'"""
    )
    transformed_yaml_string = transform_results_output_to_yaml(yaml_string)

    expected_yaml_string = textwrap.dedent(
        """test_results:

- name: happy path
  pass_status: true
  expected_steps:
   - user: "Hi!"
   - bot: "Hey! How are you?"
   - user: "I am feeling amazing."
   - bot: "Great, carry on!"
  difference: []

- name: sad path with utter template (failing status)
  pass_status: false
  expected_steps:
   - user: "Hey"
   - utter: utter_greet
   - user: "I feel overwhelmed."
   - utter: utter_cheer_up
   - utter: utter_did_that_help
  difference:
   - '  user: Hey'
   - '  bot: Hey! How are you?'
   - '  user: I feel overwhelmed.'
   - '- bot: Great, carry on!'
   - '- * No Response *'
   - '+ bot: utter_cheer_up'
   - '+ bot: utter_did_that_help'"""
    )

    assert transformed_yaml_string == expected_yaml_string


@pytest.mark.parametrize(
    "results, output_file, result_type",
    [
        (
            [
                TestResult(
                    test_case=TestCase(
                        name="test case 1", steps=[TestStep(actor="user")]
                    ),
                    pass_status=True,
                    difference=["something", "different"],
                ),
            ],
            "some/file/path/e2e_results_passed.yml",
            "Passing",
        ),
        (
            [
                TestResult(
                    test_case=TestCase(
                        name="test case 2", steps=[TestStep(actor="bot")]
                    ),
                    pass_status=False,
                    difference=["something", "different"],
                ),
            ],
            "some/file/path/e2e_results_failed.yml",
            "Failing",
        ),
    ],
)
@patch("rasa.shared.utils.cli.print_info")
@patch("rasa.utils.io.write_yaml")
@patch("rasa.e2e_test.utils.io.Path")
def test_write_test_results_to_file(
    mock_path: MagicMock,
    mock_write_yaml: MagicMock,
    mock_print_info: MagicMock,
    results: List[TestResult],
    output_file: str,
    result_type: str,
) -> None:
    path_instance = mock_path.return_value
    path_instance.touch = MagicMock()
    write_test_results_to_file(results, output_file)

    mock_path.assert_called_with(output_file)

    expected_result = {
        "test_results": [test_result.as_dict() for test_result in results]
    }
    mock_write_yaml.assert_called_with(
        expected_result, target=output_file, transform=transform_results_output_to_yaml
    )

    mock_print_info.assert_called_with(
        f"{result_type} test results have been saved at path: {output_file}."
    )


@pytest.mark.parametrize(
    "results, output_file, expected_content",
    [
        (
            [
                TestResult(
                    test_case=TestCase(
                        name="test case 1",
                        steps=[
                            TestStep(
                                actor="user", text="Hi!", _underlying={"user": "Hi!"}
                            )
                        ],
                    ),
                    pass_status=True,
                    difference=[],
                ),
            ],
            "e2e_results_passed.yml",
            """test_results:

- name: test case 1
  pass_status: true
  expected_steps:
  - user: Hi!""",
        ),
        (
            [
                TestResult(
                    test_case=TestCase(
                        name="test case 2",
                        steps=[
                            TestStep(
                                actor="bot", text="Hey!", _underlying={"bot": "Hey!"}
                            )
                        ],
                    ),
                    pass_status=False,
                    difference=["something", "different"],
                ),
            ],
            "e2e_results_failed.yml",
            """test_results:

- name: test case 2
  pass_status: false
  expected_steps:
  - bot: Hey!
  difference:
  - something
  - different""",
        ),
        ([], "e2e_results_empty.yml", "test_results: []"),
    ],
)
def test_write_test_results_to_file_already_exists(
    results: List[TestResult], output_file: str, tmp_path: Path, expected_content: str
) -> None:
    existing_content = """
    test_results:
    - name: test case 3
      pass_status: false
      expected_steps:
      - {}
      difference:
      - something
      - different
    """
    output_file_path = tmp_path / output_file
    output_file_path.touch()
    output_file_path.write_text(existing_content)

    write_test_results_to_file(results, str(output_file_path))

    content = output_file_path.read_text()
    assert content.strip() == expected_content.strip()


@pytest.mark.parametrize(
    "full_path_to_test_case, expected_path_to_test_cases, expected_test_case",
    [
        ("some/file/path/e2e_one_test.yml", "some/file/path/e2e_one_test.yml", ""),
        (
            "some/file/path/e2e_one_test.yml::test_case1",
            "some/file/path/e2e_one_test.yml",
            "test_case1",
        ),
    ],
)
def test_extract_test_case_from_path(
    full_path_to_test_case: str,
    expected_path_to_test_cases: Path,
    expected_test_case: str,
) -> None:
    """Test that test case are correctly extracted from the path to test cases."""
    path_to_test_cases, test_case = extract_test_case_from_path(
        str(full_path_to_test_case)
    )
    assert path_to_test_cases == expected_path_to_test_cases
    assert test_case == expected_test_case


@pytest.mark.parametrize(
    "test_cases_file, test_case, number_of_steps",
    [
        ("e2e_test_cases.yml", "test_booking", 6),
        ("e2e_test_cases.yml", "test_mood_great", 4),
        ("", "test_booking", 6),
        ("", "test_standard_booking", 8),
    ],
)
def test_read_single_test_case(
    e2e_input_folder: Path, test_cases_file: str, test_case: str, number_of_steps: int
) -> None:
    path_to_test_case = str(e2e_input_folder / f"{test_cases_file}::{test_case}")
    test_suite = read_test_cases(path_to_test_case)
    input_test_cases = test_suite.test_cases

    assert len(input_test_cases) == 1
    assert input_test_cases[0].name == test_case
    assert len(input_test_cases[0].steps) == number_of_steps


@pytest.mark.parametrize(
    "test_result",
    [
        TestResult(
            test_case=TestCase(
                name="some test case",
                steps=[TestStep(actor="some actor")],
                file="test.yaml",
            ),
            pass_status=False,
            difference=[],
            assertion_failure=AssertionFailure(
                assertion=FlowStartedAssertion(flow_id="test_flow_id"),
                error_message="Test error message",
                actual_events_transcript=["test_event"],
            ),
            error_line=1,
        ),
    ],
)
@patch("rasa.shared.utils.cli.print_error")
def test_print_failed_case(
    mock_print_error: MagicMock,
    test_result: TestResult,
) -> None:
    print_failed_case(test_result)

    mock_print_error.assert_any_call("Mismatch starting at test.yaml:1: \n")
    mock_print_error.assert_any_call(
        "Assertion type 'flow_started' failed with this error message: "
        "Test error message\n"
    )
    mock_print_error.assert_any_call("Actual events transcript:\n")
    mock_print_error.assert_called_with("test_event")


def test_save_coverage_report(tmp_path: Path):
    df_mock = MagicMock()
    df_mock.to_string.return_value = "Coverage Report"
    df_mock.to_csv.return_value = None
    df_mock.columns = [
        FLOW_NAME_COL_NAME,
        NUM_STEPS_COL_NAME,
        MISSING_STEPS_COL_NAME,
        LINE_NUMBERS_COL_NAME,
        COVERAGE_COL_NAME,
    ]
    df_mock.__getitem__.side_effect = lambda key: {
        FLOW_NAME_COL_NAME: ["flow1", "flow2", "Total"],
        NUM_STEPS_COL_NAME: [2, 3, 5],
        MISSING_STEPS_COL_NAME: [1, 2, 3],
        LINE_NUMBERS_COL_NAME: ["[[1-2], [3-4]]", "[[5-6], [7-8]]", ""],
        COVERAGE_COL_NAME: ["50.00%", "33.33%", "40.00%"],
    }[key]

    # Check the report is saved correctly
    output_filename = tmp_path / f"coverage_report_for_{STATUS_PASSED}_tests.csv"

    _save_coverage_report(df_mock, STATUS_PASSED, str(tmp_path))

    df_mock.to_csv.assert_called_with(str(output_filename), index=False)

    # Check the content of the DataFrame
    assert df_mock[FLOW_NAME_COL_NAME] == ["flow1", "flow2", "Total"]
    assert df_mock[NUM_STEPS_COL_NAME] == [2, 3, 5]
    assert df_mock[MISSING_STEPS_COL_NAME] == [1, 2, 3]
    assert df_mock[LINE_NUMBERS_COL_NAME] == [
        "[[1-2], [3-4]]",
        "[[5-6], [7-8]]",
        "",
    ]
    assert df_mock[COVERAGE_COL_NAME] == ["50.00%", "33.33%", "40.00%"]


def test_save_test_cases_to_yaml(tmp_path: Path):
    test_case = TestCase(
        "test_case",
        fixture_names=["fixture"],
        metadata_name="metadata",
        steps=[
            TestStep.from_dict({"user": "I need to check my balance!"}),
            TestStep.from_dict({"bot": "You have $40 in your account."}),
        ],
    )
    test_results = [TestResult(test_case, pass_status=True, difference=[])]
    test_suite = TestSuite(
        test_cases=[test_case],
        metadata=[Metadata("metadata", {"key": "value"})],
        stub_custom_actions={},
        fixtures_per_test=_fixtures_per_test_from_list(
            [Fixture("fixture", {"key": "value"})],
            test_case_name="test_case",
            file_name="",
        ),
    )

    with capture_logs() as logs:
        save_test_cases_to_yaml(test_results, str(tmp_path), STATUS_PASSED, test_suite)
        output_file_path = str(tmp_path / f"{STATUS_PASSED}.yml")
        assert len(logs) == 2
        assert logs[0]["log_level"] == "info"
        assert logs[0]["message"] == (
            f"E2e tests with 'passed' status are written to file: '{output_file_path}'."
        )
        assert logs[1]["log_level"] == "info"
        assert logs[1]["message"] == (
            f"You can use the file: '{output_file_path}' in case you want to create "
            "training data for fine-tuning an LLM via 'rasa llm finetune prepare-data'."
        )

    actual_test_suite = read_test_cases(str(tmp_path / "passed.yml"))

    assert _get_all_fixtures_from_resolved(
        actual_test_suite.fixtures_per_test
    ) == _get_all_fixtures_from_resolved(test_suite.fixtures_per_test)
    assert actual_test_suite.metadata == test_suite.metadata
    assert actual_test_suite.test_cases[0].name == test_suite.test_cases[0].name
    assert (
        actual_test_suite.test_cases[0].steps[0].actor
        == test_suite.test_cases[0].steps[0].actor
    )
    assert (
        actual_test_suite.test_cases[0].steps[0].text
        == test_suite.test_cases[0].steps[0].text
    )
    assert (
        actual_test_suite.test_cases[0].steps[1].actor
        == test_suite.test_cases[0].steps[1].actor
    )
    assert (
        actual_test_suite.test_cases[0].steps[1].text
        == test_suite.test_cases[0].steps[1].text
    )


@patch("rasa.e2e_test.utils.io.print_aggregate_stats")
def test_print_test_result_with_aggregate_stats(
    mock_print_aggregate_stats: MagicMock,
) -> None:
    accuracy_calculations = [Mock()]

    with pytest.raises(SystemExit):
        print_test_result(
            [
                TestResult(
                    test_case=TestCase(
                        name="some test case",
                        steps=[TestStep(actor="some actor")],
                        file="test.yaml",
                    ),
                    pass_status=True,
                    difference=[],
                )
            ],
            [],
            fail_fast=False,
            accuracy_calculations=accuracy_calculations,
        )

    mock_print_aggregate_stats.assert_called_with(accuracy_calculations)


@patch("rasa.e2e_test.utils.io.print_aggregate_stats")
def test_print_test_result_without_aggregate_stats(
    mock_print_aggregate_stats: MagicMock,
) -> None:
    accuracy_calculations = []

    with pytest.raises(SystemExit):
        print_test_result(
            [
                TestResult(
                    test_case=TestCase(
                        name="some test case",
                        steps=[TestStep(actor="some actor")],
                        file="test.yaml",
                    ),
                    pass_status=True,
                    difference=[],
                )
            ],
            [],
            fail_fast=False,
            accuracy_calculations=accuracy_calculations,
        )

    mock_print_aggregate_stats.assert_not_called()


@patch("rasa.e2e_test.utils.io.plt.savefig")
@patch("rasa.e2e_test.utils.io.structlogger.info")
@patch(
    "rasa.e2e_test.utils.io.pathlib.Path.joinpath",
    side_effect=lambda *args: os.sep.join(args),
)
def test_save_tested_commands_histogram(
    mock_joinpath: MagicMock,
    mock_info: MagicMock,
    mock_savefig: MagicMock,
    tmpdir: TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Define test data
    count_dict: Dict[str, int] = {"command1": 10, "command2": 5, "command3": 15}
    test_status: str = "passing"
    output_dir: str = str(tmpdir)
    output_filename: str = f"commands_histogram_for_{test_status}_tests.png"

    # Call the function
    _save_tested_commands_histogram(count_dict, test_status, output_dir)

    # Check that savefig was called with the correct path
    expected_output_file_path: str = os.path.join(output_dir, output_filename)
    mock_savefig.assert_called_once_with(expected_output_file_path)
    plt.close()  # Close the plot to clean up the state for other tests

    # Check that structlogger.info was called with the correct parameters
    mock_info.assert_called_once_with(
        "rasa.e2e_test._save_tested_commands_histogram",
        message=f"Commands histogram for {test_status} e2e tests "
        f"are written to '{expected_output_file_path}'.",
    )

    # Ensure that the file path was joined correctly
    mock_joinpath.assert_called_once_with(output_dir, output_filename)


@patch("rasa.e2e_test.utils.io.structlogger.info")
@patch("rasa.e2e_test.utils.io.plt.savefig")
def test_save_tested_commands_histogram_empty_dict(
    mock_savefig: MagicMock, mock_info: MagicMock
) -> None:
    # Define test data
    count_dict: Dict[str, int] = {}
    test_status: str = "failing"
    output_dir: str = "/some/fake/dir"

    # Call the function
    _save_tested_commands_histogram(count_dict, test_status, output_dir)

    # Check that savefig and info were never called because the dict is empty
    mock_savefig.assert_not_called()
    mock_info.assert_not_called()


def test_writing_test_suite():
    test_suite = read_test_cases(
        "data/end_to_end_testing_input_files/e2e_test_case_with_slot_was_set.yml"
    )
    test_suite_dict = test_suite.as_writable()

    test_case = TestCase.from_dict(test_suite_dict[KEY_TEST_CASES][0])

    assert test_case.steps == test_suite.test_cases[0].steps


def test_read_test_cases_with_utf_characters(tmp_path: Path):
    # 1) Generate a test case with utterances containing special characters
    test_case_name = "user_utterance_with_special_characters"
    utterances = [
        "Grüß dich! Wie läuft's bei dir?",
        "Mir geht's großartig, danke. Möchtest du darüber sprechen?",
        "Mir ist heute ein bisschen langweilig. Irgendwelche Vorschläge?",
        "Vielleicht könntest du ein neues Buch über Geschichte lesen.",
    ]
    tests = f"""{KEY_TEST_CASES}:
  - {KEY_TEST_CASE}: {test_case_name}
    steps:
      - {KEY_USER_INPUT}: {utterances[0]}
      - {KEY_BOT_INPUT}: {utterances[1]}
      - {KEY_USER_INPUT}: {utterances[2]}
      - {KEY_BOT_INPUT}: {utterances[3]}
    """

    # 2) Write a test case to a temporary YAML file
    e2e_test_file = tmp_path / f"{test_case_name}.yml"
    e2e_test_file.write_text(tests)

    # 3) Confirm that read test case is in the proper format
    test_suite = read_test_cases(str(e2e_test_file))
    test_case = test_suite.test_cases[0]
    for idx, step in enumerate(test_case.steps):
        assert step.text == utterances[idx]


@pytest.mark.parametrize(
    "fixtures, metadata, expected_result",
    [
        (
            [],
            [],
            {
                "test_cases": [TestCase(name="test case 1", steps=[]).as_dict()],
            },
        ),
        (
            [Fixture(name="premium", slots_set={"membership_type": "premium"})],
            [],
            {
                "test_cases": [TestCase(name="test case 1", steps=[]).as_dict()],
                "fixtures": [
                    Fixture(
                        name="premium", slots_set={"membership_type": "premium"}
                    ).as_dict()
                ],
            },
        ),
        (
            [],
            [Metadata(name="device_info", metadata={"os": "linux"})],
            {
                "test_cases": [TestCase(name="test case 1", steps=[]).as_dict()],
                "metadata": [
                    Metadata(name="device_info", metadata={"os": "linux"}).as_dict()
                ],
            },
        ),
        (
            [Fixture(name="premium", slots_set={"membership_type": "premium"})],
            [Metadata(name="device_info", metadata={"os": "linux"})],
            {
                "test_cases": [TestCase(name="test case 1", steps=[]).as_dict()],
                "fixtures": [
                    Fixture(
                        name="premium", slots_set={"membership_type": "premium"}
                    ).as_dict()
                ],
                "metadata": [
                    Metadata(name="device_info", metadata={"os": "linux"}).as_dict()
                ],
            },
        ),
    ],
)
@patch("rasa.shared.utils.cli.print_info")
@patch("rasa.utils.io.write_yaml")
@patch("rasa.e2e_test.utils.io.Path")
def test_write_failed_tests_to_file(
    mock_path: MagicMock,
    mock_write_yaml: MagicMock,
    mock_print_info: MagicMock,
    fixtures: List[Fixture],
    metadata: List[Metadata],
    expected_result: Dict[str, List[Dict[str, str]]],
) -> None:
    """Test writing failed tests to file."""

    path_instance = mock_path.return_value
    path_instance.touch = MagicMock()
    results = [
        TestResult(
            test_case=TestCase(name="test case 1", steps=[]),
            pass_status=True,
            difference=[],
        ),
    ]
    test_file = "some/file/path/e2e_failed_tests.yml"
    test_cases = [
        TestCase(name="test case 1", steps=[]),
        TestCase(name="test case 2", steps=[]),
        TestCase(name="test case 3", steps=[]),
    ]

    write_failed_tests_to_file(
        test_cases,
        _fixtures_per_test_from_list(
            fixtures, test_case_name="test case 1", file_name=""
        ),
        metadata,
        results,
        test_file,
    )

    mock_path.assert_called_with(test_file)

    mock_write_yaml.assert_called_with(
        expected_result, target=test_file, transform=transform_results_output_to_yaml
    )

    mock_print_info.assert_called_with(
        f"Failing tests have been saved at path: {test_file}."
    )


def test_write_failed_tests_to_file_creates_parent_directories(tmp_path: Path) -> None:
    """Test that parent directories are created if they don't exist."""
    test_file = tmp_path / "my_tests" / "failed_runs" / "failed.yml"

    results = [
        TestResult(
            test_case=TestCase(name="test case 1", steps=[]),
            pass_status=False,
            difference=[],
        ),
    ]
    test_cases = [TestCase(name="test case 1", steps=[])]

    write_failed_tests_to_file(test_cases, [], [], results, str(test_file))

    assert test_file.exists()
    assert test_file.parent.exists()


@patch("rasa.utils.io.write_yaml")
@patch("rasa.e2e_test.utils.io.Path")
def test_write_failed_tests_to_file_filters_only_failed_tests(
    mock_path: MagicMock,
    mock_write_yaml: MagicMock,
) -> None:
    """Test that only failed tests are written to the file."""
    path_instance = mock_path.return_value
    path_instance.touch = MagicMock()
    path_instance.parent.mkdir = MagicMock()

    test_cases = [
        TestCase(name="test case 1", steps=[]),
        TestCase(name="test case 2", steps=[]),
        TestCase(name="test case 3", steps=[]),
    ]

    # Only test case 1 and 3 failed
    results = [
        TestResult(
            test_case=TestCase(name="test case 1", steps=[]),
            pass_status=False,
            difference=[],
        ),
        TestResult(
            test_case=TestCase(name="test case 3", steps=[]),
            pass_status=False,
            difference=[],
        ),
    ]

    write_failed_tests_to_file(test_cases, [], [], results, "failed.yml")

    written_data = mock_write_yaml.call_args[0][0]
    written_test_cases = [
        test_case.get("test_case") for test_case in written_data["test_cases"]
    ]

    assert len(written_test_cases) == 2
    assert "test case 1" in written_test_cases
    assert "test case 3" in written_test_cases
    assert "test case 2" not in written_test_cases
