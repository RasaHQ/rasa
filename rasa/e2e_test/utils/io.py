import math
import os
import pathlib
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import rich
import structlog
from rich.table import Table

import rasa.shared.data
import rasa.shared.utils.cli
import rasa.utils.io
from rasa.e2e_test.constants import (
    KEY_FIXTURES,
    KEY_METADATA,
    KEY_STUB_CUSTOM_ACTIONS,
    KEY_TEST_CASE,
    KEY_TEST_CASES,
    STATUS_FAILED,
    STATUS_PASSED,
    STUB_CUSTOM_ACTION_NAME_SEPARATOR,
)
from rasa.e2e_test.e2e_test_case import Fixture, Metadata, TestSuite, TestCase
from rasa.e2e_test.utils.validation import (
    read_e2e_test_schema,
    validate_path_to_test_cases,
    validate_test_case,
)
from rasa.shared.utils.yaml import (
    is_key_in_yaml,
    parse_raw_yaml,
    validate_yaml_data_using_schema_with_assertions,
)
from rasa.utils.beta import BetaNotEnabledException, ensure_beta_feature_is_enabled

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_test_result import TestResult
    from rasa.e2e_test.aggregate_test_stats_calculator import AccuracyCalculation


RASA_PRO_BETA_E2E_ASSERTIONS_ENV_VAR_NAME = "RASA_PRO_BETA_E2E_ASSERTIONS"
RASA_PRO_BETA_STUB_CUSTOM_ACTION_ENV_VAR_NAME = "RASA_PRO_BETA_STUB_CUSTOM_ACTION"

structlogger = structlog.get_logger()


def color_difference(diff: List[str]) -> Generator[str, None, None]:
    """Colorize the difference between two strings.

    Example:
        >>> color_difference(["+ Hello", "- World"])
        ["<ansigreen>+ Hello</ansigreen>", "<ansired>- World</ansired>"]

    Args:
        diff: List of lines of the diff.

    Returns:
    Generator of colored lines.
    """
    for line in diff:
        if line.startswith("+"):
            yield "[green3]" + line + "[/green3]"
        elif line.startswith("-"):
            yield "[red3]" + line + "[/red3]"
        elif line.startswith("^"):
            yield "[blue3]" + line + "[/blue3]"
        elif line.startswith("?"):
            yield "[grey37]" + line + "[/grey37]"
        else:
            yield line


def print_failed_case(fail: "TestResult") -> None:
    """Print the details of a failed test case.

    Example:
        >>> print_failed_case(TestResult(TestCase([TestStep()]), True,
        ...                  ["- Hello", "+ World"]))
        ---------------------- test in test.md failed ----------------------
        Mismatch starting at test.md:1:
        <ansired>- Hello</ansired>
        <ansigreen>+ World</ansigreen>
    """
    fail_headline = (
        f"'{fail.test_case.name}' in {fail.test_case.file_with_line()} failed"
    )
    rasa.shared.utils.cli.print_error(
        f"{rasa.shared.utils.cli.pad(fail_headline, char='-')}\n"
    )
    rasa.shared.utils.cli.print_error(
        f"Mismatch starting at {fail.test_case.file}:{fail.error_line}: \n"
    )
    if fail.difference:
        rich.print(("\n".join(color_difference(fail.difference))))

    if fail.assertion_failure:
        rasa.shared.utils.cli.print_error(
            f"Assertion type '{fail.assertion_failure.assertion.type()}' failed "
            f"with this error message: {fail.assertion_failure.error_message}\n"
        )
        rasa.shared.utils.cli.print_error("Actual events transcript:\n")
        rasa.shared.utils.cli.print_error(
            "\n".join(fail.assertion_failure.actual_events_transcript)
        )


def print_test_summary(failed: List["TestResult"]) -> None:
    """Print the summary of the test run.

    Example:
        >>> print_test_summary([TestResult(TestCase([TestStep()]), True,
        ...                  ["- Hello", "+ World"])])
        =================== short test summary info ===================
        FAILED test.md::test
    """
    rasa.shared.utils.cli.print_info(
        rasa.shared.utils.cli.pad("short test summary info")
    )

    for f in failed:
        rasa.shared.utils.cli.print_error(
            f"FAILED {f.test_case.file}::{f.test_case.name}"
        )


def print_final_line(
    passed: List["TestResult"], failed: List["TestResult"], has_failed: bool
) -> None:
    """Print the final line of the test output.

    Args:
        passed: List of passed test cases.
        failed: List of failed test cases.
        has_failed: Boolean, true if the test run has failed.
    """
    final_line_color = "green3" if not has_failed else "red3"

    width = shutil.get_terminal_size((80, 20)).columns

    # calculate the length of the text - this is a bit hacky but works
    text_lengt = (
        math.ceil(len(passed) / 10)  # length of the number of passed tests
        + math.ceil(len(failed) / 10)  # length of the number of failed tests
        + 18  # length of the text "  failed, passed  "
    )
    # we can't use the padding function here as the text contains html tags
    # which are not taken into account when calculating the length
    padding = max(6, width - text_lengt)
    pre_pad = "=" * max(3, padding // 2)
    post_pad = "=" * max(3, math.ceil(padding / 2))
    rich.print(
        f"[{final_line_color}]{pre_pad} "
        f"[bold red3]{len(failed)} failed[/bold red3]"
        f"[bright_white], [/bright_white]"
        f"[bold green3]{len(passed)} passed[/bold green3]"
        f" {post_pad}[/{final_line_color}]"
    )


def print_aggregate_stats(accuracy_calculations: List["AccuracyCalculation"]) -> None:
    """Print the aggregate statistics of the test run."""
    rasa.shared.utils.cli.print_info(
        rasa.shared.utils.cli.pad("Accuracy By Assertion Type")
    )
    table = Table()
    table.add_column("Assertion Type", justify="center", style="cyan")
    table.add_column("Accuracy", justify="center", style="cyan")

    for accuracy_calculation in accuracy_calculations:
        table.add_row(
            accuracy_calculation.assertion_type, f"{accuracy_calculation.accuracy:.2%}"
        )

    rich.print(table)


def print_test_result(
    passed: List["TestResult"],
    failed: List["TestResult"],
    fail_fast: bool = False,
    **kwargs: Any,
) -> None:
    """Print the result of the test run.

    Args:
        passed: List of passed test cases.
        failed: List of failed test cases.
        fail_fast: If true, stop after the first failure.
        **kwargs: additional arguments
    """
    if failed:
        # print failure headline
        print("\n")
        rich.print(f"[bold]{rasa.shared.utils.cli.pad('FAILURES', char='=')}[/bold]")

    # print failed test_Case
    for fail in failed:
        print_failed_case(fail)

    accuracy_calculations = kwargs.get("accuracy_calculations", [])
    if accuracy_calculations:
        print_aggregate_stats(accuracy_calculations)

    print_test_summary(failed)

    if fail_fast:
        rasa.shared.utils.cli.print_error(
            rasa.shared.utils.cli.pad("stopping after 1 failure", char="!")
        )
        has_failed = True
    elif len(failed) + len(passed) == 0:
        # no tests were run, print error
        rasa.shared.utils.cli.print_error(
            rasa.shared.utils.cli.pad("no test cases found", char="!")
        )
        print_e2e_help()
        has_failed = True
    elif failed:
        has_failed = True
    else:
        has_failed = False

    print_final_line(passed, failed, has_failed=has_failed)
    sys.exit(1 if has_failed else 0)


def print_e2e_help() -> None:
    """Print help guiding users how to write e2e tests."""
    rasa.shared.utils.cli.print_info(
        dedent(
            """\
        To start using e2e tests create a yaml file in a test directory, e.g.
        'tests/test_cases.yml'. You can find example test cases in the starter
        pack at

            https://github.com/RasaHQ/starter-pack-intentless-policy#testing-the-policy

        Here is an example of a test case in yaml format:

            test_cases:
            - test_case: "test_greet"
              steps:
              - user: "hello there!"
              - bot: "Hey! How are you?"

        To run the e2e tests, execute:
            >>> rasa test e2e <path-to-test-cases>
    """
        )
    )


def split_into_passed_failed(
    results: List["TestResult"],
) -> Tuple[List["TestResult"], List["TestResult"]]:
    """Get the summary of the test results.

    Args:
        results: List of test results.

    Returns:
        Tuple consisting of passed count, failed count and failed test cases.
    """
    passed_cases = [r for r in results if r.pass_status]
    failed_cases = [r for r in results if not r.pass_status]

    return passed_cases, failed_cases


def has_test_case_with_assertions(test_cases: List["TestCase"]) -> bool:
    """Check if the test cases contain assertions."""
    try:
        next(test_case for test_case in test_cases if test_case.uses_assertions())
    except StopIteration:
        return False

    return True


@lru_cache(maxsize=1)
def extract_test_case_from_path(path: str) -> Tuple[str, str]:
    """Extract test case from path if specified.

    Args:
        path: Path to the file or folder containing test cases.

    Returns:
        Tuple consisting of the path to test cases and the extracted test case name.
    """
    test_case_name = ""

    if "::" in str(path):
        splitted_path = path.split("::")
        test_case_name = splitted_path[-1]
        path = splitted_path[0]

    return path, test_case_name


def is_test_case_file(file_path: Union[str, Path]) -> bool:
    """Check if file contains test cases.

    Args:
        file_path: Path of the file to check.

    Returns:
        `True` if the file contains test cases, `False` otherwise.
    """
    return rasa.shared.data.is_likely_yaml_file(file_path) and is_key_in_yaml(
        file_path, KEY_TEST_CASES
    )


def read_test_cases(path: str) -> TestSuite:
    """Read test cases from the given path.

    Args:
        path: Path to the file or folder containing test cases.

    Returns:
        TestSuite.
    """
    from rasa.e2e_test.stub_custom_action import (
        StubCustomAction,
        get_stub_custom_action_key,
    )

    path, test_case_name = extract_test_case_from_path(path)
    validate_path_to_test_cases(path)

    test_files = rasa.shared.data.get_data_files([path], is_test_case_file)
    e2e_test_schema = read_e2e_test_schema()

    input_test_cases = []
    fixtures: Dict[str, Fixture] = {}
    metadata: Dict[str, Metadata] = {}
    stub_custom_actions: Dict[str, StubCustomAction] = {}

    beta_flag_verified = False

    for test_file in test_files:
        test_file_content = parse_raw_yaml(Path(test_file).read_text(encoding="utf-8"))

        validate_yaml_data_using_schema_with_assertions(
            yaml_data=test_file_content, schema_content=e2e_test_schema
        )

        test_cases_content = test_file_content.get(KEY_TEST_CASES) or []

        if test_case_name:
            test_cases = [
                TestCase.from_dict(test_case_dict, file=test_file)
                for test_case_dict in test_cases_content
                if test_case_name == test_case_dict.get(KEY_TEST_CASE)
            ]
        else:
            test_cases = [
                TestCase.from_dict(test_case_dict, file=test_file)
                for test_case_dict in test_cases_content
            ]

        beta_flag_verified = verify_beta_feature_flag_for_assertions(
            test_cases, beta_flag_verified
        )

        input_test_cases.extend(test_cases)
        fixtures_content = test_file_content.get(KEY_FIXTURES) or []
        for fixture in fixtures_content:
            fixture_obj = Fixture.from_dict(fixture_dict=fixture)

            # avoid adding duplicates from across multiple files
            if fixtures.get(fixture_obj.name) is None:
                fixtures[fixture_obj.name] = fixture_obj

        metadata_contents = test_file_content.get(KEY_METADATA) or []
        for metadata_content in metadata_contents:
            metadata_obj = Metadata.from_dict(metadata_dict=metadata_content)

            # avoid adding duplicates from across multiple files
            if metadata.get(metadata_obj.name) is None:
                metadata[metadata_obj.name] = metadata_obj

        stub_custom_actions_contents = (
            test_file_content.get(KEY_STUB_CUSTOM_ACTIONS) or {}
        )

        for action_name, stub_data in stub_custom_actions_contents.items():
            if STUB_CUSTOM_ACTION_NAME_SEPARATOR in action_name:
                stub_custom_action_key = action_name
            else:
                test_file_name = Path(test_file).name
                stub_custom_action_key = get_stub_custom_action_key(
                    test_file_name, action_name
                )
            stub_custom_actions[stub_custom_action_key] = StubCustomAction.from_dict(
                action_name=action_name,
                stub_data=stub_data,
            )

    validate_test_case(test_case_name, input_test_cases)
    try:
        if stub_custom_actions:
            ensure_beta_feature_is_enabled(
                "enabling stubs for custom actions",
                RASA_PRO_BETA_STUB_CUSTOM_ACTION_ENV_VAR_NAME,
            )
    except BetaNotEnabledException as exc:
        rasa.shared.utils.cli.print_error_and_exit(str(exc))

    return TestSuite(
        input_test_cases,
        list(fixtures.values()),
        list(metadata.values()),
        stub_custom_actions,
    )


def verify_beta_feature_flag_for_assertions(
    test_cases: List["TestCase"], beta_flag_verified: bool
) -> bool:
    """Verify the beta feature flag for assertions."""
    if beta_flag_verified:
        return True

    if not has_test_case_with_assertions(test_cases):
        return beta_flag_verified

    try:
        ensure_beta_feature_is_enabled(
            "end-to-end testing with assertions",
            RASA_PRO_BETA_E2E_ASSERTIONS_ENV_VAR_NAME,
        )
    except BetaNotEnabledException as exc:
        rasa.shared.utils.cli.print_error_and_exit(str(exc))

    return True


def _save_coverage_report(
    report: Optional[pd.DataFrame], test_status: str, output_dir: str
) -> None:
    if report is None:
        return

    if report is not None and not report.empty:
        if test_status == STATUS_PASSED:
            rasa.shared.utils.cli.print_success(report.to_string(index=False))
        else:
            rasa.shared.utils.cli.print_error(report.to_string(index=False))

    output_filename = f"coverage_report_for_{test_status}_tests.csv"
    output_file_path = os.path.join(output_dir, output_filename)
    report.to_csv(output_file_path, index=False)
    structlogger.info(
        "rasa.e2e_test.save_coverage_report",
        message=f"Coverage result for {test_status} e2e tests"
        f" is written to '{output_file_path}'.",
    )


def write_test_results_to_file(results: List["TestResult"], output_file: str) -> None:
    """Write test results to a file.

    Args:
        results: List of test results.
        output_file: Path to the output file.
    """
    Path(output_file).touch()

    data = {"test_results": [test_result.as_dict() for test_result in results]}

    rasa.utils.io.write_yaml(
        data, target=output_file, transform=transform_results_output_to_yaml
    )

    if STATUS_PASSED in output_file:
        rasa.shared.utils.cli.print_info(
            f"Passing test results have been saved at path: {output_file}."
        )
    elif STATUS_FAILED in output_file:
        rasa.shared.utils.cli.print_info(
            f"Failing test results have been saved at path: {output_file}."
        )


def transform_results_output_to_yaml(yaml_string: str) -> str:
    """Transform the output of the YAML writer to make it more readable.

    Args:
        yaml_string: The YAML string to transform.

    Returns:
        The transformed YAML string.
    """
    result = []
    for s in yaml_string.splitlines(True):
        if s.startswith("- name"):
            result.append("\n")
            result.append(s)
        elif s.startswith("\n"):
            result.append(s.strip())
        elif s.strip().startswith("#"):
            continue
        else:
            result.append(s)
    return "".join(result)


def _save_tested_commands_histogram(
    count_dict: Dict[str, int], test_status: str, output_dir: str
) -> None:
    """Creates a command histogram and saves it to the specified directory.

    Args:
        count_dict (Dict[str, int]): A dictionary where keys are commands
        and values are counts.
        test_status (str): passing or failing
        output_dir (str): The directory path where the histogram
        image will be saved.
    """
    if not count_dict:
        return

    # Sort the dictionary by keys
    sorted_count_dict = dict(sorted(count_dict.items()))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_count_dict.keys(), sorted_count_dict.values(), color="blue")
    plt.xlabel("Commands")
    plt.ylabel("Counts")
    plt.title(f"Command histogram for {test_status} tests")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Add total number to each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            int(yval),
            ha="center",
            va="bottom",
        )

    output_filename = f"commands_histogram_for_{test_status}_tests.png"
    output_file_path = pathlib.Path().joinpath(output_dir, output_filename)
    plt.savefig(output_file_path)
    plt.close()

    structlogger.info(
        "rasa.e2e_test._save_tested_commands_histogram",
        message=f"Commands histogram for {test_status} e2e tests "
        f"are written to '{output_file_path}'.",
    )


def save_test_cases_to_yaml(
    test_results: List["TestResult"],
    output_dir: str,
    status: str,
    test_suite: TestSuite,
) -> None:
    """Extracts TestCases from a list of TestResults and saves them to a YAML file."""
    if not test_results:
        return

    test_cases = [result.test_case for result in test_results]
    new_test_suite = TestSuite(
        test_cases=test_cases,
        fixtures=test_suite.fixtures,
        metadata=test_suite.metadata,
        stub_custom_actions=test_suite.stub_custom_actions,
    )

    output_filename = f"{status}.yml"
    output_file_path = os.path.join(output_dir, output_filename)
    rasa.utils.io.write_yaml(new_test_suite.as_dict(), target=output_file_path)

    structlogger.info(
        "rasa.e2e_test.save_e2e_test_cases",
        message=f"E2e tests with '{status}' status are written to file: "
        f"'{output_file_path}'.",
    )
    if status == STATUS_PASSED:
        structlogger.info(
            "rasa.e2e_test.save_e2e_test_cases",
            message=f"You can use the file: '{output_file_path}' in case you want to "
            f"create training data for fine-tuning an LLM via "
            f"'rasa llm finetune prepare-data'.",
        )
