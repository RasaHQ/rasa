import argparse
import asyncio
import logging
import math
import shutil
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional, Text, Tuple, Union

import rasa.cli.arguments.run
import rasa.cli.utils
import rasa.shared.data
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rich
from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import add_endpoint_param, add_model_param
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.exceptions import RasaException
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH, DEFAULT_MODELS_PATH

from rasa.e2e_test.constants import SCHEMA_FILE_PATH, KEY_TEST_CASE
from rasa.e2e_test.e2e_test_case import (
    KEY_FIXTURES,
    KEY_METADATA,
    Fixture,
    Metadata,
    TestCase,
    TestSuite,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
import rasa.utils.io
from rasa.shared.utils.yaml import (
    parse_raw_yaml,
    read_schema_file,
    validate_yaml_content_using_schema,
    is_key_in_yaml,
)

DEFAULT_E2E_INPUT_TESTS_PATH = "tests/e2e_test_cases.yml"
DEFAULT_E2E_OUTPUT_TESTS_PATH = "tests/e2e_results.yml"
KEY_TEST_CASES = "test_cases"

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the e2e subparser to `rasa test`.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    for subparser in subparsers.choices.values():
        if subparser.prog == "rasa test":
            e2e_test_subparser = create_e2e_test_subparser(parents)

            for action in subparser._subparsers._actions:
                if action.choices is not None:
                    action.choices["e2e"] = e2e_test_subparser
                    return

    # If we get here, we couldn't hook the subparser to `rasa test`
    raise RasaException(
        "Hooking the e2e subparser to `rasa test` command "
        "could not be completed. Cannot run end-to-end testing."
    )


def create_e2e_test_subparser(
    parents: List[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Create e2e test subparser."""
    e2e_test_subparser = argparse.ArgumentParser(
        prog="rasa test e2e",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runs end-to-end testing.",
    )

    e2e_test_subparser.set_defaults(func=execute_e2e_tests)

    add_e2e_test_arguments(e2e_test_subparser)
    add_model_param(e2e_test_subparser, add_positional_arg=False)
    add_endpoint_param(
        e2e_test_subparser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )

    return e2e_test_subparser


def add_e2e_test_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running E2E tests directly using `rasa e2e`."""
    e2e_arguments = parser.add_argument_group("Testing Settings")
    e2e_arguments.add_argument(
        "path-to-test-cases",
        nargs="?",
        type=str,
        help="Input file or folder containing end-to-end test cases.",
        default=DEFAULT_E2E_INPUT_TESTS_PATH,
    )
    e2e_arguments.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail the test suite as soon as a unit test fails.",
    )

    parser.add_argument(
        "-o",
        "--e2e-results",
        action="store_const",
        const=DEFAULT_E2E_OUTPUT_TESTS_PATH,
        help="Results file containing end-to-end testing summary.",
    )

    parser.add_argument(
        "--remote-storage",
        help="Set the remote location where your Rasa model is stored, e.g. on AWS.",
    )


def split_into_passed_failed(
    results: List[TestResult],
) -> Tuple[List[TestResult], List[TestResult]]:
    """Get the summary of the test results.

    Args:
        results: List of test results.

    Returns:
        Tuple consisting of passed count, failed count and failed test cases.
    """
    passed_cases = [r for r in results if r.pass_status]
    failed_cases = [r for r in results if not r.pass_status]

    return passed_cases, failed_cases


def is_test_case_file(file_path: Union[Text, Path]) -> bool:
    """Check if file contains test cases.

    Args:
        file_path: Path of the file to check.

    Returns:
        `True` if the file contains test cases, `False` otherwise.
    """
    return rasa.shared.data.is_likely_yaml_file(file_path) and is_key_in_yaml(
        file_path, KEY_TEST_CASES
    )


def validate_path_to_test_cases(path: Text) -> None:
    """Validate that path to test cases exists."""
    if not Path(path).exists():
        rasa.shared.utils.io.raise_warning(
            f"Path to test cases does not exist: {path}. "
            f"Please provide a valid path to test cases. "
            f"Exiting...",
            UserWarning,
        )
        sys.exit(1)


def extract_test_case_from_path(path: Text) -> Tuple[Text, Text]:
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


def validate_test_case(test_case_name: Text, input_test_cases: List[TestCase]) -> None:
    """Validate that test case exists."""
    if test_case_name and not input_test_cases:
        rasa.shared.utils.io.raise_warning(
            f"Test case does not exist: {test_case_name}. "
            f"Please check for typos and provide a valid test case name. "
            f"Exiting...",
            UserWarning,
        )
        sys.exit(1)


def read_test_cases(path: Text) -> TestSuite:
    """Read test cases from the given path.

    Args:
        path: Path to the file or folder containing test cases.

    Returns:
        TestSuite.
    """
    path, test_case_name = extract_test_case_from_path(path)
    validate_path_to_test_cases(path)

    test_files = rasa.shared.data.get_data_files([path], is_test_case_file)
    e2e_test_schema = read_e2e_test_schema()

    input_test_cases = []
    fixtures: Dict[Text, Fixture] = {}
    metadata: Dict[Text, Metadata] = {}

    for test_file in test_files:
        test_file_content = parse_raw_yaml(Path(test_file).read_text())
        validate_yaml_content_using_schema(test_file_content, e2e_test_schema)

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

        input_test_cases.extend(test_cases)
        fixtures_content = test_file_content.get(KEY_FIXTURES) or []
        metadata_contents = test_file_content.get(KEY_METADATA) or []
        for fixture in fixtures_content:
            fixture_obj = Fixture.from_dict(fixture_dict=fixture)

            # avoid adding duplicates from across multiple files
            if fixtures.get(fixture_obj.name) is None:
                fixtures[fixture_obj.name] = fixture_obj

        for metadata_content in metadata_contents:
            metadata_obj = Metadata.from_dict(metadata_dict=metadata_content)

            # avoid adding duplicates from across multiple files
            if metadata.get(metadata_obj.name) is None:
                metadata[metadata_obj.name] = metadata_obj

    validate_test_case(test_case_name, input_test_cases)
    return TestSuite(input_test_cases, list(fixtures.values()), list(metadata.values()))


def execute_e2e_tests(args: argparse.Namespace) -> None:
    """Run the end-to-end tests.

    Args:
        args: Commandline arguments.
    """
    args.endpoints = rasa.cli.utils.get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

    # Ignore all endpoints apart from action server, model, nlu and nlg
    # to ensure InMemoryTrackerStore is being used instead of production
    # tracker store
    endpoints.tracker_store = None
    endpoints.lock_store = None
    endpoints.event_broker = None

    if endpoints.model is None:
        args.model = validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)

    path_to_test_cases = getattr(
        args, "path-to-test-cases", DEFAULT_E2E_INPUT_TESTS_PATH
    )

    test_suite = read_test_cases(path_to_test_cases)

    try:
        test_runner = E2ETestRunner(
            remote_storage=args.remote_storage,
            model_path=args.model,
            model_server=endpoints.model,
            endpoints=endpoints,
        )
    except AgentNotReady as error:
        logger.error(msg=error.message)
        sys.exit(1)

    results = asyncio.run(
        test_runner.run_tests(
            test_suite.test_cases,
            test_suite.fixtures,
            args.fail_fast,
            input_metadata=test_suite.metadata,
        )
    )

    if args.e2e_results is not None:
        write_test_results_to_file(results, args.e2e_results)

    passed, failed = split_into_passed_failed(results)
    print_test_result(passed, failed, args.fail_fast)


def write_test_results_to_file(results: List[TestResult], output_file: Text) -> None:
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

    rasa.shared.utils.cli.print_info(
        f"Overall results have been saved at path: {output_file}."
    )


def transform_results_output_to_yaml(yaml_string: Text) -> Text:
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
        else:
            result.append(s)
    return "".join(result)


def pad(text: Text, char: Text = "=", min: int = 3) -> Text:
    """Pad text to a certain length.

    Uses `char` to pad the text to the specified length. If the text is longer
    than the specified length, at least `min` are used.

    The padding is applied to the left and right of the text (almost) equally.

    Example:
        >>> pad("Hello")
        "========= Hello ========"
        >>> pad("Hello", char="-")
        "--------- Hello --------"

    Args:
        text: Text to pad.
        min: Minimum length of the padding.
        char: Character to pad with.

    Returns:
        Padded text.
    """
    width = shutil.get_terminal_size((80, 20)).columns
    padding = max(width - len(text) - 2, min * 2)

    return char * (padding // 2) + " " + text + " " + char * math.ceil(padding / 2)


def color_difference(diff: List[Text]) -> Generator[Text, None, None]:
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


def print_failed_case(fail: TestResult) -> None:
    """Print the details of a failed test case.

    Example:
        >>> print_failed_case(TestResult(TestCase("test", "test.md"), 1,
        ...                  ["- Hello", "+ World"]))
        ---------------------- test in test.md failed ----------------------
        Mismatch starting at test.md:1:
        <ansired>- Hello</ansired>
        <ansigreen>+ World</ansigreen>
    """
    fail_headline = (
        f"'{fail.test_case.name}' in {fail.test_case.file_with_line()} failed"
    )
    rasa.shared.utils.cli.print_error(f"{pad(fail_headline, char='-')}\n")
    print(f"Mismatch starting at {fail.test_case.file}:{fail.error_line}: \n")
    rich.print(("\n".join(color_difference(fail.difference))))


def print_test_summary(failed: List[TestResult]) -> None:
    """Print the summary of the test run.

    Example:
        >>> print_test_summary([TestResult(TestCase("test", "test.md"), 1,
        ...                  ["- Hello", "+ World"])])
        =================== short test summary info ===================
        FAILED test.md::test
    """
    rasa.shared.utils.cli.print_info(pad("short test summary info"))

    for f in failed:
        rasa.shared.utils.cli.print_error(
            f"FAILED {f.test_case.file}::{f.test_case.name}"
        )


def print_final_line(
    passed: List[TestResult], failed: List[TestResult], has_failed: bool
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


def print_test_result(
    passed: List[TestResult],
    failed: List[TestResult],
    fail_fast: bool = False,
) -> None:
    """Print the result of the test run.

    Args:
        passed: List of passed test cases.
        failed: List of failed test cases.
        fail_fast: If true, stop after the first failure.
    """
    if failed:
        # print failure headline
        print("\n")
        rich.print(f"[bold]{pad('FAILURES', char='=')}[/bold]")

    # print failed test_Case
    for fail in failed:
        print_failed_case(fail)

    print_test_summary(failed)

    if fail_fast:
        rasa.shared.utils.cli.print_error(pad("stopping after 1 failure", char="!"))
        has_failed = True
    elif len(failed) + len(passed) == 0:
        # no tests were run, print error
        rasa.shared.utils.cli.print_error(pad("no test cases found", char="!"))
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


def validate_model_path(
    model_path: Optional[Text], parameter: Text, default: Text
) -> Text:
    """Validate the model path.

    Args:
        model_path: Path to the model.
        parameter: Name of the parameter.
        default: Default path to the model.

    Returns:
    Path to the model.
    """
    if model_path and Path(model_path).exists():
        return model_path

    if model_path and not Path(model_path).exists():
        rasa.shared.utils.io.raise_warning(
            f"The provided model path '{model_path}' could not be found. "
            f"Using default location '{default}' instead.",
            UserWarning,
        )

    elif model_path is None:
        logger.info(
            f"Parameter '{parameter}' is not set. "
            f"Using default location '{default}' instead."
        )

    Path(default).mkdir(exist_ok=True)
    return default


def read_e2e_test_schema() -> Union[List[Any], Dict[Text, Any]]:
    """Read the schema for the e2e test files.

    Returns:
        The content of the schema.
    """
    return read_schema_file(SCHEMA_FILE_PATH)
