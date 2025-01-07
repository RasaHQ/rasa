import argparse
import asyncio
import datetime
import sys
from typing import List

import structlog

import rasa.cli.utils
from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_endpoint_param,
    add_model_param,
    add_remote_storage_param,
)
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.dialogue_understanding_test.command_metric_calculation import (
    calculate_command_metrics,
)
from rasa.dialogue_understanding_test.constants import (
    DEFAULT_INPUT_TESTS_PATH,
    KEY_STUB_CUSTOM_ACTIONS,
)
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
)
from rasa.dialogue_understanding_test.du_test_runner import (
    DialogueUnderstandingTestRunner,
)
from rasa.dialogue_understanding_test.io import (
    read_test_suite,
    write_test_results_to_file,
    print_test_results,
)
from rasa.dialogue_understanding_test.validation import (
    validate_cli_arguments,
    validate_test_cases,
)
from rasa.e2e_test.e2e_test_case import TestSuite
from rasa.exceptions import RasaException
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH
from rasa.utils.beta import ensure_beta_feature_is_enabled
from rasa.utils.endpoints import EndpointConfig

RASA_PRO_BETA_DIALOGUE_UNDERSTANDING_TEST_ENV_VAR_NAME = (
    "RASA_PRO_BETA_DIALOGUE_UNDERSTANDING_TEST"
)

structlogger = structlog.get_logger()


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the dialogue understanding test subparser to `rasa test`.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    for subparser in subparsers.choices.values():
        if subparser.prog == "rasa test":
            du_test_subparser = create_du_test_subparser(parents)

            for action in subparser._subparsers._actions:
                if action.choices is not None:
                    action.choices["du"] = du_test_subparser
                    return

    # If we get here, we couldn't hook the subparser to `rasa test`
    raise RasaException(
        "Hooking the dialogue understanding (du) test subparser to "
        "`rasa test` command could not be completed. "
        "Cannot run dialogue understanding testing."
    )


def create_du_test_subparser(
    parents: List[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Create dialogue understanding test subparser."""
    du_test_subparser = argparse.ArgumentParser(
        prog="rasa test du",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runs dialogue understanding testing.",
    )

    du_test_subparser.set_defaults(func=execute_dialogue_understanding_tests)

    add_du_test_arguments(du_test_subparser)
    add_bot_arguments(du_test_subparser)

    return du_test_subparser


def add_bot_arguments(parser: argparse.ArgumentParser) -> None:
    bot_arguments = parser.add_argument_group("Bot Settings")
    add_model_param(bot_arguments, add_positional_arg=False)
    add_endpoint_param(
        bot_arguments,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )
    add_remote_storage_param(bot_arguments)


def add_du_test_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running dialogue understanding tests."""
    du_arguments = parser.add_argument_group("Testing Settings")
    du_arguments.add_argument(
        "path-to-test-cases",
        nargs="?",
        type=str,
        default=DEFAULT_INPUT_TESTS_PATH,
        help="Input file or folder containing dialogue understanding test cases.",
    )
    du_arguments.add_argument(
        "--output-file",
        type=str,
        default="dialogue_understanding_test_{date:%Y%m%d-%H%M%S}.yml".format(
            date=datetime.datetime.now()
        ),
        help="Path to the output file to write the results to.",
    )
    du_arguments.add_argument(
        "--no-output",
        action="store_true",
        help="If set, no output file will be written to disk.",
    )
    du_arguments.add_argument(
        "--output-prompt",
        action="store_true",
        help="If set, the dialogue understanding test output will contain "
        "prompts for each failure.",
    )


def execute_dialogue_understanding_tests(args: argparse.Namespace) -> None:
    """Run the dialogue understanding tests.

    Args:
        args: Commandline arguments.
    """
    ensure_beta_feature_is_enabled(
        "Dialogue Understanding (DU) Testing",
        env_flag=RASA_PRO_BETA_DIALOGUE_UNDERSTANDING_TEST_ENV_VAR_NAME,
    )

    # basic validation of the passed CLI arguments
    validate_cli_arguments(args)

    # initialization of endpoints
    endpoints = set_up_available_endpoints(args)

    # read test cases from the given path
    test_suite = get_valid_test_suite(args)

    # setup stub custom actions if they are used
    set_up_stub_custom_actions(test_suite, endpoints)

    # set up the test runner, e.g. start the agent
    try:
        test_runner = DialogueUnderstandingTestRunner(
            endpoints=endpoints,
            model_path=args.model,
            model_server=endpoints.model,
            remote_storage=args.remote_storage,
        )
    except AgentNotReady as error:
        structlogger.error(
            "rasa.dialogue_understanding_test.agent_not_ready", message=error.message
        )
        sys.exit(1)

    # run the actual test cases
    test_results = asyncio.run(
        test_runner.run_tests(
            test_suite.test_cases, test_suite.fixtures, test_suite.metadata
        )
    )

    # evaluate test results
    failed_tests, passed_tests = split_test_results(test_results)
    command_metrics = calculate_command_metrics(test_results)

    # write results to console and file
    print_test_results(failed_tests, passed_tests, command_metrics, args.output_prompt)
    if not args.no_output:
        write_test_results_to_file(
            failed_tests,
            passed_tests,
            command_metrics,
            args.output_file,
            args.output_prompt,
        )


def get_valid_test_suite(args: argparse.Namespace) -> TestSuite:
    """Read the test cases from the given test case path and validate them."""
    path_to_test_cases = getattr(args, "path-to-test-cases", DEFAULT_INPUT_TESTS_PATH)
    test_suite = read_test_suite(path_to_test_cases)
    validate_test_cases(test_suite.test_cases)
    return test_suite


def set_up_available_endpoints(args: argparse.Namespace) -> AvailableEndpoints:
    """Set up the available endpoints for the test runner."""
    args.endpoints = rasa.cli.utils.get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    endpoints = AvailableEndpoints.get_instance(args.endpoints)

    # Ignore all endpoints apart from action server, model, and nlu
    # to ensure InMemoryTrackerStore is being used instead of production
    # tracker store
    endpoints.tracker_store = None
    endpoints.lock_store = None
    endpoints.event_broker = None

    # disable nlg endpoint as we don't need it for dialogue understanding tests
    endpoints.nlg = None

    return endpoints


def set_up_stub_custom_actions(
    test_suite: TestSuite, endpoints: AvailableEndpoints
) -> None:
    """Set up the stub custom actions if they are used."""
    if test_suite.stub_custom_actions:
        if not endpoints.action:
            endpoints.action = EndpointConfig()

        endpoints.action.kwargs[KEY_STUB_CUSTOM_ACTIONS] = (
            test_suite.stub_custom_actions
        )


def split_test_results(
    results: List[DialogueUnderstandingTestResult],
) -> tuple[
    List[DialogueUnderstandingTestResult], List[DialogueUnderstandingTestResult]
]:
    """Split the test results into passed and failed test cases."""
    passed_cases = [r for r in results if r.passed]
    failed_cases = [r for r in results if not r.passed]

    return passed_cases, failed_cases
