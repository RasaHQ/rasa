import argparse
import asyncio
import sys
from pathlib import Path
from typing import List

import structlog

import rasa.cli.arguments.run
import rasa.cli.utils
import rasa.shared.data
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.utils.io
from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_endpoint_param,
    add_model_param,
    add_remote_storage_param,
)
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.aggregate_test_stats_calculator import (
    AggregateTestStatsCalculator,
)
from rasa.e2e_test.constants import (
    DEFAULT_COVERAGE_OUTPUT_PATH,
    DEFAULT_E2E_INPUT_TESTS_PATH,
    DEFAULT_E2E_OUTPUT_TESTS_PATH,
    STATUS_FAILED,
    STATUS_PASSED,
)
from rasa.e2e_test.e2e_test_case import (
    KEY_STUB_CUSTOM_ACTIONS,
)
from rasa.e2e_test.e2e_test_coverage_report import (
    create_coverage_report,
    extract_tested_commands,
)
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.e2e_test.utils.io import (
    _save_coverage_report,
    _save_tested_commands_histogram,
    extract_test_case_from_path,
    print_test_result,
    read_test_cases,
    save_test_cases_to_yaml,
    split_into_passed_failed,
    write_test_results_to_file,
)
from rasa.e2e_test.utils.validation import validate_model_path
from rasa.exceptions import RasaException
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH, DEFAULT_MODELS_PATH
from rasa.utils.beta import ensure_beta_feature_is_enabled
from rasa.utils.endpoints import EndpointConfig

RASA_PRO_BETA_FINE_TUNING_RECIPE_ENV_VAR_NAME = "RASA_PRO_BETA_FINE_TUNING_RECIPE"

structlogger = structlog.get_logger()


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

    add_remote_storage_param(parser)

    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate a coverage report on flow paths and commands covered in e2e "
        "tests.",
    )

    parser.add_argument(
        "--coverage-output-path",
        default=DEFAULT_COVERAGE_OUTPUT_PATH,
        help="Directory where to save coverage report to.",
    )


def execute_e2e_tests(args: argparse.Namespace) -> None:
    """Run the end-to-end tests.

    Args:
        args: Commandline arguments.
    """
    if args.coverage_report:
        ensure_beta_feature_is_enabled(
            "LLM fine-tuning recipe",
            env_flag=RASA_PRO_BETA_FINE_TUNING_RECIPE_ENV_VAR_NAME,
        )

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

    if test_suite.stub_custom_actions:
        if not endpoints.action:
            endpoints.action = EndpointConfig()

        endpoints.action.kwargs[KEY_STUB_CUSTOM_ACTIONS] = (
            test_suite.stub_custom_actions
        )

    test_case_path, _ = extract_test_case_from_path(path_to_test_cases)

    try:
        test_runner = E2ETestRunner(
            remote_storage=args.remote_storage,
            model_path=args.model,
            model_server=endpoints.model,
            endpoints=endpoints,
            test_case_path=Path(test_case_path),
        )
    except AgentNotReady as error:
        structlogger.error(
            "rasa.e2e_test.execute_e2e_tests.agent_not_ready", message=error.message
        )
        sys.exit(1)

    results = asyncio.run(
        test_runner.run_tests(
            test_suite.test_cases,
            test_suite.fixtures,
            args.fail_fast,
            input_metadata=test_suite.metadata,
            coverage=args.coverage_report,
        )
    )

    passed, failed = split_into_passed_failed(results)

    if args.e2e_results is not None:
        results_path = Path(args.e2e_results)

        if passed:
            passed_file = rasa.cli.utils.get_e2e_results_file_name(
                results_path, STATUS_PASSED
            )
            write_test_results_to_file(passed, passed_file)

        if failed:
            failed_file = rasa.cli.utils.get_e2e_results_file_name(
                results_path, STATUS_FAILED
            )
            write_test_results_to_file(failed, failed_file)

    aggregate_stats_calculator = AggregateTestStatsCalculator(
        passed_results=passed, failed_results=failed, test_cases=test_suite.test_cases
    )
    accuracy_calculations = aggregate_stats_calculator.calculate()

    if args.coverage_report and test_runner.agent.processor:
        coverage_output_path = args.coverage_output_path
        rasa.shared.utils.io.create_directory(coverage_output_path)
        flows = asyncio.run(test_runner.agent.processor.get_flows())

        for results, status in [(passed, STATUS_PASSED), (failed, STATUS_FAILED)]:
            report = create_coverage_report(flows, results)
            _save_coverage_report(report, status, coverage_output_path)
            tested_commands = extract_tested_commands(results)
            _save_tested_commands_histogram(
                tested_commands, status, coverage_output_path
            )
            save_test_cases_to_yaml(results, coverage_output_path, status, test_suite)

        rasa.shared.utils.cli.print_info(
            f"Coverage data and result is written to {coverage_output_path}."
        )

    print_test_result(
        passed, failed, args.fail_fast, accuracy_calculations=accuracy_calculations
    )
