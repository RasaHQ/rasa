import argparse
import asyncio
import datetime
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, cast

import structlog

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_endpoint_param,
    add_model_param,
    add_remote_storage_param,
    add_sub_agents_param,
)
from rasa.core.agent import Agent
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.core.exceptions import AgentNotReady
from rasa.core.processor import MessageProcessor
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator import LLMBasedCommandGenerator
from rasa.dialogue_understanding.generator.command_parser import DEFAULT_COMMANDS
from rasa.dialogue_understanding_test.command_metric_calculation import (
    calculate_command_metrics,
)
from rasa.dialogue_understanding_test.constants import (
    DEFAULT_INPUT_TESTS_PATH,
    KEY_STUB_CUSTOM_ACTIONS,
)
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
    DialogueUnderstandingTestSuiteResult,
)
from rasa.dialogue_understanding_test.du_test_runner import (
    DialogueUnderstandingTestRunner,
)
from rasa.dialogue_understanding_test.io import (
    read_test_suite,
    write_test_results_to_file,
)
from rasa.dialogue_understanding_test.validation import (
    validate_cli_arguments,
    validate_test_cases,
)
from rasa.e2e_test.e2e_test_case import TestSuite
from rasa.exceptions import RasaException
from rasa.shared.agents.agent_setup import AgentsConnectionCleanup
from rasa.shared.constants import (
    LLM_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.utils.llm import (
    combine_custom_and_default_config,
    resolve_model_client_config,
)
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
    du_arguments.add_argument(
        "--additional-commands",
        type=str,
        nargs="*",
        help=(
            "List of additional custom command classes to add, separated by spaces. "
            "For example: --additional-commands my_module.MyCustomCommand"
        ),
    )
    du_arguments.add_argument(
        "--remove-default-commands",
        type=str,
        nargs="*",
        help=(
            f"List of default commands to remove, separated by spaces. "
            f"Default commands include: "
            f"{', '.join([command.__name__ for command in DEFAULT_COMMANDS])}. "
            f"For example: --remove-default-commands ClarifyCommand HumanHandoffCommand"
        ),
    )

    add_sub_agents_param(du_arguments)


def ensure_calm_only_bot(agent: Agent) -> None:
    if agent.domain is None or agent.processor is None:
        return

    if ROUTE_TO_CALM_SLOT in [slot.name for slot in agent.domain.slots]:
        rasa.shared.utils.cli.print_error(
            "You are using coexistence. Dialogue Understanding Tests do only work for "
            "CALM only assistants."
        )
        sys.exit(0)

    if not agent.processor.is_calm_assistant:
        rasa.shared.utils.cli.print_error(
            "Dialogue Understanding Tests do only work for CALM assistants. "
            "Your assistant does not use CALM."
        )
        sys.exit(0)


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
    Configuration.initialise_sub_agents(args.sub_agents)

    # set up the test runner, e.g. start the agent
    try:
        test_runner = DialogueUnderstandingTestRunner(
            endpoints=endpoints,
            model_path=args.model,
            model_server=endpoints.model,
            remote_storage=args.remote_storage,
            sub_agents_path=args.sub_agents,
        )
    except AgentNotReady as error:
        structlogger.error(
            "rasa.dialogue_understanding_test.agent_not_ready", message=error.message
        )
        sys.exit(1)

    # Exit if the bot is not calm only
    ensure_calm_only_bot(test_runner.agent)

    # Ensure processor is not None so that we can extract the flows and the llm config
    if test_runner.agent.processor is None:
        rasa.shared.utils.cli.print_error(
            "No processor: Not able to retrieve flows and config from trained model."
        )
        sys.exit(0)

    # flows are needed in order to parse the commands when reading the test cases
    flows = asyncio.run(test_runner.agent.processor.get_flows())
    # llm config is needed for instrumentation
    llm_config = _get_llm_command_generator_config(test_runner.agent.processor)

    # read test cases from the given path
    test_suite: TestSuite = get_valid_test_suite(args, flows, test_runner.agent.domain)

    # setup stub custom actions if they are used
    set_up_stub_custom_actions(test_suite, endpoints)

    async def run_test_cases_with_agents_connection_cleanup() -> (
        List[DialogueUnderstandingTestResult]
    ):
        # Create a wrapper coroutine that handles graceful agents connection
        # cleanup, this prevents abrupt closure when asyncio.run finishes.
        async with AgentsConnectionCleanup():
            return await test_runner.run_test_cases(
                test_suite.test_cases, test_suite.fixtures, test_suite.metadata
            )
        # Defensive return for mypy - never actually reached
        return  # type: ignore[return-value]

    # run the actual test cases
    test_results = asyncio.run(run_test_cases_with_agents_connection_cleanup())

    # evaluate test results
    passing_test_results, failing_test_results = split_test_results(test_results)
    command_metrics = calculate_command_metrics(test_results)

    test_suite_result = DialogueUnderstandingTestSuiteResult.from_results(
        failing_test_results, passing_test_results, command_metrics, llm_config
    )

    # Do not move this import to the top of the file as it will break the
    # instrumentation of this function: the CLI module is initialized before the
    # instrumentation is set up, and we won't be able to "replace" the function
    # with the instrumented wrapper
    from rasa.dialogue_understanding_test.io import print_test_results

    # write results to console and file
    print_test_results(test_suite_result, output_prompt=args.output_prompt)
    if not args.no_output:
        write_test_results_to_file(
            test_suite_result,
            args.output_file,
            args.output_prompt,
        )


def _import_custom_command_class(class_path: str) -> Command:
    """Dynamically import a command class from a string path."""
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        clz = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f"Failed to import class '{class_path}': {e}")
    if not issubclass(clz, Command):
        structlogger.error(
            "rasa.dialogue_understanding_test.invalid_additional_command",
            event_info="The custom command class must be a subclass of Command.",
            class_path=class_path,
        )
        sys.exit(1)
    return clz


def _extract_additional_command_classes_from_cli_args(
    args: argparse.Namespace,
) -> List[Command]:
    """Extract additional command classes from the CLI arguments."""
    additional_commands = getattr(args, "additional_commands", [])
    if not additional_commands:
        return []
    return [
        _import_custom_command_class(command_module)
        for command_module in additional_commands
    ]


def get_valid_test_suite(
    args: argparse.Namespace, flows: FlowsList, domain: Optional[Domain]
) -> TestSuite:
    """Read the test cases from the given test case path and validate them."""
    path_to_test_cases = getattr(args, "path-to-test-cases", DEFAULT_INPUT_TESTS_PATH)
    remove_default_commands = getattr(args, "remove_default_commands", [])
    custom_command_classes = _extract_additional_command_classes_from_cli_args(args)
    test_suite = read_test_suite(
        path_to_test_cases, flows, custom_command_classes, remove_default_commands
    )
    validate_test_cases(test_suite.test_cases, domain)
    return test_suite


def set_up_available_endpoints(args: argparse.Namespace) -> AvailableEndpoints:
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=Path(args.endpoints)
    ).endpoints

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


def _get_llm_command_generator_config(
    processor: MessageProcessor,
) -> Optional[Dict[str, Any]]:
    train_schema = processor.model_metadata.train_schema

    for node_name, node in train_schema.nodes.items():
        if node.matches_type(LLMBasedCommandGenerator, include_subtypes=True):
            # Configurations can reference model groups defined in the endpoints.yml
            resolved_llm_config = resolve_model_client_config(
                node.config.get(LLM_CONFIG_KEY, {}), node_name
            )
            llm_command_generator = cast(Type[LLMBasedCommandGenerator], node.uses)
            return combine_custom_and_default_config(
                resolved_llm_config, llm_command_generator.get_default_llm_config()
            )

    return None
