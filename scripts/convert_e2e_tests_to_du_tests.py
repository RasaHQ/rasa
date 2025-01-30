import argparse
import asyncio
import logging
import os
from typing import List, Optional

import structlog
from mypy.binder import defaultdict

from rasa.cli.arguments.default_arguments import (
    add_endpoint_param,
    add_model_param,
    add_remote_storage_param,
)
from rasa.cli.llm_fine_tuning import DEFAULT_INPUT_E2E_TEST_PATH, set_up_e2e_test_runner
from rasa.core.agent import Agent
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.dialogue_understanding_test.constants import (
    ACTOR_BOT,
    ACTOR_USER,
    PLACEHOLDER_GENERATED_ANSWER_TEMPLATE,
)
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.e2e_test.e2e_test_case import (
    ActualStepOutput,
    Fixture,
    Metadata,
    TestCase,
    TestStep,
    TestSuite,
)
from rasa.e2e_test.e2e_test_runner import TEST_TURNS_TYPE
from rasa.e2e_test.utils.io import read_test_cases
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.constants import USER
from rasa.shared.nlu.constants import PREDICTED_COMMANDS
from rasa.shared.utils.io import create_directory
from rasa.utils.io import write_yaml

structlogger = structlog.get_logger()

READY_FOLDER = "ready"
TO_REVIEW_FOLDER = "to_review"

ELIGIBLE_UTTER_SOURCE_METADATA = [
    "EnterpriseSearchPolicy",
    "ContextualResponseRephraser",
    "IntentlessPolicy",
]


def _coexistence_used(agent: Agent) -> bool:
    # check if coexistence is used by looking at the routing slot
    return ROUTE_TO_CALM_SLOT in [slot.name for slot in agent.domain.slots]


def convert_e2e_tests_to_du_tests(args: argparse.Namespace) -> None:
    e2e_test_runner = set_up_e2e_test_runner(args)
    # disable NLG
    e2e_test_runner.agent.endpoints.nlg = None

    # read e2e test cases
    path_to_test_cases = getattr(args, "path_to_e2e_tests", DEFAULT_INPUT_E2E_TEST_PATH)
    e2e_test_suite = read_test_cases(path_to_test_cases)

    structlogger.info(
        "convert_e2e_tests_to_du_tests.started",
    )

    if _coexistence_used(e2e_test_runner.agent):
        structlogger.warning(
            "convert_e2e_tests_to_du_tests.coexistence_used",
            event_info=f"You are utilizing coexistence. Dialogue understanding tests "
            f"are applicable only for CALM assistants. Test cases that "
            f"utilize the NLU-based system will be skipped. Please ensure "
            f"to review the test cases in the '{TO_REVIEW_FOLDER}' folder, "
            f"as some may involve the NLU-based system and lack command "
            f"annotations, which are required for Dialogue understanding "
            f"tests.",
        )

    # run e2e tests and convert test cases into dialogue understanding test cases
    with set_record_commands_and_prompts():
        ready_du_test_cases, to_review_du_test_cases = asyncio.run(
            e2e_test_runner.run_tests_to_convert_tests_to_du_tests(
                e2e_test_suite.test_cases,
                e2e_test_suite.fixtures,
                e2e_test_suite.metadata,
                convert_test_case,
            )
        )

    # write dialogue understanding test cases to file
    _write_du_test_cases(
        args.output_folder, e2e_test_suite, ready_du_test_cases, to_review_du_test_cases
    )

    structlogger.info(
        "convert_e2e_tests_to_du_tests.finished",
        output_folder=args.output_folder,
        original_e2e_test_cases=len(e2e_test_suite.test_cases),
        du_test_cases_to_review=len(to_review_du_test_cases),
        du_test_cases_ready=len(ready_du_test_cases),
    )


def _write_du_test_cases(
    output_folder: str,
    e2e_test_suite: TestSuite,
    ready_du_test_cases: List[DialogueUnderstandingTestCase],
    to_review_du_test_cases: List[DialogueUnderstandingTestCase],
) -> None:
    _prepare_output_directory(output_folder)
    _write_du_test_cases_to_file(
        f"{output_folder}/{READY_FOLDER}", e2e_test_suite, ready_du_test_cases
    )
    _write_du_test_cases_to_file(
        f"{output_folder}/{TO_REVIEW_FOLDER}", e2e_test_suite, to_review_du_test_cases
    )


def _prepare_output_directory(output_folder: str) -> None:
    create_directory(output_folder)
    create_directory(f"{output_folder}/{READY_FOLDER}")
    create_directory(f"{output_folder}/{TO_REVIEW_FOLDER}")


def _write_du_test_cases_to_file(
    output_folder: str,
    e2e_test_suite: TestSuite,
    du_test_cases: List[DialogueUnderstandingTestCase],
):
    # group test cases by file name
    file_to_du_test_cases = defaultdict(list)
    for test_case in du_test_cases:
        file_to_du_test_cases[test_case.file].append(test_case)

    # create test suites for test cases that should end up in one file
    for file, test_cases in file_to_du_test_cases.items():
        # filter fixtures and metadata for the test cases
        fixtures = _filter_fixtures(e2e_test_suite.fixtures, test_cases)
        metadata = _filter_metadata(e2e_test_suite.metadata, test_cases)

        test_suite = TestSuite(
            test_cases, fixtures, metadata, e2e_test_suite.stub_custom_actions
        )
        data = test_suite.as_dict()
        # remove empty fixtures, metadata and stub custom actions
        data = {k: v for k, v in data.items() if v}

        write_yaml(
            data,
            f"{output_folder}/{os.path.basename(file)}",
        )


def _filter_fixtures(
    fixtures: List[Fixture], test_cases: List[DialogueUnderstandingTestCase]
) -> List[Fixture]:
    if not fixtures:
        return []

    filtered_fixtures = []
    for test_case in test_cases:
        if not test_case.fixture_names:
            continue
        for fixture_name in test_case.fixture_names:
            filtered_fixtures.extend(
                [fixture for fixture in fixtures if fixture.name == fixture_name]
            )

    return filtered_fixtures


def _filter_metadata(
    metadata: List[Metadata], test_cases: List[DialogueUnderstandingTestCase]
) -> List[Metadata]:
    if not metadata:
        return []

    filtered_metadata = []
    for test_case in test_cases:
        if not test_case.metadata_name:
            continue
        filtered_metadata.extend(
            [meta for meta in metadata if meta.name == test_case.metadata_name]
        )

    return filtered_metadata


def convert_test_case(
    test_turns: TEST_TURNS_TYPE,
    e2e_test_case: TestCase,
    assertions_used: bool,
    test_passing: bool,
) -> Optional[DialogueUnderstandingTestCase]:
    structlogger.debug(
        "convert_e2e_tests_to_du_tests.convert_test_case",
        file=e2e_test_case.file,
        test_case=e2e_test_case.name,
    )

    if assertions_used:
        steps = _convert_test_case_with_assertions(e2e_test_case, test_turns)
    else:
        steps = _convert_standard_test_case(e2e_test_case, test_turns)

    if test_passing:
        # if the test is passing, all user steps should have commands
        # in case of a failing test, not all steps of the e2e tests might
        # have run, so not all commands could be recorded
        for step in steps:
            if step.actor == USER and not step.commands:
                structlogger.warning(
                    "convert_e2e_tests_to_du_tests.skip_test_case",
                    test_case=e2e_test_case.name,
                    file=e2e_test_case.file,
                    user_message=step.text,
                    reason="missing commands for user message",
                )
                return None

    return DialogueUnderstandingTestCase(
        name=e2e_test_case.name,
        steps=steps,
        file=e2e_test_case.file,
        line=e2e_test_case.line,
        fixture_names=e2e_test_case.fixture_names,
        metadata_name=e2e_test_case.metadata_name,
    )


def _convert_standard_test_case(
    e2e_test_case: TestCase, test_turns: TEST_TURNS_TYPE
) -> List[DialogueUnderstandingTestStep]:
    steps = []

    for i, original_step in enumerate(e2e_test_case.steps):
        # the e2e test case stops after the first failing step, i.e. if the test is
        # failing we don't have test turns for all steps.
        # in case no test turn is available for a particular step, we cannot extract
        # any commands and simple convert the step with the given text/template
        if i < len(test_turns) - 1:
            if original_step.actor == ACTOR_USER:
                steps.append(
                    _convert_to_dialogue_understanding_step(
                        original_step, test_turns[i], e2e_test_case.name
                    )
                )
                # as not all bot steps need to be present in the e2e test,
                # generate the bot steps from the bot uttered events of the test turn
                steps.extend(
                    _convert_to_bot_test_steps(
                        test_turns[i],
                    )
                )
        else:
            if original_step.actor == ACTOR_USER:
                steps.append(
                    DialogueUnderstandingTestStep(
                        actor=original_step.actor,
                        text=original_step.text,
                        metadata_name=original_step.metadata_name,
                    )
                )
            elif original_step.actor == ACTOR_BOT and (
                original_step.template or original_step.text
            ):
                steps.append(
                    DialogueUnderstandingTestStep(
                        actor=ACTOR_BOT,
                        text=original_step.text,
                        template=original_step.template,
                    )
                )

    return steps


def _convert_test_case_with_assertions(
    e2e_test_case: TestCase, test_turns: TEST_TURNS_TYPE
) -> List[DialogueUnderstandingTestStep]:
    steps = []

    for i, original_step in enumerate(e2e_test_case.steps):
        steps.append(
            _convert_to_dialogue_understanding_step(
                original_step, test_turns[i], e2e_test_case.name
            )
        )
        # we only have user steps, extract the bot response from the bot uttered
        # events of the test turn
        steps.extend(
            _convert_to_bot_test_steps(
                test_turns[i],
            )
        )

    return steps


def _convert_to_bot_test_steps(
    current_turn: ActualStepOutput,
) -> List[DialogueUnderstandingTestStep]:
    steps = []

    for bot_event in current_turn.bot_uttered_events:
        template = None
        if "utter_action" in bot_event.metadata:
            template = bot_event.metadata["utter_action"]
        elif (
            "utter_source" in bot_event.metadata
            and bot_event.metadata["utter_source"] in ELIGIBLE_UTTER_SOURCE_METADATA
        ):
            template = PLACEHOLDER_GENERATED_ANSWER_TEMPLATE

        steps.append(
            DialogueUnderstandingTestStep(
                actor=ACTOR_BOT,
                text=bot_event.text,
                template=template,
            )
        )

    return steps


def _convert_to_dialogue_understanding_step(
    current_step: TestStep,
    current_turn: ActualStepOutput,
    test_case_name: str,
) -> DialogueUnderstandingTestStep:
    # default dialogue understanding test step without commands
    dialogue_understanding_test_step = DialogueUnderstandingTestStep(
        actor=current_step.actor,
        text=current_step.text,
        template=current_step.template,
        line=current_step.line,
        metadata_name=current_step.metadata_name,
    )

    if not current_step.text == current_turn.text or not isinstance(
        current_turn, ActualStepOutput
    ):
        # There should be a one to one mapping between test steps (steps read from file)
        # and test turns (test result of e2e test). Verify that the current step is
        # aligned with the current turn.
        structlogger.debug(
            "convert_e2e_tests_to_du_tests.skip_user_message",
            test_case=test_case_name,
            user_message=current_step.text,
        )
        return dialogue_understanding_test_step

    commands = _extract_commands(current_turn)

    if not commands:
        structlogger.debug(
            "convert_e2e_tests_to_du_tests.no_commands_for_user_message",
            test_case=test_case_name,
            user_message=current_step.text,
        )
        return dialogue_understanding_test_step

    dialogue_understanding_test_step.commands = commands

    return dialogue_understanding_test_step


def _extract_commands(
    turn: ActualStepOutput,
) -> Optional[List[Command]]:
    # There should be exactly one 'UserUttered' event
    if not turn.user_uttered_events or len(turn.user_uttered_events) != 1:
        return None

    # Check if 'parse_data' contains the commands
    if (
        not turn.user_uttered_events[0].parse_data
        or PREDICTED_COMMANDS not in turn.user_uttered_events[0].parse_data
    ):
        return None

    extracted_commands = turn.user_uttered_events[0].parse_data[PREDICTED_COMMANDS]

    # convert the extracted commands to Command objects
    commands = set(
        Command.command_from_json(command_data)
        for command_list in extracted_commands.values()
        for command_data in command_list
    )

    # make sure that the commands are unique
    return list(commands)


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Convert e2e tests to DU tests.")

    # script specific arguments
    parser.add_argument(
        "path_to_e2e_tests",
        type=str,
        help="Path to the e2e test cases. Can be a single file or a folder.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="dialogue_understanding_tests",
        help="Path to the output folder to write the new test cases to.",
    )

    # default arguments
    add_model_param(parser, add_positional_arg=False)
    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )
    add_remote_storage_param(parser)

    return parser.parse_args()


if __name__ == "__main__":
    # Configure standard logging to only show INFO and above
    logging.basicConfig(level=logging.INFO)
    # Configure structlog to only show INFO and above
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    args = _parse_arguments()
    convert_e2e_tests_to_du_tests(args)
