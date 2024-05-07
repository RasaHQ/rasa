import os
import argparse
import asyncio
import uuid
import structlog
from typing import List, Any, Dict, Union

from rasa.cli.e2e_test import (
    read_test_cases,
    split_into_passed_failed,
    print_test_result,
)

from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.e2e_test_case import TestCase, Fixture

from ruamel.yaml import YAML

yaml = YAML()
yaml.representer.ignore_aliases = lambda *data: True

structlogger = structlog.get_logger()


def persist_tests(
    test_cases: List[Any],
    path: str,
) -> None:
    """Saves e2e tests.

    Args:
        test_cases: list of tests
        path: path to where to save the e2e tests
    """
    if not test_cases:
        # Handle the case where tests list is empty
        structlogger.info(
            "e2e.tests.command_annotation.persist_tests.failed.empty.list.provided",
        )
        return

    os.makedirs(path, exist_ok=True)

    for test_case in test_cases:
        test_case_name = test_case["test_cases"][0]["test_case"]
        file_path = f"{path}/{test_case_name.replace(' ', '_')}.yml"
        structlogger.info(
            "e2e.tests.command_annotation.persist_tests",
            test_name=test_case_name,
            path=file_path,
        )
        with open(file_path, "w") as f:
            yaml.dump(test_case, f)


def convert_commands(commands: List[Dict[str, Any]]) -> List[Union[str, Dict]]:
    """Processes a list of command dictionaries into a structured list of
    commands suitable for end-to-end tests,
    iterating over the list only once for efficiency.

    Args:
        commands (List[Dict[str, Any]]): List of command dictionaries.

    Returns:
        List[Union[str, Dict]]: A structured and ordered list of commands.
    """
    if not commands:
        return ["no_command"]

    categorized_commands = {
        "start_flow": [],
        "set_slot": [],
        "correct_slot": [],
        "cancel_flow": [],
        "clarify": [],
        "chitchat": [],
        "human_handoff": [],
        "knowledge": [],
        "skip_question": [],
        "error": [],
    }

    # Single pass over the commands list to categorize them
    for command in commands:
        command_type = command["command"]
        if command_type == "start flow":
            categorized_commands["start_flow"].append({"start_flow": command["flow"]})
        elif command_type == "set slot":
            categorized_commands["set_slot"].append({command["name"]: command["value"]})
        elif command_type == "correct slot":
            categorized_commands["correct_slot"].append(
                {command["name"]: command["value"]}
            )
        elif command_type == "cancel flow":
            categorized_commands["cancel_flow"].append("cancel_flow")
        elif command_type == "clarify":
            categorized_commands["clarify"].append({"clarify": command["options"]})
        elif command_type == "chitchat":
            categorized_commands["chitchat"].append("chitchat")
        elif command_type == "human handoff":
            categorized_commands["human_handoff"].append("human_handoff")
        elif command_type == "knowledge":
            categorized_commands["knowledge"].append("knowledge")
        elif command_type == "skip question":
            categorized_commands["skip_question"].append("skip question")
        elif command_type == "error":
            categorized_commands["error"].append("error")

    # Compiling the output in the desired structured format
    commands_output = []
    for key in [
        "start_flow",
        "set_slot",
        "correct_slot",
        "cancel_flow",
        "clarify",
        "chitchat",
        "human_handoff",
        "knowledge",
        "skip_question",
        "error",
    ]:
        if categorized_commands[key]:
            if key in ["set_slot", "correct_slot"]:
                commands_output.append({key: categorized_commands[key]})
            else:
                commands_output.extend(categorized_commands[key])

    return commands_output


async def annotate_test_with_commands(
    test: TestCase, fixtures: List[Fixture], test_runner: E2ETestRunner, sender_id: str
) -> Dict[str, Any]:
    command_annotated_steps = []

    for step in test.steps:
        if step.actor != "user":
            command_annotated_steps.append(step.to_dict())
            continue

        # get responce
        try:
            await test_runner.agent.handle_message(
                UserMessage(
                    step.text,
                    CollectingOutputChannel(),
                    sender_id,
                )
            )
        except asyncio.CancelledError:
            structlogger.error(
                "e2e.tests.command_annotation.handle_message.cancelled",
                error=f"Message handling timed out for user message '{step.text}'.",
                exc_info=True,
            )
        except Exception:
            structlogger.exception(
                "e2e.tests.command_annotation.handle_message.failed",
                error=f"An exception occurred handling user message '{step.text}'.",
            )
        command_annotated_steps.append({"user": step.text})

        # get commands
        tracker = await test_runner.agent.tracker_store.retrieve(sender_id)
        commands = tracker.latest_message.parse_data["commands"]  # type: ignore
        converted_commands = convert_commands(commands)

        command_annotated_steps.append({"commands": converted_commands})

    relevant_fixtures = test_runner.filter_fixtures_for_test_case(test, fixtures)
    command_annotated_test: Dict[str, Any] = {
        "fixtures": [fixture.as_dict() for fixture in relevant_fixtures],
        "test_cases": [{"test_case": test.name, "steps": command_annotated_steps}],
    }

    return command_annotated_test


async def command_annotate_tests(
    tests: List[TestCase],
    input_fixtures: List[Fixture],
    test_runner: E2ETestRunner,
) -> List[Dict[str, Any]]:
    command_annotated_tests = []
    for test_case in tests:
        command_annotated_test = await annotate_test_with_commands(
            test_case, input_fixtures, test_runner, uuid.uuid4().hex
        )
        command_annotated_tests.append(command_annotated_test)
    return command_annotated_tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotating e2e tests with commands.")
    parser.add_argument(
        "--domain_path",
        type=str,
        required=False,
        default="domain",
        help="Path to the domain.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="models",
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        required=False,
        default="endpoints.yml",
        help="Path to the endpoints file to use.",
    )
    parser.add_argument(
        "--tests_path",
        type=str,
        required=False,
        default="tests",
        help="Path to the e2e tests directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="e2e_tests_annotated",
        help="Path to save command-annotated test cases.",
    )

    # set-up
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    input_tests, input_fixtures = read_test_cases(args.tests_path)
    test_runner = E2ETestRunner(
        model_path=args.model_path,
        endpoints=AvailableEndpoints.read_endpoints(args.endpoints),
    )

    # filter passing tests
    results = asyncio.run(test_runner.run_tests(input_tests, input_fixtures))
    passed, failed = split_into_passed_failed(results)
    print_test_result(passed, failed)

    # annotate passing
    command_annotated_tests = asyncio.run(
        command_annotate_tests(
            passed,
            input_fixtures,
            test_runner,
        )
    )

    # persist
    persist_tests(
        command_annotated_tests,
        args.output_path,
    )
    structlogger.info(
        "e2e.tests.command_annotation",
        number_of_command_annotated_tests=len(command_annotated_tests),
    )


if __name__ == "__main__":
    main()
