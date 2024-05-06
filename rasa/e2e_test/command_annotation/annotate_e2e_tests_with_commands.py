import os
import argparse
import asyncio
import uuid
import structlog
from typing import List, Any, Dict

from rasa.cli.e2e_test import read_test_cases

from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.e2e_test_case import TestCase

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


async def annotate_test_with_commands(
    test: TestCase, test_runner: E2ETestRunner, sender_id: str
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
        commands = tracker.latest_message.parse_data["commands"] # type: ignore
        commands_output = []

        # if the llm did not predict any commands, we should add no_command
        if not commands:
            commands_output.append("no_command")
        else:
            # commands are in a random order, but we want them in a specific order
            # for the e2e tests
            # start
            start_flow_commands = [
                command for command in commands if command["command"] == "start flow"
            ]
            for start_flow_command in start_flow_commands:
                commands_output.append({"start_flow": start_flow_command["flow"]})

            # set slot
            set_slot_commands = [
                command for command in commands if command["command"] == "set slot"
            ]
            if set_slot_commands:
                slots = []
                for set_slot_command in set_slot_commands:
                    if isinstance(set_slot_command, str):
                        slots.append(set_slot_command)
                    else:
                        slots.append({set_slot_command["name"]: set_slot_command["value"]})
                commands_output.append({"set_slot": slots})

            # correct slot
            correct_slot_commands = [
                command for command in commands if command["command"] == "correct slot"
            ]
            if correct_slot_commands:
                slots = []
                for correct_slot_command in correct_slot_commands:
                    if isinstance(correct_slot_command, str):
                        slots.append(correct_slot_command)
                    else:
                        slots.append({correct_slot_command["name"]: correct_slot_command["value"]})
                commands_output.append({"correct_slot": slots})

            # cancel
            cancel_flow_commands = [
                command for command in commands if command["command"] == "cancel flow"
            ]
            for cancel_flow_command in cancel_flow_commands:
                commands_output.append("cancel_flow")

            # clarify
            clarify_commands = [
                command for command in commands if command["command"] == "clarify"
            ]
            for clarify_command in clarify_commands:
                commands_output.append({"clarify": clarify_command["options"]})

            # chitchat
            chitchat_commands = [
                command for command in commands if command["command"] == "chitchat"
            ]
            for chitchat_command in chitchat_commands:
                commands_output.append("chitchat")

            # human handoff
            human_handoff_commands = [
                command for command in commands if command["command"] == "human handoff"
            ]
            for human_handoff_command in human_handoff_commands:
                commands_output.append("human_handoff")

            # knowledge
            knowledge_commands = [
                command for command in commands if command["command"] == "knowledge"
            ]
            for knowledge_command in knowledge_commands:
                commands_output.append("knowledge")

            # skip question
            skip_question_commands = [
                command for command in commands if command["command"] == "skip question"
            ]
            for skip_question_command in skip_question_commands:
                commands_output.append("skip question")

            # error
            error_commands = [
                command for command in commands if command["command"] == "error"
            ]
            for error_command in error_commands:
                commands_output.append("error")

        command_annotated_steps.append({"commands": commands_output})

    command_annotated_test: Dict[str, Any] = {
        "test_cases": [{"test_case": test.name, "steps": command_annotated_steps}]
    }

    return command_annotated_test


async def command_annotate_tests(
    tests: List[TestCase],
    test_runner: E2ETestRunner,
) -> List[Dict[str, Any]]:
    command_annotated_tests = []
    for test_case in tests:
        command_annotated_test = await annotate_test_with_commands(
            test_case, test_runner, uuid.uuid4().hex
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

    # annotate
    command_annotated_tests = asyncio.run(
        command_annotate_tests(
            input_tests,
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
