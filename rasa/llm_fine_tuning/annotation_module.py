import asyncio
from contextlib import contextmanager
from typing import List, Generator, Optional, Tuple, Union

import structlog

from rasa.dialogue_understanding.commands import Command
from rasa.e2e_test.e2e_test_case import TestSuite, TestCase, ActualStepOutput, TestStep
from rasa.e2e_test.e2e_test_runner import E2ETestRunner, TEST_TURNS_TYPE
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.storage import StorageContext
from rasa.shared.core.constants import USER
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import LLM_PROMPT, LLM_COMMANDS
from rasa.shared.utils.llm import tracker_as_readable_transcript

ANNOTATION_MODULE_STORAGE_LOCATION = "1_command_annotations"

preparing_fine_tuning_data = False

structlogger = structlog.get_logger()


@contextmanager
def set_preparing_fine_tuning_data() -> Generator:
    global preparing_fine_tuning_data
    preparing_fine_tuning_data = True
    try:
        yield
    finally:
        preparing_fine_tuning_data = False


def annotate_e2e_tests(
    e2e_test_runner: E2ETestRunner,
    test_suite: TestSuite,
    storage_context: StorageContext,
) -> List[Conversation]:
    with set_preparing_fine_tuning_data():
        converations = asyncio.run(
            e2e_test_runner.run_tests_for_fine_tuning(
                test_suite.test_cases,
                test_suite.fixtures,
                test_suite.metadata,
            )
        )

    storage_context.write_conversations(
        converations, ANNOTATION_MODULE_STORAGE_LOCATION
    )

    return converations


def _get_previous_actual_step_output(
    test_turns: TEST_TURNS_TYPE, i: int
) -> Optional[ActualStepOutput]:
    while i > 0:
        i = i - 1
        if isinstance(test_turns[i], ActualStepOutput):
            return test_turns[i]  # type:ignore[return-value]
    return None


def generate_conversation(
    test_turns: TEST_TURNS_TYPE,
    test_case: TestCase,
    tracker: DialogueStateTracker,
    assertions_used: bool = False,
) -> Optional[Conversation]:
    """Generates a conversation object in case of e2e test passing.

    Args:
        test_turns: the turns that happened when running the test case or test step.
        test_case: the `TestCase` instance.
        tracker: the dialogue state tracker.
        assertions_used: if True the e2e test format with assertions was used.

    Returns:
        Conversation.
    """
    steps = []

    if assertions_used:
        # we only have user steps, extract the bot response from the bot uttered
        # events of the test turn
        for i, original_step in enumerate(test_case.steps):
            previous_turn = _get_previous_actual_step_output(test_turns, i)
            steps.append(
                _convert_to_conversation_step(
                    original_step, test_turns[i], test_case.name, previous_turn
                )
            )
            steps.extend(_create_bot_test_steps(test_turns[i]))
    else:
        for i, original_step in enumerate(test_case.steps):
            if original_step.actor == USER:
                previous_turn = _get_previous_actual_step_output(test_turns, i)
                steps.append(
                    _convert_to_conversation_step(
                        original_step, test_turns[i], test_case.name, previous_turn
                    )
                )
            else:
                steps.append(original_step)

    # Some messages in an e2e test case could be mapped to commands via
    # 'NLUCommandAdapter', e.g. the message will not be annotated with a prompt and
    # commands pair. Only convert steps that have a prompt and commands present into a
    # ConversationStep.
    # The conversation needs to have at least one 'ConversationStep' to be valid for
    # fine-tuning.
    if not any([isinstance(step, ConversationStep) for step in steps]):
        structlogger.warning(
            "annotation_module.skip_test_case.missing_llm_commands_and_prompts",
            test_case=test_case.name,
            file=test_case.file,
        )
        return None

    transcript = tracker_as_readable_transcript(tracker, max_turns=None)

    return Conversation(test_case.name, test_case, steps, transcript)


def _create_bot_test_steps(current_turn: ActualStepOutput) -> List[TestStep]:
    test_steps = []
    for bot_event in current_turn.bot_uttered_events:
        template = None
        if "utter_action" in bot_event.metadata:
            template = bot_event.metadata["utter_action"]

        test_steps.append(TestStep(actor="bot", text=bot_event.text, template=template))

    return test_steps


def _convert_to_conversation_step(
    current_step: TestStep,
    current_turn: ActualStepOutput,
    test_case_name: str,
    previous_turn: Optional[ActualStepOutput],
) -> Union[TestStep, ConversationStep]:
    if not current_step.text == current_turn.text or not isinstance(
        current_turn, ActualStepOutput
    ):
        # There should be a one to one mapping between test steps (steps read from file)
        # and test turns (test result of e2e test). Verify that the current step is
        # aligned with the current turn.
        structlogger.debug(
            "annotation_module.convert_to_conversation_step.skip_user_message",
            test_case=test_case_name,
            user_message=current_step.text,
        )
        return current_step

    llm_prompt, llm_commands = _extract_llm_prompt_and_commands(current_turn)
    if not llm_commands or not llm_prompt:
        # If no commands or no prompt is present we cannot create a data point
        # for fine-tuning, skipping this step.
        structlogger.debug(
            "annotation_module.convert_to_conversation_step.skip_user_message",
            test_case=test_case_name,
            user_message=current_step.text,
            message="No commands/prompt associated with the message.",
        )
        return current_step

    commands = [Command.command_from_json(data) for data in llm_commands]
    rephrase = _should_be_rephrased(current_turn.text, previous_turn, test_case_name)

    return ConversationStep(current_step, commands, llm_prompt, rephrase=rephrase)


def _should_be_rephrased(
    current_user_message: str,
    previous_turn: Optional[ActualStepOutput],
    test_case_name: str,
) -> bool:
    """Checks if the current user message should be rephrased or not.

    A user message should not be rephrased in case the user message comes from a button
    payload, i.e. the user clicked on a button.

    Args:
        current_user_message: The current user message.
        previous_turn: The previous turn containing the bot uttered event that came
            before.
        test_case_name: The name of the test case.

    Returns:
        True, in case the user message should be rephrased, False otherwise.
    """
    # there is no previous turn, we are at the beginning of the conversation
    if not previous_turn:
        return True

    buttons_present = (
        previous_turn.bot_uttered_events
        and "buttons" in previous_turn.bot_uttered_events[-1].data
        and previous_turn.bot_uttered_events[-1].data["buttons"] is not None
    )

    if not buttons_present:
        return True

    # if the user utterance comes from a button payload we should not rephrase
    # the user utterance in later steps
    button_data = previous_turn.bot_uttered_events[-1].data["buttons"]
    button_payloads = [data["payload"].lower() for data in button_data]
    if current_user_message.lower() in button_payloads:
        structlogger.debug(
            "annotation_module.user_message_should_not_be_rephrased",
            rephrase=False,
            user_message=current_user_message,
            test_case_name=test_case_name,
        )
        return False

    return True


def _extract_llm_prompt_and_commands(
    turn: ActualStepOutput,
) -> Tuple[Optional[str], Optional[str]]:
    # There should be exactly one 'UserUttered' event
    if not turn.user_uttered_events or len(turn.user_uttered_events) != 1:
        return None, None

    # Check if 'parse_data' contains the prompt and the commands
    if (
        not turn.user_uttered_events[0].parse_data
        or LLM_PROMPT not in turn.user_uttered_events[0].parse_data
        or LLM_COMMANDS not in turn.user_uttered_events[0].parse_data
    ):
        return None, None

    return (
        turn.user_uttered_events[0].parse_data[LLM_PROMPT],
        turn.user_uttered_events[0].parse_data[LLM_COMMANDS],
    )
