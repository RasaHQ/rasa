import asyncio
import copy
import datetime
import difflib
from asyncio import CancelledError
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Text, Tuple, Union
from urllib.parse import urlparse

import requests
import structlog
from tqdm import tqdm

import rasa.shared.utils.io
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.constants import ACTIVE_FLOW_METADATA_KEY, STEP_ID_METADATA_KEY
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.constants import TEST_CASE_NAME, TEST_FILE_NAME
from rasa.e2e_test.e2e_config import create_llm_judge_config
from rasa.e2e_test.e2e_test_case import (
    KEY_STUB_CUSTOM_ACTIONS,
    ActualStepOutput,
    Fixture,
    Metadata,
    TestCase,
    TestStep,
)
from rasa.e2e_test.e2e_test_result import (
    NO_RESPONSE,
    NO_SLOT,
    TestFailure,
    TestResult,
)
from rasa.llm_fine_tuning.conversations import Conversation
from rasa.nlu.persistor import StorageType
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    FlowCompleted,
    FlowStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.flows.flow_path import FlowPath, PathNode
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import COMMANDS
from rasa.telemetry import track_e2e_test_run
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger()

TEST_TURNS_TYPE = Dict[int, Union[TestStep, ActualStepOutput]]


class E2ETestRunner:
    def __init__(
        self,
        model_path: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[StorageType] = None,
        endpoints: Optional[AvailableEndpoints] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the E2E test suite runner.

        Args:
            model_path: Path to the model.
            model_server: Model server configuration.
            remote_storage: Remote storage to use for model retrieval.
            endpoints: Endpoints configuration.
            **kwargs: Additional arguments
        """
        import rasa.core.agent

        structlogger.info(
            "e2e_test_runner.init",
            event_info="Started running end-to-end testing.",
        )

        test_case_path = kwargs.get("test_case_path")
        self.llm_judge_config = create_llm_judge_config(test_case_path)

        are_custom_actions_stubbed = (
            endpoints
            and endpoints.action
            and endpoints.action.kwargs.get(KEY_STUB_CUSTOM_ACTIONS)
        )
        if endpoints and not are_custom_actions_stubbed:
            self._action_server_is_reachable(endpoints)

        self.agent = asyncio.run(
            rasa.core.agent.load_agent(
                model_path=model_path,
                model_server=model_server,
                remote_storage=remote_storage,
                endpoints=endpoints,
            )
        )

        if not self.agent.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. "
                "Please check that the agent was able to "
                "load the trained model."
            )

    async def run_prediction_loop(
        self,
        collector: CollectingOutputChannel,
        steps: List[TestStep],
        sender_id: Text,
        test_case_metadata: Optional[Metadata] = None,
        input_metadata: Optional[List[Metadata]] = None,
    ) -> TEST_TURNS_TYPE:
        """Runs dialogue prediction.

        Args:
            collector: Output channel.
            steps: List of steps to run.
            sender_id: The test case name with added timestamp suffix.
            test_case_metadata: Metadata of test case.
            input_metadata: List of metadata.

        Returns:
            Test turns: {turn_sequence (int) : TestStep or ActualStepOutput}.
        """
        turns: TEST_TURNS_TYPE = {}
        event_cursor = 0

        if not self.agent.processor:
            return turns

        tracker = await self.agent.processor.fetch_tracker_with_initial_session(
            sender_id
        )
        # turn -1 i used to contain events that happen during
        # the start of the session and before the first user message
        # TestStep is a placeholder just for the sake of having a turn
        # to specify the actor
        turns[-1], event_cursor = self.get_actual_step_output(
            tracker,
            TestStep(
                actor="bot",
                text=None,
            ),
            event_cursor,
        )

        for position, step in enumerate(steps):
            if step.actor != "user":
                turns[position] = step
                continue
            elif not step.text:
                rasa.shared.utils.io.raise_warning(
                    f"The test case '{sender_id}' contains a `user` step in line "
                    f"{position + 1} without a text value. "
                    f"Skipping this step and proceeding to the next user step.",
                    UserWarning,
                )
                continue

            metadata = test_case_metadata.metadata if test_case_metadata else {}

            if input_metadata:
                step_metadata = self.filter_metadata_for_input(
                    step.metadata_name, input_metadata
                )
                step_metadata_dict = step_metadata.metadata if step_metadata else {}
                metadata = self.merge_metadata(
                    sender_id, step.text, metadata, step_metadata_dict
                )

            try:
                await self.agent.handle_message(
                    UserMessage(
                        step.text,
                        collector,
                        sender_id,
                        metadata=metadata,
                    )
                )
            except CancelledError:
                structlogger.error(
                    "e2e_test_runner.run_prediction_loop",
                    error=f"Message handling timed out for user message '{step.text}'.",
                    exc_info=True,
                )
            except Exception:
                structlogger.error(
                    "e2e_test_runner.run_prediction_loop",
                    error=f"An exception occurred while handling "
                    f"user message '{step.text}'.",
                )
            tracker = await self.agent.tracker_store.retrieve(sender_id)  # type: ignore[assignment]
            turns[position], event_cursor = self.get_actual_step_output(
                tracker, step, event_cursor
            )
        return turns

    @staticmethod
    def merge_metadata(
        sender_id: Text,
        step_text: Text,
        test_case_metadata: Dict[Text, Text],
        step_metadata: Dict[Text, Text],
    ) -> Dict[Text, Text]:
        """Merges the test case and user step metadata.

        Args:
            sender_id: The test case name with added timestamp suffix.
            step_text: The user step text.
            test_case_metadata: The test case metadata dict.
            step_metadata: The user step metadata dict.

        Returns:
            A dictionary with the merged metadata.
        """
        if not test_case_metadata:
            return step_metadata
        if not step_metadata:
            return test_case_metadata

        keys_to_overwrite = []

        for key in step_metadata.keys():
            if key in test_case_metadata.keys():
                keys_to_overwrite.append(key)

        if keys_to_overwrite:
            test_case_name = sender_id.rsplit("_", 1)[0]
            structlogger.warning(
                "e2e_test_runner.merge_metadata",
                message=f"Metadata {keys_to_overwrite} exist in both the test case "
                f"'{test_case_name}' and the user step '{step_text}'. "
                "The user step metadata takes precedence and will "
                "override the test case metadata.",
            )

        merged_metadata = copy.deepcopy(test_case_metadata)
        merged_metadata.update(step_metadata)

        return merged_metadata

    @staticmethod
    def get_actual_step_output(
        tracker: DialogueStateTracker,
        test_step: TestStep,
        event_cursor: int,
    ) -> Tuple[ActualStepOutput, int]:
        """Returns the events that are generated from the current step.

        Args:
            tracker: The tracker for the current test case.
            test_step: The test step.
            event_cursor: The event cursor where the previous step left off.

        Returns:
        The events generated from the current step and the updated event
        cursor position.
        """
        session_events = list(tracker.events)[event_cursor:]
        if session_events:
            event_cursor += len(session_events)
            return (
                ActualStepOutput.from_test_step(
                    test_step,
                    [
                        event
                        for event in session_events
                        if isinstance(event, (BotUttered, UserUttered, SlotSet))
                    ],
                ),
                event_cursor,
            )
        else:
            structlogger.warning(
                "e2e_test_runner.get_actual_step_output",
                message=f"No events found for '{tracker.sender_id}' after processing "
                f"test step '{test_step.text}'.",
            )
            # if there are no events, we still want to return an
            # ActualStepOutput object with the test step as the
            # response to be able to generate a diff
            return (
                ActualStepOutput.from_test_step(
                    test_step,
                    [
                        UserUttered(text=test_step.text),
                        BotUttered(text=NO_RESPONSE),
                    ],
                ),
                event_cursor,
            )

    @classmethod
    def generate_test_result(
        cls,
        test_turns: TEST_TURNS_TYPE,
        test_case: TestCase,
    ) -> TestResult:
        """Generates test result.

        Args:
            test_turns: the turns that happened when running the test case or test step.
            test_case: the `TestCase` instance.

        Returns:
        Test result.
        """
        difference = []
        error_line = None
        test_failures = cls.find_test_failures(test_turns, test_case)
        if test_failures:
            first_failure = test_failures[0][0]
            difference = cls.human_readable_diff(test_turns, test_failures)
            error_line = first_failure.error_line if first_failure else None

        return TestResult(
            pass_status=len(test_failures) == 0,
            test_case=test_case,
            difference=difference,
            error_line=error_line,
        )

    def _get_additional_splitting_conditions(
        self,
        step: TestStep,
        input_metadata: List[Metadata],
        tracker: DialogueStateTracker,
        test_case: TestCase,
    ) -> Dict[str, Any]:
        """Returns additional splitting conditions for the user message."""
        additional_splitting_conditions: Dict[str, Any] = {"text": step.text}

        if not step.metadata_name:
            return additional_splitting_conditions

        step_metadata = self.filter_metadata_for_input(
            step.metadata_name, input_metadata
        )
        step_metadata_dict = step_metadata.metadata if step_metadata else {}

        test_case_metadata = self.filter_metadata_for_input(
            test_case.metadata_name, input_metadata
        )
        test_case_metadata_as_dict = (
            test_case_metadata.metadata if test_case_metadata else {}
        )

        metadata: Dict[str, Any] = self.merge_metadata(
            tracker.sender_id,
            step.text,
            test_case_metadata_as_dict,
            step_metadata_dict,
        )
        metadata["model_id"] = tracker.model_id
        metadata["assistant_id"] = tracker.assistant_id

        additional_splitting_conditions["metadata"] = metadata

        return additional_splitting_conditions

    @staticmethod
    def _get_current_user_turn_and_prior_events(
        tracker: DialogueStateTracker,
        additional_splitting_conditions: Dict[str, Any],
        step: TestStep,
    ) -> Tuple[List[Event], List[Event]]:
        """Returns the current user turn and prior events."""
        actual_events = tracker.events

        # this returns 2 lists, the first list contains the events until the user
        # message and the second list contains the events after the
        # user message, including the user message
        step_events = rasa.shared.core.events.split_events(
            actual_events,
            UserUttered,
            additional_splitting_conditions=additional_splitting_conditions,
            include_splitting_event=True,
        )

        if len(step_events) < 2:
            structlogger.error(
                "e2e_test_runner.run_assertions.user_message_not_found",
                message=f"User message '{step.text}' was not found in "
                f"the actual events. The user message "
                f"properties which were searched: "
                f"{additional_splitting_conditions}",
            )
            return [], []

        post_step_events = step_events[1]
        prior_events = step_events[0]

        # subset of events until the next user message
        turn_events = []
        for event in post_step_events:
            # we reached the next user message
            if isinstance(event, UserUttered) and step.text != event.text:
                break

            turn_events.append(event)

        return turn_events, prior_events

    @staticmethod
    def _slice_turn_events(
        step: TestStep,
        matching_event: Event,
        turn_events: List[Event],
        prior_events: List[Event],
    ) -> Tuple[List[Event], List[Event]]:
        """Slices the turn events when assertion order is enabled."""
        if not step.assertion_order_enabled:
            return turn_events, prior_events

        if not matching_event:
            return turn_events, prior_events

        matching_event_index = turn_events.index(matching_event)
        if matching_event_index + 1 < len(turn_events):
            prior_events += turn_events[: matching_event_index + 1]
            turn_events = turn_events[matching_event_index + 1 :]

        return turn_events, prior_events

    async def run_assertions(
        self,
        sender_id: str,
        test_case: TestCase,
        input_metadata: Optional[List[Metadata]],
    ) -> TestResult:
        """Runs the assertions defined in the test case."""
        tracker = await self.agent.processor.get_tracker(sender_id)  # type: ignore[union-attr]

        assertion_failure = None
        assertion_failure_found = False
        input_metadata = input_metadata if input_metadata else []

        for step in test_case.steps:
            if not step.assertions:
                structlogger.debug(
                    "e2e_test_runner.run_assertions.no_assertions.skipping_step",
                    step=step,
                )
                continue

            additional_splitting_conditions = self._get_additional_splitting_conditions(
                step, input_metadata, tracker, test_case
            )

            turn_events, prior_events = self._get_current_user_turn_and_prior_events(
                tracker, additional_splitting_conditions, step
            )

            if not turn_events:
                return TestResult(
                    pass_status=False,
                    test_case=test_case,
                    difference=[],
                    error_line=step.line,
                    assertion_failure=None,
                )

            for assertion in step.assertions:
                structlogger.debug(
                    "e2e_test_runner.run_assertions.running_assertion",
                    test_case_name=test_case.name,
                    step_text=step.text,
                    assertion_type=assertion.type(),
                )

                assertion_order_error_msg = ""

                if step.assertion_order_enabled:
                    assertion_order_error_msg = (
                        " You have enabled assertion order, "
                        "you should check the order in which the "
                        "assertions are listed for this user step."
                    )

                assertion_failure, matching_event = assertion.run(
                    turn_events,
                    prior_events=prior_events,
                    assertion_order_error_message=assertion_order_error_msg,
                    llm_judge_config=self.llm_judge_config,
                    step_text=step.text,
                )

                if assertion_failure:
                    assertion_failure_found = True
                    structlogger.debug(
                        "e2e_test_runner.run_assertions.assertion_failure_found",
                        test_case_name=test_case.name,
                        error_line=assertion_failure.error_line,
                    )
                    break

                turn_events, prior_events = self._slice_turn_events(
                    step, matching_event, turn_events, copy.deepcopy(prior_events)
                )

            if assertion_failure_found:
                # don't continue with the next steps if an assertion failed
                break

        return TestResult(
            pass_status=not assertion_failure,
            test_case=test_case,
            difference=[],
            error_line=assertion_failure.error_line if assertion_failure else None,
            assertion_failure=assertion_failure,
        )

    @classmethod
    def _resolve_successful_bot_utter(
        cls,
        expected_result: TestStep,
        test_response: ActualStepOutput,
    ) -> str:
        """Returns the diff text for a successful bot utter test step."""
        for event in test_response.bot_uttered_events:
            if expected_result.matches_event(event):
                # remove the event that is already matched so
                # that we dont compare against it again
                test_response.remove_bot_uttered_event(event)
                if expected_result.text is not None:
                    text = f"{expected_result.actor}: {expected_result.text}"
                elif expected_result.template is not None:
                    text = f"{expected_result.actor}: {expected_result.template}"
                break

        return text

    @classmethod
    def _resolve_successful_set_slot(
        cls,
        expected_result: TestStep,
        test_response: ActualStepOutput,
    ) -> str:
        """Returns the diff text for a successful set slot test step."""
        slot_name = expected_result.get_slot_name()
        text = f"slot_was_set: {slot_name}"

        for event in test_response.slot_set_events:
            if expected_result.matches_event(event):
                # remove the event that is already matched so
                # that we dont compare against it again
                test_response.remove_slot_set_event(event)
                if event.value is not None:
                    text += f": {event.value} ({type(event.value).__name__})"
                break

        return text

    @classmethod
    def _resolve_successful_slot_was_not_set(cls, expected_result: TestStep) -> str:
        """Returns the diff text for a successful slot was not set test step."""
        slot_name = expected_result.get_slot_name()
        slot_value = expected_result.get_slot_value()
        text = f"slot_was_not_set: {slot_name}"
        if expected_result.is_slot_instance_dict():
            text += f": {slot_value}"

        return text

    @classmethod
    def _find_first_non_matched_utterance(
        cls,
        test_response: ActualStepOutput,
        bot_utter_test_steps: List[TestStep],
    ) -> Optional[BotUttered]:
        """Finds the first non matched utterance in events (if any)."""
        bot_events = test_response.bot_uttered_events
        matching_events = []
        for event in bot_events:
            for test_step in bot_utter_test_steps:
                if test_step.matches_event(event):
                    matching_events.append(event)
                    break
        not_matching_events = [
            event for event in bot_events if event not in matching_events
        ]
        return not_matching_events[0] if not_matching_events else None

    @classmethod
    def _handle_fail_diff(
        cls,
        failed_step: TestStep,
        test_response: ActualStepOutput,
        bot_utter_test_steps: List[TestStep],
    ) -> Tuple[str, str]:
        """Handles generating the diff text for a failed test step."""
        if failed_step.text is not None:
            diff_test_text = f"{failed_step.actor}: {failed_step.text}"
            diff_actual_text = NO_RESPONSE
            event: Optional[Union[BotUttered, SlotSet]] = (
                cls._find_first_non_matched_utterance(
                    test_response, bot_utter_test_steps
                )
            )
            if event and isinstance(event, BotUttered):
                test_response.remove_bot_uttered_event(event)
                diff_actual_text = f"bot: {event.text}"

        elif failed_step.template is not None:
            diff_test_text = f"{failed_step.actor}: {failed_step.template}"
            diff_actual_text = NO_RESPONSE
            event = cls._find_first_non_matched_utterance(
                test_response, bot_utter_test_steps
            )
            if event:
                test_response.remove_bot_uttered_event(event)
                diff_actual_text = f"bot: {event.metadata.get('utter_action')}"

        elif failed_step.slot_was_set:
            slot_name = failed_step.get_slot_name()

            diff_test_text = f"slot_was_set: {slot_name}"
            if failed_step.is_slot_instance_dict():
                slot_value = failed_step.get_slot_value()
                diff_test_text += f": {slot_value} ({type(slot_value).__name__})"

            diff_actual_text = NO_SLOT
            for event in test_response.slot_set_events:
                if slot_name == event.key:
                    diff_actual_text = (
                        f"slot_was_set: {event.key}: {event.value} "
                        f"({type(event.value).__name__})"
                    )
                    test_response.remove_slot_set_event(event)
                    break

        elif failed_step.slot_was_not_set:
            slot_name = failed_step.get_slot_name()

            diff_test_text = f"slot_was_not_set: {slot_name}"
            if failed_step.is_slot_instance_dict():
                slot_value = failed_step.get_slot_value()
                diff_test_text += f": {slot_value}"

            for event in test_response.slot_set_events:
                if slot_name == event.key:
                    diff_actual_text = (
                        f"slot_was_set: {event.key}: {event.value} "
                        f"({type(event.value).__name__})"
                    )
                    test_response.remove_slot_set_event(event)
                    break

        return diff_test_text, diff_actual_text

    @classmethod
    def _select_bot_utter_turns(
        cls,
        test_turns: TEST_TURNS_TYPE,
        start_index: int,
    ) -> List[TestStep]:
        """Selects the TestSteps from the test turns that match BotUttered events."""
        bot_utter_turns = []
        for index in range(start_index, len(test_turns) - 1):
            test_turn = test_turns[index]
            if isinstance(test_turn, TestStep):
                if test_turn.text is not None or test_turn.template is not None:
                    bot_utter_turns.append(test_turn)
            elif isinstance(test_turn, ActualStepOutput):
                break

        return bot_utter_turns

    @classmethod
    def human_readable_diff(
        cls,
        test_turns: TEST_TURNS_TYPE,
        fail_positions: List[Tuple[TestFailure, int]],
    ) -> List[str]:
        """Returns a human readable diff of the test case and the actual conversation.

        Given an ordered list of test steps and actual conversation events, this
        method will return a human readable diff of the two.
        The diff uses difflib to compare the two lists and will highlight
        differences between the two in a pytest style diff.

        Args:
            test_turns: The transcript of test cases and events.
            fail_positions: The positions of the test failures.

        Returns:
            The human readable diff.
        """
        actual_transcript = []
        expected_transcript = []
        failure_points = {position for _, position in fail_positions}
        # This will only be used when the TestCase is not started
        # with a user step
        latest_response: ActualStepOutput = test_turns[-1]  # type: ignore[assignment]

        for index in range(max(failure_points) + 1):
            test_result = test_turns[index]
            if index in failure_points:
                diff_test_text, diff_actual_text = cls._handle_fail_diff(
                    test_result,  # type: ignore[arg-type]
                    latest_response,
                    cls._select_bot_utter_turns(test_turns, index),
                )  # test_result can only be TestStep in failure_points
                actual_transcript.append(diff_actual_text)
                expected_transcript.append(diff_test_text)
                continue

            if isinstance(test_result, TestStep):
                if test_result.text is not None or test_result.template is not None:
                    diff_test_text = cls._resolve_successful_bot_utter(
                        expected_result=test_result,
                        test_response=latest_response,
                    )

                if test_result.slot_was_set:
                    diff_test_text = cls._resolve_successful_set_slot(
                        expected_result=test_result,
                        test_response=latest_response,
                    )

                if test_result.slot_was_not_set:
                    diff_test_text = cls._resolve_successful_slot_was_not_set(
                        expected_result=test_result,
                    )

            elif isinstance(test_result, ActualStepOutput):
                latest_response = test_result
                event = test_result.get_user_uttered_event()
                if event:
                    diff_test_text = f"{test_result.actor}: {event.text}"
                else:
                    raise RasaException(
                        f"Bot did not catch user "
                        f"event: {test_result.actor}: {test_result.text}."
                    )
            else:
                raise ValueError(f"Unexpected test result type: {type(test_result)}")

            # test passed in these cases so it's the same
            actual_transcript.append(diff_test_text)
            expected_transcript.append(diff_test_text)

        return list(difflib.ndiff(actual_transcript, expected_transcript))

    @classmethod
    def find_test_failures(
        cls,
        test_turns: TEST_TURNS_TYPE,
        test_case: TestCase,
    ) -> List[Tuple[TestFailure, int]]:
        """Finds the test failures in the transcript.

        Args:
            test_turns: The transcript of test cases and events.
            test_case: The test case.

        Returns:
            The test failures or an empty list if there is no test failure.
        """
        # This will only be used when the TestCase is not started
        # with a user step
        latest_response: ActualStepOutput = test_turns[-1]  # type: ignore[assignment]
        failures = []
        position = 0
        match = None
        for position in range(len(test_turns) - 1):
            turn_value = test_turns[position]
            if isinstance(turn_value, ActualStepOutput):
                latest_response = turn_value
            elif isinstance(turn_value, TestStep):
                if turn_value.text is not None or turn_value.template is not None:
                    match = cls._does_match_exist(
                        latest_response.bot_uttered_events, turn_value
                    )
                if turn_value.slot_was_set:
                    match = cls._does_match_exist(
                        latest_response.slot_set_events, turn_value
                    )
                if turn_value.slot_was_not_set:
                    match = not cls._does_match_exist(
                        latest_response.slot_set_events, turn_value
                    )
                if not match:
                    failures.append((TestFailure(test_case, turn_value.line), position))
            else:
                raise ValueError(f"Unexpected turn value type: {type(turn_value)}")

        return failures

    @classmethod
    def _does_match_exist(
        cls,
        actual_events: Optional[List[Union[UserUttered, BotUttered, SlotSet]]],
        expected: TestStep,
    ) -> bool:
        if not actual_events:
            return False

        match = False
        for event in actual_events:
            if expected.matches_event(event):
                return True
        return match

    async def set_up_fixtures(
        self,
        fixtures: List[Fixture],
        sender_id: Text,
    ) -> None:
        """Sets slots in the tracker as defined by the input fixtures.

        Args:
            fixtures: List of `Fixture` objects.
            sender_id: The conversation id.
        """
        if not fixtures:
            return
        if not self.agent.processor:
            return

        tracker = await self.agent.processor.fetch_tracker_with_initial_session(
            sender_id
        )

        for fixture in fixtures:
            for slot_name, slot_value in fixture.slots_set.items():
                tracker.update(SlotSet(slot_name, slot_value))

        await self.agent.tracker_store.save(tracker)

    @staticmethod
    def filter_fixtures_for_test_case(
        test_case: TestCase, fixtures: List[Fixture]
    ) -> List[Fixture]:
        """Filters the input fixtures for the input test case.

        Args:
            test_case: The test case.
            fixtures: The fixtures.

        Returns:
        The filtered fixtures.
        """
        return list(
            filter(
                lambda fixture: test_case.fixture_names
                and fixture.name in test_case.fixture_names,
                fixtures,
            )
        )

    @staticmethod
    def filter_metadata_for_input(
        metadata_name: Optional[Text], test_suite_metadata: List[Metadata]
    ) -> Optional[Metadata]:
        """Filters the test suite metadata for a metadata name.

        Args:
            metadata_name: The test case or user step metadata name.
            test_suite_metadata: The top level list of all metadata definitions.

        Returns:
            The filtered metadata.
        """
        if not metadata_name:
            return None

        filtered_metadata = list(
            filter(
                lambda metadata: metadata_name and metadata.name == metadata_name,
                test_suite_metadata,
            )
        )

        if not filtered_metadata:
            structlogger.warning(
                "e2e_test_runner.filter_metadata_for_input",
                message=f"Metadata '{metadata_name}' is not defined in the input "
                f"metadata.",
            )
            return None

        return filtered_metadata[0]

    async def run_tests(
        self,
        input_test_cases: List[TestCase],
        input_fixtures: List[Fixture],
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> List["TestResult"]:
        """Runs the test cases.

        Args:
            input_test_cases: Input test cases.
            input_fixtures: Input fixtures.
            fail_fast: Whether to fail fast.
            **kwargs: Additional arguments which are passed here.

        Returns:
            List of test results.
        """
        results = []
        input_metadata = kwargs.get("input_metadata", None)

        # telemetry call for tracking test runs
        track_e2e_test_run(input_test_cases, input_fixtures, input_metadata)

        for test_case in input_test_cases:
            test_case_name = test_case.name.replace(" ", "_")
            # Add the name of the file and the current test case name being
            # executed in order to properly retrieve stub custom action
            if self.agent.endpoints and self.agent.endpoints.action:
                self.agent.endpoints.action.kwargs[TEST_FILE_NAME] = Path(
                    test_case.file
                ).name
                self.agent.endpoints.action.kwargs[TEST_CASE_NAME] = test_case_name

            # add timestamp suffix to ensure sender_id is unique
            sender_id = f"{test_case_name}_{datetime.datetime.now()}"
            test_turns = await self._run_test_case(
                sender_id, input_fixtures, input_metadata, test_case
            )

            if not test_case.uses_assertions():
                test_result = self.generate_test_result(test_turns, test_case)
            else:
                test_result = await self.run_assertions(
                    sender_id, test_case, input_metadata
                )

            results.append(test_result)

            coverage = kwargs.get("coverage", False)
            if coverage:
                tracker = await self.agent.tracker_store.retrieve(sender_id)
                if tracker:
                    test_result.tested_paths, test_result.tested_commands = (
                        self._get_tested_flow_paths_and_commands(
                            tracker.events, test_result
                        )
                    )

            if fail_fast and not test_result.pass_status:
                break

        return results

    async def _run_test_case(
        self,
        sender_id: str,
        input_fixtures: List[Fixture],
        input_metadata: Optional[List[Metadata]],
        test_case: TestCase,
    ) -> TEST_TURNS_TYPE:
        collector = CollectingOutputChannel()

        if input_fixtures:
            test_fixtures = self.filter_fixtures_for_test_case(
                test_case, input_fixtures
            )
            await self.set_up_fixtures(test_fixtures, sender_id)

        test_case_metadata = None
        if input_metadata:
            test_case_metadata = self.filter_metadata_for_input(
                test_case.metadata_name, input_metadata
            )

        return await self.run_prediction_loop(
            collector,
            test_case.steps,
            sender_id,
            test_case_metadata,
            input_metadata,
        )

    async def run_tests_for_fine_tuning(
        self,
        input_test_cases: List[TestCase],
        input_fixtures: List[Fixture],
        input_metadata: Optional[List[Metadata]],
    ) -> List[Conversation]:
        """Runs the test cases for fine-tuning.

        Converts passing test cases into conversation objects containing the
        prompts and llm commands per user message.

        Args:
            input_test_cases: Input test cases.
            input_fixtures: Input fixtures.
            input_metadata: Input metadata.

        Returns:
            List of conversations.
        """
        import rasa.llm_fine_tuning.annotation_module

        conversations = []

        for i in tqdm(range(len(input_test_cases))):
            test_case = input_test_cases[i]
            # add timestamp suffix to ensure sender_id is unique
            sender_id = f"{test_case.name}_{datetime.datetime.now()}"
            test_turns = await self._run_test_case(
                sender_id, input_fixtures, input_metadata, test_case
            )

            # check if the e2e test is passing, only convert passing e2e tests into
            # conversations
            if not test_case.uses_assertions():
                test_result = self.generate_test_result(test_turns, test_case)
            else:
                test_result = await self.run_assertions(
                    sender_id, test_case, input_metadata
                )
            if not test_result.pass_status:
                structlogger.warning(
                    "annotation_module.skip_test_case.failing_e2e_test",
                    test_case=test_case.name,
                    file=test_case.file,
                )
                continue

            tracker = await self.agent.tracker_store.retrieve(sender_id)
            conversation = rasa.llm_fine_tuning.annotation_module.generate_conversation(
                test_turns, test_case, tracker, test_case.uses_assertions()
            )

            if conversation:
                conversations.append(conversation)

        return conversations

    @staticmethod
    def _action_server_is_reachable(endpoints: AvailableEndpoints) -> None:
        """Calls the action server health endpoint."""
        if not endpoints.action:
            structlogger.debug(
                "e2e_test_runner._action_server_is_reachable",
                message="No action endpoint configured. Skipping the health check "
                "of the action server.",
            )
            return

        if endpoints.action.actions_module:
            structlogger.debug(
                "e2e_test_runner._action_server_is_reachable",
                message="Rasa server is configured to run custom actions directly. "
                "Skipping the health check of the action server.",
            )
            return

        if not endpoints.action.url:
            structlogger.debug(
                "e2e_test_runner._action_server_is_reachable",
                message="Action endpoint URL is not defined in the endpoint "
                "configuration.",
            )
            return

        structlogger.debug(
            "e2e_test_runner._action_server_is_reachable",
            message="Detected action URL in the endpoint configuration.\n"
            f"Action Server URL: {endpoints.action.url}\n"
            "Sending a health request to the action endpoint.",
        )
        url = urlparse(endpoints.action.url)
        # replace /<path> with just /health
        url = url._replace(path="/health").geturl()  # type: ignore[assignment]
        try:
            response = requests.get(url, timeout=3)
        except requests.exceptions.ConnectionError as error:
            raise RasaException(
                "Action endpoint could not be reached. "
                "Actions server URL is defined in your endpoint configuration as "
                f"'{endpoints.action.url}'.\n"
                "Please make sure your action server is running and properly "
                "configured. Since running e2e tests without a action server may "
                f"lead to unpredictable results.\n{error}"
            )

        if response.status_code != 200:
            raise RasaException(
                "Action endpoint is responding, but health status responded with "
                f"code {response.status_code}. Make sure your action server"
                " is properly configured and that the '/health' endpoint is available."
            )

        structlogger.debug(
            "e2e_test_runner._action_server_is_reachable",
            message="Action endpoint has responded successfully.\n"
            f"Response message: {response.text}\n"
            f"Response status code: {response.status_code}.",
        )

    def _get_tested_flow_paths_and_commands(
        self, events: List[Event], test_result: TestResult
    ) -> Tuple[Optional[List[FlowPath]], Dict[str, Dict[str, int]]]:
        """Extract tested paths and commands from dialog events.

        A flow path consists of bot utterances and custom actions.

        Args:
            events: The list of dialog events.
            test_result: The result of the test incl. the pass status.

        Returns:
            Tuple[flow_paths: Optional[List[FlowPath]], tested_commands:
            Dict[str, Dict[str, int]]], where tested_commands is a
            dictionary like
            {"flow1": {"set slot": 5, "clarify": 1}, "flow2": {"set slot": 3}}
        """
        tested_paths = []
        # we want to create a flow path per flow the e2e test covers
        # as an e2e test can cover multiple flows, we might end up creating
        # multiple flow paths
        _tested_commands: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        flow_paths_stack = []

        for event in events:
            if isinstance(event, FlowStarted) and not event.flow_id.startswith(
                RASA_DEFAULT_FLOW_PATTERN_PREFIX
            ):
                flow_paths_stack.append(FlowPath(event.flow_id))

            elif (
                isinstance(event, FlowCompleted)
                and len(flow_paths_stack) > 0
                and event.flow_id == flow_paths_stack[-1].flow
            ):
                # flow path is completed as the flow ended
                tested_paths.append(flow_paths_stack.pop())

            elif isinstance(event, BotUttered):
                if (
                    flow_paths_stack
                    and STEP_ID_METADATA_KEY in event.metadata
                    and ACTIVE_FLOW_METADATA_KEY in event.metadata
                ):
                    flow_paths_stack[-1].nodes.append(self._create_path_node(event))

            elif isinstance(event, ActionExecuted):
                # we are only interested in custom actions
                if (
                    flow_paths_stack
                    and self.agent.domain
                    and self.agent.domain.is_custom_action(event.action_name)
                ):
                    flow_paths_stack[-1].nodes.append(self._create_path_node(event))

            # Time to gather tested commands
            elif isinstance(event, UserUttered):
                if event.parse_data and COMMANDS in event.parse_data:
                    commands = [
                        command["command"] for command in event.parse_data[COMMANDS]
                    ]
                    current_flow = (
                        flow_paths_stack[-1].flow if flow_paths_stack else "no_flow"
                    )
                    for command in commands:
                        _tested_commands[current_flow][command] += 1

        # It might be that an e2e test stops before a flow was completed.
        # Add the remaining flow paths to the tested paths list.
        while len(flow_paths_stack) > 0:
            tested_paths.append(flow_paths_stack.pop())

        # Convert _tested_commands to normal dicts
        tested_commands = {key: dict(value) for key, value in _tested_commands.items()}  # type: Dict[str, Dict[str, int]]

        return tested_paths, tested_commands

    @staticmethod
    def _create_path_node(event: Event) -> PathNode:
        flow_id = event.metadata[ACTIVE_FLOW_METADATA_KEY]
        step_id = event.metadata[STEP_ID_METADATA_KEY]
        return PathNode(step_id=step_id, flow=flow_id)
