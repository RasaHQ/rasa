import asyncio
import copy
import datetime
import difflib
import logging
from asyncio import CancelledError
from typing import Any, Dict, List, Optional, Text, Tuple, Union
from urllib.parse import urlparse

import rasa.shared.utils.io
import requests
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

from rasa.e2e_test.e2e_test_case import (
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

from rasa.telemetry import track_e2e_test_run

logger = logging.getLogger(__name__)
TEST_TURNS_TYPE = Dict[int, Union[TestStep, ActualStepOutput]]


class E2ETestRunner:
    def __init__(
        self,
        model_path: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        endpoints: Optional[AvailableEndpoints] = None,
    ) -> None:
        """Initializes the E2E test suite runner.

        Args:
            model_path: Path to the model.
            model_server: Model server configuration.
            remote_storage: Remote storage configuration.
            endpoints: Endpoints configuration.
        """
        import rasa.core.agent

        logger.warning(
            "Started running end-to-end testing. "
            "Note that this feature is not intended for use in a "
            "production environment. Don't use it to process sensitive data. "
            "If you do, it's at your own risk. "
            "We're looking forward to your feedback."
        )

        if endpoints:
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

        Returns:
            Test turns: {turn_sequence (int) : TestStep or ActualStepOutput}.
        """
        turns: TEST_TURNS_TYPE = {}
        event_cursor = 0

        tracker = await self.agent.processor.fetch_tracker_with_initial_session(  # type: ignore[union-attr]
            sender_id
        )
        # turn -1 i used to contain events that happen during
        # the start of the session and before the first user message
        # TestStep is a placeholder just for the sake of having a turn
        # to specify the actor
        turns[-1], event_cursor = self.get_actual_step_output(
            tracker, TestStep(actor="bot", text=None), event_cursor
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
                logger.error(
                    f"Message handling timed out for user message '{step.text}'.",
                    exc_info=True,
                )
            except Exception:
                logger.exception(
                    f"An exception occurred while handling "
                    f"user message '{step.text}'."
                )
            tracker = await self.agent.tracker_store.retrieve(sender_id)
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
            logger.warning(
                f"Metadata {keys_to_overwrite} exist in both the test case "
                f"'{test_case_name}' and the user step '{step_text}'. "
                "The user step metadata takes precedence and will "
                "override the test case metadata."
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
            logger.warning(
                f"No events found for '{tracker.sender_id}' after processing test "
                f"step '{test_step.text}'."
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
        test_failures = cls.find_test_failures(test_turns, test_case)
        difference = []
        first_failure = None
        if test_failures:
            first_failure = test_failures[0][0]
            difference = cls.human_readable_diff(test_turns, test_failures)
        else:
            difference = []

        return TestResult(
            pass_status=len(test_failures) == 0,
            test_case=test_case,
            difference=difference,
            error_line=first_failure.error_line if first_failure else None,
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
        tracker = await self.agent.processor.fetch_tracker_with_initial_session(  # type: ignore[union-attr]
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
            logger.warning(
                f"Metadata '{metadata_name}' is not defined in the input metadata."
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
            collector = CollectingOutputChannel()

            # add timestamp suffix to ensure sender_id is unique
            sender_id = f"{test_case.name}_{datetime.datetime.now()}"

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

            tracker = await self.run_prediction_loop(
                collector,
                test_case.steps,
                sender_id,
                test_case_metadata,
                input_metadata,
            )

            test_result = self.generate_test_result(tracker, test_case)
            results.append(test_result)

            if fail_fast and not test_result.pass_status:
                break

        return results

    @staticmethod
    def _action_server_is_reachable(endpoints: AvailableEndpoints) -> None:
        """Calls the action server health endpoint."""
        if not endpoints.action:
            logger.debug(
                "No action endpoint configured. Skipping the health check of the "
                "action server."
            )
            return

        if not endpoints.action.url:
            logger.debug(
                "Action endpoint URL is not defined in the endpoint configuration."
            )
            return

        logger.debug(
            "Detected action URL in the endpoint configuration.\n"
            f"Action Server URL: {endpoints.action.url}\n"
            "Sending a health request to the action endpoint."
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

        logger.debug(
            "Action endpoint has responded successfully.\n"
            f"Response message: {response.text}\n"
            f"Response status code: {response.status_code}."
        )
