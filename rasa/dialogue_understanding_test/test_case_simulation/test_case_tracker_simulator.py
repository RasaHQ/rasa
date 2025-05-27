from typing import List, Optional

import structlog
from pydantic import BaseModel

from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding_test.constants import (
    PLACEHOLDER_GENERATED_ANSWER_TEMPLATE,
)
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.test_case_simulation.exception import (
    TestCaseTrackerSimulatorException,
)
from rasa.dialogue_understanding_test.utils import filter_metadata
from rasa.e2e_test.e2e_test_case import Fixture, Metadata
from rasa.shared.core.constants import SlotMappingType
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS, ENTITIES, INTENT
from rasa.shared.utils.llm import generate_sender_id

structlogger = structlog.get_logger()


class TestCaseTrackerSimulatorResult(BaseModel):
    sender_id: str
    user_uttered_event_indices: List[int]


class TestCaseTrackerSimulator:
    """Builds up a tracker for a test case.

    The tracker is built up by simulating a conversation with the bot
    using the user and bot steps of the test case.
    As the user steps are annotated with commands, the user messages
    are sent to the bot with the commands, i.e. skipping any LLM call.
    """

    def __init__(
        self,
        agent: Agent,
        test_case: DialogueUnderstandingTestCase,
        output_channel: Optional[CollectingOutputChannel] = None,
    ):
        self.agent = agent
        self.test_case = test_case
        self.output_channel = output_channel or CollectingOutputChannel()

        self.sender_id = generate_sender_id(self.test_case.name)

    async def simulate_test_case(
        self,
        metadata: List[Metadata],
    ) -> TestCaseTrackerSimulatorResult:
        """Simulates a conversation with the bot using the test case.

        Args:
            metadata: The available metadata.

        Returns:
        The result of the simulation.
        """
        step_index = 0
        user_uttered_event_indices: List[int] = []

        while step_index < len(self.test_case.steps):
            user_step, bot_steps = self.test_case.get_next_user_and_bot_steps(
                step_index
            )
            if user_step is None:
                raise TestCaseTrackerSimulatorException(
                    test_case_name=self.test_case.full_name(),
                    failure_reason="Could not retrieve next user step.",
                )

            step_index += len(bot_steps) + 1  # for the user step

            # we don't need to simulate the last user message
            if step_index >= len(self.test_case.steps):
                tracker = await self.agent.tracker_store.retrieve(self.sender_id)
                if tracker is None:
                    raise TestCaseTrackerSimulatorException(
                        test_case_name=self.test_case.full_name(),
                        user_message=user_step.text,
                        failure_reason="The tracker could not be retrieved.",
                    )
                # add the index of the last user uttered event
                user_uttered_event_indices.append(len(tracker.events))

                return TestCaseTrackerSimulatorResult(
                    sender_id=self.sender_id,
                    user_uttered_event_indices=user_uttered_event_indices,
                )

            # send the user message to the agent
            try:
                await self._send_user_message_with_commands(
                    user_step,
                    metadata,
                )
            except Exception as e:
                raise TestCaseTrackerSimulatorException(
                    test_case_name=self.test_case.full_name(),
                    user_message=user_step.text,
                    failure_reason="Sending the user message failed.",
                    original_exception=e.__dict__,
                )

            tracker = await self.agent.tracker_store.retrieve(self.sender_id)
            if tracker is None:
                raise TestCaseTrackerSimulatorException(
                    test_case_name=self.test_case.full_name(),
                    user_message=user_step.text,
                    failure_reason="The tracker could not be retrieved.",
                )

            # add latest user uttered event index to the list
            user_uttered_event_indices.append(
                await self._get_latest_user_uttered_event_index(
                    tracker, user_uttered_event_indices
                )
            )

            # check if bot responses match the expected bot steps
            if not self._do_bot_responses_match(tracker, bot_steps):
                structlogger.error(
                    "dialogue_understanding_test.tracker_simulator.stop_simulation.bot_utterance_mismatch",
                    test_case=self.test_case.full_name(),
                    user_message=user_step.text,
                    bot_steps=bot_steps,
                    bot_uttered_events=TestCaseTrackerSimulator._get_latest_bot_uttered_events(
                        tracker
                    ),
                )
                # stop the simulation if the bot responses do not match
                # we can still test the steps up until the mismatch
                return TestCaseTrackerSimulatorResult(
                    sender_id=self.sender_id,
                    user_uttered_event_indices=user_uttered_event_indices,
                )

        return TestCaseTrackerSimulatorResult(
            sender_id=self.sender_id,
            user_uttered_event_indices=user_uttered_event_indices,
        )

    @staticmethod
    async def _get_latest_user_uttered_event_index(
        tracker: DialogueStateTracker, user_uttered_event_indices: List[int]
    ) -> int:
        """Get the index of the latest user uttered event in the tracker."""
        # search the tracker events for the latest user message starting from the
        # index of the user message before
        # +1 to avoid getting the same index for duplicate user messages,
        # such as, "yes"
        from_index = (
            user_uttered_event_indices[-1] + 1 if user_uttered_event_indices else 0
        )
        return tracker.events.index(tracker.latest_message, from_index)

    def _do_bot_responses_match(
        self,
        tracker: DialogueStateTracker,
        bot_steps: List[DialogueUnderstandingTestStep],
    ) -> bool:
        # get all bot uttered events until the last user message
        bot_uttered_events = self._get_latest_bot_uttered_events(tracker)

        if len(bot_uttered_events) != len(bot_steps):
            return False

        for step in bot_steps:
            if step.template:
                if not self._does_template_match(step, bot_uttered_events):
                    return False
            elif step.text:
                if not self._does_text_match(step, bot_uttered_events):
                    return False

        return True

    @staticmethod
    def _does_template_match(
        step: DialogueUnderstandingTestStep, bot_uttered_events: List[BotUttered]
    ) -> bool:
        for event in bot_uttered_events:
            if step.template == PLACEHOLDER_GENERATED_ANSWER_TEMPLATE:
                return True
            elif "utter_action" not in event.metadata:
                # a chitchat or knowledge base command was triggered,
                # or the response comes from a custom action,
                # continue with the simulation
                return True
            elif event.metadata["utter_action"] == step.template:
                return True
        return False

    @staticmethod
    def _does_text_match(
        step: DialogueUnderstandingTestStep, bot_uttered_events: List[BotUttered]
    ) -> bool:
        for event in bot_uttered_events:
            if event.text == step.text:
                return True

        return False

    @staticmethod
    def _get_latest_bot_uttered_events(
        tracker: DialogueStateTracker,
    ) -> List[BotUttered]:
        """Get the latest bot uttered events in the tracker."""
        # collect all bot uttered events until a user uttered event is reached
        # starting from the end of the tracker events
        bot_uttered_events = []
        for event in reversed(tracker.events):
            if isinstance(event, BotUttered):
                bot_uttered_events.append(event)
            if isinstance(event, UserUttered):
                break
        return bot_uttered_events

    async def initialize_tracker(self, fixtures: List[Fixture]) -> None:
        """Initializes a new tracker with fixtures."""
        assert self.agent.processor is not None

        tracker = await self.agent.processor.fetch_tracker_with_initial_session(
            self.sender_id, output_channel=self.output_channel
        )

        if fixtures and self.test_case.fixture_names:
            test_fixtures = self._filter_fixtures_for_test_case(
                self.test_case.fixture_names, fixtures
            )
            await self._set_up_fixtures(test_fixtures, tracker)

        # store the tracker with the unique sender id
        await self.agent.tracker_store.save(tracker)

    async def _set_up_fixtures(
        self,
        fixtures: List[Fixture],
        tracker: DialogueStateTracker,
    ) -> None:
        """Sets up fixtures in the tracker."""
        if not fixtures or not self.agent.processor:
            return

        for fixture in fixtures:
            for slot_name, slot_value in fixture.slots_set.items():
                tracker.update(SlotSet(slot_name, slot_value))

    @staticmethod
    def _filter_fixtures_for_test_case(
        fixture_names: Optional[List[str]], fixtures: List[Fixture]
    ) -> List[Fixture]:
        """Filters fixtures applicable to the test case."""
        return [
            fixture
            for fixture in fixtures
            if fixture_names and fixture.name in fixture_names
        ]

    async def _send_user_message_with_commands(
        self,
        user_step: DialogueUnderstandingTestStep,
        metadata: List[Metadata],
    ) -> None:
        """Sends a user message with commands to the agent."""
        user_message = self._create_user_message(user_step, metadata)
        await self.agent.handle_message(user_message)

    def _create_user_message(
        self,
        user_step: DialogueUnderstandingTestStep,
        metadata: List[Metadata],
    ) -> UserMessage:
        """Creates a user message with commands."""
        user_message = user_step.text
        # Get the metadata for the step
        metadata_for_step = filter_metadata(
            self.test_case, user_step, metadata, self.sender_id
        )
        # Update the extractor of SetSlotCommand based on the slot mapping type
        commands = self._update_extractor_of_set_slot_commands(
            user_step.commands, user_message
        )

        return UserMessage(
            user_message,
            self.output_channel,
            self.sender_id,
            parse_data={
                INTENT: {},
                ENTITIES: [],
                COMMANDS: [command.as_dict() for command in commands],
            },
            metadata=metadata_for_step,
        )

    def _update_extractor_of_set_slot_commands(
        self, commands: List[Command], user_message: str
    ) -> List[Command]:
        """Update the extractor for SetSlotCommand based on the slot mapping type."""
        slots = []
        if self.agent.domain is not None:
            slots = self.agent.domain.slots

        for command in commands:
            if isinstance(command, SetSlotCommand):
                slot_definition = next(
                    (slot for slot in slots if slot.name == command.name), slots[0]
                )

                # Use the SetSlotExtractor.COMMAND_PAYLOAD_READER extractor if the user
                # message starts with the /SetSlot.
                if user_message.startswith(r"/SetSlot"):
                    command.extractor = SetSlotExtractor.COMMAND_PAYLOAD_READER.value
                # Use the SetSlotExtractor.NLU extractor if the slot mapping type is
                # not FROM_LLM.
                elif SlotMappingType.FROM_LLM not in [
                    mapping.type for mapping in slot_definition.mappings
                ]:
                    command.extractor = SetSlotExtractor.NLU.value

        return commands
