from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel
from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding_test.constants import (
    ACTOR_BOT,
    ACTOR_USER,
    PLACEHOLDER_GENERATED_ANSWER_TEMPLATE,
)
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.test_case_simulation.test_case_tracker_simulator import (  # noqa: E501
    TestCaseTrackerSimulator,
)
from rasa.e2e_test.e2e_test_case import Fixture, Metadata
from rasa.shared.core.constants import KEY_MAPPING_TYPE, SlotMappingType
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS


@pytest.fixture
def sample_test_step() -> DialogueUnderstandingTestStep:
    return DialogueUnderstandingTestStep(
        actor=ACTOR_USER,
        text="Hello",
        commands=[StartFlowCommand("bar")],
    )


@pytest.fixture
def sample_test_case(
    sample_test_step: DialogueUnderstandingTestStep,
) -> DialogueUnderstandingTestCase:
    return DialogueUnderstandingTestCase(
        name="test_case",
        steps=[sample_test_step],
    )


@pytest.fixture
def mock_tracker_simulator(
    sample_test_case: DialogueUnderstandingTestCase, monkeypatch: MonkeyPatch
) -> TestCaseTrackerSimulator:
    agent = Agent()
    processor = AsyncMock()

    # using the actual tracker store instead of a mocked one
    async def mock_fetch_tracker(
        sender_id: str,
        output_channel: CollectingOutputChannel,
    ) -> Any:
        return await agent.tracker_store.get_or_create_tracker(sender_id)

    processor.fetch_tracker_with_initial_session = mock_fetch_tracker
    agent.processor = processor
    agent.domain = Domain.empty()

    async def mock_handle_message(self: Any, message: Any) -> None:
        tracker = await self.tracker_store.get_or_create_tracker(message.sender_id)
        tracker.update(
            UserUttered(
                message.text,
                parse_data={
                    COMMANDS: message.parse_data.get(COMMANDS)
                    if message.parse_data
                    else []
                },
            )
        )
        await self.tracker_store.save(tracker)

    monkeypatch.setattr("rasa.core.agent.Agent.handle_message", mock_handle_message)

    return TestCaseTrackerSimulator(agent, sample_test_case)


@pytest.mark.asyncio
async def test_add_latest_user_uttered_event_index():
    # Given
    tracker = DialogueStateTracker("sender_id", slots={})
    tracker.update(UserUttered("user message 1"))
    tracker.update(BotUttered("bot message 1"))
    tracker.update(BotUttered("bot message 2"))
    tracker.update(UserUttered("user message"))

    user_uttered_event_indices = []

    # When
    utter_uttered_index = (
        await TestCaseTrackerSimulator._get_latest_user_uttered_event_index(
            tracker, user_uttered_event_indices
        )
    )
    # Then
    assert utter_uttered_index == 3
    user_uttered_event_indices.append(utter_uttered_index)

    # When
    tracker.update(UserUttered("user message"))
    utter_uttered_index = (
        await TestCaseTrackerSimulator._get_latest_user_uttered_event_index(
            tracker, user_uttered_event_indices
        )
    )
    # Then
    assert utter_uttered_index == 4


@pytest.mark.parametrize(
    "bot_events, bot_steps, expected",
    [
        (
            [
                BotUttered(text="Hello", metadata={"utter_action": "utter_greet"}),
                BotUttered(
                    text="How can I help you?", metadata={"utter_action": "utter_help"}
                ),
            ],
            [
                DialogueUnderstandingTestStep(actor=ACTOR_BOT, template="utter_greet"),
                DialogueUnderstandingTestStep(actor=ACTOR_BOT, template="utter_help"),
            ],
            True,
        ),
        (
            [
                BotUttered(
                    text="How are you?",
                    metadata={"utter_action": "utter_ask_how_are_you"},
                ),
            ],
            [
                DialogueUnderstandingTestStep(actor=ACTOR_BOT, template="utter_greet"),
                DialogueUnderstandingTestStep(
                    actor=ACTOR_BOT, template="utter_goodbye"
                ),
            ],
            False,
        ),
        (
            [
                BotUttered(
                    text="This is an answer from the knowledge base.",
                    metadata={},
                ),
            ],
            [
                DialogueUnderstandingTestStep(
                    actor=ACTOR_BOT, template=PLACEHOLDER_GENERATED_ANSWER_TEMPLATE
                ),
            ],
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_do_bot_responses_match(
    bot_events: List[BotUttered],
    bot_steps: List[DialogueUnderstandingTestStep],
    expected: bool,
    mock_tracker_simulator: TestCaseTrackerSimulator,
):
    tracker = DialogueStateTracker("default", [])
    for event in bot_events:
        tracker.update(event)

    assert (
        mock_tracker_simulator._do_bot_responses_match(tracker, bot_steps) == expected
    )


def test_get_latest_bot_uttered_events():
    # Create a mock tracker with events
    events = [
        UserUttered(text="Hello"),
        BotUttered(text="Hi there!"),
        BotUttered(text="How can I help you?"),
        UserUttered(text="I need assistance"),
        BotUttered(text="Sure, what do you need help with?"),
    ]
    tracker = DialogueStateTracker(sender_id="test_sender", slots={})
    tracker.events.extend(events)

    # Call the method
    bot_uttered_events = TestCaseTrackerSimulator._get_latest_bot_uttered_events(
        tracker
    )

    # Assert the results
    assert len(bot_uttered_events) == 1
    assert bot_uttered_events[0].text == "Sure, what do you need help with?"


@pytest.mark.parametrize(
    "fixture_names, fixtures, expected_length, expected_names",
    [
        (
            ["fixture1"],
            [
                Fixture(name="fixture1", slots_set={"slot1": "value1"}),
                Fixture(name="fixture2", slots_set={"slot2": "value2"}),
            ],
            1,
            ["fixture1"],
        ),
        (
            ["fixture3"],
            [
                Fixture(name="fixture1", slots_set={"slot1": "value1"}),
                Fixture(name="fixture2", slots_set={"slot2": "value2"}),
            ],
            0,
            [],
        ),
        (
            None,
            [
                Fixture(name="fixture1", slots_set={"slot1": "value1"}),
                Fixture(name="fixture2", slots_set={"slot2": "value2"}),
            ],
            0,
            [],
        ),
    ],
)
def test_filter_fixtures_for_test_case(
    fixture_names: List[str],
    fixtures: List[Fixture],
    expected_length: int,
    expected_names: List[str],
):
    filtered_fixtures = TestCaseTrackerSimulator._filter_fixtures_for_test_case(
        fixture_names, fixtures
    )
    assert len(filtered_fixtures) == expected_length
    assert [fixture.name for fixture in filtered_fixtures] == expected_names


@pytest.mark.asyncio
async def test_set_up_fixtures(sample_test_case: DialogueUnderstandingTestCase):
    # Create mock objects
    agent = MagicMock()
    agent.processor = MagicMock()
    agent.tracker_store = AsyncMock()
    tracker = DialogueStateTracker("sender_id", slots={})
    agent.tracker_store.save = AsyncMock()

    # Create fixtures
    fixtures = [
        Fixture(name="fixture1", slots_set={"slot1": "value1", "slot2": "value2"}),
        Fixture(name="fixture2", slots_set={"slot3": "value3"}),
    ]

    # Initialize ConversationTrackerBuilder
    tracker_simulator = TestCaseTrackerSimulator(agent, sample_test_case)

    # Call the method
    await tracker_simulator._set_up_fixtures(fixtures, tracker)

    # Check that the tracker was updated correctly
    assert tracker.events[0] == SlotSet("slot1", "value1")
    assert tracker.events[1] == SlotSet("slot2", "value2")
    assert tracker.events[2] == SlotSet("slot3", "value3")


@pytest.mark.asyncio
async def test_set_up_fixtures_no_fixtures(
    sample_test_case: DialogueUnderstandingTestCase,
):
    # Create mock objects
    agent = MagicMock()
    agent.processor = MagicMock()
    agent.tracker_store = AsyncMock()
    tracker = DialogueStateTracker("sender_id", slots={})
    agent.tracker_store.save = AsyncMock()

    # Initialize ConversationTrackerBuilder
    tracker_simulator = TestCaseTrackerSimulator(agent, sample_test_case)

    # Call the method with no fixtures
    await tracker_simulator._set_up_fixtures([], tracker)

    # Check that the tracker was not saved
    agent.tracker_store.save.assert_not_called()


@pytest.mark.parametrize(
    "test_case, user_step, metadata, expected_metadata, expected_commands",
    [
        (
            DialogueUnderstandingTestCase(
                name="test_case",
                steps=[
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="dummy step")
                ],
                metadata_name="test_case_meta",
            ),
            DialogueUnderstandingTestStep(
                actor=ACTOR_USER,
                metadata_name="step_meta",
                text="cancel",
                commands=[CancelFlowCommand()],
            ),
            [
                Metadata(name="test_case_meta", metadata={"key1": "value1"}),
                Metadata(name="step_meta", metadata={"key2": "value2"}),
            ],
            {"key1": "value1", "key2": "value2"},
            [CancelFlowCommand().as_dict()],
        ),
        (
            DialogueUnderstandingTestCase(
                name="test_case",
                steps=[
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="dummy step")
                ],
                metadata_name="test_case_meta",
            ),
            DialogueUnderstandingTestStep(
                actor=ACTOR_USER,
                metadata_name="step_meta",
                text="hello",
                commands=[
                    StartFlowCommand("transfer_money"),
                    SetSlotCommand("recipient", "John"),
                ],
            ),
            [Metadata(name="test_case_meta", metadata={"key1": "value1"})],
            {"key1": "value1"},
            [
                StartFlowCommand("transfer_money").as_dict(),
                SetSlotCommand(
                    name="recipient", value="John", extractor=SetSlotExtractor.NLU.value
                ).as_dict(),
            ],
        ),
        (
            DialogueUnderstandingTestCase(
                name="test_case",
                steps=[
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="dummy step")
                ],
                metadata_name="test_case_meta",
            ),
            DialogueUnderstandingTestStep(
                actor=ACTOR_USER, metadata_name="step_meta", text="hello", commands=[]
            ),
            [],
            {},
            [],
        ),
    ],
)
def test_create_user_message(
    test_case: DialogueUnderstandingTestCase,
    user_step: DialogueUnderstandingTestStep,
    metadata: List[Metadata],
    expected_metadata: Dict[str, str],
    expected_commands: Dict[str, Any],
    mock_tracker_simulator: TestCaseTrackerSimulator,
):
    mock_tracker_simulator.test_case = test_case

    user_message = mock_tracker_simulator._create_user_message(user_step, metadata)

    assert user_message.sender_id == mock_tracker_simulator.sender_id
    assert user_message.text == user_step.text
    assert user_message.metadata == expected_metadata
    assert user_message.parse_data.get(COMMANDS, []) == expected_commands


@pytest.mark.parametrize(
    "mapping, user_message, expected_set_slot_extractor",
    (
        (
            {KEY_MAPPING_TYPE: SlotMappingType.FROM_ENTITY.value, "entity": "entity1"},
            "message",
            SetSlotExtractor.NLU,
        ),
        (
            {KEY_MAPPING_TYPE: SlotMappingType.FROM_TEXT.value},
            "message",
            SetSlotExtractor.NLU,
        ),
        (
            {
                KEY_MAPPING_TYPE: SlotMappingType.FROM_INTENT.value,
                "intent": "intent1",
                "value": "value1",
            },
            "message",
            SetSlotExtractor.NLU,
        ),
        (
            {KEY_MAPPING_TYPE: SlotMappingType.FROM_LLM.value},
            "message",
            SetSlotExtractor.LLM,
        ),
        (
            {KEY_MAPPING_TYPE: SlotMappingType.FROM_LLM.value},
            "/SetSlot(slot1, value1)",
            SetSlotExtractor.COMMAND_PAYLOAD_READER,
        ),
    ),
)
def test_update_extractor_with_command_payload_reader(
    mapping: Dict[str, Any],
    user_message: str,
    expected_set_slot_extractor: SetSlotExtractor,
    sample_test_case: DialogueUnderstandingTestCase,
):
    agent = Agent()
    agent.domain = Domain.empty()
    agent.domain.slots = [TextSlot(name="slot1", mappings=[mapping])]

    simulator = TestCaseTrackerSimulator(agent, sample_test_case)
    commands = [
        SetSlotCommand(
            name="slot1", value="value1", extractor=SetSlotExtractor.LLM.value
        )
    ]

    updated_commands = simulator._update_extractor_of_set_slot_commands(
        commands, user_message
    )

    assert isinstance(updated_commands[0], SetSlotCommand)
    assert updated_commands[0].extractor == expected_set_slot_extractor.value
