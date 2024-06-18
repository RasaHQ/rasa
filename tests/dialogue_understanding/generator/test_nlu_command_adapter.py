import uuid
from typing import List
from unittest.mock import Mock

import pytest

from rasa.dialogue_understanding.commands import (
    Command,
    StartFlowCommand,
    SetSlotCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.shared.nlu.constants import (
    ENTITIES,
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


class TestNLUCommandAdapter:
    """Tests for the NLUCommandAdapter."""

    @pytest.fixture
    def command_generator(self):
        """Create a CommandGenerator subclass."""
        return NLUCommandAdapter.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock()
        )

    @pytest.fixture
    def tracker(self):
        """Create a Tracker."""
        return DialogueStateTracker.from_events("", [])

    @pytest.fixture
    def domain(self):
        """Create a Domain."""
        return Domain.from_yaml("""
        intents:
            - foo
            - foo2
        entities:
            - bar
            - bar2
        slots:
            baz:
              type: text
              mappings:
                - type: from_entity
                  entity: bar
            baz2:
              type: text
              mappings:
                - type: from_entity
                  entity: bar2
                  intent: foo
        """)

    @pytest.fixture
    def tracker_with_routing_slot(self):
        """Create a Tracker."""
        return DialogueStateTracker.from_events(
            sender_id="",
            evts=[],
            slots=[
                BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=True),
            ],
        )

    @pytest.fixture
    def flows(self) -> FlowsList:
        """Create a FlowsList."""
        return flows_from_str(
            """
            flows:
              test_flow:
                description: flow test_flow
                nlu_trigger:
                  - intent: foo
                steps:
                - id: first_step
                  action: action_listen
            """
        )

    async def test_predict_commands_with_no_flows(
        self, command_generator: NLUCommandAdapter, tracker: DialogueStateTracker
    ):
        """Test that predict_commands returns an empty list when flows is None."""
        # Given
        empty_flows = FlowsList([])
        # When
        predicted_commands = await command_generator.predict_commands(
            Message.build("some message"), flows=empty_flows, tracker=tracker
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_with_no_tracker(
        self, command_generator: NLUCommandAdapter, flows: FlowsList
    ):
        """Test that predict_commands returns an empty list when tracker is None."""
        # When
        predicted_commands = await command_generator.predict_commands(
            Message.build("some message"), flows=flows, tracker=None
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_with_message_without_intent_nor_entities(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands returns an empty list when
        message does not have any intents."""
        # When
        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "some message",
                }
            ),
            flows=flows,
            tracker=tracker,
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_does_not_set_routing_slot_on_no_predicted_commands(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker_with_routing_slot: DialogueStateTracker,
    ):
        """Test that predict_commands returns an empty list when
        message does not have any intents."""
        # When
        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "some message",
                }
            ),
            flows=flows,
            tracker=tracker_with_routing_slot,
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_returns_start_flow_command(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands returns StartFlowCommand
        if flow with valid nlu trigger."""

        test_message = Message(
            data={
                TEXT: "some message",
                INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
            },
        )
        predicted_commands = await command_generator.predict_commands(
            test_message, flows=flows, tracker=tracker
        )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], StartFlowCommand)
        assert predicted_commands[0].as_dict()["flow"] == "test_flow"

    async def test_predict_commands_returns_start_flow_command_and_set_routing_slot(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker_with_routing_slot: DialogueStateTracker,
    ):
        """Test that predict_commands returns StartFlowCommand
        if flow with valid nlu trigger."""

        test_message = Message(
            data={
                TEXT: "some message",
                INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
            },
        )
        predicted_commands = await command_generator.predict_commands(
            test_message, flows=flows, tracker=tracker_with_routing_slot
        )

        assert len(predicted_commands) == 2
        assert isinstance(predicted_commands[0], StartFlowCommand)
        assert predicted_commands[0].as_dict()["flow"] == "test_flow"
        assert isinstance(predicted_commands[1], SetSlotCommand)
        assert predicted_commands[1] == SetSlotCommand(ROUTE_TO_CALM_SLOT, True)

    async def test_predict_commands_returns_with_multiple_flows_triggered(
        self,
        command_generator: NLUCommandAdapter,
        tracker: DialogueStateTracker,
        domain: Domain,
    ):
        """Test that predict_commands returns just one StartFlowCommand
        if multiple flows can be triggered by the predicted intent."""
        # create flows one by one to avoid DuplicateNLUTriggerException
        first_flow = flows_from_str(
            """
            flows:
              first_flow:
                description: flow first_flow
                nlu_trigger:
                  - intent: foo
                steps:
                - id: first_step
                  action: action_listen
            """
        ).underlying_flows[0]
        second_flow = flows_from_str(
            """
            flows:
              second_flow:
                description: flow second_flow
                nlu_trigger:
                  - intent: foo
                steps:
                - id: first_step
                  action: action_listen
            """
        ).underlying_flows[0]

        flows = FlowsList([first_flow, second_flow])

        test_message = Message(
            data={
                TEXT: "some message",
                INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
            },
        )
        predicted_commands = await command_generator.predict_commands(
            test_message, flows=flows, tracker=tracker, domain=domain
        )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], StartFlowCommand)
        assert predicted_commands[0].as_dict()["flow"] == "first_flow"

    async def test_predict_commands_with_message_with_intent_and_no_entities(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands returns an empty list when
        message does not have any entities."""
        # When
        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
                }
            ),
            flows=flows,
            tracker=tracker,
        )
        # Then
        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], StartFlowCommand)
        assert predicted_commands[0].as_dict()["flow"] == "test_flow"

    async def test_predict_commands_with_message_with_entities_and_no_intent(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        domain: Domain,
    ):
        entity_name = "bar"
        entity_value = "test_value"
        expected_slot_name = "baz"
        sender_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)

        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "some message",
                    ENTITIES: [{"entity": entity_name, "value": entity_value}],
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], SetSlotCommand)
        assert predicted_commands[0] == SetSlotCommand(
            name=expected_slot_name,
            value=entity_value,
            extractor=SetSlotExtractor.NLU.value,
        )

    @pytest.mark.parametrize(
        "intent_name, expected_commands",
        [
            (
                "foo",
                [
                    StartFlowCommand(flow="test_flow"),
                    SetSlotCommand(
                        name="baz2",
                        value="test_value",
                        extractor=SetSlotExtractor.NLU.value,
                    ),
                ],
            ),
            ("foo2", []),
        ],
    )
    async def test_predict_commands_with_message_with_entities_and_intent(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        domain: Domain,
        intent_name: str,
        expected_commands: List[Command],
    ):
        entity_name = "bar2"
        entity_value = "test_value"
        sender_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)

        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: intent_name,
                        PREDICTED_CONFIDENCE_KEY: 1.0,
                    },
                    ENTITIES: [{"entity": entity_name, "value": entity_value}],
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        assert predicted_commands == expected_commands
