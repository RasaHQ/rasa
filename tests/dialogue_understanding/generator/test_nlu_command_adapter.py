from unittest.mock import Mock

import pytest

from rasa.dialogue_understanding.commands import (
    StartFlowCommand,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


class TestNLUCommandAdapter:
    """Tests for the LLMCommandGenerator."""

    @pytest.fixture
    def command_generator(self):
        """Create an LLMCommandGenerator."""
        return NLUCommandAdapter.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock()
        )

    @pytest.fixture
    def tracker(self):
        """Create a Tracker."""
        return DialogueStateTracker.from_events("", [])

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

    def test_predict_commands_with_no_flows(
        self, command_generator: NLUCommandAdapter, tracker: DialogueStateTracker
    ):
        """Test that predict_commands returns an empty list when flows is None."""
        # Given
        empty_flows = FlowsList([])
        # When
        predicted_commands = command_generator.predict_commands(
            Message.build("some message"), flows=empty_flows, tracker=tracker
        )
        # Then
        assert not predicted_commands

    def test_predict_commands_with_no_tracker(
        self, command_generator: NLUCommandAdapter, flows: FlowsList
    ):
        """Test that predict_commands returns an empty list when tracker is None."""
        # When
        predicted_commands = command_generator.predict_commands(
            Message.build("some message"), flows=flows, tracker=None
        )
        # Then
        assert not predicted_commands

    def test_predict_commands_with_message_without_intent(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands returns an empty list when
        message does not have any intents."""
        # When
        predicted_commands = command_generator.predict_commands(
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

    def test_predict_commands_returns_start_flow_command(
        self, command_generator: NLUCommandAdapter, flows: FlowsList
    ):
        """Test that predict_commands returns StartFlowCommand
        if flow with valid nlu trigger."""

        test_message = Message(
            data={
                TEXT: "some message",
                INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
            },
        )
        predicted_commands = command_generator.predict_commands(
            test_message, flows=flows, tracker=Mock()
        )

        assert isinstance(predicted_commands[0], StartFlowCommand)
        assert predicted_commands[0].as_dict()["flow"] == "test_flow"

    def test_predict_commands_returns_with_multiple_flows_triggered(
        self, command_generator: NLUCommandAdapter
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
        predicted_commands = command_generator.predict_commands(
            test_message, flows=flows, tracker=Mock()
        )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], StartFlowCommand)
        assert predicted_commands[0].as_dict()["flow"] == "first_flow"
