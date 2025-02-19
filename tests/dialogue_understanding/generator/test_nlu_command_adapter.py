import uuid
from typing import List, Optional, Type
from unittest.mock import Mock, patch

import pytest
from pytest import MonkeyPatch

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    ChitChatAnswerCommand,
    Command,
    CorrectedSlot,
    CorrectSlotsCommand,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    RestartCommand,
    SessionStartCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.generator.nlu_command_adapter import (
    NLUCommandAdapter,
    _issue_set_slot_commands,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import KEY_INTENTS, Domain
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.slots import BooleanSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.importer import FlowSyncImporter
from rasa.shared.nlu.constants import (
    COMMANDS,
    ENTITIES,
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_COMMANDS,
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
        return Domain.from_yaml(
            """
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
            another_slot:
              type: text
              mappings:
                - type: from_text
            qux:
              type: text
              mappings:
                - type: from_text
                  intent: foo2
        """
        )

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
                  collect: baz2
                - collect: baz
                - collect: another_slot
                  ask_before_filling: true
                - action: action_listen
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
        message does not have any intents.
        """
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
        message does not have any intents.
        """
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
        if flow with valid nlu trigger.
        """
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
        if flow with valid nlu trigger.
        """
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
        if multiple flows can be triggered by the predicted intent.
        """
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
        message does not have any entities.
        """
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

    async def test_predict_commands_skips_slot_not_collected_by_flows(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        domain: Domain,
    ):
        sender_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)

        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "some message",
                    INTENT: {
                        INTENT_NAME_KEY: "foo2",
                        PREDICTED_CONFIDENCE_KEY: 1.0,
                    },
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        assert (
            SetSlotCommand("qux", "some message", SetSlotExtractor.NLU.value)
            not in predicted_commands
        )
        assert len(predicted_commands) == 0

    async def test_predict_start_session_command(
        self, command_generator: NLUCommandAdapter
    ):
        """Test whether start session is triggerable by default."""
        sender_id = uuid.uuid4().hex
        domain = FlowSyncImporter.load_default_pattern_flows_domain()
        flows = FlowSyncImporter.load_default_pattern_flows()
        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)
        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "/session_start",
                    INTENT: {
                        INTENT_NAME_KEY: "session_start",
                        PREDICTED_CONFIDENCE_KEY: 1.0,
                    },
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], SessionStartCommand)

    async def test_predict_restart_command(self, command_generator: NLUCommandAdapter):
        """Test whether start session is triggerable by default."""
        sender_id = uuid.uuid4().hex
        domain = FlowSyncImporter.load_default_pattern_flows_domain()
        flows = FlowSyncImporter.load_default_pattern_flows()
        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)
        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: "/restart",
                    INTENT: {
                        INTENT_NAME_KEY: "restart",
                        PREDICTED_CONFIDENCE_KEY: 1.0,
                    },
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], RestartCommand)

    @pytest.mark.parametrize(
        "pattern,expected_command_class",
        [
            ("pattern_cancel_flow", CancelFlowCommand),
            ("pattern_search", KnowledgeAnswerCommand),
            ("pattern_human_handoff", HumanHandoffCommand),
            ("pattern_skip_question", SkipQuestionCommand),
            ("pattern_cannot_handle", CannotHandleCommand),
            ("pattern_collect_information", None),
            ("pattern_continue_interrupted", None),
            ("pattern_completed", None),
            ("pattern_internal_error", None),
            ("pattern_clarification", None),
            ("pattern_correction", None),
            ("pattern_code_change", None),
            ("pattern_nonexistent_XYZABC", None),
        ],
    )
    async def test_predict_other_pattern_commands(
        self,
        command_generator: NLUCommandAdapter,
        pattern: str,
        expected_command_class: Optional[Type[Command]],
    ):
        """Test whether other patterns can be made triggerable via intents."""
        sender_id = uuid.uuid4().hex
        test_trigger_intent = "test_trigger_intent"
        domain = FlowSyncImporter.load_default_pattern_flows_domain()
        domain_addon = Domain.from_dict({KEY_INTENTS: [test_trigger_intent]})
        domain = domain.merge(domain_addon)

        flows = FlowSyncImporter.load_default_pattern_flows()
        flows_pattern_overwritten = flows_from_str(
            f"""
            flows:
              {pattern}:
                description: overwritten pattern
                nlu_trigger:
                  - intent: {test_trigger_intent}
                steps:
                - id: first_step
                  action: action_listen
            """
        )
        flows = flows_pattern_overwritten.merge(flows)

        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)

        predicted_commands = await command_generator.predict_commands(
            Message(
                data={
                    TEXT: f"/{test_trigger_intent}",
                    INTENT: {
                        INTENT_NAME_KEY: test_trigger_intent,
                        PREDICTED_CONFIDENCE_KEY: 1.0,
                    },
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        if expected_command_class:
            assert len(predicted_commands) == 1
            assert isinstance(predicted_commands[0], expected_command_class)
        else:
            assert len(predicted_commands) == 0

    @pytest.mark.parametrize(
        "pattern,expected_command_class",
        [
            ("pattern_chitchat", ChitChatAnswerCommand),
        ],
    )
    async def test_predict_chitchat_pattern_commands(
        self,
        command_generator: NLUCommandAdapter,
        pattern: str,
        expected_command_class: Optional[Type[Command]],
    ):
        """Test whether other patterns can be made triggerable via intents."""
        sender_id = uuid.uuid4().hex
        test_trigger_intent = "test_trigger_intent"
        domain = FlowSyncImporter.load_default_pattern_flows_domain()
        domain_addon = Domain.from_dict({KEY_INTENTS: [test_trigger_intent]})
        domain = domain.merge(domain_addon)

        flows = FlowSyncImporter.load_default_pattern_flows()
        flows_pattern_overwritten = flows_from_str(
            f"""
                flows:
                  {pattern}:
                    description: overwritten pattern
                    nlu_trigger:
                      - intent: {test_trigger_intent}
                    steps:
                    - id: first_step
                      action: action_listen
                """
        )
        flows = flows_pattern_overwritten.merge(flows)

        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)

        with patch(
            "rasa.dialogue_understanding.processor.command_processor.clean_up_chitchat_command"
        ) as mock_clean_up_chitchat_command:
            mock_clean_up_chitchat_command.return_value = [ChitChatAnswerCommand()]

            predicted_commands = await command_generator.predict_commands(
                Message(
                    data={
                        TEXT: f"/{test_trigger_intent}",
                        INTENT: {
                            INTENT_NAME_KEY: test_trigger_intent,
                            PREDICTED_CONFIDENCE_KEY: 1.0,
                        },
                    }
                ),
                flows=flows,
                tracker=tracker,
                domain=domain,
            )

        assert len(predicted_commands) == 1
        assert isinstance(predicted_commands[0], expected_command_class)

    def test_issue_set_slot_commands_considers_slot_of_current_collect_step(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        domain: Domain,
    ):
        sender_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(sender_id, [], slots=domain.slots)

        tracker.update_stack(
            DialogueStack(
                [
                    UserFlowStackFrame(
                        flow_id="test_flow",
                        step_id="test_flow_2_collect_another_slot",
                        frame_id="some-frame-id",
                    ),
                ]
            )
        )

        set_slot_commands = _issue_set_slot_commands(
            Message(
                data={
                    TEXT: "some message",
                }
            ),
            flows=flows,
            tracker=tracker,
            domain=domain,
        )

        assert len(set_slot_commands) == 1
        assert set_slot_commands[0] == SetSlotCommand(
            "another_slot", "some message", SetSlotExtractor.NLU.value
        )

    async def test_convert_nlu_to_commands_adds_commands_to_message_object(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        # Given
        message = Message(
            data={
                TEXT: "some message",
                INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
            }
        )

        # When
        with set_record_commands_and_prompts():
            command_generator.convert_nlu_to_commands(message, tracker, flows, None)

            # Then
        assert message.get(PREDICTED_COMMANDS)[NLUCommandAdapter.__name__] == [
            {"command": "start flow", "flow": "test_flow"}
        ]

    async def test_convert_nlu_to_commands_does_not_add_commands_by_default(
        self,
        command_generator: NLUCommandAdapter,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        # Given
        message = Message(
            data={
                TEXT: "some message",
                INTENT: {INTENT_NAME_KEY: "foo", PREDICTED_CONFIDENCE_KEY: 1.0},
            }
        )

        # When
        command_generator.convert_nlu_to_commands(message, tracker, flows, None)

        # Then
        assert message.get(PREDICTED_COMMANDS) is None

    async def test_process_predict_commands_different_start_flow_names(
        self, command_generator: NLUCommandAdapter, monkeypatch: MonkeyPatch
    ):
        """Test that predict_commands retains only the NLU StartFlow predicted command."""  # noqa: E501
        prior_command = StartFlowCommand("some_flow").as_dict()

        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [prior_command], add_to_output=True)

        assert len(test_message.get(COMMANDS)) == 1
        assert test_message.get(COMMANDS) == [prior_command]

        test_tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])

        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(
            return_value=FlowsList([Flow("some_flow"), Flow("other_flow")])
        )
        nlu_command = StartFlowCommand("other_flow")

        def mock_predict_commands(*args, **kwargs) -> List[Command]:
            return [nlu_command]

        monkeypatch.setattr(
            command_generator, "convert_nlu_to_commands", mock_predict_commands
        )
        monkeypatch.setattr(
            command_generator, "get_startable_flows", mock_startable_flows
        )
        monkeypatch.setattr(
            command_generator, "get_active_flows", mock_get_active_flows
        )

        returned_message = (
            await command_generator.process(
                [test_message],
                flows=FlowsList([Flow("some_flow"), Flow("other_flow")]),
                tracker=test_tracker,
            )
        )[0]

        assert len(returned_message.get(COMMANDS)) == 1
        assert returned_message.get(COMMANDS) == [nlu_command.as_dict()]

    @pytest.mark.parametrize(
        "predicted_command",
        [
            SetSlotCommand(
                "test-slot", "test-value-123", extractor=SetSlotExtractor.NLU.value
            ),
            CorrectSlotsCommand(
                [
                    CorrectedSlot(
                        "test-slot",
                        "test-value-123",
                        filled_by=SetSlotExtractor.NLU.value,
                    )
                ]
            ),
        ],
    )
    async def test_process_predict_commands_same_slot(
        self,
        command_generator: NLUCommandAdapter,
        monkeypatch: MonkeyPatch,
        predicted_command: Command,
    ):
        """Test that predict_commands filters out the LLM SetSlot predicted command."""
        command = SetSlotCommand(
            "test-slot", "test-value", SetSlotExtractor.LLM.value
        ).as_dict()

        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [command], add_to_output=True)

        assert len(test_message.get(COMMANDS)) == 1
        assert test_message.get(COMMANDS) == [command]

        test_tracker = DialogueStateTracker.from_events(
            uuid.uuid4().hex, [], slots=[TextSlot("test-slot", [{"type": "from_text"}])]
        )
        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(return_value=FlowsList([Flow("some_flow")]))

        def mock_predict_commands(*args, **kwargs) -> List[Command]:
            return [predicted_command]

        monkeypatch.setattr(
            command_generator, "convert_nlu_to_commands", mock_predict_commands
        )
        monkeypatch.setattr(
            command_generator, "get_startable_flows", mock_startable_flows
        )
        monkeypatch.setattr(
            command_generator, "get_active_flows", mock_get_active_flows
        )

        returned_message = (
            await command_generator.process(
                [test_message],
                flows=FlowsList([Flow("some_flow")]),
                tracker=test_tracker,
            )
        )[0]

        assert len(returned_message.get(COMMANDS)) == 1
        assert returned_message.get(COMMANDS) == [predicted_command.as_dict()]
