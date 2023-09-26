from typing import Optional
from unittest.mock import Mock, patch

import pytest
from structlog.testing import capture_logs

from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)
from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
)
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows.flow import (
    CollectInformationFlowStep,
    FlowsList,
    SlotRejection,
)
from rasa.shared.core.slots import (
    Slot,
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


EXPECTED_PROMPT_PATH = "./tests/dialogue_understanding/generator/rendered_prompt.txt"


class TestLLMCommandGenerator:
    """Tests for the LLMCommandGenerator."""

    @pytest.fixture
    def command_generator(self):
        """Create an LLMCommandGenerator."""
        return LLMCommandGenerator.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock()
        )

    @pytest.fixture
    def flows(self) -> FlowsList:
        """Create a FlowsList."""
        return flows_from_str(
            """
            flows:
              test_flow:
                steps:
                - id: first_step
                  action: action_listen
            """
        )

    def test_predict_commands_with_no_flows(
        self, command_generator: LLMCommandGenerator
    ):
        """Test that predict_commands returns an empty list when flows is None."""
        # Given
        empty_flows = FlowsList([])
        # When
        predicted_commands = command_generator.predict_commands(
            Mock(), flows=empty_flows, tracker=Mock()
        )
        # Then
        assert not predicted_commands

    def test_predict_commands_with_no_tracker(
        self, command_generator: LLMCommandGenerator
    ):
        """Test that predict_commands returns an empty list when tracker is None."""
        # When
        predicted_commands = command_generator.predict_commands(
            Mock(), flows=Mock(), tracker=None
        )
        # Then
        assert not predicted_commands

    def test_generate_action_list_calls_llm_factory_correctly(
        self,
        command_generator: LLMCommandGenerator,
    ):
        """Test that _generate_action_list calls llm correctly."""
        # Given
        llm_config = {
            "_type": "openai",
            "request_timeout": 7,
            "temperature": 0.0,
            "model_name": "gpt-4",
        }
        # When
        with patch(
            "rasa.dialogue_understanding.generator.llm_command_generator.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            command_generator._generate_action_list_using_llm("some prompt")
            # Then
            mock_llm_factory.assert_called_once_with(None, llm_config)

    def test_generate_action_list_calls_llm_correctly(
        self,
        command_generator: LLMCommandGenerator,
    ):
        """Test that _generate_action_list calls llm correctly."""
        # Given
        with patch(
            "rasa.dialogue_understanding.generator.llm_command_generator.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            mock_llm_factory.return_value = Mock()
            # When
            command_generator._generate_action_list_using_llm("some prompt")
            # Then
            mock_llm_factory.return_value.assert_called_once_with("some prompt")

    def test_generate_action_list_catches_llm_exception(
        self,
        command_generator: LLMCommandGenerator,
    ):
        """Test that _generate_action_list calls llm correctly."""
        # When
        mock_llm = Mock(side_effect=Exception("some exception"))
        with patch(
            "rasa.dialogue_understanding.generator.llm_command_generator.llm_factory",
            Mock(return_value=mock_llm),
        ):
            with capture_logs() as logs:
                command_generator._generate_action_list_using_llm("some prompt")
                # Then
                print(logs)
                assert len(logs) == 1
                assert logs[0]["event"] == "llm_command_generator.llm.error"

    def test_render_template(
        self,
        command_generator: LLMCommandGenerator,
    ):
        """Test that render_template renders the correct template string."""
        # Given
        test_message = Message.build(text="some message")
        test_slot = TextSlot(
            name="test_slot",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[test_slot],
        )
        test_flows = flows_from_str(
            """
            flows:
              test_flow:
                description: some description
                steps:
                - id: first_step
                  collect_information: test_slot
            """
        )
        with open(EXPECTED_PROMPT_PATH, "r", encoding="unicode_escape") as f:
            expected_template = f.read()
        # # When
        rendered_template = command_generator.render_template(
            message=test_message, tracker=test_tracker, flows=test_flows
        )

        # # Then
        assert rendered_template == expected_template

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, [ErrorCommand()]),
            (
                "SetSlot(transfer_money_amount_of_money, )",
                [SetSlotCommand(name="transfer_money_amount_of_money", value=None)],
            ),
            ("SetSlot(flow_name, some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(check_balance)", [StartFlowCommand(flow="check_balance")]),
            ("CancelFlow()", [CancelFlowCommand()]),
            ("ChitChat()", [ChitChatAnswerCommand()]),
            ("SearchAndReply()", [KnowledgeAnswerCommand()]),
            ("HumanHandoff()", [HumanHandoffCommand()]),
            ("Clarify(transfer_money)", [ClarifyCommand(options=["transfer_money"])]),
            (
                "Clarify(list_contacts, add_contact, remove_contact)",
                [
                    ClarifyCommand(
                        options=["list_contacts", "add_contact", "remove_contact"]
                    )
                ],
            ),
        ],
    )
    def test_parse_commands_identifies_correct_command(
        self,
        input_action: Optional[str],
        expected_command: Command,
    ):
        """Test that parse_commands identifies the correct commands."""
        # When
        with patch.object(
            LLMCommandGenerator, "coerce_slot_value", Mock(return_value=None)
        ):
            parsed_commands = LLMCommandGenerator.parse_commands(input_action, Mock())
        # Then
        assert parsed_commands == expected_command

    @pytest.mark.parametrize(
        "slot_name, slot, slot_value, expected_output",
        [
            ("some_other_slot", FloatSlot("some_float", []), None, None),
            ("some_float", FloatSlot("some_float", []), 40, 40.0),
            ("some_float", FloatSlot("some_float", []), 40.0, 40.0),
            ("some_text", TextSlot("some_text", []), "fourty", "fourty"),
            ("some_bool", BooleanSlot("some_bool", []), "True", True),
            ("some_bool", BooleanSlot("some_bool", []), "false", False),
        ],
    )
    def test_coerce_slot_value(
        self,
        slot_name: str,
        slot: Slot,
        slot_value: Optional[str | int | float | bool],
        expected_output: Optional[str | int | float | bool],
    ):
        """Test that coerce_slot_value coerces the slot value correctly."""
        # Given
        tracker = DialogueStateTracker.from_events("test", evts=[], slots=[slot])
        # When
        coerced_value = LLMCommandGenerator.coerce_slot_value(
            slot_value, slot_name, tracker
        )
        # Then
        assert coerced_value == expected_output

    @pytest.mark.parametrize(
        "input_value, expected_output",
        [
            ("text", "text"),
            (" text ", "text"),
            ('"text"', "text"),
            ("'text'", "text"),
            ("' \"text' \"  ", "text"),
            ("", ""),
        ],
    )
    def test_clean_extracted_value(self, input_value: str, expected_output: str):
        """Test that clean_extracted_value removes
        the leading and trailing whitespaces.
        """
        # When
        cleaned_value = LLMCommandGenerator.clean_extracted_value(input_value)
        # Then
        assert cleaned_value == expected_output

    @pytest.mark.parametrize(
        "input_value, expected_truthiness",
        [
            ("", False),
            (" ", False),
            ("none", False),
            ("some text", False),
            ("[missing information]", True),
            ("[missing]", True),
            ("None", True),
            ("undefined", True),
            ("null", True),
        ],
    )
    def test_is_none_value(self, input_value: str, expected_truthiness: bool):
        """Test that is_none_value returns True when the value is None."""
        assert LLMCommandGenerator.is_none_value(input_value) == expected_truthiness

    @pytest.mark.parametrize(
        "slot, slot_name, expected_output",
        [
            (TextSlot("test_slot", [], initial_value="hello"), "test_slot", "hello"),
            (TextSlot("test_slot", []), "some_other_slot", "undefined"),
        ],
    )
    def test_slot_value(self, slot: Slot, slot_name: str, expected_output: str):
        """Test that slot_value returns the correct string."""
        # Given
        tracker = DialogueStateTracker.from_events("test", evts=[], slots=[slot])
        # When
        slot_value = LLMCommandGenerator.slot_value(tracker, slot_name)

        assert slot_value == expected_output

    @pytest.mark.parametrize(
        "input_slot, expected_slot_values",
        [
            (FloatSlot("test_slot", []), None),
            (TextSlot("test_slot", []), None),
            (BooleanSlot("test_slot", []), "[True, False]"),
            (
                CategoricalSlot("test_slot", [], values=["Value1", "Value2"]),
                "['value1', 'value2']",
            ),
        ],
    )
    def test_allowed_values_for_slot(
        self,
        command_generator: LLMCommandGenerator,
        input_slot: Slot,
        expected_slot_values: Optional[str],
    ):
        """Test that allowed_values_for_slot returns the correct values."""
        # When
        allowed_values = command_generator.allowed_values_for_slot(input_slot)
        # Then
        assert allowed_values == expected_slot_values

    @pytest.fixture
    def collect_info_step(self) -> CollectInformationFlowStep:
        """Create a CollectInformationFlowStep."""
        return CollectInformationFlowStep(
            collect_information="test_slot",
            ask_before_filling=True,
            utter="hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            id="collect_information",
            description="test_slot",
            metadata={},
            next="next_step",
        )

    def test_is_extractable_with_no_slot(
        self,
        command_generator: LLMCommandGenerator,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns False
        when there are no slots to be filled.
        """
        # Given
        tracker = DialogueStateTracker.from_events(sender_id="test", evts=[], slots=[])
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert not is_extractable

    def test_is_extractable_when_slot_can_be_filled_without_asking(
        self,
        command_generator: LLMCommandGenerator,
    ):
        """Test that is_extractable returns True when 
        collect_information slot can be filled.
        """
        # Given
        tracker = DialogueStateTracker.from_events(
            sender_id="test", evts=[], slots=[TextSlot(name="test_slot", mappings=[])]
        )
        collect_info_step = CollectInformationFlowStep(
            collect_information="test_slot",
            ask_before_filling=False,
            utter="hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            id="collect_information",
            description="test_slot",
            metadata={},
            next="next_step",
        )
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert is_extractable

    def test_is_extractable_when_slot_has_already_been_set(
        self,
        command_generator: LLMCommandGenerator,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns True
        when collect_information can be filled.
        """
        # Given
        slot = TextSlot(name="test_slot", mappings=[])
        tracker = DialogueStateTracker.from_events(
            sender_id="test", evts=[SlotSet("test_slot", "hello")], slots=[slot]
        )
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert is_extractable

    def test_is_extractable_with_current_step(
        self,
        command_generator: LLMCommandGenerator,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns True when the current step is a collect
        information step and matches the information step.
        """
        # Given
        tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[TextSlot(name="test_slot", mappings=[])],
        )
        # When
        is_extractable = command_generator.is_extractable(
            collect_info_step, tracker, current_step=collect_info_step
        )
        # Then
        assert is_extractable
