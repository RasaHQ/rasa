from unittest.mock import Mock, patch

import pytest
from langchain.llms.fake import FakeListLLM
from structlog.testing import capture_logs

from rasa.dialogue_understanding.generator.llm_command_generator import LLMCommandGenerator
from rasa.dialogue_understanding.commands import (
    # Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.slots import BooleanSlot, FloatSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker


class TestLLMCommandGenerator:
    """Tests for the LLMCommandGenerator."""

    @pytest.fixture
    def command_generator(self):
        """Create an LLMCommandGenerator."""
        return LLMCommandGenerator.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock())
    
    @pytest.fixture
    def mock_command_generator(
        self,
        default_model_storage: ModelStorage,
        default_execution_context: ExecutionContext,
    ) -> LLMCommandGenerator:
        """Create a patched LLMCommandGenerator."""
        with patch(
            "rasa.dialogue_understanding.generator.llm_command_generator.llm_factory",
            Mock(return_value=FakeListLLM(responses=["StartFlow(check_balance)"])),
        ) as mock_llm:
            return LLMCommandGenerator.create(
                config=LLMCommandGenerator.get_default_config(),
                model_storage=default_model_storage,
                resource=Resource("llmcommandgenerator"),
                execution_context=default_execution_context)

    def test_predict_commands_with_no_flows(self, mock_command_generator: LLMCommandGenerator):
        """Test that predict_commands returns an empty list when flows is None."""
        # When
        predicted_commands = mock_command_generator.predict_commands(Mock(), flows=None, tracker=Mock())
        # Then
        assert not predicted_commands

    def test_predict_commands_with_no_tracker(self, mock_command_generator: LLMCommandGenerator):
        """Test that predict_commands returns an empty list when tracker is None."""
        # When
        predicted_commands = mock_command_generator.predict_commands(Mock(), flows=Mock(), tracker=None)
        # Then
        assert not predicted_commands

    @patch.object(LLMCommandGenerator, "render_template", Mock(return_value="some prompt"))
    @patch.object(LLMCommandGenerator, "parse_commands", Mock())
    def test_predict_commands_calls_llm_correctly(self, command_generator: LLMCommandGenerator):
        """Test that predict_commands calls llm correctly."""
        # When
        mock_llm = Mock()
        with patch(
            "rasa.dialogue_understanding.generator.llm_command_generator.llm_factory",
            Mock(return_value=mock_llm),
        ):
            command_generator.predict_commands(Mock(), flows=Mock(), tracker=Mock())
        # Then
            mock_llm.assert_called_once_with("some prompt")

    @patch.object(LLMCommandGenerator, "render_template", Mock(return_value="some prompt"))
    @patch.object(LLMCommandGenerator, "parse_commands", Mock())
    def test_generate_action_list_catches_llm_exception(self, command_generator: LLMCommandGenerator):
        """Test that predict_commands calls llm correctly."""
        # Given
        mock_llm = Mock(side_effect=Exception("some exception"))
        with patch(
            "rasa.dialogue_understanding.generator.llm_command_generator.llm_factory",
            Mock(return_value=mock_llm),
        ):
        # When
            with capture_logs() as logs:
                command_generator.predict_commands(Mock(), flows=Mock(), tracker=Mock())
        # Then
            print(logs)
            assert len(logs) == 4
            assert isinstance(logs[1]["error"]) == isinstance(Exception("some exception"))



    def test_render_template(self, mock_command_generator: LLMCommandGenerator):
        """Test that render_template renders a template."""
        pass
        # # Given
        # message = Mock()

        # tracker = Mock()

        # flows = Mock()
        # # When
        # rendered_template = command_generator.render_template()

        # # Then
        # assert rendered_template == "template"

    # def test_generate_action_list_calls_llm_with_correct_promt(self):
    #     # Given
    #     prompt = "some prompt"
    #     with patch(
    #         "rasa.rasa.shared.utils.llm.llm_factory",
    #         Mock(return_value=FakeListLLM(responses=["hello"]))
    #     ) as mock_llm:
    #         LLMCommandGenerator._generate_action_list(prompt)
    #         mock_llm.assert_called_once_with(prompt)
    
    @pytest.mark.parametrize(
            "input_action, expected_command",
            [
                (
                    None,
                    [ErrorCommand()]
                ),
                (
                    "SetSlot(transfer_money_amount_of_money, )",
                    [SetSlotCommand(name="transfer_money_amount_of_money", value=None)]
                ),
                (
                    "SetSlot(flow_name, some_flow)",
                    [StartFlowCommand(flow="some_flow")]
                ),
                (
                    "StartFlow(check_balance)",
                    [StartFlowCommand(flow="check_balance")]
                ),
                (
                    "CancelFlow()",
                    [CancelFlowCommand()]
                ),
                (
                    "ChitChat()",
                    [ChitChatAnswerCommand()]
                ),
                (
                    "SearchAndReply()",
                    [KnowledgeAnswerCommand()]
                ),
                (
                    "HumanHandoff()", 
                    [HumanHandoffCommand()]
                ),
                (
                    "Clarify(transfer_money)",
                    [ClarifyCommand(options=["transfer_money"])]
                ),
                (
                    "Clarify(list_contacts, add_contact, remove_contact)",
                    [ClarifyCommand(options=["list_contacts", "add_contact", "remove_contact"])]
                ),
            ])
    def test_parse_commands_identifies_correct_command(self, input_action, expected_command):
        """Test that parse_commands identifies the correct commands."""
        # When
        with patch.object(LLMCommandGenerator, "coerce_slot_value", Mock(return_value=None)):
            parsed_commands = LLMCommandGenerator.parse_commands(input_action, Mock())
        # Then
        assert parsed_commands == expected_command

    @pytest.mark.parametrize(
            "slot_name, slot, slot_value, expected_coerced_value",
            [
                ("some_other_slot", FloatSlot("some_float", []), None, None),
                ("some_float", FloatSlot("some_float", []), 40, 40.0),
                ("some_float", FloatSlot("some_float", []), 40.0, 40.0),
                ("some_text", TextSlot("some_text", []),"fourty", "fourty"),
                ("some_bool", BooleanSlot("some_bool", []), "True", True),
                ("some_bool", BooleanSlot("some_bool", []), "false", False)
            ])
    def test_coerce_slot_value(self, slot_name, slot, slot_value, expected_coerced_value):
        """Test that coerce_slot_value coerces the slot value correctly."""
        # Given
        tracker = DialogueStateTracker.from_events(
            "test",
            evts=[],
            slots=[slot]
        )
        # When
        coerced_value = LLMCommandGenerator.coerce_slot_value(slot_value, slot_name, tracker)
        # Then
        assert coerced_value == expected_coerced_value

    @pytest.mark.parametrize(
            "input_string, expected_string",
            [
                ("text", "text"),
                (" text ", "text"),
                ("\"text\"", "text"),
                ("'text'", "text"),
                ("' \"text' \"  ", "text"),
                ("", "")
            ])
    def test_clean_extracted_value(self, input_string, expected_string):
        """Test that clean_extracted_value removes the leading and trailing whitespaces."""
        # When
        cleaned_extracted_value =  LLMCommandGenerator.clean_extracted_value(input_string)
        # Then
        assert cleaned_extracted_value == expected_string

    
    
    
    
    
    
    
    
    
    
    # def test_allowd_values_for_slot(self, command_generator):
    #     """Test that allowed_values_for_slot returns the allowed values for a slot."""
    #     # When
    #     allowed_values = command_generator.allowed_values_for_slot("slot_name")

    #     # Then
    #     assert allowed_values == []

    # @pytest.mark.parametrize("input_value, expected_truthiness",
    #                          [(None, True),
    #                           ("", False),

    #                           )]
    # def test_is_none_value(self):
    #     """Test that is_none_value returns True when the value is None."""
    #     assert LLMCommandGenerator.is_none_value(None)
