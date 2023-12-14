import uuid
from pathlib import Path
from typing import Optional, Dict, Text, Any
from unittest.mock import Mock, patch

import pytest
from _pytest.tmpdir import TempPathFactory
from structlog.testing import capture_logs

from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    SkipQuestionCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
)
from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps.collect import (
    SlotRejection,
    CollectInformationFlowStep,
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
from rasa.shared.utils.llm import DEFAULT_MAX_USER_INPUT_CHARACTERS
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

    @pytest.fixture(scope="session")
    def resource(self) -> Resource:
        return Resource(uuid.uuid4().hex)

    @pytest.fixture(scope="session")
    def model_storage(self, tmp_path_factory: TempPathFactory) -> ModelStorage:
        return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))

    async def test_llm_command_generator_prompt_init_custom(
        self,
        model_storage: ModelStorage,
    ) -> None:
        generator = LLMCommandGenerator(
            {
                "prompt": "data/test_prompt_templates/test_prompt.jinja2",
            },
            model_storage,
            Resource("llmcmdgen"),
        )
        assert generator.prompt_template.startswith("This is a test prompt.")

        resource = generator.train([])
        loaded = LLMCommandGenerator.load({}, model_storage, resource, None)
        assert loaded.prompt_template.startswith("This is a test prompt.")

    async def test_llm_command_generator_prompt_init_default(
        self,
        model_storage: ModelStorage,
    ) -> None:
        generator = LLMCommandGenerator({}, model_storage, Resource("llmcmdgen"))
        assert generator.prompt_template.startswith(
            "Your task is to analyze the current conversation"
        )
        assert (
            generator.user_input_config.max_characters
            == DEFAULT_MAX_USER_INPUT_CHARACTERS
        )

        resource = generator.train([])
        loaded = LLMCommandGenerator.load({}, model_storage, resource, None)
        assert loaded.prompt_template.startswith(
            "Your task is to analyze the current conversation"
        )

    @pytest.mark.parametrize(
        "config, expected_limit",
        [
            ({"user_input": {"max_characters": 100}}, 100),
            ({"user_input": {"max_characters": -1}}, -1),
            (
                {"user_input": {"max_characters": None}},
                DEFAULT_MAX_USER_INPUT_CHARACTERS,
            ),
            ({"user_input": None}, DEFAULT_MAX_USER_INPUT_CHARACTERS),
            ({"user_input": {}}, DEFAULT_MAX_USER_INPUT_CHARACTERS),
        ],
    )
    def test_llm_command_generator_init_with_message_length_limit(
        self,
        config: Dict[Text, Any],
        expected_limit: Optional[int],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        generator = LLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        assert generator.user_input_config.max_characters == expected_limit

    def test_predict_commands_with_no_flows(
        self, command_generator: LLMCommandGenerator
    ):
        """Test that predict_commands returns an empty list when flows is None."""
        # Given
        empty_flows = FlowsList(underlying_flows=[])
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
            "max_tokens": 256,
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
                  collect: test_slot
            """
        )
        with open(EXPECTED_PROMPT_PATH, "r", encoding="unicode_escape") as f:
            expected_template = f.readlines()
        # When
        rendered_template = command_generator.render_template(
            message=test_message, tracker=test_tracker, flows=test_flows
        )
        # Then
        for rendered_line, expected_line in zip(
            rendered_template.splitlines(True), expected_template
        ):
            assert rendered_line == expected_line

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, [ErrorCommand()]),
            (
                "SetSlot(transfer_money_amount_of_money, )",
                [SetSlotCommand(name="transfer_money_amount_of_money", value=None)],
            ),
            ("SetSlot(flow_name, some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(does_not_exist)", []),
            (
                "StartFlow(02_benefits_learning_days)",
                [StartFlowCommand(flow="02_benefits_learning_days")],
            ),
            ("CancelFlow()", [CancelFlowCommand()]),
            ("ChitChat()", [ChitChatAnswerCommand()]),
            ("SkipQuestion()", [SkipQuestionCommand()]),
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
            (
                "Here is a list of commands:\nSetSlot(flow_name, some_flow)\n",
                [StartFlowCommand(flow="some_flow")],
            ),
            (
                """SetSlot(flow_name, some_flow)
                   SetSlot(transfer_money_amount_of_money,)""",
                [
                    StartFlowCommand(flow="some_flow"),
                    SetSlotCommand(name="transfer_money_amount_of_money", value=None),
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
        test_flows = flows_from_str(
            """
            flows:
              some_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
              02_benefits_learning_days:
                description: some foo
                steps:
                - id: some_id
                  collect: some_slot
            """
        )
        with patch.object(
            LLMCommandGenerator, "get_nullable_slot_value", Mock(return_value=None)
        ):
            parsed_commands = LLMCommandGenerator.parse_commands(
                input_action, Mock(), test_flows
            )
        # Then
        assert parsed_commands == expected_command

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
        slot_value = LLMCommandGenerator.get_slot_value(tracker, slot_name)

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
            collect="test_slot",
            idx=0,
            ask_before_filling=True,
            utter="hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            custom_id="collect",
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
            collect="test_slot",
            ask_before_filling=False,
            utter="hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            custom_id="collect_information",
            idx=0,
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

    @pytest.mark.parametrize(
        "message, max_characters, expected_exceeds_limit",
        [
            ("Hello", 5, False),
            ("Hello! I'm a long message", 3, True),
            ("Hello! I'm a long message", -1, False),
        ],
    )
    def test_check_if_message_exceeds_limit(
        self,
        message: Text,
        max_characters: int,
        expected_exceeds_limit: bool,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # Given
        generator = LLMCommandGenerator(
            {"user_input": {"max_characters": max_characters}},
            model_storage,
            resource,
        )
        message = Message.build(text=message)
        # When
        exceeds_limit = generator.check_if_message_exceeds_limit(message)
        assert exceeds_limit == expected_exceeds_limit

    async def test_llm_command_generator_fingerprint_addon_diff_in_prompt_template(
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {"prompt": str(prompt_file)}
        generator = LLMCommandGenerator(config, model_storage, Resource("llmcmdgen"))
        fingerprint_1 = generator.fingerprint_addon(config)

        prompt_file.write_text("This is a test prompt. It has been changed.")
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 != fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_no_diff_in_prompt_template(
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {"prompt": str(prompt_file)}
        generator = LLMCommandGenerator(config, model_storage, Resource("llmcmdgen"))

        fingerprint_1 = generator.fingerprint_addon(config)
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_default_prompt_template(
        model_storage: ModelStorage,
    ) -> None:
        generator = LLMCommandGenerator({}, model_storage, Resource("llmcmdgen"))
        fingerprint_1 = generator.fingerprint_addon({})
        fingerprint_2 = generator.fingerprint_addon({})
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2
