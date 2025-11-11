import os.path
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Text
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import structlog
from _pytest.tmpdir import TempPathFactory
from pytest import MonkeyPatch

import rasa.shared.utils.io
from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    Command,
    CorrectedSlot,
    CorrectSlotsCommand,
    ErrorCommand,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.constants import KEY_MINIMIZE_NUM_CALLS
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_ACTIVE_KEY,
    FLOW_RETRIEVAL_FLOW_THRESHOLD,
    FLOW_RETRIEVAL_KEY,
    LLM_BASED_COMMAND_GENERATOR_CONFIG_FILE,
    LLM_CONFIG_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator import (  # noqa: E501
    DEFAULT_COMMAND_PROMPT_TEMPLATE,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import AgentStackFrame
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.llm_fine_tuning.annotation_module import set_preparing_fine_tuning_data
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.constants import SetSlotExtractor
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import AgentCompleted, BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.slots import BooleanSlot, CategoricalSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import (
    COMMANDS,
    KEY_COMPONENT_NAME,
    KEY_USER_PROMPT,
    LLM_COMMANDS,
    LLM_PROMPT,
    PREDICTED_COMMANDS,
    PROMPTS,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage
from rasa.shared.utils.llm import DEFAULT_MAX_USER_INPUT_CHARACTERS
from tests.utilities import filter_logs, flows_from_str

EXPECTED_PROMPT_PATH = "./tests/dialogue_understanding/generator/rendered_prompt.txt"
EXPECTED_RENDERED_FLOW_DESCRIPTION_PATH = (
    "./tests/dialogue_understanding/generator/rendered_flow.txt"
)
PROMPT_TEMPLATE_WITH_CURRENT_SLOT_INFORMATION_PATH = "./tests/dialogue_understanding/generator/prompt_template_with_current_slot_information.jinja2"  # noqa: E501
EXPECTED_RENDERED_PROMPT_WITH_CURRENT_SLOT_INFORMATION = "./tests/dialogue_understanding/generator/rendered_prompt_with_current_slot_information.txt"  # noqa: E501


@pytest.fixture(autouse=True)
def set_mock_openai_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv(
        OPENAI_API_KEY_ENV_VAR, "mock key in test_single_step_llm_command_generator"
    )


@pytest.fixture(autouse=True)
def mock_configuration_available_agents(monkeypatch: MonkeyPatch) -> None:
    """Use empty, but initialised configuration for all tests by default.
    AvailableAgents will be empty unless explicitly set otherwise in a test.
    """
    Configuration.initialise_empty()


class TestSingleStepLLMCommandGenerator:
    """Tests for the SingleStepLLMCommandGenerator."""

    @pytest.fixture
    def command_generator(self):
        """Create an SingleStepLLMCommandGenerator."""
        # Reset the command syntax version.
        CommandSyntaxManager.reset_syntax_version()

        return SingleStepLLMCommandGenerator.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock()
        )

    @pytest.fixture
    def command_generator_with_custom_prompt_template(self):
        """Create an SingleStepLLMCommandGenerator."""
        return SingleStepLLMCommandGenerator.create(
            config={
                PROMPT_TEMPLATE_CONFIG_KEY: PROMPT_TEMPLATE_WITH_CURRENT_SLOT_INFORMATION_PATH  # noqa: E501
            },
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )

    @pytest.fixture
    def flows(self) -> FlowsList:
        """Create a FlowsList."""
        return flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
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

    @pytest.fixture
    def tracker(self):
        """Create a Tracker."""
        return DialogueStateTracker.from_events("", [])

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

    async def test_deprecation_warning_with_prompt(self, model_storage: ModelStorage):
        # Given
        resource = Resource("llmcmdgen")
        config = {PROMPT_CONFIG_KEY: "data/test_prompt_templates/test_prompt.jinja2"}

        # When
        with patch("rasa.shared.utils.llm.structlogger.warning") as mock_warning:
            SingleStepLLMCommandGenerator(
                config,
                model_storage,
                resource,
            )
        mock_warning.assert_any_call(
            "single_step_llm_command_generator.init.deprecated_config_key",
            event_info=(
                "The config parameter 'prompt' is deprecated "
                "and will be removed in Rasa 4.0.0. "
                "Please use the config parameter 'prompt_template' instead. "
            ),
        )

    async def test_prompt_template_handling(self, model_storage: ModelStorage):
        # Given
        resource = Resource("llmcmdgen")
        expected_template = "data/test_prompt_templates/test_prompt.jinja2"
        config = {PROMPT_TEMPLATE_CONFIG_KEY: expected_template}

        # When
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )

        # Then
        assert generator.prompt_template.startswith("This is a test prompt.")

    async def test_default_template_when_no_prompt_template_provided(
        self, model_storage: ModelStorage
    ):
        # Given
        resource = Resource("llmcmdgen")
        config = {}  # No prompt or prompt_template provided

        # When
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )

        # Then
        assert generator.prompt_template == DEFAULT_COMMAND_PROMPT_TEMPLATE

    async def test_single_step_llm_command_generator_init_custom(
        self,
        model_storage: ModelStorage,
    ) -> None:
        # Given
        resource = Resource("llmcmdgen")
        # When
        generator = SingleStepLLMCommandGenerator(
            {
                PROMPT_TEMPLATE_CONFIG_KEY: "data/test_prompt_templates/test_prompt.jinja2",  # noqa: E501
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
            },
            model_storage,
            resource,
        )
        # Then
        assert generator.prompt_template.startswith("This is a test prompt.")
        assert generator.flow_retrieval is None

    async def test_single_step_llm_command_generator_init_default(
        self,
        model_storage: ModelStorage,
    ) -> None:
        # When
        generator = SingleStepLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )
        # Then
        assert generator.prompt_template.startswith(
            "Your task is to analyze the current conversation"
        )
        assert (
            generator.user_input_config.max_characters
            == DEFAULT_MAX_USER_INPUT_CHARACTERS
        )
        assert generator.flow_retrieval is not None

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
    def test_single_step_llm_command_generator_init_with_message_length_limit(
        self,
        config: Dict[Text, Any],
        expected_limit: Optional[int],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        assert generator.user_input_config.max_characters == expected_limit

    async def test_predict_commands_with_no_flows(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands returns an empty list when flows is None."""
        # Given
        empty_flows = FlowsList(underlying_flows=[])
        # When
        predicted_commands = await command_generator.predict_commands(
            Message(), flows=empty_flows, tracker=tracker
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_with_no_tracker(
        self, command_generator: SingleStepLLMCommandGenerator
    ):
        """Test that predict_commands returns an empty list when tracker is None."""
        # When
        predicted_commands = await command_generator.predict_commands(
            Message(), flows=Mock(), tracker=None
        )
        # Then
        assert not predicted_commands

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_sets_routing_slot(
        self,
        mock_llm_factory: Mock,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker_with_routing_slot: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        # Given
        mock_llm_client = AsyncMock()
        llm_response_object.choices = ["StartFlow(test_flow)"]
        mock_llm_client.acompletion.return_value = llm_response_object
        mock_llm_factory.return_value = mock_llm_client

        # When
        predicted_commands = await command_generator.predict_commands(
            Message.build(text="start test_flow"),
            flows=flows,
            tracker=tracker_with_routing_slot,
        )

        # Then
        assert StartFlowCommand("test_flow") in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_does_not_set_llm_commands_and_prompt(
        self,
        mock_llm_factory: Mock,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        # Given
        message = Message.build(text="start test_flow")
        mock_llm_client = AsyncMock()
        llm_response_object.choices = ["StartFlow(test_flow)"]
        mock_llm_client.acompletion.return_value = llm_response_object
        mock_llm_factory.return_value = mock_llm_client

        # When
        await command_generator.predict_commands(
            message,
            flows=flows,
            tracker=tracker,
        )

        # Then
        assert message.get(LLM_PROMPT) is None
        assert message.get(LLM_COMMANDS) is None

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_sets_llm_commands_and_prompt(
        self,
        mock_llm_factory: Mock,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        with set_preparing_fine_tuning_data():
            mock_llm_client = AsyncMock()
            llm_response_object.choices = ["StartFlow(test_flow)"]
            mock_llm_client.acompletion.return_value = llm_response_object
            mock_llm_factory.return_value = mock_llm_client

            # When
            await command_generator.predict_commands(
                message,
                flows=flows,
                tracker=tracker,
            )

        # Then
        assert message.get(LLM_PROMPT) is not None
        assert message.get(LLM_PROMPT).startswith(
            "Your task is to analyze the current conversation context"
        )
        assert message.get(LLM_COMMANDS) == [
            {"command": "start flow", "flow": "test_flow"}
        ]

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_adds_commands_and_prompt_to_message_object(
        self,
        mock_llm_factory: Mock,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        with set_record_commands_and_prompts():
            mock_llm_client = AsyncMock()
            llm_response_object.choices = ["StartFlow(test_flow)"]
            mock_llm_client.acompletion.return_value = llm_response_object
            mock_llm_factory.return_value = mock_llm_client

            # When
            await command_generator.predict_commands(
                message,
                flows=flows,
                tracker=tracker,
            )

        # Then
        prompts = message.get(PROMPTS)
        assert prompts is not None
        assert (
            prompts[0].get(KEY_COMPONENT_NAME) == SingleStepLLMCommandGenerator.__name__
        )
        assert prompts[0][KEY_USER_PROMPT].startswith(
            "Your task is to analyze the current conversation context"
        )
        assert message.get(PREDICTED_COMMANDS)[
            SingleStepLLMCommandGenerator.__name__
        ] == [{"command": "start flow", "flow": "test_flow"}]

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_does_not_add_commands_and_prompt_by_default(
        self,
        mock_llm_factory: Mock,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock

        # When
        await command_generator.predict_commands(
            message,
            flows=flows,
            tracker=tracker,
        )

        # Then
        assert message.get(PROMPTS) is None
        assert message.get(PREDICTED_COMMANDS) is None

    @pytest.mark.parametrize(
        "flow_guard_value, expected_flow_ids",
        (
            (None, {"flow_regular"}),
            (False, {"flow_regular"}),
            (True, {"flow_regular", "flow_with_guard"}),
            ("false", {"flow_regular", "flow_with_guard"}),
            ("true", {"flow_regular", "flow_with_guard"}),
        ),
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator"
        ".SingleStepLLMCommandGenerator"
        ".render_template"
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator"
        ".SingleStepLLMCommandGenerator"
        ".invoke_llm"
    )
    async def test_predict_commands_calls_prompt_rendering_with_startable_flows_only(
        self,
        mock_generate_action_list_using_llm: Mock,
        mock_render_template: Mock,
        flow_guard_value: Any,
        expected_flow_ids: Set[Text],
        command_generator: SingleStepLLMCommandGenerator,
    ):
        # Given
        test_flows = flows_from_str(
            """
            flows:
                flow_with_guard:
                    if: slots.some_slot
                    name: flow with guard
                    description: description for flow a
                    steps:
                        - id: step_a
                          action: action_listen
                flow_link:
                    if: False
                    name: flow link
                    description: description for flow b
                    steps:
                        - id: step_b
                          action: action_listen
                flow_regular:
                    name: flow regular
                    description: description for flow b
                    steps:
                        - id: step_c
                          action: action_listen
            """
        )
        tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[SlotSet(key="some_slot", value=flow_guard_value)],
        )
        mock_message = Message()
        mock_message.data = {TEXT: "some_message"}
        mock_render_template.return_value = "some rendered template"
        # regardless of flow retrieval we want to make sure we are calling the
        # prompt rendering only with startable flows.
        config = {"flow_retrieval": {"active": False}}
        command_generator = SingleStepLLMCommandGenerator.create(
            config=config,
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )
        # the return value doesn't matter
        mock_generate_action_list_using_llm.return_value = None

        # When
        await command_generator.predict_commands(
            message=mock_message, flows=test_flows, tracker=tracker
        )
        mock_render_template.assert_called_once()
        filtered_flows = mock_render_template.call_args.args[2]
        all_available_flows = mock_render_template.call_args.args[3]

        # Then
        assert filtered_flows.flow_ids == expected_flow_ids
        assert all_available_flows.flow_ids == test_flows.flow_ids

    @pytest.mark.parametrize(
        "llm_response, expected_commands",
        [
            (
                LLMResponse.from_dict({"id": None, "choices": None, "created": None}),
                [ErrorCommand()],
            ),
            (
                LLMResponse(
                    id="mock-id",
                    created=123456,
                    choices=["StartFlow(this_flow_does_not_exists)"],
                    model="test-model",
                    usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
                ),
                [
                    CannotHandleCommand(),
                ],
            ),
            (
                LLMResponse(
                    id="mock-id",
                    created=123456,
                    choices=["A random response from LLM"],
                    model="test-model",
                    usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
                ),
                [
                    CannotHandleCommand(),
                ],
            ),
            (
                LLMResponse(
                    id="mock-id",
                    created=123456,
                    choices=["SetSlot(flow_name, some_flow)"],
                    model="test-model",
                    usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
                ),
                [
                    StartFlowCommand(flow="some_flow"),
                ],
            ),
        ],
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator."
        "SingleStepLLMCommandGenerator.invoke_llm"
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator."
        "SingleStepLLMCommandGenerator.render_template"
    )
    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows"
    )
    async def test_predict_commands(
        self,
        mock_flow_retrieval_filter_flows: Mock,
        mock_render_template: Mock,
        mock_generate_action_list_using_llm: Mock,
        llm_response: Text,
        expected_commands: List[Command],
        command_generator: SingleStepLLMCommandGenerator,
        tracker_with_routing_slot: DialogueStateTracker,
    ):
        # Given
        test_flows = flows_from_str(
            """
            flows:
              some_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
            """
        )
        mock_render_template.return_value = "some_template"
        mock_generate_action_list_using_llm.return_value = llm_response
        mock_flow_retrieval_filter_flows.return_value = FlowsList(underlying_flows=[])
        mock_message = Message()
        mock_message.data = {TEXT: "some_message"}
        # When
        predicted_commands = await command_generator.predict_commands(
            message=mock_message, flows=test_flows, tracker=tracker_with_routing_slot
        )
        # Then
        mock_flow_retrieval_filter_flows.assert_called_once()
        assert len(predicted_commands) == len(expected_commands) + 1
        for expected_command in expected_commands:
            assert expected_command in predicted_commands

        # route session must be present when there is a
        # tracker with routing slot
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows"
    )
    async def test_predict_commands_and_flow_retrieval_api_error_throws_exception(
        self,
        mock_flow_retrieval_filter_flows: AsyncMock,
        command_generator: SingleStepLLMCommandGenerator,
        tracker_with_routing_slot: DialogueStateTracker,
    ) -> None:
        # Given
        test_flows = flows_from_str(
            """
            flows:
              some_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
            """
        )
        mock_message = Message()
        mock_message.data = {TEXT: "some_message"}
        mock_flow_retrieval_filter_flows.side_effect = ProviderClientAPIException(
            message="Test Exception", original_exception=Exception("API exception")
        )
        # When
        predicted_commands = await command_generator.predict_commands(
            message=mock_message,
            flows=test_flows,
            tracker=tracker_with_routing_slot,
        )

        # Then
        mock_flow_retrieval_filter_flows.assert_called_once()

        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    def test_render_template(
        self,
        command_generator: SingleStepLLMCommandGenerator,
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
            expected_template = f.read()
        # When
        rendered_template = command_generator.render_template(
            message=test_message,
            tracker=test_tracker,
            startable_flows=test_flows,
            all_flows=test_flows,
        )
        # Then
        assert rendered_template == expected_template

    def test_render_template_with_current_slot_info(
        self,
        command_generator_with_custom_prompt_template: SingleStepLLMCommandGenerator,
    ):
        """Test that rendered template includes information about the current slot
        type and allowed values if available.
        """
        # Given
        test_message = Message.build(text="some message")
        test_slot = CategoricalSlot(
            name="test_slot",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
            values=["A", "B"],
        )
        stack = DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "test_flow",
                    "step_id": "first_step",
                    "frame_id": "some-frame-id",
                },
            ]
        )
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[test_slot],
        )
        test_tracker.update_stack(stack)
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
        with open(
            EXPECTED_RENDERED_PROMPT_WITH_CURRENT_SLOT_INFORMATION,
            "r",
            encoding="unicode_escape",
        ) as f:
            expected_template = f.readlines()
        # When
        rendered_template = (
            command_generator_with_custom_prompt_template.render_template(
                message=test_message,
                tracker=test_tracker,
                startable_flows=test_flows,
                all_flows=test_flows,
            )
        )
        # Then
        for rendered_line, expected_line in zip(
            rendered_template.splitlines(True), expected_template
        ):
            assert rendered_line == expected_line

    def test_render_template_call(
        self,
        command_generator: SingleStepLLMCommandGenerator,
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
        stack = DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "test_flow",
                    "step_id": "call_step",
                    "frame_id": "some-frame-id",
                },
                {
                    "type": "flow",
                    "frame_type": "call",
                    "frame_id": "some-other-id",
                    "step_id": "first_step",
                    "flow_id": "called_flow",
                },
            ]
        )
        test_flows = flows_from_str(
            """
            flows:
              called_flow:
                if: False
                description: a flows that's called
                steps:
                - id: first_step
                  collect: test_slot
              test_flow:
                description: some description
                steps:
                - id: call_step
                  call: called_flow
            """
        )
        startable_test_flows = test_flows.exclude_link_only_flows()
        test_tracker.update_stack(stack)
        # When
        rendered_template = command_generator.render_template(
            message=test_message,
            tracker=test_tracker,
            startable_flows=startable_test_flows,
            all_flows=test_flows,
        )
        # Then
        # make sure non-startable flow isn't there
        assert "called_flow" not in rendered_template
        # make sure it looks like we are in the calling flow
        assert 'You are currently in the flow "test_flow".' in rendered_template
        # make sure the slot from the called flow is available in the template
        assert (
            'You have just asked the user for the slot "test_slot".'
            in rendered_template
        )

    @pytest.mark.parametrize(
        "agents_present",
        [
            True,
            False,
        ],
    )
    def test_render_template_agent_inputs_minimal_template(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        monkeypatch: MonkeyPatch,
        agents_present: bool,
    ) -> None:
        # Toggle agents presence
        mock_available_agents = Mock()
        mock_available_agents.agents = (
            {"test-agent": Mock(), "test-agent-2": Mock()} if agents_present else {}
        )
        mock_available_agents.has_agents.return_value = agents_present

        mock_configuration_instance = Mock()
        mock_configuration_instance.available_agents = mock_available_agents
        monkeypatch.setattr(
            "rasa.core.config.configuration.Configuration.get_instance",
            Mock(return_value=mock_configuration_instance),
        )

        # Use a minimal prompt that checks for variables being defined
        command_generator.prompt_template = (
            "{% if active_agent %}HAS_ACTIVE{% endif %}"
            "{% if completed_agents %}HAS_COMPLETED{% endif %}"
            "User input: {{ current_conversation }}"
        )

        # Prepare tracker/flows
        test_message = Message.build(text="hi")
        test_tracker = DialogueStateTracker.from_events(
            "sender", [AgentCompleted("test-agent-2", "test_flow")]
        )
        if agents_present:
            from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
            from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
                FlowStackFrameType,
            )

            # Create a user flow frame to represent the active flow
            user_flow_frame = UserFlowStackFrame(
                frame_id="user_flow_frame",
                flow_id="test_flow",
                step_id="START",
                frame_type=FlowStackFrameType.REGULAR,
            )

            stack = DialogueStack(
                frames=[
                    user_flow_frame,
                    AgentStackFrame(
                        flow_id="test_flow",
                        step_id="START",
                        frame_id="some-frame-id",
                        agent_id="test-agent",
                    ),
                ]
            )
            test_tracker.update_stack(stack)
        test_flows = flows_from_str(
            """
            flows:
              test_flow:
                description: some description
                steps:
                  - id: step
                    action: action_listen
            """
        )

        # When
        rendered = command_generator.render_template(
            message=test_message,
            tracker=test_tracker,
            startable_flows=test_flows,
            all_flows=test_flows,
        )

        # Then: agent fields are conditionally present
        assert ("HAS_ACTIVE" in rendered) is agents_present
        assert ("HAS_COMPLETED" in rendered) is agents_present

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, []),
            (
                "SetSlot(transfer_money_amount_of_money, )",
                [SetSlotCommand(name="transfer_money_amount_of_money", value=None)],
            ),
            ("SetSlot(name, value)", [SetSlotCommand(name="name", value="value")]),
            ("SetSlot('name', 'value')", [SetSlotCommand(name="name", value="value")]),
            ('SetSlot("name", "value")', [SetSlotCommand(name="name", value="value")]),
            # Start flow
            ("SetSlot(flow_name, some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow('some_flow')", [StartFlowCommand(flow="some_flow")]),
            ('StartFlow("some_flow")', [StartFlowCommand(flow="some_flow")]),
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
            # Clarify of non-existent option is dropped
            ("Clarify(transfer_money)", []),
            # Clarify orders options
            (
                "Clarify(some_flow, 02_benefits_learning_days)",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            # Clarify with quotes around the flow names
            (
                "Clarify('some_flow', 'another_flow')",
                [ClarifyCommand(options=["another_flow", "some_flow"])],
            ),
            (
                'Clarify("some_flow", "another_flow")',
                [ClarifyCommand(options=["another_flow", "some_flow"])],
            ),
            # Clarify with single option is converted to a StartFlowCommand
            ("Clarify(some_flow)", [StartFlowCommand(flow="some_flow")]),
            # Clarify with multiple but same options is converted to a StartFlowCommand
            (
                "Clarify(some_flow, some_flow, some_flow, some_flow)",
                [StartFlowCommand(flow="some_flow")],
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
              another_flow:
                description: some other description
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
        parsed_commands = SingleStepLLMCommandGenerator.parse_commands(
            input_action, Mock(), test_flows
        )
        # Then
        assert parsed_commands == expected_command

    async def test_llm_command_generator_fingerprint_addon_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {PROMPT_TEMPLATE_CONFIG_KEY: str(prompt_file)}
        generator = SingleStepLLMCommandGenerator(
            config, model_storage, Resource("llmcmdgen")
        )
        fingerprint_1 = generator.fingerprint_addon(config)

        prompt_file.write_text("This is a test prompt. It has been changed.")
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 != fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_no_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {PROMPT_TEMPLATE_CONFIG_KEY: str(prompt_file)}
        generator = SingleStepLLMCommandGenerator(
            config, model_storage, Resource("llmcmdgen")
        )

        fingerprint_1 = generator.fingerprint_addon(config)
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_default_prompt_template(
        self,
        model_storage: ModelStorage,
    ) -> None:
        generator = SingleStepLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )
        fingerprint_1 = generator.fingerprint_addon({})
        fingerprint_2 = generator.fingerprint_addon({})
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    @pytest.mark.parametrize(
        "config_1, model_groups_1, config_2, model_groups_2, fingerprint_differs",
        [
            (
                {"user_input": {"max_characters": 100}},
                [],
                {"user_input": {"max_characters": 200}},
                [],
                False,
            ),
            (
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt"}},
                [
                    {
                        "id": "openai_gpt",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                ],
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt"}},
                [
                    {
                        "id": "openai_gpt",
                        "models": [{"provider": "openai", "model": "gpt-4o"}],
                    },
                ],
                True,
            ),
            (
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-1"}},
                [
                    {
                        "id": "openai_gpt-1",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                ],
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-2"}},
                [
                    {
                        "id": "openai_gpt-2",
                        "models": [{"provider": "openai", "model": "gpt-4o"}],
                    },
                ],
                True,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            MODEL_GROUP_CONFIG_KEY: "openai_embedding"
                        }
                    },
                },
                [
                    {
                        "id": "openai_gpt-4",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                    {
                        "id": "openai_embedding",
                        "models": [{"provider": "openai", "model": "embedding-model"}],
                    },
                ],
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            MODEL_GROUP_CONFIG_KEY: "openai_embedding_2"
                        }
                    },
                },
                [
                    {
                        "id": "openai_gpt-4",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                    {
                        "id": "openai_embedding_2",
                        "models": [
                            {"provider": "openai", "model": "different-embedding-model"}
                        ],
                    },
                ],
                True,
            ),
        ],
    )
    async def test_fingerprint_addon_with_different_model_configs(
        self,
        config_1: Dict[str, Any],
        model_groups_1: List[Dict[str, Any]],
        config_2: Dict[str, Any],
        model_groups_2: List[Dict[str, Any]],
        fingerprint_differs: bool,
        model_storage: ModelStorage,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generator = SingleStepLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        mock_available_endpoints.model_groups = model_groups_1
        fingerprint_1 = generator.fingerprint_addon(config_1)

        mock_available_endpoints.model_groups = model_groups_2
        fingerprint_2 = generator.fingerprint_addon(config_2)

        assert fingerprint_1 is not None
        assert fingerprint_2 is not None
        if fingerprint_differs:
            assert fingerprint_1 != fingerprint_2
        else:
            assert fingerprint_1 == fingerprint_2

    def test_train_with_flow_retrieval_disabled(
        self,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        # When
        with structlog.testing.capture_logs() as caplog:
            generator.train(TrainingData(), flows, Mock())
        # Then
        expected_event = "llm_based_command_generator.flow_retrieval.disabled"
        expected_log_level = "warning"
        logs = filter_logs(caplog, expected_event, expected_log_level, [])
        assert generator.flow_retrieval is None
        assert len(logs) == 0

        new_flows = """
        flows:
        """
        for i in range(1, FLOW_RETRIEVAL_FLOW_THRESHOLD + 2):
            new_flows += f"""
              test_flow_{i}:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  action: action_listen
            """
        generator = SingleStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        # When
        with structlog.testing.capture_logs() as caplog:
            generator.train(TrainingData(), flows_from_str(new_flows), Mock())
        # Then
        expected_event = "llm_based_command_generator.flow_retrieval.disabled"
        expected_log_level = "warning"
        logs = filter_logs(caplog, expected_event, expected_log_level, [])
        assert generator.flow_retrieval is None
        assert len(logs) == 1
        assert (
            "It is recommended to enable flow retrieval if the total "
            "number of user flows exceed "
            + str(FLOW_RETRIEVAL_FLOW_THRESHOLD)
            in logs[0].get("event_info")
        )

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    def test_train_with_flow_retrieval_enabled(
        self,
        mock_flow_search_populate: Mock,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {},
            model_storage,
            resource,
        )
        domain = Mock()
        # When
        with structlog.testing.capture_logs() as caplog:
            generator.train(TrainingData(), flows, domain)
        # Then
        mock_flow_search_populate.assert_called_once_with(flows, domain)
        expected_event = "llm_based_command_generator.flow_retrieval.disabled"
        expected_log_level = "warning"
        logs = filter_logs(caplog, expected_event, expected_log_level, [])
        assert len(logs) == 0

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    def test_train_with_flow_retrieval_enabled_and_api_error_throws_exception(
        self,
        mock_flow_search_populate: Mock,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {},
            model_storage,
            resource,
        )
        domain = Mock()
        mock_flow_search_populate.side_effect = Exception("Test Exception")
        # When
        with pytest.raises(Exception) as exc_info:
            generator.train(TrainingData(), flows, domain)
        # Then
        assert "Test Exception" in str(exc_info.value), "Expected exception not raised"
        mock_flow_search_populate.assert_called_once_with(flows, domain)

    def test_load_with_flow_retrieval_disabled(
        self,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        domain = Mock()
        domain.slots = []
        train_resource = generator.train(TrainingData(), flows, domain)
        # When
        loaded = SingleStepLLMCommandGenerator.load(
            generator.config,
            model_storage,
            train_resource,
            Mock(),
        )
        # Then
        assert loaded is not None
        assert loaded.flow_retrieval is None
        assert not loaded.config[FLOW_RETRIEVAL_KEY][FLOW_RETRIEVAL_ACTIVE_KEY]

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    def test_load_with_flow_retrieval_enabled(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        config = {FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config()}
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        domain = Mock()
        train_resource = generator.train(TrainingData(), flows, domain)
        # When
        loaded = SingleStepLLMCommandGenerator.load(
            generator.config,
            model_storage,
            train_resource,
            Mock(),
        )
        # Then
        mock_flow_retrieval_load.assert_called_once_with(
            config=config[FLOW_RETRIEVAL_KEY],
            model_storage=model_storage,
            resource=resource,
        )
        assert loaded is not None
        assert loaded.flow_retrieval is not None

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    def test_load_with_custom_prompt(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        resource = Resource("llmcmdgen")
        config = {
            PROMPT_TEMPLATE_CONFIG_KEY: os.path.join(
                "data", "test_prompt_templates", "test_prompt.jinja2"
            )
        }
        generator = SingleStepLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # When
        loaded = SingleStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        # Then
        assert loaded.prompt_template.startswith("This is a test prompt.")

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    def test_load_with_default_prompt(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        resource = Resource("llmcmdgen")
        generator = SingleStepLLMCommandGenerator({}, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # When
        loaded = SingleStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        # Then
        assert loaded.prompt_template.startswith(
            "Your task is to analyze the current conversation"
        )

    async def test_single_step_llm_command_generator_load_prompt_from_model_storage(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "some key")
        # Create and write prompt file.
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a custom prompt")

        # Add the prompt file path to the config.
        config = {PROMPT_TEMPLATE_CONFIG_KEY: str(prompt_file)}

        # Persist the prompt file to the model storage.
        resource = Resource("llmcmdgen")
        generator = SingleStepLLMCommandGenerator(config, model_storage, resource)
        generator.persist()

        # Test loading the prompt from the model storage.
        # Case 1: No prompt in the config.
        loaded = SingleStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        assert loaded.prompt_template == "This is a custom prompt"
        assert loaded.config[PROMPT_TEMPLATE_CONFIG_KEY] is None

        # Case 2: Specifying a invalid prompt path in the config.
        loaded = SingleStepLLMCommandGenerator.load(
            {PROMPT_TEMPLATE_CONFIG_KEY: "test_prompt.jinja2"},
            model_storage,
            resource,
            Mock(),
        )
        assert loaded.prompt_template == "This is a custom prompt"
        assert loaded.config[PROMPT_TEMPLATE_CONFIG_KEY] == "test_prompt.jinja2"

    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval")
    def test_train_with_no_flows(
        self,
        mock_flow_retrieval: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        mock_flow_retrieval.__name__ = "FlowRetrieval"
        resource = Resource("llmcmdgen")
        generator = SingleStepLLMCommandGenerator({}, model_storage, resource)
        # When
        generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # Then
        assert mock_flow_retrieval.populate.call_count == 0

    @pytest.mark.parametrize(
        "config, expected_llm_config, expected_flow_retrieval_embedding_config",
        [
            (
                {
                    LLM_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
                    FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
                },
                {"provider": "openai", "model": "gpt-4"},
                None,
            ),
            (
                {
                    "user_input": {"max_characters": -1},
                    FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
                },
                None,
                None,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            MODEL_GROUP_CONFIG_KEY: "openai_embedding"
                        }
                    },
                },
                {
                    "id": "openai_gpt-4",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
                {
                    "id": "openai_embedding",
                    "models": [
                        {"model": "text-embedding-3-large", "provider": "openai"}
                    ],
                },
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            "provider": "openai",
                            "model": "text-embedding-3-large",
                        }
                    },
                },
                {
                    "id": "openai_gpt-4",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
                {"provider": "openai", "model": "text-embedding-3-large"},
            ),
        ],
    )
    def test_single_step_llm_command_generator_init_with_different_llm_configs(
        self,
        config: Optional[Dict[str, Any]],
        expected_llm_config: Optional[Dict[str, Any]],
        expected_flow_retrieval_embedding_config: Optional[Dict[str, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: MonkeyPatch,
    ) -> None:
        mock_available_endpoints.model_groups = [
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
            {
                "id": "openai_embedding",
                "models": [{"provider": "openai", "model": "text-embedding-3-large"}],
            },
        ]

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        assert generator.config[LLM_CONFIG_KEY] == expected_llm_config

        if expected_flow_retrieval_embedding_config is None:
            assert EMBEDDINGS_CONFIG_KEY not in generator.config[FLOW_RETRIEVAL_KEY]
        else:
            assert (
                generator.config[FLOW_RETRIEVAL_KEY][EMBEDDINGS_CONFIG_KEY]
                == expected_flow_retrieval_embedding_config
            )

    def test_single_step_llm_command_generator_persist_config(
        self,
        model_storage: LocalModelStorage,
        resource: Resource,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_available_endpoints.model_groups = [
            {
                "id": "model_group_id",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            }
        ]

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        config = {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "model_group_id"}}
        generator = SingleStepLLMCommandGenerator(config, model_storage, resource)

        # Ensure the config is resolved
        assert generator.config[LLM_CONFIG_KEY] == {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }

        # Persist the generator
        generator.persist()

        # Check that the persisted config is equal to our config
        with model_storage.read_from(resource) as path:
            persisted_config = rasa.shared.utils.io.read_json_file(
                path / LLM_BASED_COMMAND_GENERATOR_CONFIG_FILE
            )

        assert persisted_config[LLM_CONFIG_KEY] == {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }

    async def test_process_predict_commands_if_commands_already_present(
        self, command_generator: SingleStepLLMCommandGenerator, monkeypatch: MonkeyPatch
    ):
        """Test that predict_commands adds commands to the prior set commands on the Message object."""  # noqa: E501
        command_generator.config[KEY_MINIMIZE_NUM_CALLS] = False

        command = StartFlowCommand("some_flow").as_dict()

        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [command], add_to_output=True)

        assert len(test_message.get(COMMANDS)) == 1
        assert test_message.get(COMMANDS) == [command]

        test_tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])

        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(return_value=FlowsList([Flow("some_flow")]))

        async def mock_predict_commands(*args, **kwargs) -> List[Command]:
            return [SetSlotCommand("some slot", "some value")]

        monkeypatch.setattr(
            command_generator, "_predict_commands", mock_predict_commands
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

        assert len(returned_message.get(COMMANDS)) == 2
        assert returned_message.get(COMMANDS) == [
            command,
            SetSlotCommand("some slot", "some value").as_dict(),
        ]

    @pytest.mark.parametrize(
        "command",
        [
            StartFlowCommand("some_flow").as_dict(),
            SetSlotCommand("some_slot", "some_value").as_dict(),
        ],
    )
    async def test_process_should_skip_llm_call(
        self,
        command: Dict[str, Any],
        command_generator: SingleStepLLMCommandGenerator,
        monkeypatch: MonkeyPatch,
    ):
        """Test that predict_commands does not add commands when should_skip_llm_call is True."""  # noqa: E501
        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [command], add_to_output=True)

        assert len(test_message.get(COMMANDS)) == 1
        assert test_message.get(COMMANDS) == [command]

        mock_predict_commands = AsyncMock()
        monkeypatch.setattr(
            command_generator,
            "_predict_commands",
            mock_predict_commands,
        )

        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(return_value=FlowsList([Flow("some_flow")]))
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
                tracker=None,
            )
        )[0]

        assert len(returned_message.get(COMMANDS)) == 1
        assert returned_message.get(COMMANDS) == [command]

    @pytest.mark.parametrize(
        "active_flow, input_commands, expected_commands",
        [
            (
                "auth_user",
                [SetSlotCommand("auth_token", "ABCD12EF", SetSlotExtractor.LLM.value)],
                [SetSlotCommand("auth_token", "ABCD12EF", SetSlotExtractor.LLM.value)],
            ),
            (
                "auth_user_2",
                [SetSlotCommand("auth_token", "ABCD12EF", SetSlotExtractor.LLM.value)],
                [],
            ),
            (
                "loyalty_points",
                [StartFlowCommand("loyalty_points")],
                [StartFlowCommand("loyalty_points")],
            ),
            (
                "some_flow",
                [SetSlotCommand("nlu_slot", "some_value", SetSlotExtractor.NLU.value)],
                [SetSlotCommand("nlu_slot", "some_value", SetSlotExtractor.NLU.value)],
            ),
        ],
    )
    def test_command_generator_check_commands_against_slot_mappings_active_flow(
        self,
        active_flow: Text,
        input_commands: List[Command],
        expected_commands: List[Command],
        command_generator: SingleStepLLMCommandGenerator,
    ):
        # Given
        slot_name = "auth_token"
        flow_id = "auth_user"
        domain = Domain.from_yaml(f"""
        entities:
        - nlu_entity
        slots:
          {slot_name}:
            type: text
            mappings:
                - type: from_llm
                  conditions:
                    - active_flow: {flow_id}
          nlu_slot:
            type: text
            mappings:
            - type: from_entity
              entity: nlu_entity
        """)
        tracker = DialogueStateTracker.from_events("test", [], slots=domain.slots)
        user_frame = UserFlowStackFrame(
            flow_id=active_flow, step_id="first_step", frame_id="some-frame-id"
        )
        stack = DialogueStack(frames=[user_frame])
        tracker.update_stack(stack)

        # When
        actual_commands = command_generator._check_commands_against_slot_mappings(
            input_commands, tracker, domain
        )

        # Then
        assert actual_commands == expected_commands

    async def test_process_predict_commands_different_start_flow_names(
        self, command_generator: SingleStepLLMCommandGenerator, monkeypatch: MonkeyPatch
    ):
        """Test that predict_commands filters out the LLM StartFlow predicted command."""  # noqa: E501
        command = StartFlowCommand("some_flow").as_dict()

        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [command], add_to_output=True)

        assert len(test_message.get(COMMANDS)) == 1
        assert test_message.get(COMMANDS) == [command]

        test_tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])

        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(
            return_value=FlowsList([Flow("some_flow"), Flow("other_flow")])
        )

        async def mock_predict_commands(*args, **kwargs) -> List[Command]:
            return [StartFlowCommand("other_flow")]

        monkeypatch.setattr(
            command_generator, "_predict_commands", mock_predict_commands
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
        assert returned_message.get(COMMANDS) == [command]

    @pytest.mark.parametrize(
        "predicted_command",
        [
            SetSlotCommand("test-slot", "test-value-123"),
            CorrectSlotsCommand([CorrectedSlot("test-slot", "test-value-123")]),
        ],
    )
    async def test_process_predict_commands_same_slot(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        monkeypatch: MonkeyPatch,
        predicted_command: Command,
    ):
        """Test that predict_commands filters out the LLM SetSlot predicted command."""
        command = SetSlotCommand(
            "test-slot", "test-value", SetSlotExtractor.NLU.value
        ).as_dict()

        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [command], add_to_output=True)

        assert len(test_message.get(COMMANDS)) == 1
        assert test_message.get(COMMANDS) == [command]

        test_tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])

        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(return_value=FlowsList([Flow("some_flow")]))

        async def mock_predict_commands(*args, **kwargs) -> List[Command]:
            return [predicted_command]

        monkeypatch.setattr(
            command_generator, "_predict_commands", mock_predict_commands
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
        assert returned_message.get(COMMANDS) == [command]

    def test_command_syntax_version(self):
        assert CommandSyntaxManager.get_syntax_version() == CommandSyntaxVersion.v1
        CommandSyntaxManager.reset_syntax_version()

    async def test_process_keeps_start_flow(
        self, command_generator: SingleStepLLMCommandGenerator, monkeypatch: MonkeyPatch
    ):
        """Test that predict_commands does not filter out the LLM StartFlow predicted command."""  # noqa: E501
        command = SetSlotCommand(name="test_slot", value="test_value").as_dict()

        test_message = Message.build(text="some message")
        test_message.set(COMMANDS, [command], add_to_output=True)

        test_tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])

        mock_get_active_flows = Mock(return_value=FlowsList([]))
        mock_startable_flows = Mock(return_value=FlowsList([Flow("some_flow")]))

        llm_command = [StartFlowCommand("some_flow")]

        async def mock_predict_commands(*args, **kwargs) -> List[Command]:
            return llm_command

        monkeypatch.setattr(
            command_generator, "_predict_commands", mock_predict_commands
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

        assert len(returned_message.get(COMMANDS)) == 2
        assert returned_message.get(COMMANDS) == [command, llm_command[0].as_dict()]
