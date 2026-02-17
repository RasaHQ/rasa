import os.path
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Text
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest
import structlog
from _pytest.tmpdir import TempPathFactory
from pytest import LogCaptureFixture, MonkeyPatch

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
    RepeatBotMessagesCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.constants import KEY_MINIMIZE_NUM_CALLS
from rasa.dialogue_understanding.generator.command_parser_validator import (
    CommandParserValidatorSingleton,
)
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_ACTIVE_KEY,
    FLOW_RETRIEVAL_FLOW_THRESHOLD,
    FLOW_RETRIEVAL_KEY,
    LLM_BASED_COMMAND_GENERATOR_CONFIG_FILE,
    LLM_CONFIG_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator import (  # noqa: E501
    SearchReadyLLMCommandGenerator,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.exceptions import ValidationError
from rasa.llm_fine_tuning.annotation_module import set_preparing_fine_tuning_data
from rasa.shared.constants import (
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    EMBEDDINGS_CONFIG_KEY,
    INCLUDE_DATE_TIME_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    ROUTE_TO_CALM_SLOT,
    TIMEZONE_CONFIG_KEY,
)
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT, SetSlotExtractor
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.slots import BooleanSlot, CategoricalSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import (
    InvalidPromptTemplateException,
    ProviderClientAPIException,
)
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

PROMPT_TEMPLATE_WITH_CURRENT_SLOT_INFORMATION_PATH = "./tests/dialogue_understanding/generator/prompt_template_with_current_slot_information.jinja2"  # noqa: E501
EXPECTED_RENDERED_PROMPT_WITH_CURRENT_SLOT_INFORMATION = "./tests/dialogue_understanding/generator/rendered_prompt_with_current_slot_information.txt"  # noqa: E501

# Path to the test prompt templates directory. We maintain a separate copy of the prompt
# templates, so that changes to the original prompt will fail the tests. This is to
# ensure that the changes to the prompt templates are intentional and have to be updated
# in the test directory as well.
TEST_PROMPT_TEMPLATE_DIR = "./tests/dialogue_understanding/generator/prompt_templates"

# Load the prompt templates for testing from the test directory.
command_prompt_v3_claude_3_5_sonnet_20240620_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/command_prompt_v3_claude_3_5_sonnet_20240620_template.jinja2"
)
command_prompt_v3_claude_sonnet_4_5_20250929_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/command_prompt_v3_claude_sonnet_4_5_20250929_template.jinja2"
)
command_prompt_v3_fallback_other_models_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"
)
command_prompt_v3_gpt_4o_2024_11_20_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"
)
command_prompt_v3_gpt_5_2_2025_12_11_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/command_prompt_v3_gpt_5_2_2025_12_11_template.jinja2"
)
# Agent versions of the prompt templates
agent_command_prompt_v3_fallback_other_models_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/agent_command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"
)
agent_command_prompt_v3_gpt_4o_2024_11_20_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/agent_command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"
)
agent_command_prompt_v3_gpt_5_2_2025_12_11_template = rasa.shared.utils.io.read_file(
    f"{TEST_PROMPT_TEMPLATE_DIR}/agent_command_prompt_v3_gpt_5_2_2025_12_11_template.jinja2"
)
agent_command_prompt_v3_claude_3_5_sonnet_20240620_template = rasa.shared.utils.io.read_file(  # noqa: E501
    f"{TEST_PROMPT_TEMPLATE_DIR}/agent_command_prompt_v3_claude_3_5_sonnet_20240620_template.jinja2"
)
agent_command_prompt_v3_claude_sonnet_4_5_20250929_template = rasa.shared.utils.io.read_file(  # noqa: E501
    f"{TEST_PROMPT_TEMPLATE_DIR}/agent_command_prompt_v3_claude_sonnet_4_5_20250929_template.jinja2"
)


@pytest.fixture(autouse=True)
def mock_configuration_available_agents(monkeypatch: MonkeyPatch) -> None:
    """Use empty, but initialised configuration for all tests by default.
    AvailableAgents will be empty unless explicitly set otherwise in a test.
    """
    Configuration.initialise_empty()


@pytest.fixture(autouse=True)
def set_mock_openai_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv(
        OPENAI_API_KEY_ENV_VAR, "mock key in test_search_ready_llm_command_generator"
    )


@pytest.fixture(autouse=True)
def reset_validator_state():
    CommandParserValidatorSingleton.reset_command_parser_validation()


class TestSearchReadyLLMCommandGenerator:
    """Tests for the SearchReadyLLMCommandGenerator."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        CommandSyntaxManager.reset_syntax_version()

    @pytest.fixture
    def command_generator(self):
        """Create an SearchReadyLLMCommandGenerator."""
        # Reset the command syntax version.
        CommandSyntaxManager.reset_syntax_version()

        return SearchReadyLLMCommandGenerator.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock()
        )

    @pytest.fixture
    def command_generator_with_custom_prompt_template(self):
        return SearchReadyLLMCommandGenerator.create(
            config={
                "prompt_template": PROMPT_TEMPLATE_WITH_CURRENT_SLOT_INFORMATION_PATH
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

    @pytest.fixture
    def set_agents_presence(self, monkeypatch: MonkeyPatch) -> Callable[[bool], None]:
        def _setter(present: bool) -> None:
            monkeypatch.setattr(
                "rasa.core.available_agents.AvailableAgents.has_agents",
                classmethod(lambda cls: present),
            )

        return _setter

    async def test_prompt_template_handling(self, model_storage):
        # Given
        resource = Resource("llmcmdgen")
        expected_template = "data/test_prompt_templates/test_prompt.jinja2"
        config = {"prompt_template": expected_template}

        # When
        generator = SearchReadyLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )

        # Then
        assert generator.prompt_template.startswith("This is a test prompt.")

    @pytest.mark.parametrize(
        "agents_present,expected_prompt_template",
        [
            (
                False,
                command_prompt_v3_gpt_4o_2024_11_20_template,
            ),
            (True, agent_command_prompt_v3_gpt_4o_2024_11_20_template),
        ],
    )
    async def test_default_template_when_no_prompt_template_provided(
        self,
        model_storage,
        set_agents_presence: Callable[[bool], None],
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)
        resource = Resource("llmcmdgen")
        config = {}  # No prompt or prompt_template provided

        # When
        generator = SearchReadyLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )

        # Then
        assert generator.prompt_template == expected_prompt_template

    async def test_search_ready_llm_command_generator_init_custom(
        self,
        model_storage: ModelStorage,
    ) -> None:
        # Given
        resource = Resource("llmcmdgen")
        # When
        generator = SearchReadyLLMCommandGenerator(
            {
                "prompt_template": "data/test_prompt_templates/test_prompt.jinja2",
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
            },
            model_storage,
            resource,
        )
        # Then
        assert generator.prompt_template.startswith("This is a test prompt.")
        assert generator.flow_retrieval is None

    async def test_search_ready_llm_command_generator_init_default(
        self,
        model_storage: ModelStorage,
    ) -> None:
        # When
        generator = SearchReadyLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )
        # Then
        assert generator.prompt_template.startswith("## Task Description")
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
    def test_search_ready_llm_command_generator_init_with_message_length_limit(
        self,
        config: Dict[Text, Any],
        expected_limit: Optional[int],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        generator = SearchReadyLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        assert generator.user_input_config.max_characters == expected_limit

    async def test_predict_commands_with_no_flows(
        self,
        command_generator: SearchReadyLLMCommandGenerator,
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
        self, command_generator: SearchReadyLLMCommandGenerator
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
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker_with_routing_slot: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        # Given
        mock_llm_client = AsyncMock()
        llm_response_object.choices = ["start flow test_flow"]
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
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        # Given
        message = Message.build(text="start test_flow")
        mock_llm_client = AsyncMock()
        llm_response_object.choices = ["start test_flow"]
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
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        with set_preparing_fine_tuning_data():
            mock_llm_client = AsyncMock()
            llm_response_object.choices = ["start flow test_flow"]
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
        assert message.get(LLM_PROMPT).startswith("## Task Description")
        assert message.get(LLM_COMMANDS) == [
            {"command": "start flow", "flow": "test_flow"}
        ]

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_adds_commands_and_prompt_to_message_object(
        self,
        mock_llm_factory: Mock,
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        with set_record_commands_and_prompts():
            mock_llm_client = AsyncMock()
            llm_response_object.choices = ["start flow test_flow"]
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
            prompts[0].get(KEY_COMPONENT_NAME)
            == SearchReadyLLMCommandGenerator.__name__
        )
        assert prompts[0][KEY_USER_PROMPT].startswith("## Task Description")
        assert message.get(PREDICTED_COMMANDS)[
            SearchReadyLLMCommandGenerator.__name__
        ] == [{"command": "start flow", "flow": "test_flow"}]

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_does_not_add_commands_and_prompt_by_default(
        self,
        mock_llm_factory: Mock,
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["start test_flow"]
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
        "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator"
        ".SearchReadyLLMCommandGenerator"
        ".render_template"
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator"
        ".SearchReadyLLMCommandGenerator"
        ".invoke_llm"
    )
    async def test_predict_commands_calls_prompt_rendering_with_startable_flows_only(
        self,
        mock_generate_action_list_using_llm: Mock,
        mock_render_template: Mock,
        flow_guard_value: Any,
        expected_flow_ids: Set[Text],
        command_generator: SearchReadyLLMCommandGenerator,
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
        command_generator = SearchReadyLLMCommandGenerator.create(
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
                    choices=["start flow this_flow_does_not_exists"],
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
                    choices=["set slot flow_name some_flow"],
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
        "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator."
        "SearchReadyLLMCommandGenerator.invoke_llm"
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator."
        "SearchReadyLLMCommandGenerator.render_template"
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
        command_generator: SearchReadyLLMCommandGenerator,
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
        command_generator: SearchReadyLLMCommandGenerator,
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

    def test_render_template_call(
        self,
        command_generator: SearchReadyLLMCommandGenerator,
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
        assert (
            "\nUse the following structured data:\n"
            "```json\n"
            '{"flows":[{"name":"test_flow",'
        ) in rendered_template
        # make sure the slot from the called flow is available in the template
        assert """current_step":{"requested_slot":"test_slot",""" in rendered_template

    def test_render_template_with_multiline_flow_and_descriptions(
        self,
        command_generator: SearchReadyLLMCommandGenerator,
    ):
        """Test that render_template renders the template strings with valid JSON
        (newline, tabs and quotes are escaped)
        """
        # Given
        test_message = Message.build(text="Hey I want to test this")
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
                    "flow_id": "test_flow_multiline_descriptions",
                    "step_id": "test_slot_with_multiline_description",
                    "frame_id": "some-frame-id",
                },
            ]
        )
        test_flows = flows_from_str(
            """
            flows:
              test_flow_inline_descriptions:
                description: some inline flow description
                steps:
                  - id: test_slot_with_inline_description
                    collect: test_slot
                    description: some inline slot description
              test_flow_multiline_descriptions:
                description: |
                  some multiline flow description
                  * numbering 1
                  * numbering 2
                  lorem ipsum dolor sit amet
                steps:
                  - id: test_slot_with_multiline_description
                    collect: test_slot
                    description: |
                      some multiline slot description
                      * numbering 1
                      * numbering 2
                      lorem ipsum dolor sit amet
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

        # Make sure it looks like we are in the calling flow
        assert (
            "\nUse the following structured data:\n"
            "```json\n"
            '{"flows":[{"name":"test_flow_inline_descriptions",'
        ) in rendered_template

        # Make sure the valid JSON strings are present in the template
        # Flow inline description
        assert ('"description":"some inline flow description"') in rendered_template
        # Slot inline description
        assert ('"description":"some inline slot description"') in rendered_template
        # Flow multiline description
        assert (
            '"description":'
            '"'
            "some multiline flow description"
            "\\n* numbering 1"
            "\\n* numbering 2"
            "\\nlorem ipsum dolor sit amet"
            "\\n"
            '"'
        ) in rendered_template
        # Slot multiline description
        assert (
            '"description":'
            '"'
            "some multiline slot description"
            "\\n* numbering 1"
            "\\n* numbering 2"
            "\\nlorem ipsum dolor sit amet"
            "\\n"
            '"'
        ) in rendered_template

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, []),
            (
                "set slot transfer_money_amount_of_money None",
                [SetSlotCommand(name="transfer_money_amount_of_money", value=None)],
            ),
            ("set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("set slot 'name' 'value'", [SetSlotCommand(name="name", value="value")]),
            ('set slot "name" "value"', [SetSlotCommand(name="name", value="value")]),
            ('"set slot "name" "value""', [SetSlotCommand(name="name", value="value")]),
            ("set slot name 'value'", [SetSlotCommand(name="name", value="value")]),
            ("set slot name ''value''", [SetSlotCommand(name="name", value="value")]),
            (
                "set slot name 'value a'  ",
                [SetSlotCommand(name="name", value="value a")],
            ),
            (
                'set slot name "value a"  ',
                [SetSlotCommand(name="name", value="value a")],
            ),
            ("set slot name 'value'", [SetSlotCommand(name="name", value="value")]),
            (
                "set slot name \"value with 'nested' quotes\"",
                [SetSlotCommand(name="name", value="value with 'nested' quotes")],
            ),
            ("set slot 'name' 'value'", [SetSlotCommand(name="name", value="value")]),
            ("*** set slot name value", [SetSlotCommand(name="name", value="value")]),
            (" - set slot name value", [SetSlotCommand(name="name", value="value")]),
            (" + set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("1.set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("2. set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("\t3. set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("   4. set slot name value", [SetSlotCommand(name="name", value="value")]),
            (" 1,. set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("1)set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("2) set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("3)\tset slot name value", [SetSlotCommand(name="name", value="value")]),
            ("\t4)\tset slot name value", [SetSlotCommand(name="name", value="value")]),
            ('""set slot name value', [SetSlotCommand(name="name", value="value")]),
            ("''set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("`set slot name value`", [SetSlotCommand(name="name", value="value")]),
            ("`set slot name value", [SetSlotCommand(name="name", value="value")]),
            ("set slot name value`", [SetSlotCommand(name="name", value="value")]),
            ("'`set slot name value`'", [SetSlotCommand(name="name", value="value")]),
            ("'set slot name value'", [SetSlotCommand(name="name", value="value")]),
            (
                "*+```set slot name value   ",
                [SetSlotCommand(name="name", value="value")],
            ),
            (
                "```\nset slot document_type passport\n```",
                [SetSlotCommand(name="document_type", value="passport")],
            ),
            (
                "```plaintext\nset slot confirm_slot_correction True"
                "\nset slot document_type national id\n```",
                [
                    SetSlotCommand(name="confirm_slot_correction", value="True"),
                    SetSlotCommand(name="document_type", value="national id"),
                ],
            ),
            # Start flow
            ("set slot flow_name some_flow", [StartFlowCommand(flow="some_flow")]),
            ("start flow some_flow", [StartFlowCommand(flow="some_flow")]),
            ("start flow 'some_flow'", [StartFlowCommand(flow="some_flow")]),
            ('start flow "some_flow"', [StartFlowCommand(flow="some_flow")]),
            ("start flow does_not_exist", []),
            (
                "start flow 02_benefits_learning_days",
                [StartFlowCommand(flow="02_benefits_learning_days")],
            ),
            ("* start flow 'some_flow'", [StartFlowCommand(flow="some_flow")]),
            ("--->start flow 'some_flow'", [StartFlowCommand(flow="some_flow")]),
            ("```start flow 'some_flow'```", [StartFlowCommand(flow="some_flow")]),
            ("```start flow some_flow```", [StartFlowCommand(flow="some_flow")]),
            ("1.start flow some_flow", [StartFlowCommand(flow="some_flow")]),
            ("2. start flow some_flow", [StartFlowCommand(flow="some_flow")]),
            ("\t3. start flow some_flow", [StartFlowCommand(flow="some_flow")]),
            ("    4.  start flow some_flow", [StartFlowCommand(flow="some_flow")]),
            ("`start flow some_flow`", [StartFlowCommand(flow="some_flow")]),
            ("`start flow 'some_flow'`", [StartFlowCommand(flow="some_flow")]),
            ("```start flow 'some_flow'```", [StartFlowCommand(flow="some_flow")]),
            (
                "```plaintext\nstart flow 'some_flow'```",
                [StartFlowCommand(flow="some_flow")],
            ),
            (
                '```plaintext\nstart flow "some_flow"```',
                [StartFlowCommand(flow="some_flow")],
            ),
            # Cancel flow
            ("cancel flow", [CancelFlowCommand()]),
            ("'cancel flow'", [CancelFlowCommand()]),
            ("`cancel flow", [CancelFlowCommand()]),
            ("`cancel flow`", [CancelFlowCommand()]),
            ("```cancel flow```", [CancelFlowCommand()]),
            ("```plaintext\ncancel flow```", [CancelFlowCommand()]),
            ("```plaintext\n'cancel flow'```", [CancelFlowCommand()]),
            (" 1. cancel flow", [CancelFlowCommand()]),
            (" 2) cancel flow", [CancelFlowCommand()]),
            (" 3. 'cancel flow'", [CancelFlowCommand()]),
            ("'`cancel flow`'", [CancelFlowCommand()]),
            ("`'cancel flow'`", [CancelFlowCommand()]),
            ('"cancel flow"', [CancelFlowCommand()]),
            (" * cancel flow", [CancelFlowCommand()]),
            (" *** cancel flow", [CancelFlowCommand()]),
            (" --> cancel flow", [CancelFlowCommand()]),
            # ChitChat
            ("offtopic reply", [ChitChatAnswerCommand()]),
            (" - offtopic reply", [ChitChatAnswerCommand()]),
            (" ** offtopic reply", [ChitChatAnswerCommand()]),
            ("1. offtopic reply", [ChitChatAnswerCommand()]),
            ("  2. offtopic reply", [ChitChatAnswerCommand()]),
            ("  3) offtopic reply", [ChitChatAnswerCommand()]),
            ("'offtopic reply'", [ChitChatAnswerCommand()]),
            ("'offtopic reply'", [ChitChatAnswerCommand()]),
            ('"offtopic reply"', [ChitChatAnswerCommand()]),
            ("offtopic reply'`", [ChitChatAnswerCommand()]),
            ("`'offtopic reply'`", [ChitChatAnswerCommand()]),
            ("```offtopic reply```", [ChitChatAnswerCommand()]),
            ("```plaintext\nofftopic reply```", [ChitChatAnswerCommand()]),
            ("```plaintext\n'offtopic reply'```", [ChitChatAnswerCommand()]),
            # Knowledge
            ("search and reply", [KnowledgeAnswerCommand()]),
            (" - search and reply", [KnowledgeAnswerCommand()]),
            (" --> search and reply", [KnowledgeAnswerCommand()]),
            (" *** search and reply", [KnowledgeAnswerCommand()]),
            (" 1. search and reply", [KnowledgeAnswerCommand()]),
            (" 2) search and reply", [KnowledgeAnswerCommand()]),
            ("'search and reply'", [KnowledgeAnswerCommand()]),
            ("```search and reply```", [KnowledgeAnswerCommand()]),
            ("```plaintext\nsearch and reply```", [KnowledgeAnswerCommand()]),
            ("```plaintext\n'search and reply'```", [KnowledgeAnswerCommand()]),
            ("'`search and reply`'", [KnowledgeAnswerCommand()]),
            ("`search and reply`", [KnowledgeAnswerCommand()]),
            ('"search and reply"', [KnowledgeAnswerCommand()]),
            # Human handoff
            ("hand over", [HumanHandoffCommand()]),
            ("1. hand over", [HumanHandoffCommand()]),
            (" 2. hand over", [HumanHandoffCommand()]),
            (" 3) hand over", [HumanHandoffCommand()]),
            ("'hand over'", [HumanHandoffCommand()]),
            ("`hand over`", [HumanHandoffCommand()]),
            ("```hand over```", [HumanHandoffCommand()]),
            ("```plaintext\nhand over```", [HumanHandoffCommand()]),
            ("```plaintext\n'hand over'```", [HumanHandoffCommand()]),
            (
                "Here is a list of commands:\nset slot flow_name some_flow\n",
                [StartFlowCommand(flow="some_flow")],
            ),
            (
                """set slot flow_name some_flow
                       set slot transfer_money_amount_of_money None""",
                [
                    StartFlowCommand(flow="some_flow"),
                    SetSlotCommand(name="transfer_money_amount_of_money", value=None),
                ],
            ),
            # Clarify of non-existent option is dropped
            ("disambiguate flows transfer_money", [ClarifyCommand(options=[])]),
            # Clarify orders options
            (
                "disambiguate flows some_flow 02_benefits_learning_days",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            # Clarify with quotes around the flow names
            (
                "disambiguate flows 'some_flow' 'another_flow'",
                [ClarifyCommand(options=["another_flow", "some_flow"])],
            ),
            (
                'disambiguate flows "some_flow" "another_flow"',
                [ClarifyCommand(options=["another_flow", "some_flow"])],
            ),
            # Clarify with single option is converted to a StartFlowCommand
            ("disambiguate flows some_flow", [StartFlowCommand(flow="some_flow")]),
            # Clarify with multiple but same options is converted to a StartFlowCommand
            (
                "disambiguate flows some_flow some_flow some_flow some_flow",
                [StartFlowCommand(flow="some_flow")],
            ),
            (
                "1. disambiguate flows some_flow 02_benefits_learning_days",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                " 2. disambiguate flows some_flow 02_benefits_learning_days",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                "  3) disambiguate flows some_flow 02_benefits_learning_days",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                " -- disambiguate flows some_flow 02_benefits_learning_days",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                " *** disambiguate flows some_flow 02_benefits_learning_days",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                "```disambiguate flows some_flow 02_benefits_learning_days```",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                "`disambiguate flows some_flow 02_benefits_learning_days`",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                "`disambiguate flows 'some_flow' '02_benefits_learning_days'`",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            (
                "```plaintext\ndisambiguate flows some_flow 02_benefits_learning_days```",  # noqa: E501
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            # RepeatBotMessagesCommand
            ("repeat message", [RepeatBotMessagesCommand()]),
            (" - repeat message", [RepeatBotMessagesCommand()]),
            (" --> repeat message", [RepeatBotMessagesCommand()]),
            (" *** repeat message", [RepeatBotMessagesCommand()]),
            (" 1. repeat message", [RepeatBotMessagesCommand()]),
            (" 2) repeat message", [RepeatBotMessagesCommand()]),
            ("'repeat message'", [RepeatBotMessagesCommand()]),
            ("```repeat message```", [RepeatBotMessagesCommand()]),
            ("```plaintext\nrepeat message```", [RepeatBotMessagesCommand()]),
            ("```plaintext\n'repeat message'```", [RepeatBotMessagesCommand()]),
            # SkipQuestionCommand
            ("skip question", [SkipQuestionCommand()]),
            (" - skip question", [SkipQuestionCommand()]),
            (" --> skip question", [SkipQuestionCommand()]),
            (" *** skip question", [SkipQuestionCommand()]),
            (" 1. skip question", [SkipQuestionCommand()]),
            (" 2) skip question", [SkipQuestionCommand()]),
            ("'skip question'", [SkipQuestionCommand()]),
            ("```skip question```", [SkipQuestionCommand()]),
            ("```plaintext\nskip question```", [SkipQuestionCommand()]),
            ("```plaintext\n'skip question'```", [SkipQuestionCommand()]),
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
        CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)
        parsed_commands = SearchReadyLLMCommandGenerator.parse_commands(
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

        config = {"prompt_template": str(prompt_file)}
        generator = SearchReadyLLMCommandGenerator(
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

        config = {"prompt_template": str(prompt_file)}
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
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
        generator = SearchReadyLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        domain = Mock()
        domain.slots = []
        train_resource = generator.train(TrainingData(), flows, domain)
        # When
        loaded = SearchReadyLLMCommandGenerator.load(
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
        generator = SearchReadyLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        domain = Mock()
        train_resource = generator.train(TrainingData(), flows, domain)
        # When
        loaded = SearchReadyLLMCommandGenerator.load(
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
            "prompt_template": os.path.join(
                "data", "test_prompt_templates", "test_prompt.jinja2"
            )
        }
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )
        # Then
        assert loaded.prompt_template.startswith("This is a test prompt.")

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @pytest.mark.parametrize(
        "agents_present,expected_prompt_template",
        [
            (False, command_prompt_v3_gpt_4o_2024_11_20_template),
            (True, agent_command_prompt_v3_gpt_4o_2024_11_20_template),
        ],
    )
    def test_load_with_default_prompt(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
        set_agents_presence: Callable[[bool], None],
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)
        resource = Resource("llmcmdgen")
        generator = SearchReadyLLMCommandGenerator({}, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert loaded.prompt_template.find("## Available Flows and Slots\n") > 0
        assert loaded.prompt_template.find("```json\n") > 0
        assert loaded.prompt_template.find("| search and reply   |") > 0
        assert loaded.prompt_template == expected_prompt_template

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @pytest.mark.parametrize(
        "agents_present,expected_prompt_template",
        [
            (False, command_prompt_v3_fallback_other_models_template),
            (True, agent_command_prompt_v3_fallback_other_models_template),
        ],
    )
    def test_load_with_fallback_prompt(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
        set_agents_presence: Callable[[bool], None],
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)
        resource = Resource("llmcmdgen")
        generator = SearchReadyLLMCommandGenerator(
            {"llm": {"provider": "unknown", "model": "test"}}, model_storage, resource
        )
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert loaded.prompt_template.find("## Available Flows and Slots\n") > 0
        assert (
            loaded.prompt_template.find(
                "\nUse the following structured data:\n```json\n"
            )
            > 0
        )
        assert loaded.prompt_template.find("| search and reply   |") > 0
        assert loaded.prompt_template == expected_prompt_template

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
    @pytest.mark.parametrize(
        "model_name,agents_present,expected_prompt_template",
        [
            (
                "claude-3-5-sonnet-20240620",
                False,
                command_prompt_v3_claude_3_5_sonnet_20240620_template,
            ),
            (
                "claude-3-5-sonnet-20240620",
                True,
                agent_command_prompt_v3_claude_3_5_sonnet_20240620_template,
            ),
            (
                "claude-sonnet-4-5-20250929",
                False,
                command_prompt_v3_claude_sonnet_4_5_20250929_template,
            ),
            (
                "claude-sonnet-4-5-20250929",
                True,
                agent_command_prompt_v3_claude_sonnet_4_5_20250929_template,
            ),
        ],
    )
    def test_load_default_prompt_based_on_model_name_claude(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        mock_perform_health_check: Mock,
        model_storage: ModelStorage,
        set_agents_presence: Callable[[bool], None],
        model_name: str,
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)
        resource = Resource("llmcmdgen")
        config = {
            "llm": {
                "provider": "anthropic",
                "model": model_name,
            },
        }
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert (
            loaded.prompt_template.find(
                "\nUse the following structured data:\n```json\n"
            )
            > 0
        )
        assert """{"flows":[""" in loaded.prompt_template  # minified JSON
        assert "`search and reply`" in loaded.prompt_template  # correct DSL
        assert loaded.prompt_template == expected_prompt_template

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
    @pytest.mark.parametrize(
        "model_name,agents_present,expected_prompt_template",
        [
            (
                "claude-3-5-sonnet-20240620",
                False,
                command_prompt_v3_claude_3_5_sonnet_20240620_template,
            ),
            (
                "claude-3-5-sonnet-20240620",
                True,
                agent_command_prompt_v3_claude_3_5_sonnet_20240620_template,
            ),
            (
                "claude-sonnet-4-5-20250929",
                False,
                command_prompt_v3_claude_sonnet_4_5_20250929_template,
            ),
            (
                "claude-sonnet-4-5-20250929",
                True,
                agent_command_prompt_v3_claude_sonnet_4_5_20250929_template,
            ),
        ],
    )
    def test_load_default_prompt_based_on_model_name_from_model_group_claude(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        mock_perform_health_check: Mock,
        model_storage: ModelStorage,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        set_agents_presence: Callable[[bool], None],
        model_name: str,
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)

        mock_available_endpoints.model_groups = [
            {
                "id": "anthropic_claude",
                "models": [
                    {
                        "provider": "anthropic",
                        "model": model_name,
                    }
                ],
            }
        ]

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)
        resource = Resource("llmcmdgen")

        config = {"llm": {"model_group": "anthropic_claude"}}
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert (
            loaded.prompt_template.find(
                "\nUse the following structured data:\n```json\n"
            )
            > 0
        )
        assert """{"flows":[""" in loaded.prompt_template  # minified JSON
        assert "`search and reply`" in loaded.prompt_template  # correct DSL
        assert loaded.prompt_template == expected_prompt_template

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @pytest.mark.parametrize(
        "agents_present,expected_prompt_template",
        [
            (False, command_prompt_v3_gpt_4o_2024_11_20_template),
            (True, agent_command_prompt_v3_gpt_4o_2024_11_20_template),
        ],
    )
    def test_load_deafult_prompt_based_on_model_name_gpt_4o(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
        set_agents_presence: Callable[[bool], None],
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)
        resource = Resource("llmcmdgen")
        config = {"llm": {"provider": "openai", "model": "gpt-4o-2024-11-20"}}
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert (
            loaded.prompt_template.find(
                "Flows and Slots\nUse the following structured data:\n```json\n"
            )
            > 0
        )
        assert loaded.prompt_template.find("| search and reply   |") > 0
        assert loaded.prompt_template == expected_prompt_template

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @pytest.mark.parametrize(
        "agents_present,expected_prompt_template",
        [
            (False, command_prompt_v3_gpt_5_2_2025_12_11_template),
            (True, agent_command_prompt_v3_gpt_5_2_2025_12_11_template),
        ],
    )
    def test_load_default_prompt_based_on_model_name_gpt_5_2(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
        set_agents_presence: Callable[[bool], None],
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)
        resource = Resource("llmcmdgen")
        config = {"llm": {"provider": "openai", "model": "gpt-5.2-2025-12-11"}}
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert (
            loaded.prompt_template.find(
                "\nUse the following structured data:\n```json\n"
            )
            > 0
        )
        assert loaded.prompt_template == expected_prompt_template

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    @patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
    @pytest.mark.parametrize(
        "agents_present,expected_prompt_template",
        [
            (False, command_prompt_v3_gpt_4o_2024_11_20_template),
            (True, agent_command_prompt_v3_gpt_4o_2024_11_20_template),
        ],
    )
    def test_load_default_prompt_based_on_model_name_from_model_group_gpt_4o(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        mock_perform_health_check: Mock,
        model_storage: ModelStorage,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        set_agents_presence: Callable[[bool], None],
        agents_present: bool,
        expected_prompt_template: Any,
    ):
        # Given
        set_agents_presence(agents_present)

        mock_available_endpoints.model_groups = [
            {
                "id": "openai-gpt-4o-direct",
                "models": [
                    {
                        "provider": "openai",
                        "model": "gpt-4o-2024-11-20",
                    }
                ],
            }
        ]

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        resource = Resource("llmcmdgen")
        config = {"llm": {"model_group": "openai-gpt-4o-direct"}}
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())

        # When
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )

        # Then
        assert loaded.prompt_template.startswith("## Task Description")
        assert (
            loaded.prompt_template.find(
                "Flows and Slots\nUse the following structured data:\n```json\n"
            )
            > 0
        )
        assert loaded.prompt_template.find("| search and reply   |") > 0
        assert loaded.prompt_template == expected_prompt_template

    async def test_search_ready_llm_command_generator_load_prompt_from_model_storage(
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
        config = {"prompt_template": str(prompt_file)}

        # Persist the prompt file to the model storage.
        resource = Resource("llmcmdgen")
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)
        generator.persist()

        # Test loading the prompt from the model storage.
        # Case 1: No prompt in the config.
        loaded = SearchReadyLLMCommandGenerator.load(
            {}, model_storage, resource, Mock()
        )
        assert loaded.prompt_template == "This is a custom prompt"
        assert loaded.config["prompt_template"] is None

        # Case 2: Specifying a invalid prompt path in the config.
        loaded = SearchReadyLLMCommandGenerator.load(
            {"prompt_template": "test_prompt.jinja2"},
            model_storage,
            resource,
            Mock(),
        )
        assert loaded.prompt_template == "This is a custom prompt"
        assert loaded.config["prompt_template"] == "test_prompt.jinja2"

    @patch(
        "rasa.shared.utils.health_check.health_check.perform_llm_health_check",
        return_value=None,
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval")
    def test_train_with_no_flows(
        self,
        mock_flow_retrieval: Mock,
        mock_perform_llm_health_check: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        mock_flow_retrieval.__name__ = "FlowRetrieval"
        resource = Resource("llmcmdgen")
        generator = SearchReadyLLMCommandGenerator({}, model_storage, resource)
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
    def test_search_ready_llm_command_generator_init_with_different_llm_configs(
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

        generator = SearchReadyLLMCommandGenerator(
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

    def test_search_ready_llm_command_generator_persist_config(
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
        generator = SearchReadyLLMCommandGenerator(config, model_storage, resource)

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
        self,
        command_generator: SearchReadyLLMCommandGenerator,
        monkeypatch: MonkeyPatch,
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
        command_generator: SearchReadyLLMCommandGenerator,
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
        command_generator: SearchReadyLLMCommandGenerator,
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
        self,
        command_generator: SearchReadyLLMCommandGenerator,
        monkeypatch: MonkeyPatch,
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
        command_generator: SearchReadyLLMCommandGenerator,
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

    @pytest.mark.parametrize(
        "model,expected_prompt_template,expected_agent_prompt_template",
        [
            (
                "openai/gpt-4o-2024-11-20",
                "command_prompt_v3_gpt_4o_2024_11_20_template.jinja2",
                "agent_command_prompt_v3_gpt_4o_2024_11_20_template.jinja2",
            ),
            (
                "openai/gpt-5.2-2025-12-11",
                "command_prompt_v3_gpt_5_2_2025_12_11_template.jinja2",
                "agent_command_prompt_v3_gpt_5_2_2025_12_11_template.jinja2",
            ),
            (
                "azure/gpt-4o-2024-11-20",
                "command_prompt_v3_gpt_4o_2024_11_20_template.jinja2",
                "agent_command_prompt_v3_gpt_4o_2024_11_20_template.jinja2",
            ),
            (
                "azure/gpt-5.2-2025-12-11",
                "command_prompt_v3_gpt_5_2_2025_12_11_template.jinja2",
                "agent_command_prompt_v3_gpt_5_2_2025_12_11_template.jinja2",
            ),
            (
                "anthropic/claude-3-5-sonnet-20240620",
                "command_prompt_v3_claude_3_5_sonnet_20240620_template.jinja2",
                "agent_command_prompt_v3_claude_3_5_sonnet_20240620_template.jinja2",
            ),
            (
                "anthropic/claude-sonnet-4-5-20250929",
                "command_prompt_v3_claude_sonnet_4_5_20250929_template.jinja2",
                "agent_command_prompt_v3_claude_sonnet_4_5_20250929_template.jinja2",
            ),
            (
                "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                "command_prompt_v3_claude_3_5_sonnet_20240620_template.jinja2",
                "agent_command_prompt_v3_claude_3_5_sonnet_20240620_template.jinja2",
            ),
            (
                "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
                "command_prompt_v3_claude_sonnet_4_5_20250929_template.jinja2",
                "agent_command_prompt_v3_claude_sonnet_4_5_20250929_template.jinja2",
            ),
        ],
    )
    def test_model_prompt_mapper(
        self,
        model: str,
        expected_prompt_template: str,
        expected_agent_prompt_template: str,
        set_agents_presence: Callable[[bool], None],
    ):
        # Given
        set_agents_presence(False)
        # Then
        assert (
            SearchReadyLLMCommandGenerator.get_model_prompt_mapper().get(model)
            == expected_prompt_template
        )

        # Given
        set_agents_presence(True)
        # Then
        assert (
            SearchReadyLLMCommandGenerator.get_model_prompt_mapper().get(model)
            == expected_agent_prompt_template
        )

    def test_command_syntax_version(self):
        assert CommandSyntaxManager.get_syntax_version() == CommandSyntaxVersion.v3
        CommandSyntaxManager.reset_syntax_version()

    def test_render_template_with_current_slot_info(
        self,
        command_generator_with_custom_prompt_template: SearchReadyLLMCommandGenerator,
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

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_validation_command_parser_unable_to_parse_commands_N_turns(
        self,
        mock_llm_factory: Mock,
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        caplog: LogCaptureFixture,
    ):
        # Given
        message = Message.build(text="start test_flow")
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock

        # When
        with structlog.testing.capture_logs() as caplog:
            # Predict empty commands for 6 turns.
            for _ in range(6):
                commands = await command_generator.predict_commands(
                    message,
                    flows=flows,
                    tracker=tracker,
                )
                assert len(commands) == 1
                assert commands[0] == CannotHandleCommand()

        # Then
        assert (
            CommandParserValidatorSingleton.get_no_command_predicted_turn_counter() == 6
        )
        assert CommandParserValidatorSingleton.should_validate_command_parser() is True

        event = "llm_command_generator.predict_commands.command_parser_not_working"
        found_validation_log = False
        for record in caplog:
            if record["event"] == event:
                found_validation_log = True
                break

        # Check if the validation log was found.
        assert found_validation_log

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_validation_command_parser_unable_to_parse_commands_intermittently(
        self,
        mock_llm_factory: Mock,
        command_generator: SearchReadyLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        caplog: LogCaptureFixture,
    ):
        """Test that predict_commands sets the routing slot to True."""
        # Given, invalid command as the return value of the mock.
        message = Message.build(text="start test_flow")
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock

        # When
        with structlog.testing.capture_logs() as caplog:
            # Predict empty commands for 3 turns.
            for _ in range(3):
                commands = await command_generator.predict_commands(
                    message,
                    flows=flows,
                    tracker=tracker,
                )
                assert len(commands) == 1
                assert commands[0] == CannotHandleCommand()

        # Then
        assert (
            CommandParserValidatorSingleton.get_no_command_predicted_turn_counter() == 3
        )
        assert CommandParserValidatorSingleton.should_validate_command_parser() is True

        # Change the return value of the mock to a valid command.
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["start flow test_flow"]
        )
        mock_llm_factory.return_value = llm_mock

        # Predict valid command once, causing the command parser to be validated.
        commands = await command_generator.predict_commands(
            message,
            flows=flows,
            tracker=tracker,
        )

        # Then
        assert len(commands) == 1
        assert commands[0] == StartFlowCommand("test_flow")
        assert (
            CommandParserValidatorSingleton.get_no_command_predicted_turn_counter() == 0
        )
        assert CommandParserValidatorSingleton.should_validate_command_parser() is False

        # Given, invalid command as the return value of the mock.
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock

        # When
        with structlog.testing.capture_logs() as caplog:
            # Predict invalid command for 6 turns now.
            for _ in range(6):
                commands = await command_generator.predict_commands(
                    message,
                    flows=flows,
                    tracker=tracker,
                )
                assert len(commands) == 1
                assert commands[0] == CannotHandleCommand()

        # Then
        assert (
            CommandParserValidatorSingleton.get_no_command_predicted_turn_counter() == 0
        )
        assert CommandParserValidatorSingleton.should_validate_command_parser() is False

        event = "llm_command_generator.predict_commands.command_parser_not_working"
        found_validation_log = False
        for record in caplog:
            if record["event"] == event:
                found_validation_log = True
                break

        # Check if the validation log is not found as the command parser is working now.
        assert found_validation_log is False

    def test_resolve_component_prompt_template_custom_prompt_success(self):
        """Test that custom prompt template is used when custom prompt template is provided and loads successfully."""  # noqa: E501
        # Given
        config = {"prompt_template": "custom_prompt.jinja2", "llm": {"model": "gpt-4o"}}

        # Mock get_prompt_template to return custom content
        with patch(
            "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator.get_prompt_template"
        ) as mock_get_prompt_template:
            mock_get_prompt_template.return_value = "Custom prompt template content"

            # Mock get_default_prompt_template_based_on_model (should not be called)
            with patch(
                "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator.get_default_prompt_template_based_on_model"
            ) as mock_get_default:
                mock_get_default.return_value = "Default prompt template"

                # When
                result = (
                    SearchReadyLLMCommandGenerator._resolve_component_prompt_template(
                        config
                    )
                )
                # Then
                assert result == "Custom prompt template content"

                mock_get_prompt_template.assert_called_once()
                mock_get_default.assert_not_called()

    def test_resolve_component_prompt_template_custom_prompt_read_error(self):
        """Test that an exception is raised and error is logged when custom prompt template file is not found."""  # noqa: E501
        # Given
        config = {
            "prompt_template": "nonexistent_prompt.jinja2",
            "llm": {"model": "gpt-4o"},
        }

        # Patch structlogger.error to verify error is logged
        with patch("rasa.shared.utils.llm.structlogger.error") as mock_error:
            # When/Then - should raise exception
            with pytest.raises(InvalidPromptTemplateException) as exc_info:
                SearchReadyLLMCommandGenerator._resolve_component_prompt_template(
                    config
                )

            # Verify exception contains file path information
            assert exc_info.value.file_path == "nonexistent_prompt.jinja2"
            assert exc_info.value.resolved_path is not None

            # Verify error was logged with file path information
            mock_error.assert_called_once()
            call_kwargs = mock_error.call_args[1]
            assert call_kwargs["prompt_file_path"] == "nonexistent_prompt.jinja2"
            assert "resolved_path" in call_kwargs

    def test_resolve_component_prompt_template_no_custom_prompt(self):
        """Test that default prompt template is used when no custom prompt template is provided."""  # noqa: E501
        # Given
        config = {"llm": {"model": "gpt-4o"}}

        # Mock get_default_prompt_template_based_on_model to return default content
        with patch(
            "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator.get_default_prompt_template_based_on_model"
        ) as mock_get_default:
            mock_get_default.return_value = "Default prompt template"

            # When
            result = SearchReadyLLMCommandGenerator._resolve_component_prompt_template(
                config
            )

            # Then
            assert result == "Default prompt template"
            mock_get_default.assert_called_once()

    def test_resolve_component_prompt_template_model_specific_prompt(self):
        """Test that model-specific prompt is used when model is found in prompt mapper."""  # noqa: E501
        # Given
        config = {"llm": {"model": "gpt-4o"}}

        # Mock get_default_prompt_template_based_on_model to return
        # model-specific prompt template
        with patch(
            "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator.get_default_prompt_template_based_on_model"
        ) as mock_get_default:
            mock_get_default.return_value = "GPT-4o specific prompt template"

            # When
            result = SearchReadyLLMCommandGenerator._resolve_component_prompt_template(
                config
            )

            # Then
            assert result == "GPT-4o specific prompt template"
            mock_get_default.assert_called_once()

    def test_resolve_component_prompt_template_fallback_prompt(self):
        """Test that fallback prompt is used when model is not found in prompt mapper."""  # noqa: E501
        # Given
        config = {"llm": {"model": "unknown-model"}}

        # Mock get_default_prompt_template_based_on_model to return fallback content
        with patch(
            "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator.get_default_prompt_template_based_on_model"
        ) as mock_get_default:
            mock_get_default.return_value = "Fallback prompt template"

            # When
            result = SearchReadyLLMCommandGenerator._resolve_component_prompt_template(
                config
            )

            # Then
            assert result == "Fallback prompt template"
            mock_get_default.assert_called_once()

    def test_resolve_component_prompt_template_direct_prompt_parameter(self):
        """Test that direct prompt_template parameter takes precedence over config."""
        # Given
        config = {"prompt_template": "config_prompt.jinja2", "llm": {"model": "gpt-4o"}}
        direct_prompt = "Direct prompt template content"

        # When
        result = SearchReadyLLMCommandGenerator._resolve_component_prompt_template(
            config=config, prompt_template=direct_prompt
        )

        # Then
        assert result == "Direct prompt template content"

    @pytest.mark.parametrize(
        "config, expected_error_code",
        [
            # Invalid timezone when include_date_time is True (default)
            (
                {TIMEZONE_CONFIG_KEY: "Invalid/Timezone"},
                "datetime_utils.validate_datetime_configuration.invalid_timezone",
            ),
            # Invalid timezone when include_date_time is explicitly True
            (
                {
                    INCLUDE_DATE_TIME_CONFIG_KEY: True,
                    TIMEZONE_CONFIG_KEY: "Invalid/Timezone",
                },
                "datetime_utils.validate_datetime_configuration.invalid_timezone",
            ),
            # Empty timezone string
            (
                {TIMEZONE_CONFIG_KEY: ""},
                "datetime_utils.validate_datetime_configuration.invalid_timezone",
            ),
        ],
    )
    def test_search_ready_llm_command_generator_invalid_timezone_raises_error(
        self,
        config: Dict[str, Any],
        expected_error_code: str,
        model_storage: ModelStorage,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that invalid timezone raises ValidationError."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "test")

        # When/Then
        with pytest.raises(ValidationError) as exc_info:
            SearchReadyLLMCommandGenerator.create(
                config=config,
                resource=resource,
                model_storage=model_storage,
                execution_context=Mock(),
            )

        assert exc_info.value.code == expected_error_code

    @pytest.mark.parametrize(
        "config, expected_log_event",
        [
            # Timezone provided when include_date_time is False
            (
                {
                    INCLUDE_DATE_TIME_CONFIG_KEY: False,
                    TIMEZONE_CONFIG_KEY: "America/New_York",
                },
                "datetime_utils.validate_datetime_configuration.timezone_not_allowed",
            ),
            (
                {
                    INCLUDE_DATE_TIME_CONFIG_KEY: False,
                    TIMEZONE_CONFIG_KEY: "Europe/London",
                },
                "datetime_utils.validate_datetime_configuration.timezone_not_allowed",
            ),
        ],
    )
    def test_search_ready_llm_command_generator_timezone_warning_when_date_time_disable(
        self,
        config: Dict[str, Any],
        expected_log_event: str,
        model_storage: ModelStorage,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that timezone warning is logged when include_date_time is False."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "test")
        expected_log_level = "warning"

        with structlog.testing.capture_logs() as caplog:
            # When
            SearchReadyLLMCommandGenerator.create(
                config=config,
                resource=resource,
                model_storage=model_storage,
                execution_context=Mock(),
            )
            logs = filter_logs(caplog, expected_log_event, expected_log_level)

        # Then
        assert len(logs) == 1
        # Verify the generator is still created successfully despite the warning
        generator = SearchReadyLLMCommandGenerator.create(
            config=config,
            resource=resource,
            model_storage=model_storage,
            execution_context=Mock(),
        )
        assert generator.include_date_time is False
        assert generator.timezone == config[TIMEZONE_CONFIG_KEY]

    @pytest.mark.parametrize(
        "include_date_time, timezone, expected_datetime_present, expected_date_format,"
        "expected_time_format, expected_day",
        [
            # include_date_time is True (default), should include datetime
            (
                DEFAULT_INCLUDE_DATE_TIME,
                DEFAULT_TIMEZONE,
                True,
                "15 January, 2024",
                "14:30:45",
                "Monday",
            ),
            # include_date_time is True with custom timezone
            (
                True,
                "America/New_York",
                True,
                "15 January, 2024",
                "14:30:45",
                "Monday",
            ),
            # include_date_time is False, should NOT include datetime
            (False, DEFAULT_TIMEZONE, False, None, None, None),
            # include_date_time is False with custom timezone
            # should NOT include datetime
            (False, "America/New_York", False, None, None, None),
        ],
    )
    def test_render_template_includes_current_datetime_when_enabled(
        self,
        model_storage: ModelStorage,
        resource: Resource,
        include_date_time: bool,
        timezone: str,
        expected_datetime_present: bool,
        expected_date_format: Optional[str],
        expected_time_format: Optional[str],
        expected_day: Optional[str],
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "test")

        # Create generator with datetime configuration
        config = {
            INCLUDE_DATE_TIME_CONFIG_KEY: include_date_time,
            TIMEZONE_CONFIG_KEY: timezone,
        }
        generator = SearchReadyLLMCommandGenerator.create(
            config=config,
            resource=resource,
            model_storage=model_storage,
            execution_context=Mock(),
        )

        # Create test message and tracker
        test_message = Message.build(text="test message")
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
        )
        test_flows = flows_from_str(
            """
            flows:
              test_flow:
                description: some description
                steps:
                - id: first_step
                  action: action_listen
            """
        )

        # Mock get_current_datetime to return a fixed datetime
        mock_now = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo(timezone))
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = mock_now

            rendered_template = generator.render_template(
                message=test_message,
                tracker=test_tracker,
                startable_flows=test_flows,
                all_flows=test_flows,
            )

            if expected_datetime_present:
                # Verify datetime section is present
                assert "### Date & Time Context" in rendered_template
                assert expected_date_format in rendered_template
                assert expected_time_format in rendered_template
                assert expected_day in rendered_template
                assert mock_now.tzname() in rendered_template
                # Verify get_current_datetime was called
                mock_get_current_datetime.assert_called_once_with(timezone=timezone)
            else:
                # Verify datetime section is NOT present
                assert "### Date & Time Context" not in rendered_template
                assert "Current date:" not in rendered_template
                assert "Current time:" not in rendered_template
                assert "Current day:" not in rendered_template
                # Verify get_current_datetime was NOT called
                mock_get_current_datetime.assert_not_called()

    # ============================================================================
    # _resolve_datetime Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "mocked_dt, expected_tzname",
        [
            (
                "2024-01-15T10:30:00+00:00",
                "UTC",
            ),
            (
                "2024-01-15T10:30:00-05:00",
                "UTC-05:00",
            ),
        ],
    )
    def test_search_ready_llm_command_generator_resolve_datetime_with_mocked_datetime(
        self,
        command_generator: SearchReadyLLMCommandGenerator,
        mocked_dt: str,
        expected_tzname: str,
    ) -> None:
        """render_template uses mocked_datetime when present in tracker."""
        domain = Domain.from_dict(
            {
                "slots": {
                    MOCKED_DATETIME_SLOT: {
                        "type": "any",
                        "mappings": [],
                        "influence_conversation": False,
                    }
                }
            }
        )
        tracker = DialogueStateTracker.from_events(
            "test_resolve_datetime",
            domain=domain,
            slots=domain.slots,
            evts=[SlotSet(MOCKED_DATETIME_SLOT, mocked_dt)],
        )

        test_message = {TEXT: "test message"}
        test_flows = FlowsList(underlying_flows=[])

        # Call render_template which internally calls resolve_datetime
        rendered_template = command_generator.render_template(
            message=test_message,
            tracker=tracker,
            startable_flows=test_flows,
            all_flows=test_flows,
        )

        # Verify the mocked datetime is used in the rendered template
        assert "15 January, 2024" in rendered_template
        assert "10:30:00" in rendered_template
        assert "Monday" in rendered_template
        assert expected_tzname in rendered_template

    def test_search_ready_llm_command_generator_resolve_without_mocked_datetime(
        self, command_generator: SearchReadyLLMCommandGenerator
    ) -> None:
        """render_template uses current datetime when mocked_datetime undefined."""
        domain = Domain.empty()
        tracker = DialogueStateTracker.from_events(
            "test_resolve_datetime",
            domain=domain,
            slots=domain.slots,
            evts=[UserUttered("test message")],
        )

        test_message = {TEXT: "test message"}
        test_flows = FlowsList(underlying_flows=[])

        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current:
            expected_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            mock_get_current.return_value = expected_dt

            # Call render_template which internally calls resolve_datetime
            # which calls get_current_datetime when mocked_datetime is None
            rendered_template = command_generator.render_template(
                message=test_message,
                tracker=tracker,
                startable_flows=test_flows,
                all_flows=test_flows,
            )

            # Verify get_current_datetime was called through resolve_datetime
            mock_get_current.assert_called_once_with(
                timezone=command_generator.timezone
            )

            # Verify the current datetime is used in the rendered template
            assert "15 January, 2024" in rendered_template
            assert "10:30:00" in rendered_template
            assert "Monday" in rendered_template

    def test_search_ready_llm_command_generator_resolve_with_mocked_datetime_none(
        self, command_generator: SearchReadyLLMCommandGenerator
    ) -> None:
        """render_template uses current datetime when mocked_datetime is None."""
        domain = Domain.from_dict(
            {
                "slots": {
                    MOCKED_DATETIME_SLOT: {
                        "type": "any",
                        "mappings": [],
                        "influence_conversation": False,
                    }
                }
            }
        )
        tracker = DialogueStateTracker.from_events(
            "test_resolve_datetime",
            domain=domain,
            slots=domain.slots,
            evts=[SlotSet(MOCKED_DATETIME_SLOT, None)],
        )

        test_message = {TEXT: "test message"}
        test_flows = FlowsList(underlying_flows=[])

        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current:
            expected_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            mock_get_current.return_value = expected_dt

            # Call render_template which internally calls resolve_datetime
            # which calls get_current_datetime when mocked_datetime is None
            rendered_template = command_generator.render_template(
                message=test_message,
                tracker=tracker,
                startable_flows=test_flows,
                all_flows=test_flows,
            )

            # Verify get_current_datetime was called through resolve_datetime
            mock_get_current.assert_called_once_with(
                timezone=command_generator.timezone
            )

            # Verify the current datetime is used in the rendered template
            assert "15 January, 2024" in rendered_template
            assert "10:30:00" in rendered_template
            assert "Monday" in rendered_template
