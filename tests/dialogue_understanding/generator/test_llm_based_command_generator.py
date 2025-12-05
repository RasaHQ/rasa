import uuid
from typing import Any, Dict, List, Optional, Text
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from _pytest.tmpdir import TempPathFactory
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.commands import (
    ChitChatAnswerCommand,
    Command,
    ErrorCommand,
    SetSlotCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
)
from rasa.dialogue_understanding.generator import (
    CompactLLMCommandGenerator,
    LLMBasedCommandGenerator,
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SearchReadyLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_ACTIVE_KEY,
    FLOW_RETRIEVAL_KEY,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    PROVIDER_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
    TEMPERATURE_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.events import AgentStarted, BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps.collect import (
    CollectInformationFlowStep,
)
from rasa.shared.core.slots import (
    Slot,
    SlotRejection,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
)
from rasa.shared.utils.llm import LLMInput
from tests.utilities import flows_from_str

TEST_BASE_CLASSES: List[type] = [
    LLMCommandGenerator,
    SingleStepLLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    CompactLLMCommandGenerator,
    SearchReadyLLMCommandGenerator,
]


class TestLLMBasedCommandGenerator:
    """Tests for the LLMBasedCommandGenerator."""

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
              foo:
                name: another test flow
                description: another test flow
                steps:
                - action: action_listen
            """
        )

    @pytest.fixture(scope="session")
    def resource(self) -> Resource:
        return Resource(uuid.uuid4().hex)

    @pytest.fixture(autouse=True, scope="function")
    def empty_configuration(self) -> None:
        Configuration.initialise_empty()

    @pytest.fixture(scope="session")
    def model_storage(self, tmp_path_factory: TempPathFactory) -> ModelStorage:
        return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))

    # Fixture for the base class
    @pytest.fixture
    def base_class_fixture(self):
        class TestLLMBasedCommandGenerator(LLMBasedCommandGenerator):
            def __init__(
                self,
                config: Dict[str, Any],
                model_storage: ModelStorage,
                resource: Resource,
                prompt_template: Optional[Text] = None,
                **kwargs: Any,
            ) -> None:
                super().__init__(
                    config,
                    model_storage,
                    resource,
                    prompt_template=prompt_template,
                    **kwargs,
                )

            @staticmethod
            def get_default_config() -> dict:
                return {}

            @classmethod
            def load(
                cls,
                config: dict,
                model_storage: ModelStorage,
                resource: Resource,
                execution_context: ExecutionContext,
                **kwargs,
            ):
                return cls(config, model_storage, resource)

            def persist(self) -> None:
                pass

            async def predict_commands(
                self,
                message: Message,
                flows: FlowsList,
                tracker: DialogueStateTracker = None,
                **kwargs: Any,
            ):
                return []

            def parse_commands(cls, actions, tracker, flows):
                return []

            def fingerprint_addon(cls, config):
                return None

        return TestLLMBasedCommandGenerator

    # Fixture for the implementated classes
    @pytest.fixture
    def llm_command_generator_fixture(self, model_storage, resource):
        # Reset the command syntax version.
        CommandSyntaxManager.reset_syntax_version()

        return LLMCommandGenerator.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    @pytest.fixture
    def single_step_llm_command_generator_fixture(self, model_storage, resource):
        # Reset the command syntax version.
        CommandSyntaxManager.reset_syntax_version()

        return SingleStepLLMCommandGenerator.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    @pytest.fixture
    def multi_step_llm_command_generator_fixture(self, model_storage, resource):
        # Reset the command syntax version.
        CommandSyntaxManager.reset_syntax_version()

        return MultiStepLLMCommandGenerator.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    ### Tests for abstract methods (i.e. based on implemented classes)
    # Parameterized fixture
    @pytest.fixture(params=["implementation_single_step", "implementation_multi_step"])
    def command_generator_fixture(
        self,
        request,
        single_step_llm_command_generator_fixture,
        multi_step_llm_command_generator_fixture,
    ):
        if request.param == "implementation_single_step":
            return single_step_llm_command_generator_fixture
        elif request.param == "implementation_multi_step":
            return multi_step_llm_command_generator_fixture
        else:
            raise ValueError("Unknown fixture type")

    @pytest.mark.asyncio
    async def test_predict_commands_no_flows(self, command_generator_fixture):
        generator = command_generator_fixture
        message = Message()
        message.data = {TEXT: "some_message"}
        tracker = Mock(spec=DialogueStateTracker)
        flows = FlowsList(underlying_flows=[])

        commands = await generator.predict_commands(message, flows, tracker)

        assert commands == []

    @pytest.mark.asyncio
    async def test_predict_commands_no_tracker(self, command_generator_fixture):
        generator = command_generator_fixture
        message = Message()
        message.data = {TEXT: "some_message"}
        flows = FlowsList(underlying_flows=[])

        commands = await generator.predict_commands(message, flows, None)

        assert commands == []

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (
                "SetSlot(test_slot, 1234)",
                [SetSlotCommand(name="test_slot", value="1234")],
            ),
            (
                "SetSlot(phone_number, (412) 555-1234)",
                [SetSlotCommand(name="phone_number", value="(412) 555-1234")],
            ),
        ],
    )
    def test_parse_commands_uses_correct_regex(
        self,
        input_action: Optional[str],
        expected_command: Command,
        command_generator_fixture,
    ):
        """Test that parse_commands uses the expected regex."""
        generator = command_generator_fixture
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
        parsed_commands = generator.parse_commands(input_action, Mock(), test_flows)
        # Then
        assert parsed_commands == expected_command

    ## Test flow retrieval
    @pytest.mark.parametrize(
        "flow_retrieval_active, expected_initialization",
        [
            (True, True),  # Flow retrieval enabled
            (False, False),  # Flow retrieval disabled
        ],
    )
    async def test_flow_retrieval_initialization(
        self,
        flow_retrieval_active: bool,
        expected_initialization: bool,
        command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # When
        config = {
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: flow_retrieval_active},
        }
        generator_class = command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        # Then
        assert (generator.flow_retrieval is not None) == expected_initialization

    @pytest.mark.parametrize(
        "flow_retrieval_active, should_call_flow_retrieval",
        [
            (True, True),  # Flow retrieval enabled
            (False, False),  # Flow retrieval disabled
        ],
    )
    async def test_predict_commands_with_flow_retrieval(
        self,
        flow_retrieval_active: bool,
        should_call_flow_retrieval: bool,
        command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
        flows: FlowsList,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        config = {
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: flow_retrieval_active},
        }
        generator_class = command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        if should_call_flow_retrieval:
            # Mock the flow retrieval filter_flows method
            generator.flow_retrieval = Mock()
            generator.flow_retrieval.filter_flows = AsyncMock(return_value=flows)

        # When
        await generator.predict_commands(
            Message(),
            flows,
            DialogueStateTracker.from_events(
                "test",
                evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
            ),
        )

        # Then
        if should_call_flow_retrieval:
            generator.flow_retrieval.filter_flows.assert_called_once()
        else:
            assert generator.flow_retrieval is None

    async def test_predict_commands_with_mocked_flow_retrieval(
        self,
        command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
        flows: FlowsList,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        config = {
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: True},
        }
        generator_class = command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        # Mock the flow retrieval filter_flows method to return
        # a specific subset of flows
        mock_filtered_flows = FlowsList(underlying_flows=[])
        generator.flow_retrieval = Mock()
        generator.flow_retrieval.filter_flows = AsyncMock(
            return_value=mock_filtered_flows
        )

        # When
        await generator.predict_commands(
            Message(),
            flows,
            DialogueStateTracker.from_events(
                "test",
                evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
            ),
        )

        # Then
        generator.flow_retrieval.filter_flows.assert_called_once()

    # Tests for prepare_flows_for_template method
    @pytest.fixture
    def flows_with_collect_steps(self) -> FlowsList:
        """Create a FlowsList with flows that have collect steps."""
        return flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: test_slot
                  description: test_slot
                  ask_before_filling: false
                - id: second_step
                  collect: another_slot
                  description: another_slot
                  ask_before_filling: true
              another_flow:
                name: another flow
                description: another test flow
                steps:
                - id: third_step
                  collect: third_slot
                  description: third_slot
                  ask_before_filling: false
            """
        )

    @pytest.fixture
    def tracker_with_slots(self) -> DialogueStateTracker:
        """Create a tracker with slots."""
        slots = [
            TextSlot(name="test_slot", mappings=[]),
            TextSlot(name="another_slot", mappings=[]),
            TextSlot(name="third_slot", mappings=[]),
        ]
        return DialogueStateTracker.from_events("test", evts=[], slots=slots)

    @pytest.fixture
    def tracker_with_set_slots(self) -> DialogueStateTracker:
        """Create a tracker with some slots already set."""
        slots = [
            TextSlot(name="test_slot", mappings=[]),
            TextSlot(name="another_slot", mappings=[]),
            TextSlot(name="third_slot", mappings=[]),
        ]
        events = [SlotSet("another_slot", "already_set")]
        return DialogueStateTracker.from_events("test", evts=events, slots=slots)

    def test_prepare_flows_for_template_without_agent_info(
        self,
        base_command_generator_fixture,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template when add_agent_info is False."""
        generator = base_command_generator_fixture

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_slots, add_agent_info=False
        )

        assert len(result) == 2

        # Check first flow
        first_flow = result[0]
        assert first_flow["name"] == "test_flow"
        assert first_flow["description"] == "some test flow"
        assert "agent_info" not in first_flow
        # Only test_slot should be included (ask_before_filling=false)
        # another_slot is not included because ask_before_filling=true and not set
        assert len(first_flow["slots"]) == 1

        # Check slots in first flow
        test_slot = first_flow["slots"][0]
        assert test_slot["name"] == "test_slot"
        assert test_slot["description"] == "test_slot"

        # Check second flow
        second_flow = result[1]
        assert second_flow["name"] == "another_flow"
        assert second_flow["description"] == "another test flow"
        assert "agent_info" not in second_flow
        assert len(second_flow["slots"]) == 1

    def test_prepare_flows_for_template_with_agent_info_no_agents(
        self,
        base_command_generator_fixture,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template when add_agent_info is True but no
        agents exist.
        """
        generator = base_command_generator_fixture

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_slots, add_agent_info=True
        )

        assert len(result) == 2

        # Check that agent_info is not included when no agents exist
        for flow in result:
            assert "agent_info" not in flow

    @patch("rasa.core.available_agents.AvailableAgents.get_agent_config")
    def test_prepare_flows_for_template_with_agent_info_with_agents(
        self,
        mock_get_agent_config,
        base_command_generator_fixture,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template when add_agent_info is True and
        agents exist.
        """
        generator = base_command_generator_fixture

        # Mock agent configuration
        mock_agent_config = Mock()
        mock_agent_config.agent.name = "Test Agent"
        mock_agent_config.agent.description = "A test agent"
        mock_get_agent_config.return_value = mock_agent_config

        # Add AgentStarted events to tracker
        tracker_with_slots.update(AgentStarted("agent1", "test_flow"))
        tracker_with_slots.update(AgentStarted("agent2", "another_flow"))

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_slots, add_agent_info=True
        )

        assert len(result) == 2

        # Check first flow has agent_info
        first_flow = result[0]
        assert first_flow["name"] == "test_flow"
        assert "agent_info" in first_flow
        assert len(first_flow["agent_info"]) == 1
        assert first_flow["agent_info"][0]["name"] == "Test Agent"
        assert first_flow["agent_info"][0]["description"] == "A test agent"

        # Check second flow has agent_info
        second_flow = result[1]
        assert second_flow["name"] == "another_flow"
        assert "agent_info" in second_flow
        assert len(second_flow["agent_info"]) == 1
        assert second_flow["agent_info"][0]["name"] == "Test Agent"
        assert second_flow["agent_info"][0]["description"] == "A test agent"

        # Verify get_agent_config was called for each agent
        assert mock_get_agent_config.call_count == 2
        mock_get_agent_config.assert_any_call("agent1")
        mock_get_agent_config.assert_any_call("agent2")

    @patch("rasa.core.available_agents.AvailableAgents.get_agent_config")
    def test_prepare_flows_for_template_with_agent_info_none_agent_config(
        self,
        mock_get_agent_config,
        base_command_generator_fixture,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template when agent config is None."""
        generator = base_command_generator_fixture

        # Mock agent configuration to return None
        mock_get_agent_config.return_value = None

        # Add AgentStarted events to tracker
        tracker_with_slots.update(AgentStarted("agent1", "test_flow"))

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_slots, add_agent_info=True
        )

        assert len(result) == 2

        # Check that agent_info is not included when agent config is None
        for flow in result:
            assert "agent_info" not in flow

        # Verify get_agent_config was called
        mock_get_agent_config.assert_called_once_with("agent1")

    @patch("rasa.core.available_agents.AvailableAgents.get_agent_config")
    def test_prepare_flows_for_template_with_agent_info_multiple_agents_same_flow(
        self,
        mock_get_agent_config: MagicMock,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template with multiple agents for the same flow."""
        generator = base_command_generator_fixture

        # Mock agent configurations
        mock_agent_config1 = Mock()
        mock_agent_config1.agent.name = "Agent 1"
        mock_agent_config1.agent.description = "First agent"

        mock_agent_config2 = Mock()
        mock_agent_config2.agent.name = "Agent 2"
        mock_agent_config2.agent.description = "Second agent"

        mock_get_agent_config.side_effect = [mock_agent_config1, mock_agent_config2]

        # Add multiple AgentStarted events for the same flow
        tracker_with_slots.update(AgentStarted("agent1", "test_flow"))
        tracker_with_slots.update(AgentStarted("agent2", "test_flow"))

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_slots, add_agent_info=True
        )

        assert len(result) == 2

        # Check first flow has both agents in agent_info
        first_flow = result[0]
        assert first_flow["name"] == "test_flow"
        assert "agent_info" in first_flow
        assert len(first_flow["agent_info"]) == 2

        agent_names = [agent["name"] for agent in first_flow["agent_info"]]
        agent_descriptions = [
            agent["description"] for agent in first_flow["agent_info"]
        ]

        assert "Agent 1" in agent_names
        assert "Agent 2" in agent_names
        assert "First agent" in agent_descriptions
        assert "Second agent" in agent_descriptions

        # Check second flow has no agent_info
        second_flow = result[1]
        assert second_flow["name"] == "another_flow"
        assert "agent_info" not in second_flow

    def test_prepare_flows_for_template_with_extractable_slots(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        flows_with_collect_steps: FlowsList,
        tracker_with_set_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template with slots that are extractable."""
        generator = base_command_generator_fixture

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_set_slots, add_agent_info=False
        )

        assert len(result) == 2

        # Check that slots are included based on extractability
        first_flow = result[0]
        # Both slots should be included:
        # - test_slot (ask_before_filling=False)
        # - another_slot (already set in tracker_with_set_slots)
        assert len(first_flow["slots"]) == 2

        # test_slot should be included (ask_before_filling=False)
        test_slot = first_flow["slots"][0]
        assert test_slot["name"] == "test_slot"

        # another_slot should be included (already set)
        another_slot = first_flow["slots"][1]
        assert another_slot["name"] == "another_slot"

    def test_prepare_flows_for_template_with_no_extractable_slots(
        self,
        base_command_generator_fixture,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template with no extractable slots."""
        generator = base_command_generator_fixture

        # Create a flow with only non-extractable slots
        flows_no_extractable = flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: test_slot
                  description: test_slot
                  ask_before_filling: true
            """
        )

        result = generator.prepare_flows_for_template(
            flows_no_extractable, tracker_with_slots, add_agent_info=False
        )

        assert len(result) == 1

        # Check that no slots are included
        first_flow = result[0]
        assert first_flow["name"] == "test_flow"
        assert len(first_flow["slots"]) == 0

    def test_prepare_flows_for_template_empty_flows(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test prepare_flows_for_template with empty flows list."""
        generator = base_command_generator_fixture

        empty_flows = FlowsList(underlying_flows=[])

        result = generator.prepare_flows_for_template(
            empty_flows, tracker_with_slots, add_agent_info=True
        )

        assert len(result) == 0

    @patch("rasa.core.available_agents.AvailableAgents.get_agent_config")
    def test_prepare_flows_for_template_agent_info_structure(
        self,
        mock_get_agent_config: MagicMock,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        flows_with_collect_steps: FlowsList,
        tracker_with_slots: DialogueStateTracker,
    ):
        """Test that agent_info has the correct structure."""
        generator = base_command_generator_fixture

        # Mock agent configuration
        mock_agent_config = Mock()
        mock_agent_config.agent.name = "Test Agent"
        mock_agent_config.agent.description = "A test agent description"
        mock_get_agent_config.return_value = mock_agent_config

        # Add AgentStarted event to tracker
        tracker_with_slots.update(AgentStarted("agent1", "test_flow"))

        result = generator.prepare_flows_for_template(
            flows_with_collect_steps, tracker_with_slots, add_agent_info=True
        )

        # Check agent_info structure
        first_flow = result[0]
        agent_info = first_flow["agent_info"][0]

        assert "name" in agent_info
        assert "description" in agent_info
        assert agent_info["name"] == "Test Agent"
        assert agent_info["description"] == "A test agent description"

        # Verify no extra fields are present
        expected_keys = {"name", "description"}
        assert set(agent_info.keys()) == expected_keys

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows"
    )
    async def test_predict_commands_and_flow_retrieval_api_error_throws_exception(
        self,
        mock_flow_retrieval_filter_flows: MagicMock,
        command_generator_fixture: LLMBasedCommandGenerator,
        tracker: DialogueStateTracker,
        flows: FlowsList,
    ) -> None:
        generator = command_generator_fixture
        message = Message()
        message.data = {TEXT: "some_message"}
        mock_flow_retrieval_filter_flows.side_effect = ProviderClientAPIException(
            message="Test Exception", original_exception=Exception("API exception")
        )

        predicted_commands = await generator.predict_commands(message, flows, tracker)

        mock_flow_retrieval_filter_flows.assert_called_once()

        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    ### Tests for methods implemented in the base class
    # Parameterized fixture
    @pytest.fixture(
        params=[
            "base_class",
            "implementation_single_step",
            "implementation_multi_step",
        ]
    )
    def base_command_generator_fixture(
        self,
        request: pytest.FixtureRequest,
        base_class_fixture: LLMBasedCommandGenerator,
        single_step_llm_command_generator_fixture: LLMBasedCommandGenerator,
        multi_step_llm_command_generator_fixture: LLMBasedCommandGenerator,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        if request.param == "base_class":
            config = {
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: True},
            }
            return base_class_fixture(
                config=config, model_storage=model_storage, resource=resource
            )
        if request.param == "implementation_single_step":
            return single_step_llm_command_generator_fixture
        elif request.param == "implementation_multi_step":
            return multi_step_llm_command_generator_fixture
        else:
            raise ValueError("Unknown fixture type")

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_generate_action_list_calls_llm_factory_correctly(
        self,
        mock_llm_factory: Mock,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        llm_response_dict: Dict[Text, Any],
    ):
        """Test that _generate_action_list calls llm correctly."""
        command_generator = base_command_generator_fixture

        # Given
        expected_llm_config = {
            MODEL_CONFIG_KEY: "gpt-4-0613",
            PROVIDER_CONFIG_KEY: "openai",
            TIMEOUT_CONFIG_KEY: 7,
            TEMPERATURE_CONFIG_KEY: 0.0,
            MAX_COMPLETION_TOKENS_CONFIG_KEY: 256,
        }

        mock_raw_response = AsyncMock()
        mock_raw_response.to_dict = MagicMock(return_value=llm_response_dict)
        mock_llm_client = AsyncMock()
        mock_llm_client.acompletion.return_value = mock_raw_response
        mock_llm_factory.return_value = mock_llm_client

        # When
        await command_generator.invoke_llm(LLMInput(prompt="some prompt", metadata={}))

        # Then
        mock_llm_factory.assert_called_once_with(None, expected_llm_config)

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_generate_action_list_calls_llm_correctly(
        self,
        mock_llm_factory: Mock,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        llm_response_dict: Dict[Text, Any],
    ):
        """Test that _generate_action_list calls llm correctly."""
        # Given
        command_generator = base_command_generator_fixture

        mock_raw_response = AsyncMock()
        mock_raw_response.to_dict = MagicMock(return_value=llm_response_dict)
        mock_llm_client = AsyncMock()
        mock_llm_client.acompletion.return_value = mock_raw_response
        mock_llm_factory.return_value = mock_llm_client

        # When
        await command_generator.invoke_llm(LLMInput(prompt="some prompt", metadata={}))
        # Then
        mock_llm_client.acompletion.assert_called_once_with("some prompt", metadata={})

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_generate_action_list_catches_llm_exception(
        self,
        mock_llm_factory: Mock,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test that _generate_action_list calls llm correctly."""
        command_generator = base_command_generator_fixture
        mock_llm = AsyncMock()
        mock_llm.acompletion = AsyncMock(side_effect=Exception("API exception"))
        mock_llm_factory.return_value = mock_llm

        # When
        with capture_logs() as logs:
            with pytest.raises(ProviderClientAPIException):
                await command_generator.invoke_llm(
                    LLMInput(prompt="some prompt", metadata={})
                )

            # Then
            assert len(logs) == 1
            assert logs[0]["event"] == "llm_based_command_generator.llm.error"

    @pytest.mark.parametrize(
        "slot, slot_name, expected_output",
        [
            (TextSlot("test_slot", [], initial_value="hello"), "test_slot", "hello"),
            (TextSlot("test_slot", []), "some_other_slot", "undefined"),
        ],
    )
    def test_slot_value(
        self,
        slot: Slot,
        slot_name: str,
        expected_output: str,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test that slot_value returns the correct string."""
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events("test", evts=[], slots=[slot])
        # When
        slot_value = command_generator.get_slot_value(tracker, slot_name)

        assert slot_value == expected_output

    @pytest.fixture
    def collect_info_step(self) -> CollectInformationFlowStep:
        """Create a CollectInformationFlowStep."""
        return CollectInformationFlowStep(
            collect="test_slot",
            idx=0,
            ask_before_filling=True,
            utter="hello",
            collect_action="action_ask_hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            custom_id="collect",
            description="test_slot",
            metadata={},
            next="next_step",
            flow_id="test_flow",
        )

    def test_is_extractable_with_no_slot(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns False
        when there are no slots to be filled.
        """
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events(sender_id="test", evts=[], slots=[])
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert not is_extractable

    def test_is_extractable_when_slot_can_be_filled_without_asking(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test that is_extractable returns True when
        collect_information slot can be filled.
        """
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events(
            sender_id="test", evts=[], slots=[TextSlot(name="test_slot", mappings=[])]
        )
        collect_info_step = CollectInformationFlowStep(
            collect="test_slot",
            ask_before_filling=False,
            utter="hello",
            collect_action="action_ask_hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            custom_id="collect_information",
            idx=0,
            description="test_slot",
            metadata={},
            next="next_step",
            flow_id="test_flow",
        )
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert is_extractable

    def test_is_extractable_when_slot_has_already_been_set(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns True
        when collect_information can be filled.
        """
        command_generator = base_command_generator_fixture
        # Given
        slot = TextSlot(name="test_slot", mappings=[])
        tracker = DialogueStateTracker.from_events(
            sender_id="test", evts=[SlotSet("test_slot", "hello")], slots=[slot]
        )
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert is_extractable

    # Tests for prepare_current_flow_slots_for_template method
    def test_prepare_current_flow_slots_for_template_basic(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test prepare_current_flow_slots_for_template with basic flow and slots."""
        generator = base_command_generator_fixture

        flows = flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: slot_1
                  description: "Slot 1 description"
                  ask_before_filling: false
                - id: second_step
                  collect: slot_2
                  description: "Slot 2 description"
                  ask_before_filling: false
            """
        )

        tracker = DialogueStateTracker.from_events(
            "test",
            evts=[],
            slots=[
                TextSlot(name="slot_1", mappings=[]),
                TextSlot(name="slot_2", mappings=[]),
            ],
        )

        top_flow = flows.flow_by_id("test_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id("first_step")

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        assert len(result) == 2
        assert result[0]["name"] == "slot_1"
        assert result[0]["value"] == "undefined"
        assert result[0]["description"] == "Slot 1 description"
        assert result[1]["name"] == "slot_2"
        assert result[1]["value"] == "undefined"
        assert result[1]["description"] == "Slot 2 description"

    def test_prepare_current_flow_slots_for_template_deduplicates_by_slot_name(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test that prepare_current_flow_slots_for_template deduplicates slots by name.

        This test covers the fix for slot duplication when a flow calls other flows
        that collect the same slot multiple times.
        """
        generator = base_command_generator_fixture

        flows = flows_from_str(
            """
            flows:
              main_flow:
                name: main flow
                description: main flow that calls subflows
                steps:
                - id: step1
                  collect: slot_support_ticket_title
                  description: "Title of the ticket"
                  ask_before_filling: false
                - id: step2
                  call: subflow_1
                - id: step3
                  call: subflow_2
                - id: step4
                  collect: slot_licenses_for_escalation
                  description: "License IDs"
                  ask_before_filling: false
              subflow_1:
                name: subflow 1
                description: subflow that collects slot_support_ticket_title
                steps:
                - id: sub_step1
                  collect: slot_support_ticket_title
                  description: "Title of the ticket"
                  ask_before_filling: false
              subflow_2:
                name: subflow 2
                description: subflow that also collects slot_support_ticket_title
                steps:
                - id: sub_step2
                  collect: slot_support_ticket_title
                  description: "Title of the ticket"
                  ask_before_filling: false
                - id: sub_step3
                  call: subflow_3
              subflow_3:
                name: subflow 3
                description: nested subflow
                steps:
                - id: sub_step4
                  collect: slot_support_ticket_title
                  description: "Title of the ticket"
                  ask_before_filling: false
        """
        )

        tracker = DialogueStateTracker.from_events(
            "test",
            evts=[],
            slots=[
                TextSlot(name="slot_support_ticket_title", mappings=[]),
                TextSlot(name="slot_licenses_for_escalation", mappings=[]),
            ],
        )

        top_flow = flows.flow_by_id("main_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id("step1")

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        # Should only have 2 unique slots, not duplicates
        slot_names = [slot["name"] for slot in result]
        assert len(slot_names) == 2
        assert "slot_support_ticket_title" in slot_names
        assert "slot_licenses_for_escalation" in slot_names

        # slot_support_ticket_title should appear only once
        assert slot_names.count("slot_support_ticket_title") == 1
        assert slot_names.count("slot_licenses_for_escalation") == 1

        # The first occurrence should be kept (from main_flow)
        slot_ticket_title = next(
            slot for slot in result if slot["name"] == "slot_support_ticket_title"
        )
        assert slot_ticket_title["description"] == "Title of the ticket"

    @pytest.mark.parametrize(
        "slot_1_value, slot_2_value, expected_value_1, expected_value_2",
        [
            (None, None, "undefined", "undefined"),
            ("value_1", "value_2", "value_1", "value_2"),
            ("value_1", None, "value_1", "undefined"),
            (None, "value_2", "undefined", "value_2"),
        ],
    )
    def test_prepare_current_flow_slots_for_template_with_slot_values(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        slot_1_value: Optional[str],
        slot_2_value: Optional[str],
        expected_value_1: str,
        expected_value_2: str,
    ):
        """Test prepare_current_flow_slots_for_template with various slot value
        combinations."""
        generator = base_command_generator_fixture

        flows = flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: slot_1
                  description: "Slot 1 description"
                  ask_before_filling: false
                - id: second_step
                  collect: slot_2
                  description: "Slot 2 description"
                  ask_before_filling: false
            """
        )

        events = []
        if slot_1_value is not None:
            events.append(SlotSet("slot_1", slot_1_value))
        if slot_2_value is not None:
            events.append(SlotSet("slot_2", slot_2_value))

        tracker = DialogueStateTracker.from_events(
            "test",
            evts=events,
            slots=[
                TextSlot(name="slot_1", mappings=[]),
                TextSlot(name="slot_2", mappings=[]),
            ],
        )

        top_flow = flows.flow_by_id("test_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id("first_step")

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        assert len(result) == 2
        assert result[0]["name"] == "slot_1"
        assert result[0]["value"] == expected_value_1
        assert result[1]["name"] == "slot_2"
        assert result[1]["value"] == expected_value_2

    @pytest.mark.parametrize(
        "ask_before_filling_1, ask_before_filling_2, slot_1_set, current_step_id, expected_count, expected_slots",  # noqa: E501
        [
            # Both extractable (ask_before_filling=False)
            (False, False, False, "first_step", 2, ["slot_1", "slot_2"]),
            # One extractable, one not (ask_before_filling=True)
            (False, True, False, "first_step", 1, ["slot_1"]),
            # Both non-extractable, but slot_1 is set
            (True, True, True, "first_step", 1, ["slot_1"]),
            # Both non-extractable, neither set, but current_step is collecting slot_1
            # (so slot_1 is extractable because we're currently asking for it)
            (True, True, False, "first_step", 1, ["slot_1"]),
            # Both non-extractable, neither set, current_step is not a collect step
            # (so neither slot is extractable)
            (True, True, False, None, 0, []),
        ],
    )
    def test_prepare_current_flow_slots_for_template_filters_by_extractability(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        ask_before_filling_1: bool,
        ask_before_filling_2: bool,
        slot_1_set: bool,
        current_step_id: Optional[str],
        expected_count: int,
        expected_slots: List[str],
    ):
        """Test that prepare_current_flow_slots_for_template filters slots based on
        extractability."""
        generator = base_command_generator_fixture

        flows = flows_from_str(
            f"""
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: slot_1
                  description: "Slot 1 description"
                  ask_before_filling: {str(ask_before_filling_1).lower()}
                - id: second_step
                  collect: slot_2
                  description: "Slot 2 description"
                  ask_before_filling: {str(ask_before_filling_2).lower()}
            """
        )

        events = []
        if slot_1_set:
            events.append(SlotSet("slot_1", "value"))

        tracker = DialogueStateTracker.from_events(
            "test",
            evts=events,
            slots=[
                TextSlot(name="slot_1", mappings=[]),
                TextSlot(name="slot_2", mappings=[]),
            ],
        )

        top_flow = flows.flow_by_id("test_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id(current_step_id) if current_step_id else None

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        assert len(result) == expected_count
        slot_names = [slot["name"] for slot in result]
        assert set(slot_names) == set(expected_slots)

    def test_prepare_current_flow_slots_for_template_with_current_collect_step(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test that current collect step is included even if
        ask_before_filling=True."""
        generator = base_command_generator_fixture

        flows = flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: current_slot
                  description: "Current slot being asked"
                  ask_before_filling: true
                - id: second_step
                  collect: other_slot
                  description: "Other slot"
                  ask_before_filling: false
            """
        )

        tracker = DialogueStateTracker.from_events(
            "test",
            evts=[],
            slots=[
                TextSlot(name="current_slot", mappings=[]),
                TextSlot(name="other_slot", mappings=[]),
            ],
        )

        top_flow = flows.flow_by_id("test_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id("first_step")

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        # Both slots should be included:
        # - current_slot (currently being asked, so extractable)
        # - other_slot (ask_before_filling=False)
        assert len(result) == 2
        slot_names = [slot["name"] for slot in result]
        assert "current_slot" in slot_names
        assert "other_slot" in slot_names

    def test_prepare_current_flow_slots_for_template_no_flow(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test prepare_current_flow_slots_for_template when top_flow is None."""
        generator = base_command_generator_fixture

        tracker = DialogueStateTracker.from_events("test", evts=[], slots=[])

        result = generator.prepare_current_flow_slots_for_template(None, None, tracker)

        assert result == []

    def test_prepare_current_flow_slots_for_template_no_collect_steps(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
    ):
        """Test prepare_current_flow_slots_for_template with flow that has no collect
        steps."""
        generator = base_command_generator_fixture

        flows = flows_from_str(
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

        tracker = DialogueStateTracker.from_events("test", evts=[], slots=[])

        top_flow = flows.flow_by_id("test_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id("first_step")

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        assert result == []

    @pytest.mark.parametrize(
        "slot_type, slot_name, slot_kwargs, expected_allowed_values",
        [
            (
                "TextSlot",
                "text_slot",
                {},
                None,
            ),
            (
                "CategoricalSlot",
                "category_slot",
                {"values": ["option1", "option2", "option3"]},
                "['option1', 'option2', 'option3']",
            ),
            (
                "CategoricalSlot",
                "category_slot_empty",
                {"values": []},
                "[]",
            ),
        ],
    )
    def test_prepare_current_flow_slots_for_template_with_different_slot_types(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        slot_type: str,
        slot_name: str,
        slot_kwargs: Dict[str, Any],
        expected_allowed_values: Optional[str],
    ):
        """Test prepare_current_flow_slots_for_template with different slot types.

        Note: allowed_values_for_slot returns a string representation of the list,
        not the actual list, which is why expected_allowed_values is a string.
        """
        from rasa.shared.core.slots import CategoricalSlot, TextSlot

        generator = base_command_generator_fixture

        flows = flows_from_str(
            f"""
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  collect: {slot_name}
                  description: "{slot_name} description"
                  ask_before_filling: false
            """
        )

        slot_class = TextSlot if slot_type == "TextSlot" else CategoricalSlot
        slot = slot_class(name=slot_name, mappings=[], **slot_kwargs)

        tracker = DialogueStateTracker.from_events(
            "test",
            evts=[],
            slots=[slot],
        )

        top_flow = flows.flow_by_id("test_flow")
        assert top_flow is not None
        current_step = top_flow.step_by_id("first_step")

        result = generator.prepare_current_flow_slots_for_template(
            top_flow, current_step, tracker
        )

        assert len(result) == 1
        assert result[0]["name"] == slot_name
        if expected_allowed_values is not None:
            assert result[0]["allowed_values"] == expected_allowed_values
        else:
            # TextSlot should not have allowed_values or it should be None/empty
            assert result[0].get("allowed_values") in (None, [])

    def test_is_extractable_with_current_step(
        self,
        base_command_generator_fixture: LLMBasedCommandGenerator,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns True when the current step is a collect
        information step and matches the information step.
        """
        command_generator = base_command_generator_fixture
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
        base_command_generator_fixture: LLMBasedCommandGenerator,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # Given
        config = {
            "user_input": {"max_characters": max_characters},
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: True},
        }
        generator_class = base_command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        message = Message.build(text=message)
        # When
        exceeds_limit = generator.check_if_message_exceeds_limit(message)
        assert exceeds_limit == expected_exceeds_limit

    def test_import_rasa_generators_from_generator_module(
        self, model_storage: ModelStorage, resource: Resource
    ):
        """Test that rasa generator modules can be imported
        without errors from generator module.
        """
        from rasa.dialogue_understanding.generator import (
            LLMCommandGenerator,
            MultiStepLLMCommandGenerator,
            SingleStepLLMCommandGenerator,
        )

        assert LLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert SingleStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert MultiStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    @pytest.mark.parametrize("generator_class", TEST_BASE_CLASSES)
    def test_get_llm_tracing_metadata_in_command_generators(
        self, generator_class, model_storage: ModelStorage, resource: Resource
    ) -> None:
        """Ensure get_llm_tracing_metadata returns correct metadata for generators."""
        generator = generator_class.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        sender_id = "user123"
        assistant_id = "assistant456"
        model_id = "model789"

        tracker = DialogueStateTracker(sender_id=sender_id, slots=[])
        tracker.assistant_id = assistant_id
        tracker.model_id = model_id

        metadata = generator.get_llm_tracing_metadata(tracker)
        assert metadata == {
            LANGFUSE_METADATA_SESSION_ID: sender_id,
            LANGFUSE_METADATA_TAGS: [generator.__class__.__name__],
            LANGFUSE_METADATA_CUSTOM_METADATA: {
                LANGFUSE_METADATA_AGENT_ID: assistant_id,
                LANGFUSE_METADATA_MODEL_ID: model_id,
                LANGFUSE_METADATA_COMPONENT_NAME: generator.__class__.__name__,
            },
        }

    @pytest.mark.parametrize("generator_class", TEST_BASE_CLASSES)
    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_invoke_llm_success_command_generators(
        self,
        mock_llm_factory: Mock,
        generator_class,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """Ensure invoke_llm calls LLM with metadata and returns response."""
        generator = generator_class.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        test_prompt = "Test prompt"
        test_metadata = {
            LANGFUSE_METADATA_SESSION_ID: "test_session",
            LANGFUSE_METADATA_TAGS: ["test_tag"],
            LANGFUSE_METADATA_CUSTOM_METADATA: {"key": "value"},
        }
        llm_input = LLMInput(prompt=test_prompt, metadata=test_metadata)

        mock_llm = Mock()
        mock_llm_response = LLMResponse(
            id="test-id",
            created=123456,
            choices=["Test response"],
            model="test-model",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=2),
        )
        mock_llm.acompletion = AsyncMock(return_value=mock_llm_response)
        mock_llm_factory.return_value = mock_llm

        result = await generator.invoke_llm(llm_input)

        mock_llm.acompletion.assert_called_once_with(
            test_prompt, metadata=test_metadata
        )
        assert result is not None
        assert result.choices == ["Test response"]

    @pytest.mark.parametrize("generator_class", TEST_BASE_CLASSES)
    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_invoke_llm_passes_metadata_command_generators(
        self,
        mock_llm_factory: Mock,
        generator_class,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """Ensure invoke_llm forwards provided metadata to the LLM client."""
        generator = generator_class.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        test_prompt = "Test prompt"
        test_metadata = {
            LANGFUSE_METADATA_SESSION_ID: "test_session_id",
            LANGFUSE_METADATA_TAGS: [generator.__class__.__name__],
            LANGFUSE_METADATA_CUSTOM_METADATA: {
                LANGFUSE_METADATA_AGENT_ID: "test_agent",
                LANGFUSE_METADATA_MODEL_ID: "test_model",
                LANGFUSE_METADATA_COMPONENT_NAME: generator.__class__.__name__,
            },
        }
        llm_input = LLMInput(prompt=test_prompt, metadata=test_metadata)

        mock_llm = Mock()
        mock_llm_response = LLMResponse(
            id="test-id",
            created=123456,
            choices=["Test response"],
            model="test-model",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=2),
        )
        mock_llm.acompletion = AsyncMock(return_value=mock_llm_response)
        mock_llm_factory.return_value = mock_llm

        await generator.invoke_llm(llm_input)

        call_args = mock_llm.acompletion.call_args
        assert call_args[0][0] == test_prompt
        assert call_args[1]["metadata"] == test_metadata

    def test_import_rasa_generators_directly(self, model_storage, resource):
        """Test that rasa generator modules can be imported
        without errors directly.
        """
        from rasa.dialogue_understanding.generator.llm_command_generator import (
            LLMCommandGenerator,
        )
        from rasa.dialogue_understanding.generator.multi_step.multi_step_llm_command_generator import (  # noqa: E501
            MultiStepLLMCommandGenerator,
        )
        from rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator import (  # noqa: E501
            CompactLLMCommandGenerator,
        )
        from rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator import (  # noqa: E501
            SearchReadyLLMCommandGenerator,
        )
        from rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator import (  # noqa: E501
            SingleStepLLMCommandGenerator,
        )

        assert LLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert SingleStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert MultiStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert CompactLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert SearchReadyLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    @pytest.mark.parametrize("base_class", TEST_BASE_CLASSES)
    async def test_new_subclass_uses_own_predict_commands(
        self, base_class, flows, model_storage, resource
    ):
        """Test that if custom component has overriden the predict_commands
        method, it will be called and not the parent's.
        """

        class CustomCommandGenerator(base_class):
            async def predict_commands(
                self,
                message: Message,
                flows: FlowsList,
                tracker: DialogueStateTracker = None,
            ):
                return [ChitChatAnswerCommand()]

        message = Mock()
        message.data = {TEXT: "some_message"}
        tracker = Mock(spec=DialogueStateTracker)
        flows = FlowsList(underlying_flows=[])

        generator = CustomCommandGenerator(
            config={}, model_storage=model_storage, resource=resource
        )
        result = await generator.predict_commands(message, flows, tracker)

        assert result == [ChitChatAnswerCommand()]
