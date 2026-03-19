from __future__ import annotations

import importlib
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    List,
    Optional,
    Text,
    Tuple,
    Type,
)
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.core.cancellation import CancellationToken
from rasa.agents.core.types import AgentStatus
from rasa.agents.protocol.mcp.mcp_open_agent import MCPOpenAgent
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.agents.schemas.agent_tool_result import AgentToolResult
from rasa.core.actions.action import (
    Action,
    CustomActionExecutor,
    NoEndpointCustomActionExecutor,
    RetryCustomActionExecutor,
)
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.core.actions.http_custom_action_executor import HTTPCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EB, EventBroker
from rasa.core.channels import OutputChannel, UserMessage
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.core.information_retrieval import (
    InformationRetrieval,
    SearchResult,
    SearchResultList,
)
from rasa.core.lock import TicketLock
from rasa.core.lock_store import LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command, StartFlowCommand
from rasa.dialogue_understanding.generator import (
    CompactLLMCommandGenerator,
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SearchReadyLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.engine.caching import LocalTrainingCache, TrainingCache
from rasa.engine.graph import (
    ExecutionContext,
    GraphComponent,
    GraphModelConfiguration,
    GraphNode,
)
from rasa.engine.recipes.graph_recipe import GraphV1Recipe
from rasa.engine.recipes.recipe import Recipe
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SlotSet, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.data import TrainingType
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.llm import LLMInput, StreamingConfig
from rasa.shared.utils.yaml import read_yaml_file
from rasa.tracing.instrumentation.instrumentation import (
    FLOW_EXECUTOR_MODULE_NAME,
    _instrumented_module_boolean_attribute_name,
    _mangled_instrumented_boolean_attribute_name,
)
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer


@pytest.fixture(scope="session")
def tracer_provider() -> TracerProvider:
    return TracerProvider()


@pytest.fixture(scope="session")
def span_exporter(tracer_provider: TracerProvider) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()  # type: ignore
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture(autouse=True)
def clear_span_exporter(span_exporter: InMemorySpanExporter) -> None:
    """Clear the span exporter before each test to ensure test isolation."""
    span_exporter.clear()


@pytest.fixture(scope="function")
def previous_num_captured_spans(span_exporter: InMemorySpanExporter) -> int:
    captured_spans = span_exporter.get_finished_spans()  # type: ignore
    return len(captured_spans)


@pytest.fixture(autouse=True)
def reset_instrumentation_state():
    """Reset instrumentation state between tests to ensure test isolation.

    This fixture properly addresses the instrumentation reset challenge by:
    1. Clearing instrumentation flags to allow re-instrumentation
    2. Restoring original functions from __wrapped__ attributes (when available)
    3. Using defensive programming to handle edge cases

    This approach leverages both the instrumentation system's idempotency mechanisms
    and the fact that functools.wraps creates __wrapped__ attributes for restoration.
    """

    def _restore_function_if_wrapped(module, func_name):
        """Safely restore a function from its __wrapped__ attribute if it exists."""
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if hasattr(func, "__wrapped__"):
                # Function is wrapped, restore the original
                original_func = func.__wrapped__
                setattr(module, func_name, original_func)
                return True
        return False

    # Reset module instrumentation flags and restore original functions
    modules_to_reset = [
        FLOW_EXECUTOR_MODULE_NAME,
        "rasa.core.policies.flows.mcp_tool_executor",
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator",
        "rasa.dialogue_understanding.generator.multi_step.multi_step_llm_command_generator",
        "rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator",
        "rasa.dialogue_understanding.generator.single_step.search_ready_llm_command_generator",
        "rasa.dialogue_understanding.generator.llm_command_generator",
    ]

    for module_name in modules_to_reset:
        try:
            module = importlib.import_module(module_name)

            # Clear instrumentation flag
            flag_name = _instrumented_module_boolean_attribute_name(module_name)
            if hasattr(module, flag_name):
                delattr(module, flag_name)

            # Restore original functions for flow executor module
            if module_name == FLOW_EXECUTOR_MODULE_NAME:
                functions_to_restore = [
                    "advance_flows",
                    "advance_flows_until_next_action",
                    "run_step",
                    "_call_agent_with_retry",
                ]
                for func_name in functions_to_restore:
                    _restore_function_if_wrapped(module, func_name)

            # Restore original functions for MCP tool executor module
            elif module_name == "rasa.core.policies.flows.mcp_tool_executor":
                _restore_function_if_wrapped(module, "_execute_mcp_tool_call")

        except ImportError:
            # Module doesn't exist, skip silently
            pass

    # Reset class instrumentation flags for any instrumented classes
    try:
        flow_executor_module = importlib.import_module(FLOW_EXECUTOR_MODULE_NAME)

        # Reset any class-level instrumentation flags
        for attr_name in dir(flow_executor_module):
            attr_value = getattr(flow_executor_module, attr_name)
            if isinstance(attr_value, type):  # It's a class
                flag_name = _mangled_instrumented_boolean_attribute_name(attr_value)
                if hasattr(attr_value, flag_name):
                    delattr(attr_value, flag_name)
    except ImportError:
        pass


@pytest.fixture()
def default_model_storage(tmp_path: Path) -> ModelStorage:
    return LocalModelStorage.create(tmp_path)


class TestSpanExporter:
    def __init__(self, original_exporter: InMemorySpanExporter):
        self._original_exporter = original_exporter

    def get_previous_num_captured_spans(
        self, span_name_substrings_to_ignore: Optional[List[str]] = None
    ) -> int:
        captured_spans = self.get_finished_spans(span_name_substrings_to_ignore)
        return len(captured_spans)

    def get_finished_spans(
        self, span_name_substrings_to_ignore: Optional[List[str]] = None
    ) -> Type[Any, ...]:
        captured_spans = self._original_exporter.get_finished_spans()  # type: ignore
        if span_name_substrings_to_ignore:
            captured_spans = self._filter_out_spans(
                captured_spans, span_name_substrings_to_ignore
            )
        return captured_spans

    def _filter_out_spans(
        self,
        captured_spans: Tuple[Any, ...],
        span_name_substrings_to_ignore: List[str],
    ) -> Tuple[Any, ...]:
        if span_name_substrings_to_ignore:
            captured_spans = [
                span
                for span in captured_spans
                if not any(
                    substring in span.name
                    for substring in span_name_substrings_to_ignore
                )
            ]
        return captured_spans


class TrackerMock(DialogueStateTracker):
    def __init__(self, events: List[Any]) -> None:
        self.events = events
        self.sender_id = "test_id"
        self.slots = {
            "requested_slot": SlotSet(key="requested_slot", value="test_slot")
        }
        self.latest_message = UserUttered("Hello", {"name": "greet"})


class MockAgent(Agent):
    def __init__(self) -> None:
        self.processor = Mock(spec=MessageProcessor)
        self.processor.model_filename = "model_filename"

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        if not (
            hasattr(self.__class__.__base__, "handle_message")
            and callable(getattr(self.__class__.__base__, "handle_message"))
        ):
            pytest.fail(
                f"method handle_message not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

        return None

    @property
    def model_id(self) -> Optional[Text]:
        return "model_id"


class MockAgentWithToolCall(MockAgent):
    """Mock agent that has _execute_tool_call method for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._name = "test_agent"

    @property
    def protocol_type(self):
        from rasa.agents.core.types import ProtocolType

        return ProtocolType.MCP_OPEN

    async def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_input: Optional[AgentInput] = None,
    ) -> AgentToolResult:
        """Mock tool call execution (signature aligned with `MCPBaseAgent`)."""
        return AgentToolResult(
            tool_name=tool_name,
            result=f"Result for {tool_name}",
            is_error=False,
        )


class MockSubAgent(AgentProtocol):
    """Mock for AgentProtocol classes (subagents)."""

    def __init__(self) -> None:
        self.processor = Mock(spec=MessageProcessor)
        self.processor.model_filename = "model_filename"

    @classmethod
    def from_config(cls, config) -> "MockSubAgent":
        return cls()

    @property
    def protocol_type(self):
        from rasa.agents.core.types import ProtocolType

        return ProtocolType.MCP_OPEN

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def process_input(self, input: AgentInput) -> AgentInput:
        return input

    async def run(
        self, input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to Agent/server and return response."""
        return AgentOutput(
            id=input.id,
            status=AgentStatus.COMPLETED,
            response_message="Test response",
        )

    @property
    def model_id(self) -> Optional[Text]:
        return "model_id"


class MockMessageProcessor(MessageProcessor):
    def __init__(self, events: List[Any]) -> None:
        self.fail_if_undefined("handle_message")
        self.fail_if_undefined("log_message")
        self.fail_if_undefined("get_tracker")
        self.fail_if_undefined("_run_action")
        self.fail_if_undefined("save_tracker")
        self.fail_if_undefined("_run_prediction_loop")

        self.tracker_mock = TrackerMock(events)

    def fail_if_undefined(self, method_name: str) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method {method_name} not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        pass

    async def log_message(
        self, message: UserMessage, should_save_tracker: bool = True
    ) -> None:
        pass

    async def get_tracker(
        self, conversation_id: str, user_id: Optional[str] = None
    ) -> TrackerMock:
        return self.tracker_mock

    async def _run_action(
        self,
        action: Action,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> bool:
        return True

    def save_tracker(self, tracker: Mock) -> None:
        pass

    async def _run_prediction_loop(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> None:
        pass

    def _predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        output_channel: Optional[OutputChannel] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Mock:
        return Mock()


class MockGraphNode(GraphNode):
    pass


class MockGraphComponent(GraphComponent):
    @classmethod
    def create(
        cls,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> MockGraphComponent:
        return cls()

    def mock_fn(self) -> None:
        pass


class MockEventBroker(EventBroker):
    def __init__(self) -> None:
        pass

    @classmethod
    async def from_endpoint_config(
        cls: Type[EB],
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> Optional[EB]:
        pass

    def publish(self, event: Dict[Text, Any]) -> None:
        pass


class MockTrackerStore(TrackerStore):
    # `Optional` seems required for mypy to not throw `missing return statement` errors,
    #  although this might be a bug: https://github.com/python/mypy/issues/10297
    def __init__(self, event_broker: Optional[EventBroker]):
        self.event_broker = event_broker

    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        pass

    def keys(self) -> Optional[Iterable[Text]]:
        pass

    def save(self, tracker: DialogueStateTracker) -> None:
        pass

    async def _stream_new_events(
        self,
        event_broker: EventBroker,
        new_events: List[Event],
        sender_id: Text,
        user_id: Optional[str] = None,
    ) -> None:
        if not (
            hasattr(self.__class__.__base__, "_stream_new_events")
            and callable(getattr(self.__class__.__base__, "_stream_new_events"))
        ):
            pytest.fail(
                f"method '_stream_new_events' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockLockStore(LockStore):
    @asynccontextmanager
    async def lock(
        self,
        conversation_id: Text,
        lock_lifetime: float = 60,
        wait_time_in_seconds: float = 1,
    ) -> AsyncGenerator[TicketLock, None]:
        if not (
            hasattr(self.__class__.__base__, "lock")
            and callable(getattr(self.__class__.__base__, "lock"))
        ):
            pytest.fail(
                f"method lock not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

        yield TicketLock(conversation_id)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        pass

    def delete_lock(self, conversation_id: Text) -> None:
        pass

    def save_lock(self, lock: TicketLock) -> None:
        pass


class MockGraphTrainer(GraphTrainer):
    def __init__(
        self,
        default_model_storage: ModelStorage,
        cache: TrainingCache,
        graph_runner_class: Type[GraphRunner],
    ) -> None:
        self.fail_if_undefined("train")
        super().__init__(default_model_storage, cache, graph_runner_class)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockLLMCommandgenerator(LLMCommandGenerator):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.fail_if_undefined("invoke_llm")
        super().__init__(config, model_storage, resource)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[str]:
        pass


class MockSingleStepLLMCommandGenerator(SingleStepLLMCommandGenerator):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        self.fail_if_undefined("invoke_llm")
        super().__init__(config, model_storage, resource, prompt_template)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[str]:
        pass


class MockCompactLLMCommandGenerator(CompactLLMCommandGenerator):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        self.fail_if_undefined("invoke_llm")
        super().__init__(config, model_storage, resource, prompt_template)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[str]:
        pass


class MockSearchReadyLLMCommandGenerator(SearchReadyLLMCommandGenerator):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        self.fail_if_undefined("invoke_llm")
        super().__init__(config, model_storage, resource, prompt_template)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[str]:
        pass


class MockMultiStepLLMCommandGenerator(MultiStepLLMCommandGenerator):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_templates: Optional[Dict[Text, Optional[Text]]] = None,
    ) -> None:
        self.fail_if_undefined("invoke_llm")
        super().__init__(config, model_storage, resource, prompt_templates)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[str]:
        pass


class MockCommand(Command):
    def __init__(self) -> None:
        pass

    @classmethod
    def type(cls) -> None:
        pass

    @classmethod
    def command(cls) -> None:
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> None:
        pass

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        if not (
            hasattr(self.__class__.__base__, "run_command_on_tracker")
            and callable(getattr(self.__class__.__base__, "run_command_on_tracker"))
        ):
            pytest.fail(
                f"method 'run_command_on_tracker' not found in "
                f"{self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )
        return []


class MockContextualResponseRephraser(ContextualResponseRephraser):
    def __init__(self, endpoint_config: EndpointConfig, domain: Domain) -> None:
        self.fail_if_undefined("_generate_llm_response")
        super().__init__(endpoint_config, domain)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    async def _generate_llm_response(
        self,
        llm_input: LLMInput,
        output_channel: Any,
        recipient_id: str,
        streaming_config: Optional[StreamingConfig] = None,
    ) -> Optional[LLMResponse]:
        """Mock implementation that returns a dummy LLMResponse."""
        return LLMResponse(
            id="mock_id",
            created=0,
            choices=["Mock response"],
            model="mock_model",
        )

    async def _create_history(self, tracker: DialogueStateTracker) -> Optional[str]:
        pass

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        pass


@pytest.fixture()
def graph_trainer(
    default_model_storage: LocalModelStorage,
    temp_cache: LocalTrainingCache,
) -> MockGraphTrainer:
    return MockGraphTrainer(default_model_storage, temp_cache, DaskGraphRunner)


def model_configuration(
    config_path: Text, training_type: TrainingType
) -> GraphModelConfiguration:
    config = read_yaml_file(config_path)

    recipe = Recipe.recipe_for_name(GraphV1Recipe.name)
    model_config = recipe.graph_config_for_recipe(
        config,
        {},
        training_type=training_type,
    )

    return model_config


class MockPolicy(Policy):
    async def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        pass

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional["TrackerFeaturizer"] = None,
    ) -> None:
        self.fail_if_undefined("_prediction")
        super().__init__(config, model_storage, resource, execution_context, featurizer)

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    def _prediction(
        self,
        probabilities: List[float],
        events: Optional[List[Event]] = None,
        optional_events: Optional[List[Event]] = None,
        is_end_to_end_prediction: bool = False,
        is_no_user_prediction: bool = False,
        diagnostic_data: Optional[Dict[Text, Any]] = None,
        action_metadata: Optional[Dict[Text, Any]] = None,
    ) -> PolicyPrediction:
        pass


class MockInformationRetrieval(InformationRetrieval):
    def __init__(self) -> None:
        self.fail_if_undefined("search")

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )

    def connect(
        self,
        config: EndpointConfig,
    ) -> None:
        pass

    async def search(
        self,
        query: Text,
        tracker_state: Dict[Text, Any],
        threshold: float = 0.0,
    ) -> SearchResultList:
        return SearchResultList(
            results=[
                SearchResult(text="Some content", metadata={"source": "docs/test.txt"}),
            ],
            metadata={"total_results": 1},
        )


class MockNLUCommandAdapter(NLUCommandAdapter):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        self.fail_if_undefined("predict_commands")
        super().__init__(config, model_storage, resource, execution_context)

    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        **kwargs: Any,
    ) -> List[Command]:
        return [StartFlowCommand(flow="health_advice")]

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockEndpointConfig(EndpointConfig):
    def __init__(
        self,
        url: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        basic_auth: Optional[Dict[str, str]] = None,
        token: Optional[str] = None,
        token_name: str = "token",
        cafile: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.fail_if_undefined("request")
        super().__init__(
            url, params, headers, basic_auth, token, token_name, cafile, **kwargs
        )

    async def request(
        self,
        method: Text = "post",
        subpath: Optional[Text] = None,
        content_type: Optional[Text] = "application/json",
        compress: bool = False,
        **kwargs: Any,
    ) -> Optional[Any]:
        return None

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockCustomActionExecutor(CustomActionExecutor):
    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        if not (
            hasattr(self.__class__.__base__, "run")
            and callable(getattr(self.__class__.__base__, "run"))
        ):
            pytest.fail(
                f"method 'run' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockNoEndpointCustomActionExecutor(NoEndpointCustomActionExecutor):
    def __init__(self, action_name: str) -> None:
        self.fail_if_undefined("run")
        super().__init__(action_name)

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        pass

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockRetryCustomActionExecutor(RetryCustomActionExecutor):
    def __init__(self, custom_action_executor: CustomActionExecutor) -> None:
        self.fail_if_undefined("run")
        super().__init__(custom_action_executor)

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        pass

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockGRPCCustomActionExecutor(GRPCCustomActionExecutor):
    def __init__(
        self,
        action_name: str,
        action_endpoint: EndpointConfig,
    ) -> None:
        self.fail_if_undefined("run")
        super().__init__(action_name, action_endpoint)

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        pass

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


class MockHTTPCustomActionExecutor(HTTPCustomActionExecutor):
    def __init__(
        self,
        action_name: str,
        action_endpoint: EndpointConfig,
    ) -> None:
        self.fail_if_undefined("run")
        super().__init__(action_name, action_endpoint)

    async def run(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        pass

    def fail_if_undefined(self, method_name: Text) -> None:
        if not (
            hasattr(self.__class__.__base__, method_name)
            and callable(getattr(self.__class__.__base__, method_name))
        ):
            pytest.fail(
                f"method '{method_name}' not found in {self.__class__.__base__}. "
                f"This likely means the method was renamed, which means the "
                f"instrumentation needs to be adapted!"
            )


def get_model_groups() -> List[Dict[str, Any]]:
    return [
        {
            "id": "llm-model-group",
            "models": [
                {
                    "provider": "cohere",
                    "model": "test-cohere",
                    "api_key": "mock key in test_tracing_rephraser",
                },
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "tedst",
                },
                {
                    "provider": "azure",
                    "deployment": "my-llm-azure-deployment",
                    "api_key": "test",
                    "api_base": "test-base",
                    "api_version": "test-version",
                    "num_retries": 100,
                    "timeout": 100,
                },
            ],
            "router": {"routing_strategy": "test"},
        },
        {
            "id": "embedding-model-group",
            "models": [
                {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "api_key": "mock key in test_tracing_rephraser",
                    # configuration parsers will append these deprecated fields
                    # automatically, so it's easier for testing the value of
                    # 'embeddings' attribute to have them upfront.
                    "api_base": None,
                    "api_version": None,
                    "api_type": "openai",
                },
                {
                    "provider": "azure",
                    "deployment": "my-azure-embedding-deployment",
                    "api_key": "test",
                    "api_base": "test-base",
                    "api_version": "test-version",
                    "num_retries": 100,
                    "timeout": 100,
                    # again, configuration parsers will append these fields
                    # automatically
                    "api_type": "azure",
                    "model": None,
                },
            ],
            "router": {"routing_strategy": "test"},
        },
    ]


@pytest.fixture
def mock_available_endpoints() -> MagicMock:
    _mock_available_endpoints = MagicMock(spec=AvailableEndpoints)
    _mock_available_endpoints.config_file_path = Path("this/is/a/mock/file")

    _mock_available_endpoints.model_groups = get_model_groups()

    return _mock_available_endpoints


@pytest.fixture
def mock_configuration(
    monkeypatch: pytest.MonkeyPatch, mock_available_endpoints: MagicMock
) -> MagicMock:
    _mock_configuration_instance = MagicMock(spec=Configuration)
    _mock_configuration_instance.endpoints = mock_available_endpoints

    _mock_configuration = MagicMock()
    _mock_configuration.get_instance.return_value = _mock_configuration_instance
    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", _mock_configuration)

    return _mock_configuration


@pytest.fixture
def mock_perform_llm_health_check() -> Mock:
    with patch(
        "rasa.shared.utils.health_check.health_check" ".perform_llm_health_check"
    ) as mock_function:
        mock_function.return_value = None
        yield mock_function


@pytest.fixture
def mock_perform_embeddings_health_check() -> Mock:
    with patch(
        "rasa.shared.utils.health_check.health_check" ".perform_embeddings_health_check"
    ) as mock_function:
        mock_function.return_value = None
        yield mock_function


class MockMCPOpenAgent(MCPOpenAgent):
    """Mock MCP Open Agent for testing instrumentation.

    This class follows the same pattern as other mock classes in conftest.py
    - it inherits from the real class to allow instrumentation to work
    - it implements methods in a simple way for testing
    """

    def __init__(self) -> None:
        from rasa.agents.protocol.mcp.mcp_base_agent import DEFAULT_LLM_CONFIG

        self.llm_client = Mock()
        self.llm_client.config = DEFAULT_LLM_CONFIG
        self.build_messages_for_llm_request = Mock()
        self.build_messages_for_llm_request.return_value = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, help me with a task"},
        ]
        self._agent_output = None

    def set_agent_output(self, agent_output: Optional["AgentOutput"]) -> None:
        """Set the agent output to return from send_message."""
        self._agent_output = agent_output

    async def send_message(
        self, agent_input: "AgentInput", output_channel: Optional[OutputChannel] = None
    ) -> "AgentOutput":
        """Mock send_message method that returns the configured agent output."""
        return self._agent_output
