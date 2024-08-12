from __future__ import annotations

from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Text,
    Type,
)
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.core.actions.action import Action
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EB, EventBroker
from rasa.core.channels import OutputChannel, UserMessage
from rasa.core.information_retrieval import (
    SearchResult,
    SearchResultList,
    InformationRetrieval,
)
from rasa.core.lock import TicketLock
from rasa.core.lock_store import LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command, StartFlowCommand
from rasa.dialogue_understanding.generator import (
    LLMCommandGenerator,
    SingleStepLLMCommandGenerator,
    MultiStepLLMCommandGenerator,
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
from rasa.shared.utils.yaml import read_yaml_file
from rasa.utils.endpoints import EndpointConfig

from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser

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


@pytest.fixture(scope="function")
def previous_num_captured_spans(span_exporter: InMemorySpanExporter) -> int:
    captured_spans = span_exporter.get_finished_spans()  # type: ignore
    return len(captured_spans)


@pytest.fixture()
def default_model_storage(tmp_path: Path) -> ModelStorage:
    return LocalModelStorage.create(tmp_path)


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

    async def get_tracker(self, conversation_id: Text) -> TrackerMock:
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

    def _predict_next_with_tracker(self, tracker: DialogueStateTracker) -> Mock:
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

    async def invoke_llm(self, prompt: str) -> Optional[str]:
        pass


class MockSingleStepLLMCommandGenerator(SingleStepLLMCommandGenerator):
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

    async def invoke_llm(self, prompt: str) -> Optional[str]:
        pass


class MockMultiStepLLMCommandGenerator(MultiStepLLMCommandGenerator):
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

    async def invoke_llm(self, prompt: str) -> Optional[str]:
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

    async def _generate_llm_response(self, prompt: str) -> Optional[str]:
        pass

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
