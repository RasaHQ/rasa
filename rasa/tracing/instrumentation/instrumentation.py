import contextlib
import functools
import importlib
import inspect
import json
import logging
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Text,
    Type,
    TypeVar,
    cast,
)

from multidict import MultiDict
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import AgentInput, AgentOutput, AgentToolSchema
from rasa.core.actions.action import Action, CustomActionExecutor, RemoteAction
from rasa.core.actions.custom_action_executor import RetryCustomActionExecutor
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.channels import OutputChannel
from rasa.core.information_retrieval.information_retrieval import (
    InformationRetrieval,
    SearchResultList,
)
from rasa.core.lock_store import LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.flows.flow_step_result import FlowActionPrediction
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator import (
    CompactLLMCommandGenerator,
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SearchReadyLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.engine.graph import GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import SET_SLOT_COMMAND
from rasa.shared.nlu.training_data.message import Message
from rasa.tracing.constants import (
    AGENT_EXECUTION_DURATION_METRIC_NAME,
    MCP_TOOL_EXECUTION_DURATION_METRIC_NAME,
    REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME,
    TOOL_OUTPUT_VALUE_MAX_LENGTH,
)
from rasa.tracing.instrumentation import attribute_extractors
from rasa.tracing.instrumentation.intentless_policy_instrumentation import (
    _instrument_extract_ai_responses,
    _instrument_generate_answer,
    _instrument_select_few_shot_conversations,
    _instrument_select_response_examples,
)
from rasa.tracing.instrumentation.metrics import (
    record_callable_duration_metrics,
    record_compact_llm_command_generator_metrics,
    record_enterprise_search_policy_metrics,
    record_llm_command_generator_metrics,
    record_mcp_agent_llm_metrics,
    record_multi_step_llm_command_generator_metrics,
    record_request_size_in_bytes,
    record_search_ready_llm_command_generator_metrics,
    record_single_step_llm_command_generator_metrics,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig, concat_url

# The `TypeVar` representing the return type for a function to be wrapped.
S = TypeVar("S")
# The `TypeVar` representing the type of the argument passed to the function to be
# wrapped.
T = TypeVar("T")

logger = logging.getLogger(__name__)
INSTRUMENTED_BOOLEAN_ATTRIBUTE_NAME = "class_has_been_instrumented"
INSTRUMENTED_MODULE_BOOLEAN_ATTRIBUTE_NAME = "module_has_been_instrumented"
COMMAND_PROCESSOR_MODULE_NAME = (
    "rasa.dialogue_understanding.processor.command_processor"
)
FLOW_EXECUTOR_MODULE_NAME = "rasa.core.policies.flows.flow_executor"
AGENT_EXECUTOR_MODULE_NAME = "rasa.core.policies.flows.agent_executor"
DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME = "rasa.dialogue_understanding_test.io"


def _check_extractor_argument_list(
    fn: Callable[[T, Any, Any], S],
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
) -> bool:
    if attr_extractor is None:
        return False

    fn_args = inspect.signature(fn)
    attr_args = inspect.signature(attr_extractor)

    are_arglists_congruent = fn_args.parameters.keys() == attr_args.parameters.keys()

    if not are_arglists_congruent:
        logger.warning(
            f"Argument lists for {fn.__name__} and {attr_extractor.__name__}"
            f" do not match up. {fn.__name__} will be traced without attributes."
        )

    return are_arglists_congruent


def extract_tracing_context_from_headers(
    headers: Dict[str, Any],
) -> Optional[Context]:
    """Extracts the tracing context from the headers."""
    tracing_carrier = MultiDict(
        [
            (key, value)
            for key, value in headers.items()
            if key.lower() not in ("content-length", "content-encoding")
        ]
    )
    context = (
        TraceContextTextMapPropagator().extract(headers) if tracing_carrier else None
    )

    return context


def traceable(
    fn: Callable[[T, Any, Any], S],
    tracer: Tracer,
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
    metrics_recorder: Optional[Callable],
) -> Callable[[T, Any, Any], S]:
    """Wrap a non-`async` function by tracing functionality.

    :param fn: The function to be wrapped.
    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's instance and
        the function's arguments.
    :param metrics_recorder: A function that records metric measurements.
    :return: The wrapped function.
    """
    should_extract_args = _check_extractor_argument_list(fn, attr_extractor)

    @functools.wraps(fn)
    def wrapper(self: T, *args: Any, **kwargs: Any) -> S:
        attrs = (
            attr_extractor(self, *args, **kwargs)
            if attr_extractor and should_extract_args
            else {}
        )

        module_name = attrs.pop("module_name", "")
        if module_name in [
            "command_processor",
            FLOW_EXECUTOR_MODULE_NAME,
            DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME,
        ]:
            span_name = f"{module_name}.{fn.__name__}"
        else:
            span_name = f"{self.__class__.__name__}.{fn.__name__}"
        with tracer.start_as_current_span(span_name, attributes=attrs):
            start_time = time.perf_counter_ns()

            result = fn(self, *args, **kwargs)

            end_time = time.perf_counter_ns()
            record_callable_duration_metrics(self, start_time, end_time)

            if metrics_recorder:
                metrics_recorder(attrs)

            return result

    return wrapper


def traceable_async(
    fn: Callable[[T, Any, Any], Awaitable[S]],
    tracer: Tracer,
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
    header_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
    metrics_recorder: Optional[Callable],
) -> Callable[[T, Any, Any], Awaitable[S]]:
    """Wrap an `async` function by tracing functionality.

    :param fn: The function to be wrapped.
    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's instance and
        the function's arguments.
    :param header_extractor: A function that is applied to the function's arguments
    :param metrics_recorder: A function that records metric measurements.
    :return: The wrapped function.
    """
    should_extract_args = _check_extractor_argument_list(fn, attr_extractor)

    @functools.wraps(fn)
    async def async_wrapper(self: T, *args: Any, **kwargs: Any) -> S:
        attrs = (
            attr_extractor(self, *args, **kwargs)
            if attr_extractor and should_extract_args
            else {}
        )
        headers = header_extractor(*args, **kwargs) if header_extractor else {}
        context = extract_tracing_context_from_headers(headers)

        if issubclass(self.__class__, GraphNode) and fn.__name__ == "__call__":
            span_name = f"{self.__class__.__name__}." + attrs.get(
                "component_class", "GraphNode"
            )
        else:
            span_name = f"{self.__class__.__name__}.{fn.__name__}"

        with tracer.start_as_current_span(
            span_name,
            attributes=attrs,
            context=context,
        ) as span:
            TraceContextTextMapPropagator().inject(headers)

            ctx = span.get_span_context()
            logger.debug(
                f"The trace id for the current span '{span_name}' is '{ctx.trace_id}'."
            )

            start_time = time.perf_counter_ns()

            result = await fn(self, *args, **kwargs)

            end_time = time.perf_counter_ns()
            record_callable_duration_metrics(self, start_time, end_time, **attrs)

            if metrics_recorder:
                metrics_recorder(attrs)

            return result

    return async_wrapper


def traceable_module_async(
    fn: Callable[..., Awaitable[S]],
    tracer: Tracer,
    attr_extractor: Optional[Callable[..., Dict[str, Any]]],
    header_extractor: Optional[Callable[..., Dict[str, Any]]],
    metrics_recorder: Optional[Callable],
) -> Callable[..., Awaitable[S]]:
    """Wrap a module-level async function with tracing functionality.

    :param fn: The async function to be wrapped.
    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's arguments.
    :param header_extractor: A function that is applied to the function's arguments.
    :param metrics_recorder: A function that records metric measurements.
    :return: The wrapped function.
    """
    should_extract_args = _check_extractor_argument_list(fn, attr_extractor)

    @functools.wraps(fn)
    async def async_wrapper(*args: Any, **kwargs: Any) -> S:
        attrs = (
            attr_extractor(*args, **kwargs)
            if attr_extractor and should_extract_args
            else {}
        )
        headers = header_extractor(*args, **kwargs) if header_extractor else {}
        context = extract_tracing_context_from_headers(headers)

        # Use module name from attrs or fallback to function module
        module_name = attrs.pop("module_name", "")
        if module_name in [
            "command_processor",
            FLOW_EXECUTOR_MODULE_NAME,
            DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME,
        ]:
            span_name = f"{module_name}.{fn.__name__}"
        else:
            span_name = f"{fn.__module__}.{fn.__name__}"

        with tracer.start_as_current_span(
            span_name,
            attributes=attrs,
            context=context,
        ) as span:
            TraceContextTextMapPropagator().inject(headers)

            ctx = span.get_span_context()
            logger.debug(
                f"The trace id for the current span '{span_name}' is '{ctx.trace_id}'."
            )

            result = await fn(*args, **kwargs)

            if metrics_recorder:
                metrics_recorder(attrs)

            return result

    return async_wrapper


def traceable_async_generator(
    fn: Callable[[T, Any, Any], AsyncIterator[S]],
    tracer: Tracer,
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
) -> Callable[[T, Any, Any], AsyncIterator[AsyncIterator[S]]]:
    """Wrap an `async` function that `yield`s by tracing functionality.

    :param fn: The function to be wrapped.
    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's instance and
        the function's arguments.
    :return: The wrapped function.
    """
    should_extract_args = _check_extractor_argument_list(fn, attr_extractor)

    @functools.wraps(fn)
    async def async_wrapper(
        self: T, *args: Any, **kwargs: Any
    ) -> AsyncIterator[AsyncIterator[S]]:
        attrs = (
            attr_extractor(self, *args, **kwargs)
            if attr_extractor and should_extract_args
            else {}
        )
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{fn.__name__}", attributes=attrs
        ):
            yield fn(self, *args, **kwargs)

    return async_wrapper


# This `TypeVar` restricts the agent_class to be instrumented to subclasses of `Agent`.
AgentType = TypeVar("AgentType", bound=Agent)
# This `TypeVar` restricts the subagent_class to be instrumented to subclasses of
# `AgentProtocol`.
AgentProtocolType = TypeVar("AgentProtocolType", bound=AgentProtocol)
# This `TypeVar` restricts the processor_class to be instrumented to subclasses of
# `MessageProcessor`.
ProcessorType = TypeVar("ProcessorType", bound=MessageProcessor)
TrackerStoreType = TypeVar("TrackerStoreType", bound=TrackerStore)
GraphNodeType = TypeVar("GraphNodeType", bound=GraphNode)
# This `TypeVar` restricts the lock_store_class to be instrumented to subclasses of
# `LockStore`.
LockStoreType = TypeVar("LockStoreType", bound=LockStore)
GraphTrainerType = TypeVar("GraphTrainerType", bound=GraphTrainer)
LLMCommandGeneratorType = TypeVar("LLMCommandGeneratorType", bound=LLMCommandGenerator)
SingleStepLLMCommandGeneratorType = TypeVar(
    "SingleStepLLMCommandGeneratorType", bound=SingleStepLLMCommandGenerator
)
CompactLLMCommandGeneratorType = TypeVar(
    "CompactLLMCommandGeneratorType", bound=CompactLLMCommandGenerator
)
SearchReadyLLMCommandGeneratorType = TypeVar(
    "SearchReadyLLMCommandGeneratorType", bound=SearchReadyLLMCommandGenerator
)
MultiStepLLMCommandGeneratorType = TypeVar(
    "MultiStepLLMCommandGeneratorType", bound=MultiStepLLMCommandGenerator
)
FlowRetrievalType = TypeVar("FlowRetrievalType", bound=FlowRetrieval)
CommandType = TypeVar("CommandType", bound=Command)
PolicyType = TypeVar("PolicyType", bound=Policy)
InformationRetrievalType = TypeVar(
    "InformationRetrievalType", bound=InformationRetrieval
)
NLUCommandAdapterType = TypeVar("NLUCommandAdapterType", bound=NLUCommandAdapter)
EndpointConfigType = TypeVar("EndpointConfigType", bound=EndpointConfig)


def instrument(
    tracer_provider: TracerProvider,
    agent_class: Optional[Type[AgentType]] = None,
    processor_class: Optional[Type[ProcessorType]] = None,
    tracker_store_class: Optional[Type[TrackerStoreType]] = None,
    graph_node_class: Optional[Type[GraphNodeType]] = None,
    lock_store_class: Optional[Type[LockStoreType]] = None,
    graph_trainer_class: Optional[Type[GraphTrainerType]] = None,
    llm_command_generator_class: Optional[Type[LLMCommandGeneratorType]] = None,
    command_subclasses: Optional[List[Type[CommandType]]] = None,
    contextual_response_rephraser_class: Optional[Any] = None,
    policy_subclasses: Optional[List[Type[PolicyType]]] = None,
    vector_store_subclasses: Optional[List[Type[InformationRetrievalType]]] = None,
    nlu_command_adapter_class: Optional[Type[NLUCommandAdapterType]] = None,
    endpoint_config_class: Optional[Type[EndpointConfigType]] = None,
    grpc_custom_action_executor_class: Optional[Type[GRPCCustomActionExecutor]] = None,
    single_step_llm_command_generator_class: Optional[
        Type[SingleStepLLMCommandGeneratorType]
    ] = None,
    compact_llm_command_generator_class: Optional[
        Type[CompactLLMCommandGeneratorType]
    ] = None,
    search_ready_llm_command_generator_class: Optional[
        Type[SearchReadyLLMCommandGeneratorType]
    ] = None,
    multi_step_llm_command_generator_class: Optional[
        Type[MultiStepLLMCommandGeneratorType]
    ] = None,
    custom_action_executor_subclasses: Optional[
        List[Type[CustomActionExecutor]]
    ] = None,
    flow_retrieval_class: Optional[Type[FlowRetrievalType]] = None,
    subagent_classes: Optional[List[Type[AgentProtocolType]]] = None,
) -> None:
    """Substitute methods to be traced by their traced counterparts.

    Because of type bounds on `AgentType` and `ProcessorType`,
    we ensure that only subtypes of `Agent`
    and `MessageProcessor` can be instrumented, respectively.

    :param tracer_provider: The `TracerProvider` to be used for configuring tracing
        on the substituted methods.
    :param agent_class: The `Agent` to be instrumented. If `None` is given, no `Agent`
        will be instrumented.
    :param processor_class: The `MessageProcessor` to be instrumented. If `None` is
        given, no `MessageProcessor` will be instrumented.
    :param tracker_store_class: The `TrackerStore` to be instrumented. If `None` is
        given, no `TrackerStore` will be instrumented.
    :param graph_node_class: The `GraphNode` to be instrumented. If `None` is
        given, no `GraphNode` will be instrumented.
    :param lock_store_class: The `LockStore` to be instrumented. If `None` is
        given, no `LockStore` will be instrumented.
    :param graph_trainer_class: The `GraphTrainer` to be instrumented. If `None` is
        given, no `GraphTrainer` will be instrumented.
    :param llm_command_generator_class: The `LLMCommandGenerator` to be instrumented.
        If `None` is given, no `LLMCommandGenerator` will be instrumented.
    :param command_subclasses: The subclasses of `Command` to be instrumented. If `None`
        is given, no subclass of `Command` will be instrumented.
    :param contextual_response_rephraser_class: The `ContextualResponseRephraser` to
        be instrumented. If `None` is given, no `ContextualResponseRephraser` will be
        instrumented.
    :param policy_subclasses: The subclasses of `Policy` to be instrumented. If `None`
        is given, no subclass of `Policy` will be instrumented.
    :param vector_store_subclasses: The subclasses of `InformationRetrieval` to be
        instrumented. If `None` is given, no subclass of `InformationRetrieval` will be
        instrumented.
    :param nlu_command_adapter_class: The `NLUCommandAdapter` to be instrumented. If
        `None` is given, no `NLUCommandAdapter` will be instrumented.
    :param endpoint_config_class: The `EndpointConfig` to be instrumented. If
        `None` is given, no `EndpointConfig` will be instrumented.
    :param grpc_custom_action_executor_class: The `GRPCCustomActionExecution` to be
        instrumented. If `None` is given, no `GRPCCustomActionExecution`
        will be instrumented.
    :param single_step_llm_command_generator_class: The `SingleStepLLMCommandGenerator`
        to be instrumented. If `None` is given, no `SingleStepLLMCommandGenerator` will
        be instrumented.
    :param compact_llm_command_generator_class: The `CompactLLMCommandGenerator`
        to be instrumented. If `None` is given, no `CompactLLMCommandGenerator` will
        be instrumented.
    :param search_ready_llm_command_generator_class: The`SearchReadyLLMCommandGenerator`
        to be instrumented. If `None` is given, no `SearchReadyLLMCommandGenerator` will
        be instrumented.
    :param multi_step_llm_command_generator_class: The `MultiStepLLMCommandGenerator`
        to be instrumented. If `None` is given, no `MultiStepLLMCommandGenerator` will
        be instrumented.
    :param custom_action_executor_subclasses: The subclasses of `CustomActionExecutor`
        to be instrumented. If `None` is given, no subclass of `CustomActionExecutor`
        will be instrumented.
    :param subagent_classes: The `AgentProtocol` classes to be instrumented.
        If `None` is given, no `AgentProtocol` classes will be instrumented.
    """
    if agent_class is not None and not class_is_instrumented(agent_class):
        _instrument_method(
            tracer_provider.get_tracer(agent_class.__module__),
            agent_class,
            "handle_message",
            attribute_extractors.extract_attrs_for_agent,
            attribute_extractors.extract_headers,
        )
        mark_class_as_instrumented(agent_class)

    if subagent_classes is not None:
        for subagent_class in subagent_classes:
            if not class_is_instrumented(subagent_class):
                # Instrument _execute_tool_call method if it exists
                if hasattr(subagent_class, "_execute_tool_call"):
                    _instrument_execute_tool_call(
                        tracer_provider.get_tracer(subagent_class.__module__),
                        subagent_class,
                    )

                # Instrument send_message method for MCP agents
                if hasattr(subagent_class, "send_message"):
                    _instrument_method(
                        tracer_provider.get_tracer(subagent_class.__module__),
                        subagent_class,
                        "send_message",
                        attribute_extractors.extract_attrs_for_mcp_agent_llm_call,
                        metrics_recorder=record_mcp_agent_llm_metrics,
                    )

                    _instrument_mcp_agent_send_message_response_capture(
                        tracer_provider.get_tracer(subagent_class.__module__),
                        subagent_class,
                    )

                # Instrument get_available_tools method for MCP agents
                if hasattr(subagent_class, "get_available_tools"):
                    _instrument_mcp_agent_get_available_tools_response_capture(
                        tracer_provider.get_tracer(subagent_class.__module__),
                        subagent_class,
                    )

                # Instrument health check method for A2A agents
                if hasattr(subagent_class, "_perform_health_check"):
                    _instrument_method(
                        tracer_provider.get_tracer(subagent_class.__module__),
                        subagent_class,
                        "_perform_health_check",
                        attribute_extractors.extract_attrs_for_a2a_agent_perform_health_check,
                    )

                mark_class_as_instrumented(subagent_class)

    if processor_class is not None and not class_is_instrumented(processor_class):
        _instrument_processor(tracer_provider, processor_class)
        mark_class_as_instrumented(processor_class)

    if tracker_store_class is not None and not class_is_instrumented(
        tracker_store_class
    ):
        _instrument_method(
            tracer_provider.get_tracer(tracker_store_class.__module__),
            tracker_store_class,
            "_stream_new_events",
            attribute_extractors.extract_attrs_for_tracker_store,
        )
        mark_class_as_instrumented(tracker_store_class)

    if graph_node_class is not None and not class_is_instrumented(graph_node_class):
        _instrument_method(
            tracer_provider.get_tracer(graph_node_class.__module__),
            graph_node_class,
            "__call__",
            attribute_extractors.extract_attrs_for_graph_node,
        )
        mark_class_as_instrumented(graph_node_class)

    if lock_store_class is not None and not class_is_instrumented(lock_store_class):
        # Not so straightforward: to wrap a function that is decorated as a
        # `@asynccontextmanager`, we need to first unwrap the original function, then
        # wrap that with the tracing functionality, and re-decorate as an
        # `@asynccontextmanager`.
        lock_method = inspect.unwrap(lock_store_class.lock)
        traced_lock_method = traceable_async_generator(
            lock_method,
            tracer_provider.get_tracer(lock_store_class.__module__),
            attribute_extractors.extract_attrs_for_lock_store,
        )
        lock_store_class.lock = contextlib.asynccontextmanager(traced_lock_method)  # type: ignore[method-assign]

        logger.debug(f"Instrumented '{lock_store_class.__name__}.lock'.")

        mark_class_as_instrumented(lock_store_class)

    if graph_trainer_class is not None and not class_is_instrumented(
        graph_trainer_class
    ):
        _instrument_method(
            tracer_provider.get_tracer(graph_trainer_class.__module__),
            graph_trainer_class,
            "train",
            attribute_extractors.extract_attrs_for_graph_trainer,
        )
        mark_class_as_instrumented(graph_trainer_class)

    if llm_command_generator_class is not None and not class_is_instrumented(
        llm_command_generator_class
    ):
        _instrument_method(
            tracer_provider.get_tracer(llm_command_generator_class.__module__),
            llm_command_generator_class,
            "invoke_llm",
            attribute_extractors.extract_attrs_for_llm_based_command_generator,
            metrics_recorder=record_llm_command_generator_metrics,
        )
        _instrument_method(
            tracer_provider.get_tracer(llm_command_generator_class.__module__),
            llm_command_generator_class,
            "_check_commands_against_startable_flows",
            attribute_extractors.extract_attrs_for_check_commands_against_startable_flows,
        )
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(llm_command_generator_class.__module__),
            llm_command_generator_class,
            "perform_llm_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(llm_command_generator_class)

    if (
        single_step_llm_command_generator_class is not None
        and not class_is_instrumented(single_step_llm_command_generator_class)
    ):
        _instrument_method(
            tracer_provider.get_tracer(
                single_step_llm_command_generator_class.__module__
            ),
            single_step_llm_command_generator_class,
            "invoke_llm",
            attribute_extractors.extract_attrs_for_llm_based_command_generator,
            metrics_recorder=record_single_step_llm_command_generator_metrics,
        )
        _instrument_method(
            tracer_provider.get_tracer(
                single_step_llm_command_generator_class.__module__
            ),
            single_step_llm_command_generator_class,
            "_check_commands_against_startable_flows",
            attribute_extractors.extract_attrs_for_check_commands_against_startable_flows,
        )
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(
                single_step_llm_command_generator_class.__module__
            ),
            single_step_llm_command_generator_class,
            "perform_llm_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(single_step_llm_command_generator_class)

    if compact_llm_command_generator_class is not None and not class_is_instrumented(
        compact_llm_command_generator_class
    ):
        _instrument_method(
            tracer_provider.get_tracer(compact_llm_command_generator_class.__module__),
            compact_llm_command_generator_class,
            "invoke_llm",
            attribute_extractors.extract_attrs_for_llm_based_command_generator,
            metrics_recorder=record_compact_llm_command_generator_metrics,
        )
        _instrument_method(
            tracer_provider.get_tracer(compact_llm_command_generator_class.__module__),
            compact_llm_command_generator_class,
            "_check_commands_against_startable_flows",
            attribute_extractors.extract_attrs_for_check_commands_against_startable_flows,
        )
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(compact_llm_command_generator_class.__module__),
            compact_llm_command_generator_class,
            "perform_llm_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(compact_llm_command_generator_class)

    if (
        search_ready_llm_command_generator_class is not None
        and not class_is_instrumented(search_ready_llm_command_generator_class)
    ):
        _instrument_method(
            tracer_provider.get_tracer(
                search_ready_llm_command_generator_class.__module__
            ),
            search_ready_llm_command_generator_class,
            "invoke_llm",
            attribute_extractors.extract_attrs_for_llm_based_command_generator,
            metrics_recorder=record_search_ready_llm_command_generator_metrics,
        )
        _instrument_method(
            tracer_provider.get_tracer(
                search_ready_llm_command_generator_class.__module__
            ),
            search_ready_llm_command_generator_class,
            "_check_commands_against_startable_flows",
            attribute_extractors.extract_attrs_for_check_commands_against_startable_flows,
        )
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(
                search_ready_llm_command_generator_class.__module__
            ),
            search_ready_llm_command_generator_class,
            "perform_llm_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(search_ready_llm_command_generator_class)

    if multi_step_llm_command_generator_class is not None and not class_is_instrumented(
        multi_step_llm_command_generator_class
    ):
        _instrument_method(
            tracer_provider.get_tracer(
                multi_step_llm_command_generator_class.__module__
            ),
            multi_step_llm_command_generator_class,
            "invoke_llm",
            attribute_extractors.extract_attrs_for_llm_based_command_generator,
            metrics_recorder=record_multi_step_llm_command_generator_metrics,
        )
        _instrument_multi_step_llm_command_generator_parse_commands(
            tracer_provider.get_tracer(
                multi_step_llm_command_generator_class.__module__
            ),
            multi_step_llm_command_generator_class,
        )
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(
                multi_step_llm_command_generator_class.__module__
            ),
            multi_step_llm_command_generator_class,
            "perform_llm_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(multi_step_llm_command_generator_class)

    if (
        any(
            llm_based_command_generator_class is not None
            for llm_based_command_generator_class in (
                llm_command_generator_class,
                single_step_llm_command_generator_class,
                compact_llm_command_generator_class,
                search_ready_llm_command_generator_class,
                multi_step_llm_command_generator_class,
            )
        )
        and flow_retrieval_class is not None
        and not class_is_instrumented(flow_retrieval_class)
    ):
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(flow_retrieval_class.__module__),
            flow_retrieval_class,
            "perform_embeddings_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(flow_retrieval_class)

    if command_subclasses:
        for command_subclass in command_subclasses:
            if command_subclass is not None and not class_is_instrumented(
                command_subclass
            ):
                _instrument_method(
                    tracer_provider.get_tracer(command_subclass.__module__),
                    command_subclass,
                    "run_command_on_tracker",
                    attribute_extractors.extract_attrs_for_command,
                )
                mark_class_as_instrumented(command_subclass)

    if contextual_response_rephraser_class is not None and not class_is_instrumented(
        contextual_response_rephraser_class
    ):
        _instrument_method(
            tracer_provider.get_tracer(contextual_response_rephraser_class.__module__),
            contextual_response_rephraser_class,
            "_generate_llm_response",
            attribute_extractors.extract_attrs_for_contextual_response_rephraser,
        )
        _instrument_method(
            tracer_provider.get_tracer(contextual_response_rephraser_class.__module__),
            contextual_response_rephraser_class,
            "_create_history",
            attribute_extractors.extract_attrs_for_create_history,
        )
        _instrument_method(
            tracer_provider.get_tracer(contextual_response_rephraser_class.__module__),
            contextual_response_rephraser_class,
            "generate",
            attribute_extractors.extract_attrs_for_generate,
        )
        _instrument_perform_health_check_method_for_component(
            tracer_provider.get_tracer(contextual_response_rephraser_class.__module__),
            contextual_response_rephraser_class,
            "perform_llm_health_check",
            attribute_extractors.extract_attrs_for_performing_health_check,
        )
        mark_class_as_instrumented(contextual_response_rephraser_class)

    if not module_is_instrumented(COMMAND_PROCESSOR_MODULE_NAME):
        _instrument_command_processor_module(tracer_provider)

    if not module_is_instrumented(FLOW_EXECUTOR_MODULE_NAME):
        _instrument_flow_executor_module(tracer_provider)

    if not module_is_instrumented(DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME):
        _instrument_dialog_understanding_test_io_module(tracer_provider)

    if policy_subclasses:
        for policy_subclass in policy_subclasses:
            if policy_subclass is not None and not class_is_instrumented(
                policy_subclass
            ):
                _instrument_method(
                    tracer_provider.get_tracer(policy_subclass.__module__),
                    policy_subclass,
                    "_prediction",
                    attribute_extractors.extract_attrs_for_policy_prediction,
                )

                _instrument_intentless_policy(
                    tracer_provider,
                    policy_subclass,
                )

                _instrument_enterprise_search_policy(
                    tracer_provider,
                    policy_subclass,
                )

                mark_class_as_instrumented(policy_subclass)

    if vector_store_subclasses:
        for vector_store_subclass in vector_store_subclasses:
            if vector_store_subclass is not None and not class_is_instrumented(
                vector_store_subclass
            ):
                _instrument_information_retrieval_search(
                    tracer_provider.get_tracer(vector_store_subclass.__module__),
                    vector_store_subclass,
                )
                mark_class_as_instrumented(vector_store_subclass)

    if nlu_command_adapter_class is not None and not class_is_instrumented(
        nlu_command_adapter_class
    ):
        _instrument_nlu_command_adapter_predict_commands(
            tracer_provider.get_tracer(nlu_command_adapter_class.__module__),
            nlu_command_adapter_class,
        )
        mark_class_as_instrumented(nlu_command_adapter_class)

    if endpoint_config_class is not None and not class_is_instrumented(
        endpoint_config_class
    ):
        _instrument_endpoint_config(
            tracer_provider.get_tracer(endpoint_config_class.__module__),
            endpoint_config_class,
        )

    if grpc_custom_action_executor_class is not None and not class_is_instrumented(
        grpc_custom_action_executor_class
    ):
        _instrument_grpc_custom_action_executor(
            tracer_provider.get_tracer(grpc_custom_action_executor_class.__module__),
            grpc_custom_action_executor_class,
        )

    if custom_action_executor_subclasses:
        for custom_action_executor_subclass in custom_action_executor_subclasses:
            if (
                custom_action_executor_subclass is not None
                and not class_is_instrumented(custom_action_executor_subclass)
            ):
                _instrument_method(
                    tracer_provider.get_tracer(
                        custom_action_executor_subclass.__module__
                    ),
                    custom_action_executor_subclass,
                    "run",
                    attribute_extractors.extract_attrs_for_custom_action_executor_run,
                )

                if issubclass(
                    custom_action_executor_subclass, GRPCCustomActionExecutor
                ):
                    _instrument_method(
                        tracer=tracer_provider.get_tracer(
                            custom_action_executor_subclass.__module__
                        ),
                        instrumented_class=custom_action_executor_subclass,
                        method_name="_request",
                        attr_extractor=attribute_extractors.extract_attrs_for_grpc_custom_action_executor_request,
                    )

                mark_class_as_instrumented(custom_action_executor_subclass)


def _instrument_mcp_agent_send_message_response_capture(
    tracer: Tracer, agent_class: Type["MCPBaseAgent"]
) -> None:
    """Add response capture to the send_message method."""

    def tracing_send_message_response_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: "MCPBaseAgent",
            agent_input: "AgentInput",
            output_channel: Optional[OutputChannel] = None,
        ) -> "AgentOutput":
            agent_output = await fn(self, agent_input)

            # Only create response span if there's actually a response to capture
            if agent_output:
                with tracer.start_as_current_span(
                    f"{self.__class__.__name__}.{fn.__name__}.llm_response"
                ) as span:
                    span.set_attributes(
                        {
                            "agent_output_response_message": (
                                agent_output.response_message or ""
                            ),
                            "agent_output_id": str(agent_output.id),
                            "agent_output_status": str(agent_output.status),
                            "agent_output_events_count": (
                                len(agent_output.events) if agent_output.events else 0
                            ),
                        }
                    )

            return agent_output

        return wrapper

    agent_class.send_message = tracing_send_message_response_wrapper(  # type: ignore[method-assign]
        agent_class.send_message
    )
    logger.debug(
        f"Instrumented '{agent_class.__name__}.send_message' for response capture."
    )


def _instrument_mcp_agent_get_available_tools_response_capture(
    tracer: Tracer, agent_class: Type["MCPBaseAgent"]
) -> None:
    """Add response capture to the get_available_tools method."""

    def tracing_get_available_tools_response_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(
            self: "MCPBaseAgent",
            agent_input: "AgentInput",
        ) -> List["AgentToolSchema"]:
            # Call the original method
            available_tools = fn(self, agent_input)

            # Create a span to capture the tool information
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                # Create a simple dictionary of tool names and descriptions
                tools_dict = {
                    tool.name: tool.description or "" for tool in available_tools
                }

                span.set_attributes(
                    {
                        "agent_name": self._name,
                        "agent_id": str(
                            make_agent_identifier(self._name, self.protocol_type)
                        ),
                        "total_available_tools_count": len(available_tools),
                        "available_tools": json.dumps(tools_dict),
                    }
                )

            return available_tools

        return wrapper

    agent_class.get_available_tools = tracing_get_available_tools_response_wrapper(  # type: ignore[method-assign]
        agent_class.get_available_tools
    )
    logger.debug(
        f"Instrumented '{agent_class.__name__}.get_available_tools' "
        f"for response capture."
    )


def _instrument_nlu_command_adapter_predict_commands(
    tracer: Tracer, nlu_command_adapter_class: Type[NLUCommandAdapterType]
) -> None:
    def tracing_nlu_command_adapter_predict_commands_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: NLUCommandAdapter,
            message: Message,
            flows: FlowsList,
            tracker: Optional[DialogueStateTracker] = None,
            **kwargs: Any,
        ) -> List[Command]:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                commands = await fn(self, message, flows, tracker, **kwargs)

                span.set_attributes(
                    {
                        "commands": json.dumps(
                            [command.as_dict() for command in commands]
                        ),
                        "intent": json.dumps(message.get("intent", {}).get("name")),
                    }
                )
                return commands

        return wrapper

    nlu_command_adapter_class.predict_commands = (  # type: ignore[method-assign]
        tracing_nlu_command_adapter_predict_commands_wrapper(
            nlu_command_adapter_class.predict_commands
        )
    )

    logger.debug(
        f"Instrumented '{nlu_command_adapter_class.__name__}.predict_commands'."
    )


def _instrument_multi_step_llm_command_generator_parse_commands(
    tracer: Tracer,
    multi_step_llm_command_generator_class: Type[MultiStepLLMCommandGeneratorType],
) -> None:
    def tracing_multi_step_llm_command_generator_parse_commands_wrapper(
        fn: Callable,
    ) -> Callable:
        @functools.wraps(fn)
        def wrapper(
            cls: MultiStepLLMCommandGenerator,
            actions: Optional[str],
            tracker: DialogueStateTracker,
            flows: FlowsList,
            is_start_or_end_prompt: bool = False,
        ) -> List[Command]:
            with tracer.start_as_current_span(
                f"{cls.__class__.__name__}.{fn.__name__}"
            ) as span:
                commands = fn(actions, tracker, flows, is_start_or_end_prompt)

                commands_list = []
                for command in commands:
                    command_as_dict = command.as_dict()
                    command_type = command.command()

                    if command_type == SET_SLOT_COMMAND:
                        slot_value = command_as_dict.pop("value", None)
                        command_as_dict["is_slot_value_missing_or_none"] = (
                            slot_value is None
                        )

                    commands_list.append(command_as_dict)

                span.set_attributes({"commands": json.dumps(commands_list)})
                return commands

        return wrapper

    multi_step_llm_command_generator_class.parse_commands = (  # type: ignore[method-assign]
        tracing_multi_step_llm_command_generator_parse_commands_wrapper(
            multi_step_llm_command_generator_class.parse_commands
        )
    )


def _instrument_information_retrieval_search(
    tracer: Tracer, vector_store_class: Type[InformationRetrievalType]
) -> None:
    def tracing_information_retrieval_search_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: InformationRetrieval,
            query: Text,
            tracker_state: Dict[str, Any],
            threshold: float = 0.0,
        ) -> SearchResultList:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                documents = await fn(self, query, tracker_state, threshold)
                span.set_attributes(
                    {
                        "query": query,
                        "document_metadata": json.dumps(
                            [document.metadata for document in documents.results]
                        ),
                    }
                )
                return documents

        return wrapper

    vector_store_class.search = tracing_information_retrieval_search_wrapper(  # type: ignore[method-assign]
        vector_store_class.search
    )

    logger.debug(f"Instrumented '{vector_store_class.__name__}.search' method.")


def _instrument_enterprise_search_policy(
    tracer_provider: TracerProvider, policy_class: Type[PolicyType]
) -> None:
    if policy_class.__module__ != "rasa.core.policies.enterprise_search_policy":
        return None

    tracer = tracer_provider.get_tracer(policy_class.__module__)

    _instrument_method(
        tracer,
        policy_class,
        "_invoke_llm",
        attribute_extractors.extract_attrs_for_enterprise_search_invoke_llm,
        metrics_recorder=record_enterprise_search_policy_metrics,
    )
    _instrument_method(
        tracer,
        policy_class,
        "_parse_llm_relevancy_check_response",
        attribute_extractors.extract_attrs_for_enterprise_search_parse_llm_relevancy_check_response,
    )
    _instrument_perform_health_check_method_for_component(
        tracer_provider.get_tracer(policy_class.__module__),
        policy_class,
        "perform_embeddings_health_check",
        attribute_extractors.extract_attrs_for_performing_health_check,
    )
    _instrument_perform_health_check_method_for_component(
        tracer_provider.get_tracer(policy_class.__module__),
        policy_class,
        "perform_llm_health_check",
        attribute_extractors.extract_attrs_for_performing_health_check,
    )


def _instrument_intentless_policy(
    tracer_provider: TracerProvider, policy_class: Type[PolicyType]
) -> None:
    if policy_class.__module__ != "rasa.core.policies.intentless_policy":
        return None

    tracer = tracer_provider.get_tracer(policy_class.__module__)

    _instrument_method(
        tracer,
        policy_class,
        "_prediction_result",
        attribute_extractors.extract_attrs_for_intentless_policy_prediction_result,
    )
    _instrument_method(
        tracer,
        policy_class,
        "find_closest_response",
        attribute_extractors.extract_attrs_for_intentless_policy_find_closest_response,
    )
    _instrument_select_response_examples(tracer, policy_class)
    _instrument_select_few_shot_conversations(tracer, policy_class)
    _instrument_extract_ai_responses(tracer, policy_class)
    _instrument_generate_answer(tracer, policy_class)
    _instrument_method(
        tracer,
        policy_class,
        "_generate_llm_answer",
        attribute_extractors.extract_attrs_for_intentless_policy_generate_llm_answer,
    )
    _instrument_perform_health_check_method_for_component(
        tracer_provider.get_tracer(policy_class.__module__),
        policy_class,
        "perform_embeddings_health_check",
        attribute_extractors.extract_attrs_for_performing_health_check,
    )
    _instrument_perform_health_check_method_for_component(
        tracer_provider.get_tracer(policy_class.__module__),
        policy_class,
        "perform_llm_health_check",
        attribute_extractors.extract_attrs_for_performing_health_check,
    )


def _instrument_processor(
    tracer_provider: TracerProvider, processor_class: Type[ProcessorType]
) -> None:
    tracer = tracer_provider.get_tracer(processor_class.__module__)
    _instrument_method(
        tracer,
        processor_class,
        "handle_message",
        None,
        attribute_extractors.extract_headers,
    )
    _instrument_method(
        tracer,
        processor_class,
        "log_message",
        None,
        attribute_extractors.extract_headers,
    )
    _instrument_run_action(tracer, processor_class)
    _instrument_method(
        tracer,
        processor_class,
        "save_tracker",
        attribute_extractors.extract_number_of_events,
    )
    _instrument_method(tracer, processor_class, "_run_prediction_loop", None)
    _instrument_method(
        tracer,
        processor_class,
        "_predict_next_with_tracker",
        attribute_extractors.extract_intent_name_and_slots,
    )
    _instrument_get_tracker(tracer, processor_class)


# Wrapping `get_tracker` works a bit differently since in this case, we actually need
# to extract the attributes from the return value.
def _instrument_get_tracker(
    tracer: Tracer, processor_class: Type[ProcessorType]
) -> None:
    def tracing_get_tracker_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: Type[ProcessorType], conversation_id: Text
        ) -> DialogueStateTracker:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                tracker: DialogueStateTracker = await fn(self, conversation_id)
                span.set_attributes({"number_of_events": len(tracker.events)})
                return tracker

        return wrapper

    processor_class.get_tracker = tracing_get_tracker_wrapper(  # type: ignore[method-assign]
        processor_class.get_tracker
    )

    logger.debug(f"Instrumented '{processor_class.__name__}.get_tracker'.")


def _instrument_command_processor_module(tracer_provider: TracerProvider) -> None:
    _instrument_function(
        tracer_provider.get_tracer(COMMAND_PROCESSOR_MODULE_NAME),
        COMMAND_PROCESSOR_MODULE_NAME,
        "execute_commands",
        attribute_extractors.extract_attrs_for_execute_commands,
    )
    _instrument_function(
        tracer_provider.get_tracer(COMMAND_PROCESSOR_MODULE_NAME),
        COMMAND_PROCESSOR_MODULE_NAME,
        "validate_state_of_commands",
        attribute_extractors.extract_attrs_for_validate_state_of_commands,
    )
    _instrument_function(
        tracer_provider.get_tracer(COMMAND_PROCESSOR_MODULE_NAME),
        COMMAND_PROCESSOR_MODULE_NAME,
        "clean_up_commands",
        attribute_extractors.extract_attrs_for_clean_up_commands,
    )
    _instrument_function(
        tracer_provider.get_tracer(COMMAND_PROCESSOR_MODULE_NAME),
        COMMAND_PROCESSOR_MODULE_NAME,
        "remove_duplicated_set_slots",
        attribute_extractors.extract_attrs_for_remove_duplicated_set_slots,
    )
    mark_module_as_instrumented(COMMAND_PROCESSOR_MODULE_NAME)


def _instrument_flow_executor_module(tracer_provider: TracerProvider) -> None:
    _instrument_function(
        tracer_provider.get_tracer(FLOW_EXECUTOR_MODULE_NAME),
        FLOW_EXECUTOR_MODULE_NAME,
        "advance_flows",
        attribute_extractors.extract_attrs_for_advance_flows,
    )
    _instrument_advance_flows_until_next_action(
        tracer_provider.get_tracer(FLOW_EXECUTOR_MODULE_NAME),
        FLOW_EXECUTOR_MODULE_NAME,
        "advance_flows_until_next_action",
    )
    _instrument_function(
        tracer_provider.get_tracer(FLOW_EXECUTOR_MODULE_NAME),
        FLOW_EXECUTOR_MODULE_NAME,
        "run_step",
        attribute_extractors.extract_attrs_for_run_step,
    )
    # Instrument the agent execution function
    _instrument_call_agent_with_retry(
        tracer_provider.get_tracer(FLOW_EXECUTOR_MODULE_NAME),
        AGENT_EXECUTOR_MODULE_NAME,
    )
    # Instrument the MCP tool execution function
    _instrument_execute_mcp_tool_call(
        tracer_provider.get_tracer("rasa.core.policies.flows.mcp_tool_executor"),
        "rasa.core.policies.flows.mcp_tool_executor",
    )
    # Instrument agent internal state transitions
    _instrument_agent_internal_state_transitions(
        tracer_provider.get_tracer(FLOW_EXECUTOR_MODULE_NAME),
        FLOW_EXECUTOR_MODULE_NAME,
    )
    mark_module_as_instrumented(FLOW_EXECUTOR_MODULE_NAME)


def _instrument_dialog_understanding_test_io_module(
    tracer_provider: TracerProvider,
) -> None:
    _instrument_function(
        tracer_provider.get_tracer(DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME),
        DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME,
        "print_test_results",
        attribute_extractors.extract_attrs_for_du_print_test_results,
    )
    mark_module_as_instrumented(DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME)


def _instrument_advance_flows_until_next_action(
    tracer: Tracer,
    module_name: str,
    function_name: str,
) -> None:
    def tracing_advance_flows_until_next_action_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            tracker: DialogueStateTracker,
            available_actions: List[str],
            flows: FlowsList,
            slots: List[Slot],
            output_channel: Optional[OutputChannel] = None,
        ) -> FlowActionPrediction:
            with tracer.start_as_current_span(f"{module_name}.{fn.__name__}") as span:
                prediction: FlowActionPrediction = await fn(
                    tracker, available_actions, flows, slots
                )

                span.set_attributes(
                    {
                        "action_name": prediction.action_name
                        if prediction.action_name
                        else "None",
                        "score": prediction.score,
                        "metadata": json.dumps(prediction.metadata)
                        if prediction.metadata
                        else "{}",
                        "events": json.dumps(
                            [event.__class__.__name__ for event in prediction.events]
                            if prediction.events
                            else [],
                        ),
                    }
                )

                return prediction

        return wrapper

    module = importlib.import_module(module_name)
    function_to_trace = getattr(module, function_name)

    traced_function = tracing_advance_flows_until_next_action_wrapper(function_to_trace)

    setattr(module, function_name, traced_function)

    logger.debug(
        f"Instrumented function '{function_name}' in the module '{module_name}'. "
    )


def _instrument_call_agent_with_retry(
    tracer: Tracer,
    module_name: str,
) -> None:
    """Instrument the actual agent execution function to capture inputs/outputs."""
    module = importlib.import_module(module_name)

    original_function = getattr(module, "_call_agent_with_retry")

    @functools.wraps(original_function)
    async def traced_call_agent_with_retry(
        agent_name: str,
        protocol_type: Any,
        agent_input: Any,
        max_retries: int,
        output_channel: Optional[OutputChannel] = None,
    ) -> Any:
        agent_input_attrs = {
            "agent_name": agent_name,
            "protocol_type": str(protocol_type),
            "max_retries": max_retries,
            "agent_input_id": agent_input.id,
            "agent_input_user_message": agent_input.user_message,
            "agent_input_slots_count": len(agent_input.slots),
            "agent_input_events_count": len(agent_input.events),
            "agent_input_conversation_history_length": len(
                agent_input.conversation_history
            ),
        }

        if agent_input.metadata:
            agent_input_attrs["agent_input_metadata"] = json.dumps(
                agent_input.metadata, sort_keys=True
            )

        span_name = f"{module_name}._call_agent_with_retry"
        with tracer.start_as_current_span(
            span_name, attributes=agent_input_attrs
        ) as span:
            start_time = time.perf_counter_ns()
            result = await original_function(
                agent_name, protocol_type, agent_input, max_retries
            )
            end_time = time.perf_counter_ns()
            duration_ns = end_time - start_time

            span.set_attribute(AGENT_EXECUTION_DURATION_METRIC_NAME, duration_ns)
            span.set_attribute("agent_output_id", str(result.id))
            span.set_attribute("agent_output_status", str(result.status))

            instrument_provider = MetricInstrumentProvider()
            if not instrument_provider.instruments:
                logger.warning(
                    "MetricInstrumentProvider has no instruments registered. "
                    "Agent execution metrics will not be recorded."
                )
            else:
                agent_metric = instrument_provider.get_instrument(
                    AGENT_EXECUTION_DURATION_METRIC_NAME
                )
                if agent_metric is None:
                    logger.warning(
                        f"Failed to get instrument "
                        f"'{AGENT_EXECUTION_DURATION_METRIC_NAME}'. "
                        f"Agent execution metrics will not be recorded."
                    )
                else:
                    agent_metric_attrs = {
                        "agent_name": agent_name,
                        "protocol_type": str(protocol_type),
                        "status": str(result.status),
                    }
                    agent_metric.record(
                        amount=duration_ns, attributes=agent_metric_attrs
                    )

            if result.response_message:
                span.set_attribute(
                    "agent_output_response_message", result.response_message
                )

            if result.events:
                span.set_attribute("agent_output_events_count", len(result.events))

            if result.structured_results:
                span.set_attribute(
                    "agent_output_structured_results_count",
                    len(result.structured_results),
                )

            if result.error_message:
                span.set_attribute("agent_output_error", result.error_message)

            if result.metadata:
                span.set_attribute("agent_output_metadata", json.dumps(result.metadata))

            return result

    setattr(module, "_call_agent_with_retry", traced_call_agent_with_retry)
    logger.debug(
        f"Instrumented function '_call_agent_with_retry' in module '{module_name}'"
    )


def _instrument_execute_mcp_tool_call(
    tracer: Tracer,
    module_name: str,
) -> None:
    """Instrument the actual MCP tool execution function to capture inputs/outputs."""
    module = importlib.import_module(module_name)

    original_function = getattr(module, "_execute_mcp_tool_call")

    @functools.wraps(original_function)
    async def traced_execute_mcp_tool_call(
        initial_events: List[Event],
        stack: DialogueStack,
        step: "CallFlowStep",
        tracker: DialogueStateTracker,
    ) -> Any:
        tool_input_attrs = {
            "tool_id": step.call,
            "mcp_server": step.mcp_server or "unknown",
            "execution_context": "flow",
            "tool_input_mapping": (
                json.dumps(step.mapping["input"])
                if step.mapping and "input" in step.mapping
                else "{}"
            ),
        }

        if step.mapping and "input" in step.mapping and step.mapping["input"]:
            try:
                from rasa.core.policies.flows.mcp_tool_executor import (
                    _prepare_tool_arguments,
                )

                arguments = _prepare_tool_arguments(step.mapping["input"], tracker)
                tool_input_attrs["tool_input_arguments"] = json.dumps(arguments)
            except Exception as e:
                logger.warning(f"Failed to prepare tool arguments: {e}")
                tool_input_attrs["tool_input_arguments"] = "{}"

        span_name = f"{module_name}._execute_mcp_tool_call"
        with tracer.start_as_current_span(
            span_name, attributes=tool_input_attrs
        ) as span:
            start_time = time.perf_counter_ns()
            result = await original_function(initial_events, stack, step, tracker)
            end_time = time.perf_counter_ns()
            duration_ns = end_time - start_time

            span.set_attribute(MCP_TOOL_EXECUTION_DURATION_METRIC_NAME, duration_ns)

            # Count events that were generated by the tool execution
            # Success: result.events contains initial_events + new tool events
            # Error: result.events contains only initial_events (unchanged)
            tool_generated_events_count = len(result.events) - len(initial_events)
            span.set_attribute("tool_output_events_count", tool_generated_events_count)

            instrument_provider = MetricInstrumentProvider()
            if not instrument_provider.instruments:
                logger.warning(
                    "MetricInstrumentProvider has no instruments registered. "
                    "MCP tool execution metrics will not be recorded."
                )
            else:
                tool_metric = instrument_provider.get_instrument(
                    MCP_TOOL_EXECUTION_DURATION_METRIC_NAME
                )
                if tool_metric is None:
                    logger.warning(
                        f"Failed to get instrument "
                        f"'{MCP_TOOL_EXECUTION_DURATION_METRIC_NAME}'. "
                        "MCP tool execution metrics will not be recorded."
                    )
                else:
                    success = "true"
                    if len(result.events) == len(initial_events):
                        for frame in stack.frames:
                            if isinstance(frame, InternalErrorPatternFlowStackFrame):
                                success = "false"
                                break
                    tool_metric_attrs = {
                        "tool_id": step.call,
                        "mcp_server": step.mcp_server or "unknown",
                        "execution_context": "flow",
                        "success": success,
                    }
                    tool_metric.record(amount=duration_ns, attributes=tool_metric_attrs)

            tool_output_events = []
            if result.events and len(result.events) > len(initial_events):
                for event in result.events[len(initial_events) :]:
                    if hasattr(event, "key") and hasattr(event, "value"):
                        tool_output_events.append(
                            {
                                "slot": str(event.key),
                                "value": (
                                    str(event.value)[:TOOL_OUTPUT_VALUE_MAX_LENGTH]
                                    if event.value
                                    else None
                                ),
                            }
                        )

            if tool_output_events:
                span.set_attribute("tool_output_slots", json.dumps(tool_output_events))

            return result

    setattr(module, "_execute_mcp_tool_call", traced_execute_mcp_tool_call)
    logger.debug(
        f"Instrumented function '_execute_mcp_tool_call' in module '{module_name}'"
    )


def _instrument_execute_tool_call(
    tracer: Tracer,
    agent_class: Type[AgentProtocolType],
) -> None:
    """Instrument the agent's _execute_tool_call method."""
    original_method = getattr(agent_class, "_execute_tool_call")

    @functools.wraps(original_method)
    async def traced_execute_tool_call(
        self: AgentProtocolType, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        tool_input_attrs = {
            "tool_name": tool_name,
            "tool_arguments": json.dumps(arguments, sort_keys=True),
            "agent_name": getattr(self, "_name", None),
            "protocol_type": str(self.protocol_type),
            "execution_context": "agent",
        }

        span_name = f"{agent_class.__name__}._execute_tool_call"
        with tracer.start_as_current_span(
            span_name, attributes=tool_input_attrs
        ) as span:
            start_time = time.perf_counter_ns()
            result = await original_method(self, tool_name, arguments)
            end_time = time.perf_counter_ns()
            duration_ns = end_time - start_time

            # Set execution time
            span.set_attribute("tool_execution_duration_ns", duration_ns)

            # Set tool result attributes
            span.set_attribute("tool_result_name", result.tool_name)
            span.set_attribute("tool_result_is_error", result.is_error)

            if result.result:
                # Truncate result if too long
                result_str = str(result.result)
                if len(result_str) > TOOL_OUTPUT_VALUE_MAX_LENGTH:
                    result_str = result_str[:TOOL_OUTPUT_VALUE_MAX_LENGTH] + "..."
                span.set_attribute("tool_result_content", result_str)

            if result.error_message:
                span.set_attribute("tool_result_error_message", result.error_message)

            # Record metrics
            instrument_provider = MetricInstrumentProvider()
            if instrument_provider.instruments:
                tool_metric = instrument_provider.get_instrument(
                    MCP_TOOL_EXECUTION_DURATION_METRIC_NAME
                )
                if tool_metric:
                    tool_metric_attrs = {
                        "tool_name": tool_name,
                        "agent_name": getattr(self, "_name", None),
                        "execution_context": "agent",
                        "success": "false" if result.is_error else "true",
                    }
                    tool_metric.record(amount=duration_ns, attributes=tool_metric_attrs)

            return result

    setattr(agent_class, "_execute_tool_call", traced_execute_tool_call)
    logger.debug(f"Instrumented '{agent_class.__name__}._execute_tool_call'.")


def _instrument_method(
    tracer: Tracer,
    instrumented_class: Type,
    method_name: Text,
    attr_extractor: Optional[Callable],
    header_extractor: Optional[Callable] = None,
    metrics_recorder: Optional[Callable] = None,
) -> None:
    method_to_trace = getattr(instrumented_class, method_name)
    traced_method = _wrap_with_tracing_decorator(
        method_to_trace, tracer, attr_extractor, header_extractor, metrics_recorder
    )
    setattr(instrumented_class, method_name, traced_method)

    logger.debug(f"Instrumented '{instrumented_class.__name__}.{method_name}'.")


def _instrument_function(
    tracer: Tracer,
    module_name: Text,
    function_name: Text,
    attr_extractor: Optional[Callable],
    header_extractor: Optional[Callable] = None,
) -> None:
    module = importlib.import_module(module_name)
    function_to_trace = getattr(module, function_name)
    traced_function = _wrap_with_tracing_decorator(
        function_to_trace,
        tracer,
        attr_extractor,
        header_extractor,
        is_module_function=True,
    )

    setattr(module, function_name, traced_function)

    logger.debug(
        f"Instrumented function '{function_name}' in the module '{module_name}'. "
    )


def _wrap_with_tracing_decorator(
    callable_to_trace: Callable,
    tracer: Tracer,
    attr_extractor: Optional[Callable],
    header_extractor: Optional[Callable] = None,
    metrics_recorder: Optional[Callable] = None,
    is_module_function: bool = False,
) -> Callable:
    if inspect.iscoroutinefunction(callable_to_trace):
        if is_module_function:
            # This is a module-level async function
            traced_callable = traceable_module_async(
                callable_to_trace,
                tracer,
                attr_extractor,
                header_extractor,
                metrics_recorder,
            )
        else:
            # This is a class method
            traced_callable = traceable_async(
                callable_to_trace,
                tracer,
                attr_extractor,
                header_extractor,
                metrics_recorder,
            )
    else:
        traced_callable = traceable(
            callable_to_trace, tracer, attr_extractor, metrics_recorder
        )

    return traced_callable


def _instrument_run_action(
    tracer: Tracer, processor_class: Type[ProcessorType]
) -> None:
    def tracing_run_action_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: Type[ProcessorType],
            action: Action,
            tracker: DialogueStateTracker,
            output_channel: OutputChannel,
            nlg: NaturalLanguageGenerator,
            prediction: PolicyPrediction,
        ) -> bool:
            attrs = {
                "action_name": action.name(),
                "sender_id": tracker.sender_id,
                "message_id": tracker.latest_message.message_id or "default",  # type: ignore[union-attr]
            }
            if isinstance(action, RemoteAction):
                if isinstance(action.executor, RetryCustomActionExecutor):
                    attrs["executor_class_name"] = type(
                        action.executor._custom_action_executor
                    ).__name__
                else:
                    attrs["executor_class_name"] = type(action.executor).__name__
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}",
                kind=SpanKind.CLIENT,
                attributes=attrs,
            ):
                return await fn(self, action, tracker, output_channel, nlg, prediction)

        return wrapper

    processor_class._run_action = tracing_run_action_wrapper(  # type: ignore[method-assign]
        processor_class._run_action
    )

    logger.debug(f"Instrumented '{processor_class.__name__}._run_action'.")


def _instrument_endpoint_config(
    tracer: Tracer, endpoint_config_class: Type[EndpointConfigType]
) -> None:
    """Instrument the `request` method of the `EndpointConfig` class.

    Args:
        tracer: The `Tracer` that shall be used for tracing.
        endpoint_config_class: The `EndpointConfig` to be instrumented.
    """

    def tracing_endpoint_config_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: Type[EndpointConfigType],
            method: Text = "post",
            subpath: Optional[Text] = None,
            content_type: Optional[Text] = "application/json",
            compress: bool = False,
            **kwargs: Any,
        ) -> bool:
            request_body = kwargs.get("json")
            attrs: Dict[str, Any] = {"url": concat_url(self.url, subpath)}

            if not request_body:
                attrs.update({REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME: 0})
            else:
                attrs.update(
                    {
                        REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME: len(
                            json.dumps(request_body).encode("utf-8")
                        )
                    }
                )
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}",
                attributes=attrs,
            ):
                TraceContextTextMapPropagator().inject(self.headers)

                start_time = time.perf_counter_ns()
                result = await fn(
                    self, method, subpath, content_type, compress, **kwargs
                )
                end_time = time.perf_counter_ns()

                record_callable_duration_metrics(self, start_time, end_time)
                record_request_size_in_bytes(attrs)

                return result

        return wrapper

    endpoint_config_class.request = tracing_endpoint_config_wrapper(  # type: ignore[method-assign]
        endpoint_config_class.request
    )

    logger.debug(f"Instrumented '{endpoint_config_class.__name__}.request'.")


def _instrument_grpc_custom_action_executor(
    tracer: Tracer, grpc_custom_action_executor_class: Type[GRPCCustomActionExecutor]
) -> None:
    """Instrument the `run` method of the `GRPCCustomActionExecutor` class.

    Args:
        tracer: The `Tracer` that shall be used for tracing.
        grpc_custom_action_executor_class: The `GRPCCustomActionExecutor` to
            be instrumented.
    """

    def tracing_grpc_custom_action_executor_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: Type[GRPCCustomActionExecutor],
            tracker: Type[DialogueStateTracker],
            domain: Type[Domain],
            include_domain: bool = False,
        ) -> bool:
            TraceContextTextMapPropagator().inject(self.action_endpoint.headers)
            result = await fn(self, tracker, domain, include_domain)
            return result

        return wrapper

    grpc_custom_action_executor_class.run = tracing_grpc_custom_action_executor_wrapper(  # type: ignore[method-assign]
        grpc_custom_action_executor_class.run
    )

    logger.debug(f"Instrumented '{grpc_custom_action_executor_class.__name__}.run.")


def _instrument_perform_health_check_method_for_component(
    tracer: Tracer,
    instrumented_class: Type,
    method_name: Text,
    attr_extractor: Optional[Callable] = None,
    return_value_attr_extractor: Optional[Callable] = None,
) -> None:
    def tracing_perform_health_check_for_component(
        fn: Callable[..., S],
    ) -> Callable[..., S]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> S:
            # Check the first argument to adjust for self/cls depending on how
            # the static method from LLMHealthCheckMixin / EmbeddingsLLMHealthCheckMixin
            # is called.
            if args and isinstance(
                args[0], (instrumented_class, type(instrumented_class))
            ):
                # The first argument is self/cls; align args to match the signature
                args = args[1:]

            span_name = f"{instrumented_class.__name__}.{fn.__name__}"
            extracted_attrs = attr_extractor(*args, **kwargs) if attr_extractor else {}

            with tracer.start_as_current_span(span_name) as span:
                result = fn(*args, **kwargs)

                # Extract attributes from the return value, if an extractor is provided
                return_value_attributes = (
                    return_value_attr_extractor(result, *args, **kwargs)
                    if return_value_attr_extractor
                    else {}
                )

                span.set_attributes({**extracted_attrs, **return_value_attributes})
                return result

        return wrapper

    method_to_trace = getattr(instrumented_class, method_name)
    traced_method = tracing_perform_health_check_for_component(method_to_trace)
    setattr(instrumented_class, method_name, traced_method)

    logger.debug(f"Instrumented '{instrumented_class.__name__}.{method_name}'.")


def _mangled_instrumented_boolean_attribute_name(instrumented_class: Type) -> Text:
    # see https://peps.python.org/pep-0008/#method-names-and-instance-variables
    # and https://stackoverflow.com/a/50401073
    return f"_{instrumented_class.__name__}__{INSTRUMENTED_BOOLEAN_ATTRIBUTE_NAME}"


def class_is_instrumented(instrumented_class: Type) -> bool:
    """Check if a class has already been instrumented."""
    return getattr(
        instrumented_class,
        _mangled_instrumented_boolean_attribute_name(instrumented_class),
        False,
    )


def mark_class_as_instrumented(instrumented_class: Type) -> None:
    """Mark a class as instrumented if it isn't already marked."""
    if not class_is_instrumented(instrumented_class):
        setattr(
            instrumented_class,
            _mangled_instrumented_boolean_attribute_name(instrumented_class),
            True,
        )


def _instrumented_module_boolean_attribute_name(module_name: Text) -> Text:
    return f"{module_name}_{INSTRUMENTED_MODULE_BOOLEAN_ATTRIBUTE_NAME}"


def module_is_instrumented(module_name: Text) -> bool:
    """Check if a module has already been instrumented."""
    module = importlib.import_module(module_name)
    return getattr(
        module,
        _instrumented_module_boolean_attribute_name(module_name),
        False,
    )


def mark_module_as_instrumented(module_name: Text) -> None:
    """Mark a module as instrumented if it isn't already marked."""
    module = importlib.import_module(module_name)
    if not module_is_instrumented(module_name):
        setattr(module, _instrumented_module_boolean_attribute_name(module_name), True)


def _instrument_agent_internal_state_transitions(
    tracer: Tracer,
    module_name: str,
) -> None:
    """Instrument agent internal state transitions."""
    # Import agent stack frame
    from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
        AgentStackFrame,
    )

    # Store original __setattr__ method
    original_setattr = AgentStackFrame.__setattr__

    def traced_setattr(self: object, name: str, value: Any) -> None:
        # Type assertion: we know this is called only on AgentStackFrame instances
        agent_frame = cast("AgentStackFrame", self)

        # Check if we're setting the state field
        if name == "state":
            # Get current state before change
            current_state = agent_frame.state

            # Call original setattr
            original_setattr(self, name, value)

            # Track state transition if state actually changed
            if current_state != value:
                _track_agent_internal_state_transition(
                    tracer=tracer,
                    agent_id=agent_frame.agent_id,
                    flow_id=agent_frame.flow_id,
                    from_state=current_state.value,
                    to_state=value.value,
                    step_id=agent_frame.step_id,
                )
        else:
            # For other attributes, just call original setattr
            original_setattr(self, name, value)

    # Replace the __setattr__ method with our traced version
    AgentStackFrame.__setattr__ = traced_setattr  # type: ignore[method-assign]


def _track_agent_internal_state_transition(
    tracer: Tracer,
    agent_id: str,
    flow_id: str,
    from_state: str,
    to_state: str,
    step_id: str,
) -> None:
    """Track an agent internal state transition."""
    # Create span for state transition
    span_name = "AgentStackFrame.state_transition"
    with tracer.start_as_current_span(
        span_name,
        attributes={
            "agent_id": agent_id,
            "flow_id": flow_id,
            "from_state": from_state,
            "to_state": to_state,
            "step_id": step_id,
        },
    ):
        # Context manager ensures proper span lifecycle (start/end)
        # No additional work needed - span attributes are set at creation
        pass
