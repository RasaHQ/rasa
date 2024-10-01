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
)

from multidict import MultiDict
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from rasa.core.actions.action import Action, RemoteAction, CustomActionExecutor
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
from rasa.core.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator import (
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.engine.graph import GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import SET_SLOT_COMMAND
from rasa.shared.nlu.training_data.message import Message
from rasa.tracing.constants import REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME
from rasa.tracing.instrumentation import attribute_extractors
from rasa.tracing.instrumentation.intentless_policy_instrumentation import (
    _instrument_extract_ai_responses,
    _instrument_generate_answer,
    _instrument_select_few_shot_conversations,
    _instrument_select_response_examples,
)
from rasa.tracing.instrumentation.metrics import (
    record_llm_command_generator_metrics,
    record_single_step_llm_command_generator_metrics,
    record_multi_step_llm_command_generator_metrics,
    record_callable_duration_metrics,
    record_request_size_in_bytes,
)
from rasa.utils.endpoints import concat_url, EndpointConfig

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
        if module_name in ["command_processor", FLOW_EXECUTOR_MODULE_NAME]:
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
MultiStepLLMCommandGeneratorType = TypeVar(
    "MultiStepLLMCommandGeneratorType", bound=MultiStepLLMCommandGenerator
)
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
    multi_step_llm_command_generator_class: Optional[
        Type[MultiStepLLMCommandGeneratorType]
    ] = None,
    custom_action_executor_subclasses: Optional[
        List[Type[CustomActionExecutor]]
    ] = None,
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
    :param multi_step_llm_command_generator_class: The `MultiStepLLMCommandGenerator`
        to be instrumented. If `None` is given, no `MultiStepLLMCommandGenerator` will
        be instrumented.
    :param custom_action_executor_subclasses: The subclasses of `CustomActionExecutor`
    to be instrumented. If `None` is given, no subclass of `CustomActionExecutor`
    will be instrumented.
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
        lock_store_class.lock = contextlib.asynccontextmanager(traced_lock_method)  # type: ignore[assignment]

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
        mark_class_as_instrumented(single_step_llm_command_generator_class)

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
        mark_class_as_instrumented(multi_step_llm_command_generator_class)

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
        mark_class_as_instrumented(contextual_response_rephraser_class)

    if not module_is_instrumented(COMMAND_PROCESSOR_MODULE_NAME):
        _instrument_command_processor_module(tracer_provider)

    if not module_is_instrumented(FLOW_EXECUTOR_MODULE_NAME):
        _instrument_flow_executor_module(tracer_provider)

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

    nlu_command_adapter_class.predict_commands = (  # type: ignore[assignment]
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

    multi_step_llm_command_generator_class.parse_commands = (  # type: ignore[assignment]
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

    vector_store_class.search = tracing_information_retrieval_search_wrapper(  # type: ignore[assignment]
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
        "_generate_llm_answer",
        attribute_extractors.extract_attrs_for_enterprise_search_generate_llm_answer,
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

    processor_class.get_tracker = tracing_get_tracker_wrapper(  # type: ignore[assignment]
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
    mark_module_as_instrumented(FLOW_EXECUTOR_MODULE_NAME)


def _instrument_advance_flows_until_next_action(
    tracer: Tracer,
    module_name: str,
    function_name: str,
) -> None:
    def tracing_advance_flows_until_next_action_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(
            tracker: DialogueStateTracker,
            available_actions: List[str],
            flows: FlowsList,
        ) -> FlowActionPrediction:
            with tracer.start_as_current_span(f"{module_name}.{fn.__name__}") as span:
                prediction: FlowActionPrediction = fn(tracker, available_actions, flows)

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
        function_to_trace, tracer, attr_extractor, header_extractor
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
) -> Callable:
    if inspect.iscoroutinefunction(callable_to_trace):
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

    processor_class._run_action = tracing_run_action_wrapper(  # type: ignore[assignment]
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

    endpoint_config_class.request = tracing_endpoint_config_wrapper(  # type: ignore[assignment]
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

    grpc_custom_action_executor_class.run = tracing_grpc_custom_action_executor_wrapper(  # type: ignore[assignment]
        grpc_custom_action_executor_class.run
    )

    logger.debug(f"Instrumented '{grpc_custom_action_executor_class.__name__}.run.")


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
