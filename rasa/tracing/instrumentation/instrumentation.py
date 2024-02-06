import contextlib
import functools
import importlib
import inspect
import logging
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Text,
    Type,
    TypeVar,
)

import rasa.shared.utils.io
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from rasa.core.actions.action import Action, RemoteAction
from rasa.core.agent import Agent
from rasa.core.channels import OutputChannel
from rasa.core.lock_store import LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)
from rasa.engine.graph import GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.constants import DOCS_BASE_URL
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

from rasa.tracing.instrumentation import attribute_extractors

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


def traceable(
    fn: Callable[[T, Any, Any], S],
    tracer: Tracer,
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
) -> Callable[[T, Any, Any], S]:
    """Wrap a non-`async` function by tracing functionality.

    :param fn: The function to be wrapped.
    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's instance and
        the function's arguments.
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
        if issubclass(self.__class__, GraphNode) and fn.__name__ == "__call__":
            span_name = f"{self.__class__.__name__}." + attrs.get(
                "component_class", "GraphNode"
            )
        elif module_name == "command_processor":
            span_name = f"{module_name}.{fn.__name__}"
        else:
            span_name = f"{self.__class__.__name__}.{fn.__name__}"
        with tracer.start_as_current_span(span_name, attributes=attrs):
            return fn(self, *args, **kwargs)

    return wrapper


def traceable_async(
    fn: Callable[[T, Any, Any], Awaitable[S]],
    tracer: Tracer,
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
    header_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]],
) -> Callable[[T, Any, Any], Awaitable[S]]:
    """Wrap an `async` function by tracing functionality.

    :param fn: The function to be wrapped.
    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's instance and
        the function's arguments.
    :param header_extractor: A function that is applied to the function's arguments
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

        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{fn.__name__}",
            attributes=attrs,
        ):
            TraceContextTextMapPropagator().inject(headers)
            return await fn(self, *args, **kwargs)

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
CommandType = TypeVar("CommandType", bound=Command)


def instrument(
    tracer_provider: TracerProvider,
    agent_class: Optional[Type[AgentType]] = None,
    processor_class: Optional[Type[ProcessorType]] = None,
    tracker_store_class: Optional[Type[TrackerStoreType]] = None,
    graph_node_class: Optional[Type[GraphNodeType]] = None,
    lock_store_class: Optional[Type[LockStoreType]] = None,
    graph_trainer_class: Optional[Type[GraphTrainerType]] = None,
    llm_command_generator_class: Optional[Type[LLMCommandGeneratorType]] = None,
    command_subclasses: Optional[Type[CommandType]] = None,
    contextual_response_rephraser_class: Optional[Any] = None,
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
        lock_store_class.lock = contextlib.asynccontextmanager(traced_lock_method)  # type: ignore[assignment]  # noqa: E501

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
            "_generate_action_list_using_llm",
            attribute_extractors.extract_attrs_for_llm_command_generator,
        )
        _instrument_method(
            tracer_provider.get_tracer(llm_command_generator_class.__module__),
            llm_command_generator_class,
            "_check_commands_against_startable_flows",
            attribute_extractors.extract_attrs_for_check_commands_against_startable_flows,
        )
        mark_class_as_instrumented(llm_command_generator_class)

    if command_subclasses:
        for command_subclass in command_subclasses:  # type: ignore[attr-defined]
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
            "generate",
            attribute_extractors.extract_attrs_for_generate,
        )
        mark_class_as_instrumented(contextual_response_rephraser_class)

    if not module_is_instrumented(COMMAND_PROCESSOR_MODULE_NAME):
        _instrument_command_processor_module(tracer_provider)


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

    processor_class.get_tracker = tracing_get_tracker_wrapper(  # type: ignore[assignment]  # noqa: E501
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


def _instrument_method(
    tracer: Tracer,
    instrumented_class: Type,
    method_name: Text,
    attr_extractor: Optional[Callable],
    header_extractor: Optional[Callable] = None,
) -> None:
    method_to_trace = getattr(instrumented_class, method_name)
    traced_method = _wrap_with_tracing_decorator(
        method_to_trace, tracer, attr_extractor, header_extractor
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
) -> Callable:
    if inspect.iscoroutinefunction(callable_to_trace):
        traced_callable = traceable_async(
            callable_to_trace, tracer, attr_extractor, header_extractor
        )
    else:
        traced_callable = traceable(callable_to_trace, tracer, attr_extractor)

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
                "message_id": tracker.latest_message.message_id or "default",  # type: ignore[union-attr]  # noqa: E501
            }
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}",
                kind=SpanKind.CLIENT,
                attributes=attrs,
            ):
                if isinstance(action, RemoteAction):
                    if not isinstance(action.action_endpoint, EndpointConfig):
                        rasa.shared.utils.io.raise_warning(
                            f"No endpoint is configured to propagate the trace of this "
                            f"custom action {action.name()}. Please take a look at "
                            f"the docs and set an endpoint configuration in the "
                            f"endpoints.yml file",
                            docs=f"{DOCS_BASE_URL}/custom-actions",
                        )
                    else:
                        propagator = TraceContextTextMapPropagator()
                        propagator.inject(action.action_endpoint.headers)

                return await fn(self, action, tracker, output_channel, nlg, prediction)

        return wrapper

    processor_class._run_action = tracing_run_action_wrapper(  # type: ignore[assignment]  # noqa: E501
        processor_class._run_action
    )

    logger.debug(f"Instrumented '{processor_class.__name__}._run_action'.")


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
