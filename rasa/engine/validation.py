import dataclasses
import inspect
import logging
import re
import typing
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import structlog
import typing_utils

import rasa.utils.common
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.core.policies.policy import PolicyPrediction
from rasa.dialogue_understanding.coexistence.constants import (
    CALM_ENTRY,
    NLU_ENTRY,
    NON_STICKY,
    STICKY,
)
from rasa.dialogue_understanding.coexistence.intent_based_router import (
    IntentBasedRouter,
)
from rasa.dialogue_understanding.coexistence.llm_based_router import LLMBasedRouter
from rasa.dialogue_understanding.generator import (
    LLMBasedCommandGenerator,
)
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_KEY,
    LLM_CONFIG_KEY,
)
from rasa.dialogue_understanding.patterns.chitchat import FLOW_PATTERN_CHITCHAT
from rasa.engine.constants import RESERVED_PLACEHOLDERS
from rasa.engine.exceptions import GraphSchemaValidationException
from rasa.engine.graph import (
    ExecutionContext,
    GraphComponent,
    GraphModelConfiguration,
    GraphSchema,
    SchemaNode,
)
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelMetadata, ModelStorage
from rasa.engine.training.fingerprinting import Fingerprintable
from rasa.exceptions import ValidationError
from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_KEY,
    API_TYPE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    DOCS_URL_GRAPH_COMPONENTS,
    EMBEDDINGS_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    REDIS_HOST_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
    ROUTER_CONFIG_KEY,
    ROUTING_STRATEGIES_NOT_REQUIRING_CACHE,
    ROUTING_STRATEGIES_REQUIRING_REDIS_CACHE,
    ROUTING_STRATEGY_CONFIG_KEY,
    SECRET_DATA_FORMAT_PATTERN,
    SENSITIVE_DATA,
    USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY,
    VALID_PROVIDERS_FOR_API_TYPE_CONFIG_KEY,
    VALID_ROUTING_STRATEGIES,
)
from rasa.shared.core.constants import ACTION_RESET_ROUTING, ACTION_TRIGGER_CHITCHAT
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.policies.utils import contains_intentless_policy_responses
from rasa.shared.core.slots import Slot
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.common import display_research_study_prompt

TypeAnnotation = Union[TypeVar, Text, Type, Optional[AvailableEndpoints]]

structlogger = structlog.get_logger()


@dataclasses.dataclass
class ParameterInfo:
    """Stores metadata about a function parameter."""

    type_annotation: TypeAnnotation
    # `True` if we have a parameter like `**kwargs`
    is_variable_length_keyword_arg: bool
    has_default: bool


KEYWORDS_EXPECTED_TYPES: Dict[Text, TypeAnnotation] = {
    "resource": Resource,
    "execution_context": ExecutionContext,
    "model_storage": ModelStorage,
    "config": Dict[Text, Any],
}


def validate(model_configuration: GraphModelConfiguration) -> None:
    """Validates a graph schema.

    This tries to validate that the graph structure is correct (e.g. all nodes pass the
    correct things into each other) as well as validates the individual graph
    components.

    Args:
        model_configuration: The model configuration (schemas, language, etc.)

    Raises:
        GraphSchemaValidationException: If the validation failed.
    """
    _validate(model_configuration.train_schema, True, model_configuration.language)
    _validate(model_configuration.predict_schema, False, model_configuration.language)

    _validate_prediction_targets(
        model_configuration.predict_schema,
        core_target=model_configuration.core_target,
        nlu_target=model_configuration.nlu_target,
    )


def _validate(
    schema: GraphSchema, is_train_graph: bool, language: Optional[Text]
) -> None:
    _validate_cycle(schema)

    for node_name, node in schema.nodes.items():
        _validate_interface_usage(node)
        _validate_supported_languages(language, node)
        _validate_required_packages(node)

        run_fn_params, run_fn_return_type = _get_parameter_information(
            node.uses, node.fn
        )
        _validate_run_fn(node, run_fn_params, run_fn_return_type, is_train_graph)

        create_fn_params, _ = _get_parameter_information(
            node.uses, node.constructor_name
        )
        _validate_constructor(node, create_fn_params)

        _validate_needs(node, schema, create_fn_params, run_fn_params)

    _validate_required_components(schema)


def _validate_prediction_targets(
    schema: GraphSchema, core_target: Optional[Text], nlu_target: Text
) -> None:
    if not nlu_target:
        raise GraphSchemaValidationException(
            "Graph schema specifies no target for the 'nlu_target'. It is required "
            "for a prediction graph to specify this. Please choose a valid node "
            "name for this."
        )

    _validate_target(nlu_target, "NLU", List[Message], schema)

    if core_target:
        _validate_target(core_target, "Core", PolicyPrediction, schema)


def _validate_target(
    target_name: Text, target_type: Text, expected_type: Type, schema: GraphSchema
) -> None:
    if target_name not in schema.nodes:
        raise GraphSchemaValidationException(
            f"Graph schema specifies invalid {target_type} target '{target_name}'. "
            f"Please make sure specify a valid node name as target."
        )

    if any(target_name in node.needs.values() for node in schema.nodes.values()):
        raise GraphSchemaValidationException(
            f"One graph node uses the {target_type} target '{target_name}' as input. "
            f"This is not allowed as NLU prediction and Core prediction are run "
            f"separately."
        )

    target_node = schema.nodes[target_name]
    _, target_return_type = _get_parameter_information(target_node.uses, target_node.fn)

    if not typing_utils.issubtype(target_return_type, expected_type):
        raise GraphSchemaValidationException(
            f"Your {target_type} model's output component "
            f"'{target_node.uses.__name__}' returns an invalid return "
            f"type '{target_return_type}'. This is not allowed. The {target_type} "
            f"model's last component is expected to return the type '{expected_type}'. "
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )


def _validate_cycle(schema: GraphSchema) -> None:
    for target_name in schema.target_names:
        parents = schema.nodes[target_name].needs.values()
        for parent_name in parents:
            _walk_and_check_for_cycles([], parent_name, schema)


def _walk_and_check_for_cycles(
    visited_so_far: List[Text], node_name: Text, schema: GraphSchema
) -> None:
    if node_name in visited_so_far:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' has itself as dependency. Cycles are not allowed "
            f"in the graph. Please make sure that '{node_name}' does not have itself "
            f"specified in 'needs' and none of '{node_name}'s dependencies have "
            f"'{node_name}' specified in 'needs'."
        )

    if node_name not in schema.nodes:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' is not part of the graph. Node was expected to be "
            f"present in the graph as it is used by another component."
        )

    parents = schema.nodes[node_name].needs.values()
    for parent_name in parents:
        if not _is_placeholder_input(parent_name):
            _walk_and_check_for_cycles(
                [*visited_so_far, node_name], parent_name, schema
            )


def _is_placeholder_input(name: Text) -> bool:
    return name in RESERVED_PLACEHOLDERS


def _validate_interface_usage(node: SchemaNode) -> None:
    if not issubclass(node.uses, GraphComponent):
        raise GraphSchemaValidationException(
            f"Your model uses a component with class '{node.uses.__name__}'. "
            f"This class does not implement the '{GraphComponent.__name__}' interface "
            f"and can hence not be run within Rasa Pro. Please use a different "
            f"component or implement the '{GraphComponent}' interface in class "
            f"'{node.uses.__name__}'. "
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )


def _validate_supported_languages(language: Optional[Text], node: SchemaNode) -> None:
    supported_languages = node.uses.supported_languages()
    not_supported_languages = node.uses.not_supported_languages()

    if supported_languages and not_supported_languages:
        raise RasaException(
            "Only one of `supported_languages` and "
            "`not_supported_languages` can return a value different from `None`."
        )

    if (
        language
        and supported_languages is not None
        and language not in supported_languages
    ):
        raise GraphSchemaValidationException(
            f"The component '{node.uses.__name__}' does not support the currently "
            f"specified language '{language}'."
        )

    if (
        language
        and not_supported_languages is not None
        and language in not_supported_languages
    ):
        raise GraphSchemaValidationException(
            f"The component '{node.uses.__name__}' does not support the currently "
            f"specified language '{language}'."
        )


def _validate_required_packages(node: SchemaNode) -> None:
    missing_packages = rasa.utils.common.find_unavailable_packages(
        node.uses.required_packages()
    )
    if missing_packages:
        raise GraphSchemaValidationException(
            f"Component '{node.uses.__name__}' requires the following packages which "
            f"are currently not installed: {', '.join(missing_packages)}."
        )


def _get_parameter_information(
    uses: Type[GraphComponent], method_name: Text
) -> Tuple[Dict[Text, ParameterInfo], TypeAnnotation]:
    fn = _get_fn(uses, method_name)

    type_hints = _get_type_hints(uses, fn)
    return_type = type_hints.pop("return", inspect.Parameter.empty)
    type_hints.pop("cls", None)

    params = inspect.signature(fn).parameters

    type_info = {}
    for param_name, type_annotation in type_hints.items():
        inspect_info = params[param_name]
        if inspect_info.kind == inspect.Parameter.VAR_POSITIONAL:
            # We always pass things using keywords so we can ignore the any variable
            # length positional arguments
            continue

        type_info[param_name] = ParameterInfo(
            type_annotation=type_annotation,
            is_variable_length_keyword_arg=inspect_info.kind
            == inspect.Parameter.VAR_KEYWORD,
            has_default=inspect_info.default != inspect.Parameter.empty,
        )

    return type_info, return_type


def _get_type_hints(
    uses: Type[GraphComponent], fn: Callable
) -> Dict[Text, TypeAnnotation]:
    try:
        return typing.get_type_hints(fn)
    except NameError as e:
        logging.debug(
            f"Failed to retrieve type annotations for component "
            f"'{uses.__name__}' due to error:\n{e}"
        )
        raise GraphSchemaValidationException(
            f"Your model uses a component '{uses.__name__}' which has "
            f"type annotations in its method '{fn.__name__}' which failed to be "
            f"retrieved. Please make sure remove any forward "
            f"reference by removing the quotes around the type "
            f"(e.g. 'def foo() -> \"int\"' becomes 'def foo() -> int'. and make sure "
            f"all type annotations can be resolved during runtime. Note that you might "
            f"need to do a 'from __future__ import annotations' to avoid forward "
            f"references."
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )


def _get_fn(uses: Type[GraphComponent], method_name: Text) -> Callable:
    fn = getattr(uses, method_name, None)
    if fn is None:
        raise GraphSchemaValidationException(
            f"Your model uses a graph component '{uses.__name__}' which does not "
            f"have the required "
            f"method '{method_name}'. Please make sure you're either using "
            f"the right component or that your component is registered with the "
            f"correct component type."
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )
    return fn


def _validate_run_fn(
    node: SchemaNode,
    run_fn_params: Dict[Text, ParameterInfo],
    run_fn_return_type: TypeAnnotation,
    is_train_graph: bool,
) -> None:
    _validate_types_of_reserved_keywords(run_fn_params, node, node.fn)
    _validate_run_fn_return_type(node, run_fn_return_type, is_train_graph)

    for param_name in _required_args(run_fn_params):
        if param_name not in node.needs:
            raise GraphSchemaValidationException(
                f"Your model uses a component '{node.uses.__name__}' which "
                f"needs the param '{param_name}' to be provided to its method "
                f"'{node.fn}'. Please make sure that you registered "
                f"your component correctly and and that your model configuration is "
                f"valid."
                f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
            )


def _required_args(fn_params: Dict[Text, ParameterInfo]) -> Set[Text]:
    keywords = set(KEYWORDS_EXPECTED_TYPES)
    return {
        param_name
        for param_name, param in fn_params.items()
        if not param.has_default
        and not param.is_variable_length_keyword_arg
        and param_name not in keywords
    }


def _validate_run_fn_return_type(
    node: SchemaNode, return_type: Type, is_training: bool
) -> None:
    if return_type == inspect.Parameter.empty:
        raise GraphSchemaValidationException(
            f"Your model uses a component '{node.uses.__name__}' whose "
            f"method '{node.fn}' does not have a type annotation for "
            f"its return value. Type annotations are required for all "
            f"components to validate your model's structure."
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )

    # TODO: Handle forward references here
    if typing_utils.issubtype(return_type, list):
        return_type = typing_utils.get_args(return_type)[0]

    if is_training and not isinstance(return_type, Fingerprintable):
        raise GraphSchemaValidationException(
            f"Your model uses a component '{node.uses.__name__}' whose method "
            f"'{node.fn}' does not return a fingerprintable "
            f"output. This is required for proper caching between model trainings. "
            f"Please make sure you're using a return type which implements the "
            f"'{Fingerprintable.__name__}' protocol."
        )


def _validate_types_of_reserved_keywords(
    params: Dict[Text, ParameterInfo], node: SchemaNode, fn_name: Text
) -> None:
    for param_name, param in params.items():
        if param_name in KEYWORDS_EXPECTED_TYPES:
            if not typing_utils.issubtype(
                param.type_annotation, KEYWORDS_EXPECTED_TYPES[param_name]
            ):
                raise GraphSchemaValidationException(
                    f"Your model uses a component '{node.uses.__name__}' which has an "
                    f"incompatible type '{param.type_annotation}' for "
                    f"the '{param_name}' parameter in its '{fn_name}' method. "
                    f"The expected type is '{KEYWORDS_EXPECTED_TYPES[param_name]}'."
                    f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
                )


def _validate_constructor(
    node: SchemaNode, create_fn_params: Dict[Text, ParameterInfo]
) -> None:
    _validate_types_of_reserved_keywords(create_fn_params, node, node.constructor_name)

    required_args = _required_args(create_fn_params)

    if required_args and node.eager:
        raise GraphSchemaValidationException(
            f"Your model uses a component '{node.uses.__name__}' which has a "
            f"method '{node.constructor_name}' which has required parameters "
            f"('{', '.join(required_args)}'). "
            f"Extra parameters can only be supplied to the constructor method which is "
            f"used during training."
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )

    for param_name in _required_args(create_fn_params):
        if not node.eager and param_name not in node.needs:
            raise GraphSchemaValidationException(
                f"Your model uses a component '{node.uses.__name__}' which "
                f"needs the param '{param_name}' to be provided to its method "
                f"'{node.constructor_name}'. Please make sure that you registered "
                f"your component correctly and and that your model configuration is "
                f"valid."
                f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
            )


def _validate_needs(
    node: SchemaNode,
    graph: GraphSchema,
    create_fn_params: Dict[Text, ParameterInfo],
    run_fn_params: Dict[Text, ParameterInfo],
) -> None:
    available_args, has_kwargs = _get_available_args(
        node, create_fn_params, run_fn_params
    )

    for param_name, parent_name in node.needs.items():
        if not has_kwargs and param_name not in available_args:
            raise GraphSchemaValidationException(
                f"Your model uses a component '{node.uses.__name__}' which is "
                f"supposed to retrieve a value for the "
                f"param '{param_name}' although "
                f"its method '{node.fn}' does not accept a parameter with this "
                f"name. Please make sure that you registered "
                f"your component correctly and and that your model configuration is "
                f"valid."
                f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
            )

        if not _is_placeholder_input(parent_name) and parent_name not in graph.nodes:
            raise GraphSchemaValidationException(
                f"Missing graph component '{parent_name}'."
                f"Your model uses a component '{node.uses.__name__}' which expects "
                f"input from the missing component. The component is missing from "
                f"your model configuration. Please make sure that you registered "
                f"your component correctly and and that your model configuration is "
                f"valid."
                f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
            )

        required_type = available_args.get(param_name)

        if not has_kwargs and required_type is not None:
            parent = None
            if _is_placeholder_input(parent_name):
                parent_return_type: TypeAnnotation
                parent_return_type = RESERVED_PLACEHOLDERS[parent_name]
            else:
                parent = graph.nodes[parent_name]
                _, parent_return_type = _get_parameter_information(
                    parent.uses, parent.fn
                )

            _validate_parent_return_type(
                node, parent, parent_return_type, required_type.type_annotation
            )


def _get_available_args(
    node: SchemaNode,
    create_fn_params: Dict[Text, ParameterInfo],
    run_fn_params: Dict[Text, ParameterInfo],
) -> Tuple[Dict[Text, ParameterInfo], bool]:
    has_kwargs = any(
        param.is_variable_length_keyword_arg for param in run_fn_params.values()
    )
    available_args = run_fn_params.copy()
    if node.eager is False:
        has_kwargs = has_kwargs or any(
            param.is_variable_length_keyword_arg for param in create_fn_params.values()
        )
        available_args.update(create_fn_params)
    return available_args, has_kwargs


def _validate_parent_return_type(
    node: SchemaNode,
    parent_node: Optional[SchemaNode],
    parent_return_type: TypeAnnotation,
    required_type: TypeAnnotation,
) -> None:
    if not typing_utils.issubtype(parent_return_type, required_type):
        parent_node_text = ""
        if parent_node:
            parent_node_text = f" by the component '{parent_node.uses.__name__}'"

        raise GraphSchemaValidationException(
            f"Your component '{node.uses.__name__}' expects an input of type "
            f"'{required_type}' but it receives an input of type '{parent_return_type}'"
            f"{parent_node_text}. "
            f"Please make sure that you registered "
            f"your component correctly and and that your model configuration is "
            f"valid."
            f"See {DOCS_URL_GRAPH_COMPONENTS} for more information."
        )


def _validate_required_components(schema: GraphSchema) -> None:
    unmet_requirements: Dict[Type, Set[Text]] = dict()
    for target_name in schema.target_names:
        unmet_requirements_for_target, _ = _recursively_check_required_components(
            node_name=target_name, schema=schema
        )
        for component_type, node_names in unmet_requirements_for_target.items():
            unmet_requirements.setdefault(component_type, set()).update(node_names)
    if unmet_requirements:
        errors = "\n".join(
            [
                f"The following components require a {component_type.__name__}: "
                f"{', '.join(sorted(required_by))}. "
                for component_type, required_by in unmet_requirements.items()
            ]
        )
        num_nodes = len(
            set(
                node_name
                for required_by in unmet_requirements.values()
                for node_name in required_by
            )
        )
        raise GraphSchemaValidationException(
            f"{num_nodes} components are missing required components which have to "
            f"run before themselves:\n"
            f"{errors}"
            f"Please add the required components to your model configuration."
        )


def _recursively_check_required_components(
    node_name: Text, schema: GraphSchema
) -> Tuple[Dict[Type, Set[Text]], Set[Type]]:
    """Collects unmet requirements and types used in the subtree rooted at `node_name`.

    Args:
        schema: the graph schema
        node_name: the name of the root of the subtree
    Returns:
       unmet requirements, i.e. a mapping from component types to names of nodes that
       are contained in the subtree rooted at `schema_node` that require that component
       type but can't find it in their respective subtrees and
       a set containing all component types of nodes that are ancestors of the
       `schema_node` (or of the`schema_node` itself)
    """
    schema_node = schema.nodes[node_name]

    unmet_requirements: Dict[Type, Set[Text]] = dict()
    component_types = set()

    # collect all component types used by ancestors and their unmet requirements
    for parent_node_name in schema_node.needs.values():
        if _is_placeholder_input(parent_node_name):
            continue
        (
            unmet_requirements_of_ancestors,
            ancestor_types,
        ) = _recursively_check_required_components(
            node_name=parent_node_name, schema=schema
        )
        for _type, nodes in unmet_requirements_of_ancestors.items():
            unmet_requirements.setdefault(_type, set()).update(nodes)
        component_types.update(ancestor_types)

    # check which requirements of the `schema_node` are not fulfilled by
    # comparing its requirements with the types found so far among the ancestor nodes
    unmet_requirements_of_current_node = set(
        required
        for required in schema_node.uses.required_components()
        if not any(
            issubclass(used_subtype, required) for used_subtype in component_types
        )
    )

    # add the unmet requirements and the type of the `schema_node`
    for component_type in unmet_requirements_of_current_node:
        unmet_requirements.setdefault(component_type, set()).add(node_name)
    component_types.add(schema_node.uses)

    return unmet_requirements, component_types


def validate_flow_component_dependencies(
    flows: FlowsList,
    domain: Domain,
    story_graph: StoryGraph,
    model_configuration: GraphModelConfiguration,
) -> None:
    if (pattern_chitchat := flows.flow_by_id(FLOW_PATTERN_CHITCHAT)) is not None:
        _validate_chitchat_dependencies(pattern_chitchat, model_configuration)

    _validate_intentless_policy_responses(
        flows, domain, story_graph, model_configuration
    )


def _validate_chitchat_dependencies(
    pattern_chitchat: Flow, model_configuration: GraphModelConfiguration
) -> None:
    """Validate that the IntentlessPolicy is configured if the pattern_chitchat
    is using action_trigger_chitchat.
    """
    has_action_trigger_chitchat = pattern_chitchat.has_action_step(
        ACTION_TRIGGER_CHITCHAT
    )
    has_intentless_policy_configured = model_configuration.predict_schema.has_node(
        IntentlessPolicy
    )

    if has_action_trigger_chitchat and not has_intentless_policy_configured:
        structlogger.warn(
            f"flow_component_dependencies"
            f".{FLOW_PATTERN_CHITCHAT}"
            f".intentless_policy_not_configured",
            event_info=(
                f"`{FLOW_PATTERN_CHITCHAT}` has an action step with "
                f"`{ACTION_TRIGGER_CHITCHAT}`, but `IntentlessPolicy` is not "
                f"configured."
            ),
        )


def _validate_intentless_policy_responses(
    flows: FlowsList,
    domain: Domain,
    story_graph: StoryGraph,
    model_configuration: GraphModelConfiguration,
) -> None:
    """If IntentlessPolicy is configured, validate that it has responses to use:
    either responses from the domain that are not part of any flow, or from
    end-to-end stories.
    """
    if not model_configuration.predict_schema.has_node(IntentlessPolicy):
        return

    if not contains_intentless_policy_responses(flows, domain, story_graph):
        raise ValidationError(
            code="engine.validation.intentless_policy.no_applicable_responses_found",
            event_info=(
                "IntentlessPolicy is configured, but no applicable responses are "
                "found. Please make sure that there are responses defined in the "
                "domain that are not part of any flow, or that there are "
                "end-to-end stories in the training data."
            ),
        )


def get_component_index(schema: GraphSchema, component_class: Type) -> Optional[int]:
    """Extracts the index of a component of the given class in the schema.
    This function assumes that each component's node name is stored in a way
    that includes the index as part of the name, formatted as
    "run_{ComponentName}{Index}", which is how it's created by the recipe.
    """
    # the index of the component is at the end of the node name
    pattern = re.compile(r"\d+$")
    for node_name, node in schema.nodes.items():
        if issubclass(node.uses, component_class):
            match = pattern.search(node_name)
            if match:
                index = int(match.group())
                return index
    # index is not found or there is no component with the given class
    return None


def get_component_config(
    schema: GraphSchema, component_class: Type
) -> Optional[Dict[str, Any]]:
    """Extracts the config of a component of the given class in the schema."""
    for node_name, node in schema.nodes.items():
        if issubclass(node.uses, component_class):
            return node.config
    return None


def validate_router_exclusivity(schema: GraphSchema) -> None:
    """Validate that intent-based and llm-based routers are not
    defined at the same time.
    """
    if schema.has_node(IntentBasedRouter) and schema.has_node(LLMBasedRouter):
        raise ValidationError(
            code="engine.validation.coexistance.both_routers_defined",
            event_info=(
                "Both LLMBasedRouter and IntentBasedRouter are in the config. "
                "Please use only one of them."
            ),
        )


def validate_intent_based_router_position(schema: GraphSchema) -> None:
    """Validate that if intent-based router is defined, it is positioned before
    the llm command generator.
    """
    intent_based_router_pos = get_component_index(schema, IntentBasedRouter)
    llm_command_generator_pos = get_component_index(schema, LLMBasedCommandGenerator)
    if (
        intent_based_router_pos is not None
        and llm_command_generator_pos is not None
        and intent_based_router_pos > llm_command_generator_pos
    ):
        raise ValidationError(
            code="engine.validation.coexistance.wrong_order_of_components",
            event_info=(
                "IntentBasedRouter should come before "
                "an LLMBasedCommandGenerator in the pipeline."
            ),
        )


def validate_that_slots_are_defined_if_router_is_defined(
    schema: GraphSchema, routing_slots: List[Slot]
) -> None:
    # check whether intent-based or llm-based type of router is present
    for router_type in [IntentBasedRouter, LLMBasedRouter]:
        router_present = schema.has_node(router_type)
        slot_has_issue = len(routing_slots) == 0 or routing_slots[0].type_name != "bool"
        if router_present and slot_has_issue:
            raise ValidationError(
                code=f"engine.validation.coexistance.{ROUTE_TO_CALM_SLOT}_not_in_domain",
                event_info=(
                    f"{router_type.__name__} is in the config, but the slot "
                    f"{ROUTE_TO_CALM_SLOT} is not in the domain or not of "
                    f"type bool."
                ),
            )


def validate_that_router_is_defined_if_router_slots_are_in_domain(
    schema: GraphSchema,
    routing_slots: List[Slot],
) -> None:
    defined_router_slots = len(routing_slots) > 0
    router_present = schema.has_node(IntentBasedRouter) or schema.has_node(
        LLMBasedRouter
    )
    if defined_router_slots and (
        not router_present or routing_slots[0].type_name != "bool"
    ):
        raise ValidationError(
            code=f"engine.validation.coexistance"
            f".{ROUTE_TO_CALM_SLOT}_in_domain_with_no_router_defined",
            event_info=(
                f"The slot {ROUTE_TO_CALM_SLOT} is in the domain but the "
                f"LLMBasedRouter or the IntentBasedRouter is not in the config or "
                f"the type of the slot is not bool."
            ),
        )


def valid_nlu_entry_config(config: Optional[Dict[str, Any]]) -> bool:
    return (
        config is not None
        and NLU_ENTRY in config
        and isinstance(config[NLU_ENTRY], dict)
        and STICKY in config[NLU_ENTRY]
        and NON_STICKY in config[NLU_ENTRY]
    )


def valid_calm_entry_config(config: Optional[Dict[str, Any]]) -> bool:
    return (
        config is not None
        and CALM_ENTRY in config
        and isinstance(config[CALM_ENTRY], dict)
        and STICKY in config[CALM_ENTRY]
    )


def validate_configuration(
    schema: GraphSchema,
) -> None:
    """Validate the configuration of the existing coexistence routers."""
    if schema.has_node(IntentBasedRouter, include_subtypes=False):
        config = get_component_config(schema, IntentBasedRouter)
        if not valid_calm_entry_config(config) or not valid_nlu_entry_config(config):
            raise ValidationError(
                code="engine.validation.coexistance.invalid_configuration",
                event_info=(
                    "The configuration of the IntentBasedRouter is invalid. "
                    "Please check the documentation."
                ),
            )

    if schema.has_node(LLMBasedRouter, include_subtypes=False):
        config = get_component_config(schema, LLMBasedRouter)
        if not valid_calm_entry_config(config) or (
            config is not None
            and NLU_ENTRY in config
            and not valid_nlu_entry_config(config)
        ):
            raise ValidationError(
                code="engine.validation.coexistance.invalid_configuration",
                event_info=(
                    "The configuration of the LLMBasedRouter is invalid. "
                    "Please check the documentation."
                ),
            )


def validate_coexistance_routing_setup(
    domain: Domain, model_configuration: GraphModelConfiguration, flows: FlowsList
) -> None:
    schema = model_configuration.predict_schema
    routing_slots = [s for s in domain.slots if s.name == ROUTE_TO_CALM_SLOT]

    def validate_that_router_or_router_slot_are_defined_if_action_reset_routing_is_used(
        schema: GraphSchema, flows: FlowsList, routing_slots: List[Slot]
    ) -> None:
        slot_has_issue = len(routing_slots) == 0 or routing_slots[0].type_name != "bool"
        router_present = schema.has_node(LLMBasedRouter) or schema.has_node(
            IntentBasedRouter
        )

        if router_present or not slot_has_issue:
            return

        faulty_flows_with_action_reset_routing = [
            flow for flow in flows if flow.has_action_step(ACTION_RESET_ROUTING)
        ]

        if faulty_flows_with_action_reset_routing:
            for flow in faulty_flows_with_action_reset_routing:
                structlogger.error(
                    f"engine.validation.coexistance.{ACTION_RESET_ROUTING}_present_in_flow"
                    f"_without_router_or_{ROUTE_TO_CALM_SLOT}_slot",
                    event_info=(
                        f"The action - {ACTION_RESET_ROUTING} is used in the flow - "
                        f"{flow.id}, but a router (LLMBasedRouter/IntentBasedRouter) or"
                        f" {ROUTE_TO_CALM_SLOT} are not defined.",
                    ),
                )

            flow_ids = [flow.id for flow in faulty_flows_with_action_reset_routing]
            raise ValidationError(
                code=(
                    "engine.validation.coexistance."
                    f"{ACTION_RESET_ROUTING}_present_in_flow_without_router_or_"
                    f"{ROUTE_TO_CALM_SLOT}_slot"
                ),
                event_info=(
                    f"The action '{ACTION_RESET_ROUTING}' is used in "
                    f"{len(flow_ids)} flow(s) "
                    f"({', '.join(flow_ids)}) without a router "
                    f"(LLMBasedRouter/IntentBasedRouter) or the "
                    f"'{ROUTE_TO_CALM_SLOT}' slot."
                ),
                flows=flow_ids,  # any extra fields you want
            )

    validate_router_exclusivity(schema)
    validate_intent_based_router_position(schema)
    validate_that_slots_are_defined_if_router_is_defined(schema, routing_slots)
    validate_that_router_is_defined_if_router_slots_are_in_domain(schema, routing_slots)
    validate_configuration(schema)
    validate_that_router_or_router_slot_are_defined_if_action_reset_routing_is_used(
        schema, flows, routing_slots
    )


def _validate_component_model_client_config(
    component_config: Dict[str, Any],
    key: str,
    model_group_syntax_used: List[bool],
    model_group_ids: List[str],
    component_name: Optional[str] = None,
) -> None:
    """Validate the LLM configuration of a component.

    Checks if the llm is defined using the new syntax or the old syntax.
    If the new syntax is used, it checks that no other parameters are present.

    Args:
        component_config: The config of the component
        key: either 'llm' or 'embeddings'
        model_group_syntax_used:
            list of booleans indicating whether the new syntax is used
        model_group_ids: list of model group ids
        component_name: the name of the component
    """
    if key not in component_config:
        # no llm configuration present
        return

    if MODEL_GROUP_CONFIG_KEY in component_config[key]:
        model_group_syntax_used.append(True)
        model_group_ids.append(component_config[key][MODEL_GROUP_CONFIG_KEY])

        if len(component_config[key]) > 1:
            raise ValidationError(
                code="engine.validation.validate_model_client_configuration_setup"
                ".only_model_group_reference_key_is_allowed",
                event_info=(
                    f"You specified a '{MODEL_GROUP_CONFIG_KEY}' for the '{key}' "
                    f"config key for the component "
                    f"'{component_name or component_config['name']}'. "
                    "No other parameters are allowed under the "
                    f"'{key}' key in that case. Please update your config."
                ),
                component_name=component_name or component_config["name"],
                component_client_config_key=key,
            )
    else:
        model_group_syntax_used.append(False)

        # check that any of the sensitive data keys is not set in config
        for secret_key in SENSITIVE_DATA:
            if secret_key in component_config[key]:
                raise ValidationError(
                    code="engine.validation.validate_model_client_configuration_setup"
                    ".secret_key_not_allowed_in_the_config",
                    event_info=(
                        f"You specified '{secret_key}' in the config for "
                        f"'{component_name or component_config['name']}', "
                        f"which is not allowed. "
                        "Set secret keys through environment variables."
                    ),
                    component_name=component_name or component_config["name"],
                    component_client_config_key=key,
                    secret_key=secret_key,
                )


def validate_model_client_configuration_setup_during_training_time(
    config: Dict[str, Any],
) -> None:
    """Validates the model client configuration setup.

    Checks the model configuration of the components in the pipeline.
    Validation fails, if
    - the LLM/embeddings is/are defined using the old and the new syntax at
      the same time (either at component level itself or across different components)
    - the LLM/embeddings is/are defined using the new syntax, but no model
      group is defined or the referenced model group does not exist
    - the LLM/embeddings provider is defined using 'api_type' key for providers other
    than 'openai' or 'azure'

    Args:
        config: The config dictionary
    """

    def is_uniform_bool_list(bool_list: List[bool]) -> bool:
        # check if list contains only True or False
        return all(bool_list) or not any(bool_list)

    model_group_syntax_used: List[bool] = []
    model_group_ids: List[str] = []

    for outer_key in ["pipeline", "policies"]:
        if outer_key not in config or config[outer_key] is None:
            continue

        for component_config in config[outer_key]:
            for key in [LLM_CONFIG_KEY, EMBEDDINGS_CONFIG_KEY]:
                _validate_component_model_client_config(
                    component_config, key, model_group_syntax_used, model_group_ids
                )
                validate_api_type_config_key_usage(component_config, key)

            # as flow retrieval is not a component itself, we need to
            # check it separately
            if FLOW_RETRIEVAL_KEY in component_config:
                if EMBEDDINGS_CONFIG_KEY in component_config[FLOW_RETRIEVAL_KEY]:
                    _validate_component_model_client_config(
                        component_config[FLOW_RETRIEVAL_KEY],
                        EMBEDDINGS_CONFIG_KEY,
                        model_group_syntax_used,
                        model_group_ids,
                        component_config["name"] + "." + FLOW_RETRIEVAL_KEY,
                    )
                    validate_api_type_config_key_usage(
                        component_config[FLOW_RETRIEVAL_KEY],
                        EMBEDDINGS_CONFIG_KEY,
                        component_config["name"] + "." + FLOW_RETRIEVAL_KEY,
                    )

    # also include the ContextualResponseRephraser component
    endpoints = Configuration.get_instance().endpoints
    if endpoints.nlg is not None:
        _validate_component_model_client_config(
            endpoints.nlg.kwargs,
            LLM_CONFIG_KEY,
            model_group_syntax_used,
            model_group_ids,
            ContextualResponseRephraser.__name__,
        )

    if not is_uniform_bool_list(model_group_syntax_used):
        raise ValidationError(
            code="engine.validation.validate_model_client_configuration_setup"
            ".inconsistent_use_of_model_group_syntax",
            event_info=(
                "Some of your components refer to an LLM using the "
                f"'{MODEL_GROUP_CONFIG_KEY}' parameter, other components directly"
                f" define the LLM under the '{LLM_CONFIG_KEY}' or the "
                f"'{EMBEDDINGS_CONFIG_KEY}' key. You cannot use"
                " both types of definitions. Please chose one syntax "
                "and update your config."
            ),
        )

    # Print a deprecation warning in case the old syntax is used.
    if len(model_group_syntax_used) > 0 and model_group_syntax_used[0] is False:
        structlogger.warning(
            "validate_llm_configuration_setup",
            event_info=(
                "Defining the LLM configuration in the config.yml file itself is"
                " deprecated and will be removed in Rasa 4.0.0. "
                "Please use the new syntax and define your LLM configuration"
                "in the endpoints.yml file."
            ),
        )

    endpoints = Configuration.get_instance().endpoints
    if len(model_group_ids) > 0 and endpoints.model_groups is None:
        raise ValidationError(
            code="engine.validation.validate_model_client_configuration_setup"
            ".referencing_model_group_but_none_are_defined",
            event_info=(
                "You are referring to (a) model group(s) in your "
                "config.yml file, but no model group was defined in "
                "the endpoints.yml file. Please define the model "
                "group(s)."
            ),
        )

    if endpoints.model_groups is None:
        return

    existing_model_group_ids = [
        model_group[MODEL_GROUP_ID_CONFIG_KEY] for model_group in endpoints.model_groups
    ]

    for model_group_id in model_group_ids:
        if model_group_id not in existing_model_group_ids:
            raise ValidationError(
                code="engine.validation.validate_model_client_configuration_setup"
                ".referencing_undefined_model_group",
                event_info=(
                    "One of your components is referring to the model group "
                    f"'{model_group_id}', but this model group does not exist in the "
                    f"endpoints.yml file. Please chose one of the existing "
                    f"model groups ({existing_model_group_ids}) or define "
                    f"the model group for '{model_group_id}'."
                ),
                referencing_model_group_id=model_group_id,
                existing_model_group_ids=existing_model_group_ids,
            )


def _validate_component_model_client_config_has_references_to_endpoints(
    component_config: Dict[Text, Any],
    key: str,
    component_name: Optional[Text] = None,
) -> None:
    """Validates that the specified client configuration references a valid model group
    defined in the `endpoints.yml` file.

    This function ensures that when the client configuration for a component uses the
    `model_group` key, the referenced model group exists in the `endpoints.yml` file.
    If the referenced model group is missing or invalid, an error is raised.

    Args:
        component_config: The configuration dictionary for the component being
            validated.
        key: 'llm' or 'embeddings'
        component_name: Optional; the name of the component being validated, used for
            error messages.

    Raises:
        SystemExit: If the referenced model group is missing or invalid.
    """
    if key not in component_config:
        # no llm/embeddings configuration present
        return

    endpoints = Configuration.get_instance().endpoints

    if MODEL_GROUP_CONFIG_KEY in component_config[key]:
        referencing_model_group_id = component_config[key][MODEL_GROUP_CONFIG_KEY]

        if endpoints.model_groups is None:
            raise ValidationError(
                code="engine.validation.validate_model_client_config_correctly_references_endpoints"
                ".no_model_groups_defined",
                event_info=(
                    f"Your {component_name or component_config.get('name') or ''} "
                    f"component's '{key}' configuration of the trained model "
                    f"references the model group '{referencing_model_group_id}', "
                    f"but NO MODEL GROUPS ARE DEFINED in the endpoints.yml file. "
                    f"Please add a definition for the required model group in the "
                    f"endpoints.yml file."
                ),
                component_name=component_name or component_config.get("name"),
                model_group_id=referencing_model_group_id,
                component_client_config_key=key,
            )

        existing_model_group_ids = [
            model_group[MODEL_GROUP_ID_CONFIG_KEY]
            for model_group in endpoints.model_groups
        ]

        if referencing_model_group_id not in existing_model_group_ids:
            raise ValidationError(
                code="engine.validation.validate_model_client_config_correctly_references_endpoints"
                ".referenced_model_group_does_not_exist",
                event_info=(
                    f"Your {component_name or component_config.get('name') or ''} "
                    f"component's '{key}' configuration of the trained model "
                    f"references the model group '{referencing_model_group_id}', "
                    f"but this model group DOES NOT EXIST in the endpoints.yml file. "
                    f"The endpoints.yml defines the following model groups: "
                    f"{existing_model_group_ids}. "
                    f"Please add a definition for the required model group in the "
                    f"endpoints.yml file."
                ),
                model_group_id=referencing_model_group_id,
                existing_model_group_ids=existing_model_group_ids,
                component_client_config_key=key,
            )


def validate_model_client_configuration_setup_during_inference_time(
    model_metadata: ModelMetadata,
) -> None:
    for (
        component_node_name,
        component_node,
    ) in model_metadata.predict_schema.nodes.items():
        for client_config_key in [EMBEDDINGS_CONFIG_KEY, LLM_CONFIG_KEY]:
            if client_config_key not in component_node.config:
                continue

            _validate_component_model_client_config_has_references_to_endpoints(
                component_config=component_node.config,
                key=client_config_key,
                component_name=component_node_name,
            )

            # as flow retrieval is not a component itself, we need to
            # check it separately
            if FLOW_RETRIEVAL_KEY in component_node.config:
                if EMBEDDINGS_CONFIG_KEY in component_node.config[FLOW_RETRIEVAL_KEY]:
                    _validate_component_model_client_config_has_references_to_endpoints(
                        component_config=component_node.config[FLOW_RETRIEVAL_KEY],
                        key=EMBEDDINGS_CONFIG_KEY,
                        component_name=component_node_name + "." + FLOW_RETRIEVAL_KEY,
                    )

    # also include the ContextualResponseRephraser component
    endpoints = Configuration.get_instance().endpoints
    if endpoints.nlg is not None:
        _validate_component_model_client_config_has_references_to_endpoints(
            component_config=endpoints.nlg.kwargs,
            key=LLM_CONFIG_KEY,
            component_name=ContextualResponseRephraser.__name__,
        )


def _validate_unique_model_group_ids(model_groups: List[Dict[str, Any]]) -> None:
    # Each model id must be unique within the model_groups
    structlogger.debug(
        "engine.validation.validate_unique_model_group_ids",
        event_info="Validating that model group IDs are unique.",
        model_groups=model_groups,
    )

    model_ids = [model_group[MODEL_GROUP_ID_CONFIG_KEY] for model_group in model_groups]
    if len(model_ids) != len(set(model_ids)):
        counts = Counter(model_ids)
        duplicate_model_group_ids = {
            model_group_id for model_group_id, count in counts.items() if count > 1
        }
        raise ValidationError(
            code="engine.validation.validate_model_group_configuration_setup.non_unique_model_group_ids",
            event_info=(
                f"Duplicate model-group IDs found: "
                f"{', '.join(duplicate_model_group_ids)}. "
                f"Each model-group ID in endpoints.yml must be unique."
            ),
        )


def _validate_model_group_with_multiple_models(
    model_groups: List[Dict[str, Any]],
) -> None:
    # You cannot define multiple models within a model group, when no router is defined.
    for model_group in model_groups:
        if (
            len(model_group[MODELS_CONFIG_KEY]) > 1
            and ROUTER_CONFIG_KEY not in model_group
        ):
            raise ValidationError(
                code="engine.validation.validate_model_group_configuration_setup.router_not_present",
                event_info=(
                    f"You defined multiple models for the model group "
                    f"'{model_group[MODEL_GROUP_ID_CONFIG_KEY]}', but no router. "
                    "If a model group contains multiple models, a router must be "
                    "defined. Please define a router for the model group "
                    f"'{model_group[MODEL_GROUP_ID_CONFIG_KEY]}'."
                ),
                model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
            )


def _validate_model_group_router_setting(
    model_groups: List[Dict[str, Any]],
) -> None:
    # You cannot define multiple models within a model group, when no router is defined.
    for model_group in model_groups:
        if ROUTER_CONFIG_KEY not in model_group:
            continue

        for model_config in model_group.get(MODELS_CONFIG_KEY, []):
            if USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY in model_config:
                raise ValidationError(
                    code="engine.validation.validate_model_group_configuration_setup"
                    f".{USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY}_set_incorrectly",
                    event_info=(
                        f"You defined the '{USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY}' "
                        f"in the model group "
                        f"'{model_group[MODEL_GROUP_ID_CONFIG_KEY]}'. This key is not "
                        f"allowed in the model configuration as the router is defined. "
                        f"Please remove this key from your model configuration and "
                        f"update it in the '{ROUTER_CONFIG_KEY} configuration, as it "
                        f"is a router level setting."
                    ),
                    model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
                )

        router_config = model_group[ROUTER_CONFIG_KEY]
        if ROUTING_STRATEGY_CONFIG_KEY in router_config:
            routing_strategy = router_config.get(ROUTING_STRATEGY_CONFIG_KEY)
            if routing_strategy and routing_strategy not in VALID_ROUTING_STRATEGIES:
                raise ValidationError(
                    code="engine.validation.validate_model_group_configuration_setup"
                    ".invalid_routing_strategy",
                    event_info=(
                        f"The routing strategy '{routing_strategy}' you defined for "
                        f"the model group '{model_group[MODEL_GROUP_ID_CONFIG_KEY]}' "
                        f"is not valid. Valid routing strategies are categorized as "
                        f"follows:\n"
                        f"- Strategies requiring Redis caching: "
                        f"{', '.join(ROUTING_STRATEGIES_REQUIRING_REDIS_CACHE)}\n"
                        f"- Strategies not requiring caching: "
                        f"{', '.join(ROUTING_STRATEGIES_NOT_REQUIRING_CACHE)}"
                    ),
                    model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
                    invalid_routing_strategy=routing_strategy,
                    supported_routing_strategies_requiring_redis_cache=(
                        ROUTING_STRATEGIES_REQUIRING_REDIS_CACHE
                    ),
                    supported_routing_strategies_not_requiring_redis_cache=(
                        ROUTING_STRATEGIES_NOT_REQUIRING_CACHE
                    ),
                )
            if (
                routing_strategy in ROUTING_STRATEGIES_REQUIRING_REDIS_CACHE
                and REDIS_HOST_CONFIG_KEY not in router_config
            ):
                structlogger.warning(
                    "engine.validation.routing_strategy.redis_host_not_defined",
                    event_info=(
                        f"The routing strategy '{routing_strategy}' requires a Redis "
                        f"host to be defined. Without a Redis host, the system "
                        f"defaults to 'in-memory' caching. Please add the "
                        f"'{REDIS_HOST_CONFIG_KEY}' to the router configuration for "
                        f"the model group '{model_group[MODEL_GROUP_ID_CONFIG_KEY]}'."
                    ),
                    model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
                )


def _validate_usage_of_environment_variables_in_model_group_config(
    model_groups: List[Dict[str, Any]],
) -> None:
    # Limit the use of ${env_var} in the model_groups config to the following variables:
    # - deployment,
    # - api_base, api_version and api_key,
    # - aws_region_name, aws_access_key_id, aws_secret_access_key, and aws_session_token
    allowed_env_vars = {
        DEPLOYMENT_CONFIG_KEY,
        API_BASE_CONFIG_KEY,
        API_KEY,
        API_VERSION_CONFIG_KEY,
        AWS_REGION_NAME_CONFIG_KEY,
        AWS_ACCESS_KEY_ID_CONFIG_KEY,
        AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
        AWS_SESSION_TOKEN_CONFIG_KEY,
    }

    for model_group in model_groups:
        for model_config in model_group[MODELS_CONFIG_KEY]:
            for key, value in model_config.items():
                if isinstance(value, str):
                    if (
                        re.match(SECRET_DATA_FORMAT_PATTERN, value)
                        and key not in allowed_env_vars
                    ):
                        raise ValidationError(
                            code="engine.validation.validate_model_group_configuration_setup"
                            ".invalid_use_of_environment_variables",
                            event_info=(
                                f"You defined '{key}' as environment variable in model "
                                f"group '{model_group[MODEL_GROUP_ID_CONFIG_KEY]}', "
                                f"which is not allowed. "
                                f"You can only use environment variables for the "
                                f"following keys: {', '.join(allowed_env_vars)}. "
                                f"Please update your config."
                            ),
                            model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
                            key=key,
                            allowed_keys_for_env_vars=allowed_env_vars,
                        )


def _validate_sensitive_keys_are_an_environment_variables_for_model_groups(
    model_groups: List[Dict[str, Any]],
) -> None:
    # the api key can only be set as an environment variable
    for model_group in model_groups:
        for model_config in model_group[MODELS_CONFIG_KEY]:
            for key, value in model_config.items():
                if key in SENSITIVE_DATA:
                    if isinstance(value, str):
                        if not re.match(SECRET_DATA_FORMAT_PATTERN, value):
                            raise ValidationError(
                                code="engine.validation.validate_model_group_configuration_setup"
                                ".sensitive_key_string_value_must_be_set_as_env_var",
                                event_info=(
                                    f"You defined the '{key}' in model group "
                                    f"'{model_group[MODEL_GROUP_ID_CONFIG_KEY]}' as a "
                                    f"string. The '{key}' must be set as an "
                                    f"environment variable. Please update your config."
                                ),
                                key=key,
                                model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
                            )
                    else:
                        raise ValidationError(
                            code="engine.validation.validate_model_group_configuration_setup"
                            ".sensitive_key_must_be_set_as_env_var",
                            event_info=(
                                f"You should define the '{key}' in model group "
                                f"'{model_group[MODEL_GROUP_ID_CONFIG_KEY]}' using the "
                                f"environment variable syntax - "
                                f"${{ENV_VARIABLE_NAME}}. "
                                f"Please update your config."
                            ),
                            key=key,
                            model_group_id=model_group[MODEL_GROUP_ID_CONFIG_KEY],
                        )


def validate_model_group_configuration_setup() -> None:
    """Validates the model group configuration setup in endpoints.yml."""
    endpoints = Configuration.get_instance().endpoints

    if endpoints.model_groups is None:
        return

    structlogger.debug(
        "engine.validation.validate_model_group_configuration_setup",
        event_info="Validating the model group configuration setup.",
        model_groups=endpoints.model_groups,
        path=endpoints.config_file_path,
    )
    _validate_unique_model_group_ids(endpoints.model_groups)
    _validate_model_group_with_multiple_models(endpoints.model_groups)
    _validate_usage_of_environment_variables_in_model_group_config(
        endpoints.model_groups
    )
    _validate_sensitive_keys_are_an_environment_variables_for_model_groups(
        endpoints.model_groups
    )
    _validate_model_group_router_setting(endpoints.model_groups)


def validate_command_generator_exclusivity(schema: GraphSchema) -> None:
    """Validate that multiple command generators are not defined at same time."""
    from rasa.dialogue_understanding.generator import (
        LLMBasedCommandGenerator,
    )

    count = schema.count_nodes_of_a_given_type(
        LLMBasedCommandGenerator, include_subtypes=True
    )

    if count > 1:
        raise ValidationError(
            code="engine.validation.command_generator.multiple_command_generator_defined",
            event_info="Multiple LLM based command generators are defined in "
            "the config. Please use only one LLM based command generator.",
        )


def validate_command_generator_setup(
    model_configuration: GraphModelConfiguration,
) -> None:
    schema = model_configuration.predict_schema
    validate_command_generator_exclusivity(schema)


def validate_api_type_config_key_usage(
    component_config: Dict[str, Any],
    key: Literal["llm", "embeddings"],
    component_name: Optional[str] = None,
) -> None:
    """Validate the LLM/embeddings configuration of a component.

    Validation fails, if
    - the LLM/embeddings provider is defined using 'api_type' key for providers other
    than 'openai' or 'azure'

    Args:
        component_config: The config of the component
        key: either 'llm' or 'embeddings'
        component_name: the name of the component
    """
    if component_config is None or key not in component_config:
        return

    if API_TYPE_CONFIG_KEY in component_config[key]:
        api_type = component_config[key][API_TYPE_CONFIG_KEY]
        if api_type not in VALID_PROVIDERS_FOR_API_TYPE_CONFIG_KEY:
            display_research_study_prompt()
            raise ValidationError(
                code="engine.validation.component.api_type_config_key_invalid",
                event_info=f"You specified '{API_TYPE_CONFIG_KEY}: {api_type}' for "
                f"'{component_name or component_config['name']}', which is not "
                f"allowed. "
                f"The '{API_TYPE_CONFIG_KEY}' key can only be used for the "
                f"following providers: {VALID_PROVIDERS_FOR_API_TYPE_CONFIG_KEY}. "
                f"For other providers, please use the '{PROVIDER_CONFIG_KEY}' key.",
            )
