import inspect
import logging
import typing
from typing import (
    Optional,
    Callable,
    Text,
    Tuple,
    Dict,
    Type,
    Any,
    Set,
    Union,
    TypeVar,
    List,
)

import dataclasses

from rasa.core.policies.policy import PolicyPrediction
from rasa.engine.training.fingerprinting import Fingerprintable
import rasa.utils.common
import typing_utils

from rasa.engine.exceptions import GraphSchemaValidationException
from rasa.engine.graph import (
    GraphSchema,
    GraphComponent,
    SchemaNode,
    ExecutionContext,
    GraphModelConfiguration,
)
from rasa.engine.constants import RESERVED_PLACEHOLDERS
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DOCS_URL_GRAPH_COMPONENTS
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.training_data.message import Message

TypeAnnotation = Union[TypeVar, Text, Type]


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
            f"and can hence not be run within Rasa Open Source. Please use a different "
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
                parent_return_type: TypeAnnotation = RESERVED_PLACEHOLDERS[parent_name]
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
