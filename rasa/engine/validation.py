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

from rasa.engine.training.fingerprinting import Fingerprintable
import rasa.utils.common
import typing_utils

from rasa.engine.exceptions import GraphSchemaValidationException
from rasa.engine.graph import (
    GraphSchema,
    GraphComponent,
    SchemaNode,
    ExecutionContext,
)
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.exceptions import RasaException


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


def validate(
    schema: GraphSchema, language: Optional[Text], is_train_graph: bool
) -> None:
    """Validates a graph schema.

    This tries to validate that the graph structure is correct (e.g. all nodes pass the
    correct things into each other) as well as validates the individual graph
    components.

    Args:
        schema: The schema which needs validating.
        language: Used to validate if all components support the language the assistant
            is used in. If the language is `None`, all components are assumed to be
            compatible.
        is_train_graph: Whether the graph is used for training.

    Raises:
        GraphSchemaValidationException: If the validation failed.
    """
    _validate_cycle(schema)

    for node_name, node in schema.nodes.items():
        _validate_interface_usage(node_name, node)
        _validate_supported_languages(language, node, node_name)
        _validate_required_packages(node, node_name)

        run_fn_params, run_fn_return_type = _get_parameter_information(
            node_name, node.uses, node.fn
        )
        _validate_run_fn(
            node_name, node, run_fn_params, run_fn_return_type, is_train_graph
        )

        create_fn_params, _ = _get_parameter_information(
            node_name, node.uses, node.constructor_name
        )
        _validate_constructor(node_name, node, create_fn_params)

        _validate_needs(
            node_name, node, schema, create_fn_params, run_fn_params,
        )

    _validate_required_components(schema)


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
            f"in the graph. Please make sure that '{node_name}' does not have itself"
            f"specified in 'needs' and none of '{node_name}'s dependencies have "
            f"'{node_name}' specified in 'needs'."
        )

    parents = schema.nodes[node_name].needs.values()
    for parent_name in parents:
        _walk_and_check_for_cycles([*visited_so_far, node_name], parent_name, schema)


def _validate_interface_usage(node_name: Text, node: SchemaNode) -> None:
    if not issubclass(node.uses, GraphComponent):
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses class '{node.uses.__name__}'. This class does "
            f"not implement the '{GraphComponent.__name__}' interface and can "
            f"hence not be used within the graph. Please use a different "
            f"component or implement the '{GraphComponent}' interface for "
            f"'{node.uses.__name__}'."
        )


def _validate_supported_languages(
    language: Optional[Text], node: SchemaNode, node_name: Text
) -> None:
    supported_languages = node.uses.supported_languages()
    not_supported_languages = node.uses.not_supported_languages()

    if supported_languages and not_supported_languages:
        raise RasaException(
            "Only one of `supported_languages` and"
            "`not_supported_languages` can return a value different from `None`"
        )

    if (
        language
        and supported_languages is not None
        and language not in supported_languages
    ):
        raise GraphSchemaValidationException(
            f"Node '{node_name}' does not support the currently specified "
            f"language '{language}'."
        )

    if (
        language
        and not_supported_languages is not None
        and language in not_supported_languages
    ):
        raise GraphSchemaValidationException(
            f"Node '{node_name}' does not support the currently specified "
            f"language '{language}'."
        )


def _validate_required_packages(node: SchemaNode, node_name: Text) -> None:
    missing_packages = rasa.utils.common.find_unavailable_packages(
        node.uses.required_packages()
    )
    if missing_packages:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' requires the following packages which are "
            f"currently not installed: {', '.join(missing_packages)}."
        )


def _get_parameter_information(
    node_name: Text, uses: Type[GraphComponent], method_name: Text
) -> Tuple[Dict[Text, ParameterInfo], TypeAnnotation]:
    fn = _get_fn(node_name, uses, method_name)

    type_hints = _get_type_hints(node_name, uses, fn)
    return_type = type_hints.pop("return", inspect.Parameter.empty)

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
    node_name: Text, uses: Type[GraphComponent], fn: Callable
) -> Dict[Text, TypeAnnotation]:
    try:
        return typing.get_type_hints(fn)
    except NameError as e:
        logging.debug(
            f"Failed to retrieve type annotations for component "
            f"'{uses.__name__}' due to error:\n{e}"
        )
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses graph component '{uses.__name__}' which has "
            f"type annotations in its method '{fn.__name__}' which failed to be "
            f"retrieved. Please make sure remove any forward "
            f"reference by removing the quotes around the type "
            f"(e.g. 'def foo() -> \"int\"' becomes 'def foo() -> int'. and make sure "
            f"all type annotations can be resolved during runtime. Note that you might "
            f"need to do a 'from __future__ import annotations' to avoid forward "
            f"references."
        )


def _get_fn(node_name: Text, uses: Type[GraphComponent], method_name: Text) -> Callable:
    fn = getattr(uses, method_name, None)
    if fn is None:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses graph component '{uses.__name__}' which does not "
            f"have the specified "
            f"method '{method_name}'. Please make sure you're either using "
            f"the right graph component or specifying a valid method "
            f"for the 'fn' and 'constructor_name' options."
        )
    return fn


def _validate_run_fn(
    node_name: Text,
    node: SchemaNode,
    run_fn_params: Dict[Text, ParameterInfo],
    run_fn_return_type: TypeAnnotation,
    is_train_graph: bool,
) -> None:
    _validate_types_of_reserved_keywords(run_fn_params, node_name, node, node.fn)
    _validate_run_fn_return_type(node_name, node, run_fn_return_type, is_train_graph)

    for param_name in _required_args(run_fn_params):
        if param_name not in node.needs:
            raise GraphSchemaValidationException(
                f"Node '{node_name}' uses a component '{node.uses.__name__}' which "
                f"needs the param '{param_name}' to be provided to its method "
                f"'{node.fn}'. Please make sure to specify the parameter in "
                f"the node's 'needs' section."
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
    node_name: Text, node: SchemaNode, return_type: Type, is_training: bool
) -> None:
    if return_type == inspect.Parameter.empty:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses a component '{node.uses.__name__}' whose "
            f"method '{node.fn}' does not have a type annotation for "
            f"its return value. Type annotations are required for all graph "
            f"components to validate the graph's structure."
        )

    if is_training and not isinstance(return_type, Fingerprintable):
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses a component '{node.uses.__name__}' whose method "
            f"'{node.fn}' does not return a fingerprintable "
            f"output. This is required for caching. Please make sure you're "
            f"using a return type which implements the "
            f"'{Fingerprintable.__name__}' protocol."
        )


def _validate_types_of_reserved_keywords(
    params: Dict[Text, ParameterInfo], node_name: Text, node: SchemaNode, fn_name: Text
) -> None:
    for param_name, param in params.items():
        if param_name in KEYWORDS_EXPECTED_TYPES:
            if not typing_utils.issubtype(
                param.type_annotation, KEYWORDS_EXPECTED_TYPES[param_name]
            ):
                raise GraphSchemaValidationException(
                    f"Node '{node_name}' uses a graph component "
                    f"'{node.uses.__name__}' which has an incompatible type "
                    f"'{param.type_annotation}' for the '{param_name}' parameter in "
                    f"its '{fn_name}' method. Expected type "
                    f"'{ KEYWORDS_EXPECTED_TYPES[param_name]}'."
                )


def _validate_constructor(
    node_name: Text, node: SchemaNode, create_fn_params: Dict[Text, ParameterInfo],
) -> None:
    _validate_types_of_reserved_keywords(
        create_fn_params, node_name, node, node.constructor_name
    )

    required_args = _required_args(create_fn_params)

    if required_args and node.eager:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' has a constructor which has required "
            f"required parameters ('{', '.join(required_args)}'). "
            f"Extra parameters can only supplied to be the constructor if the node "
            f"is being run in lazy mode."
        )

    for param_name in _required_args(create_fn_params):
        if not node.eager and param_name not in node.needs:
            raise GraphSchemaValidationException(
                f"Node '{node_name}' uses a component '{node.uses.__name__}' which "
                f"needs the param '{param_name}' to be provided to its method "
                f"'{node.constructor_name}'. Please make sure to specify the "
                f"parameter in the node's 'needs' section."
            )


def _validate_needs(
    node_name: Text,
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
                f"Node '{node_name}' is configured to retrieve a value for the "
                f"param '{param_name}' by its parent node '{parent_name}' although "
                f"its method '{node.fn}' does not accept a parameter with this "
                f"name. Please make sure your node's 'needs' section is "
                f"correctly specified."
            )

        parent = graph.nodes[parent_name]

        required_type = available_args.get(param_name)
        needs_passed_to_kwargs = has_kwargs and required_type is None

        if not needs_passed_to_kwargs:
            _validate_parent_return_type(
                node_name, node, parent_name, parent, required_type.type_annotation
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
    node_name: Text,
    node: SchemaNode,
    parent_name: Text,
    parent: SchemaNode,
    required_type: TypeAnnotation,
) -> None:
    _, parent_return_type = _get_parameter_information(
        parent_name, parent.uses, parent.fn
    )
    if not typing_utils.issubtype(parent_return_type, required_type):
        raise GraphSchemaValidationException(
            f"Parent of node '{node_name}' returns type "
            f"'{parent_return_type}' but type '{required_type}' "
            f"was expected by component '{node.uses.__name__}'."
        )


def _validate_required_components(schema: GraphSchema,) -> None:
    unmet_requirements: Dict[Type, Set[Text]] = dict()
    for target_name in schema.target_names:
        unmet_requirements_for_target, _ = _recursively_check_required_components(
            node_name=target_name, schema=schema,
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
            f"{num_nodes} nodes are missing required components:\n"
            f"{errors}"
            f"Please add the required components to the graph."
        )


def _recursively_check_required_components(
    node_name: Text, schema: GraphSchema,
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
        (
            unmet_requirements_of_ancestors,
            ancestor_types,
        ) = _recursively_check_required_components(
            node_name=parent_node_name, schema=schema
        )
        for type, nodes in unmet_requirements_of_ancestors.items():
            unmet_requirements.setdefault(type, set()).update(nodes)
        component_types.update(ancestor_types)

    # check which requirements of the `schema_node` are not fulfilled by
    # comparing its requirements with the types found so far among the ancestor nodes
    unmet_requirements_of_current_node = set(
        required
        for required in schema_node.uses.required_components
        if not any(
            issubclass(used_subtype, required) for used_subtype in component_types
        )
    )

    # add the unmet requirements and the type of the `schema_node`
    for component_type in unmet_requirements_of_current_node:
        unmet_requirements.setdefault(component_type, set()).add(node_name)
    component_types.add(schema_node.uses)

    return unmet_requirements, component_types
