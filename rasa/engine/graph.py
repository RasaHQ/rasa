from __future__ import annotations

import dataclasses
import typing
from abc import ABC, abstractmethod
from collections import ChainMap
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Text, Type, Tuple

from rasa.engine.exceptions import GraphComponentException, GraphSchemaException
import rasa.shared.utils.common
from rasa.engine.storage.resource import Resource

if typing.TYPE_CHECKING:
    from rasa.engine.storage.storage import ModelStorage


logger = logging.getLogger(__name__)


@dataclass
class SchemaNode:
    """Represents one node in the schema."""

    needs: Dict[Text, Text]
    uses: Type[GraphComponent]
    constructor_name: Text
    fn: Text
    config: Dict[Text, Any]
    eager: bool = False
    is_target: bool = False
    is_input: bool = False
    resource: Optional[Resource] = None


@dataclass
class GraphSchema:
    """Represents a graph for training a model or making predictions."""

    nodes: Dict[Text, SchemaNode]

    def as_dict(self) -> Dict[Text, Any]:
        """Returns graph schema in a serializable format.

        Returns:
            The graph schema in a format which can be dumped as JSON or other formats.
        """
        serializable_graph_schema = {}
        for node_name, node in self.nodes.items():
            serializable = dataclasses.asdict(node)

            # Classes are not JSON serializable (surprise)
            serializable["uses"] = f"{node.uses.__module__}.{node.uses.__name__}"

            serializable_graph_schema[node_name] = serializable

        return serializable_graph_schema

    @classmethod
    def from_dict(cls, serialized_graph_schema: Dict[Text, Any]) -> GraphSchema:
        """Loads a graph schema which has been serialized using `schema.as_dict()`.

        Args:
            serialized_graph_schema: A serialized graph schema.

        Returns:
            A properly loaded schema.

        Raises:
            GraphSchemaException: In case the component class for a node couldn't be
                found.
        """
        nodes = {}
        for node_name, serialized_node in serialized_graph_schema.items():
            try:
                serialized_node[
                    "uses"
                ] = rasa.shared.utils.common.class_from_module_path(
                    serialized_node["uses"]
                )
            except ImportError as e:
                raise GraphSchemaException(
                    "Error deserializing graph schema. Can't "
                    "find class for graph component type "
                    f"'{serialized_node['uses']}'."
                ) from e

            nodes[node_name] = SchemaNode(**serialized_node)

        return GraphSchema(nodes)


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    # TODO: This doesn't enforce that it exists in subclasses..
    default_config: Dict[Text, Any]

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.

        Returns: An instantiated `GraphComponent`.
        """
        ...

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """Creates a component using a persisted version of itself.

        If not overridden this method merely calls `create`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            kwargs: Output values from previous nodes might be passed in as `kwargs`.

        Returns:
            An instantiated, loaded `GraphComponent`.
        """
        return cls.create(config, model_storage, resource, execution_context)

    def supported_languages(self) -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    def required_packages(self) -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []


class GraphNodeHook(ABC):
    """Holds functionality to be run before and after a `GraphNode`."""

    @abstractmethod
    def on_before_node(
        self,
        node_name: Text,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Runs before the `GraphNode` executes.

        Args:
            node_name: The name of the node being run.
            config: The node's config.
            received_inputs: The inputs received by the node.

        Returns: Data that is then passed to `on_after_node`

        """
        ...

    @abstractmethod
    def on_after_node(
        self,
        node_name: Text,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Runs after the `GraphNode` as executed.

        Args:
            node_name: The name of the node that has run.
            config: The node's config.
            output: The output of the node.
            input_hook_data: Data returned from `on_before_node`.
        """
        ...


@dataclass
class ExecutionContext:
    """Holds information about a single graph run."""

    graph_schema: GraphSchema = field(repr=False)
    model_id: Text
    should_add_diagnostic_data: bool = False
    is_finetuning: bool = False


class GraphNode:
    """Instantiates and runs a `GraphComponent` within a graph.

    A `GraphNode` is a wrapper for a `GraphComponent` that allows it to be executed
    in the context of a graph. It is responsible for instantiating the component at the
    correct time, collecting the inputs from the parent nodes, running the run function
    of the component and passing the output onwards.
    """

    def __init__(
        self,
        node_name: Text,
        component_class: Type[GraphComponent],
        constructor_name: Text,
        component_config: Dict[Text, Any],
        fn_name: Text,
        inputs: Dict[Text, Text],
        eager: bool,
        model_storage: ModelStorage,
        resource: Optional[Resource],
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> None:
        """Initializes `GraphNode`.

        Args:
            node_name: The name of the node in the schema.
            component_class: The class to be instantiated and run.
            constructor_name: The method used to instantiate the component.
            component_config: Config to be passed to the component.
            fn_name: The function on the instantiated `GraphComponent` to be run when
                the node executes.
            inputs: A map from input name to parent node name that provides it.
            eager: Determines if the node is instantiated right away, or just before
                being run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: If given the `GraphComponent` will be loaded from the
                `model_storage` using the given resource.
            execution_context: Information about the current graph run.
            hooks: These are called before and after execution.
        """
        self._node_name: Text = node_name
        self._component_class: Type[GraphComponent] = component_class
        self._constructor_name: Text = constructor_name
        self._constructor_fn: Callable = getattr(
            self._component_class, self._constructor_name
        )
        self._component_config: Dict[Text, Any] = {
            **self._component_class.default_config,
            **component_config,
        }
        self._fn_name: Text = fn_name
        self._fn: Callable = getattr(self._component_class, self._fn_name)
        self._inputs: Dict[Text, Text] = inputs
        self._eager: bool = eager

        self._model_storage = model_storage
        self._existing_resource = resource

        self._execution_context: ExecutionContext = execution_context

        self._hooks: List[GraphNodeHook] = hooks if hooks else []

        self._component: Optional[GraphComponent] = None
        if self._eager:
            self._load_component()

    def _load_component(
        self, additional_kwargs: Optional[Dict[Text, Any]] = None
    ) -> None:
        kwargs = additional_kwargs if additional_kwargs else {}

        logger.debug(
            f"Node {self._node_name} loading "
            f"{self._component_class.__name__}.{self._constructor_name} "
            f"with config: {self._component_config}, and kwargs: {kwargs}."
        )

        constructor = getattr(self._component_class, self._constructor_name)
        try:
            self._component: GraphComponent = constructor(  # type: ignore[no-redef]
                config=self._component_config,
                model_storage=self._model_storage,
                resource=self._get_resource(kwargs),
                execution_context=self._execution_context,
                **kwargs,
            )
        except Exception as e:
            raise GraphComponentException(
                f"Error initializing graph component for node {self._node_name}."
            ) from e

    def _get_resource(self, kwargs: Dict[Text, Any]) -> Resource:
        if "resource" in kwargs:
            # A parent node provides resource during training. The component wrapped
            # by this `GraphNode` will load itself from this resource.
            return kwargs.pop("resource")

        if self._existing_resource:
            # The component should be loaded from a trained resource during inference.
            # E.g. a classifier might train and persist itself during training and will
            # then load itself from this resource during inference.
            return self._existing_resource

        # The component gets a chance to persist itself
        return Resource(self._node_name)

    def parent_node_names(self) -> List[Text]:
        """The names of the parent nodes of this node."""
        return list(self._inputs.values())

    @staticmethod
    def _collapse_inputs_from_previous_nodes(
        inputs_from_previous_nodes: Tuple[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        return dict(ChainMap(*inputs_from_previous_nodes))

    def __call__(self, *inputs_from_previous_nodes: Dict[Text, Any]) -> Dict[Text, Any]:
        """Calls the `GraphComponent` run method when the node executes in the graph.

        Args:
            *inputs_from_previous_nodes: The output of all parent nodes. Each is a
                dictionary with a single item mapping the node's name to its output.

        Returns: A dictionary with a single item mapping the node's name to the output.
        """
        received_inputs = self._collapse_inputs_from_previous_nodes(
            inputs_from_previous_nodes
        )

        input_hook_outputs = self._run_before_hooks(received_inputs)

        kwargs = {}
        for input_name, input_node in self._inputs.items():
            kwargs[input_name] = received_inputs[input_node]

        if not self._eager:
            constructor_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self._load_component(constructor_kwargs)

        run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._fn)
        logger.debug(
            f"Node {self._node_name} running "
            f"{self._component_class.__name__}.{self._fn_name} "
            f"with kwargs: {run_kwargs}."
        )

        try:
            output = self._fn(self._component, **run_kwargs)
        except Exception as e:
            raise GraphComponentException(
                f"Error running graph component for node {self._node_name}."
            ) from e

        self._run_after_hooks(input_hook_outputs, output)

        return {self._node_name: output}

    def _run_after_hooks(self, input_hook_outputs: List[Dict], output: Any) -> None:
        for hook, hook_data in zip(self._hooks, input_hook_outputs):
            try:
                hook.on_after_node(
                    node_name=self._node_name,
                    config=self._component_config,
                    output=output,
                    input_hook_data=hook_data,
                )
            except Exception as e:
                raise GraphComponentException(
                    f"Error running after hook for node {self._node_name}."
                ) from e

    def _run_before_hooks(self, received_inputs: Dict[Text, Any]) -> List[Dict]:
        input_hook_outputs = []
        for hook in self._hooks:
            try:
                hook_output = hook.on_before_node(
                    node_name=self._node_name,
                    config=self._component_config,
                    received_inputs=received_inputs,
                )
                input_hook_outputs.append(hook_output)
            except Exception as e:
                raise GraphComponentException(
                    f"Error running before hook for node {self._node_name}."
                ) from e
        return input_hook_outputs

    @classmethod
    def from_schema_node(
        cls,
        node_name: Text,
        schema_node: SchemaNode,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
    ) -> GraphNode:
        """Creates a `GraphNode` from a `SchemaNode`."""
        return cls(
            node_name=node_name,
            component_class=schema_node.uses,
            constructor_name=schema_node.constructor_name,
            component_config=schema_node.config,
            fn_name=schema_node.fn,
            inputs=schema_node.needs,
            eager=schema_node.eager,
            model_storage=model_storage,
            execution_context=execution_context,
            resource=schema_node.resource,
        )
