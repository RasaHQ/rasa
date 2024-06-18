from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Text, Type, Tuple, Union

import structlog
import asyncio

import rasa.shared.utils.common
import rasa.utils.common
from rasa.engine.exceptions import (
    GraphComponentException,
    GraphRunError,
    GraphSchemaException,
)
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.data import TrainingType
from rasa.shared.exceptions import InvalidConfigException, RasaException

structlogger = structlog.get_logger()


@dataclass
class SchemaNode:
    """Represents one node in the schema.

    Args:
        needs: describes which parameters in `fn` (or `constructor_name`
            if `eager==False`) are filled by which parent nodes.
        uses: The class which models the behavior of this specific graph node.
        constructor_name: The name of the constructor which should be used to
            instantiate the component. If `eager==False` then the `constructor` can
            also specify parameters which are filled by parent nodes. This is e.g.
            useful if a parent node returns a `Resource` and this node wants to
            directly load itself from this resource.
        fn: The name of the function which should be called on the instantiated
            component when the graph is executed. The parameters from `needs` are
            filled from the parent nodes.
        config: The user's configuration for this graph node. This configuration
            does not need to be specify all possible parameters; the default values
            for missing parameters will be filled in later.
        eager: If `eager` then the component is instantiated before the graph is run.
            Otherwise it's instantiated as the graph runs (lazily). Usually we always
            instantiated lazily during training and eagerly during inference (to
            avoid that the first prediction takes longer).
        is_target: If `True` then this node can't be pruned during fingerprinting
            (it might be replaced with a cached value though). This is e.g. used for
            all components which train as their result always needs to be added to
            the model archive so that the data is available during inference.
        is_input: Nodes with `is_input` are _always_ run (also during the fingerprint
            run). This makes sure that we e.g. detect changes in file contents.
        resource: If given, then the graph node is loaded from an existing resource
            instead of instantiated from scratch. This is e.g. used to load a trained
            component for predictions.
    """

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
        serializable_graph_schema: Dict[Text, Dict[Text, Any]] = {"nodes": {}}
        for node_name, node in self.nodes.items():
            serializable = dataclasses.asdict(node)

            # Classes are not JSON serializable (surprise)
            serializable["uses"] = f"{node.uses.__module__}.{node.uses.__name__}"

            serializable_graph_schema["nodes"][node_name] = serializable

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
        for node_name, serialized_node in serialized_graph_schema["nodes"].items():
            try:
                serialized_node["uses"] = (
                    rasa.shared.utils.common.class_from_module_path(
                        serialized_node["uses"]
                    )
                )

                resource = serialized_node["resource"]
                if resource:
                    serialized_node["resource"] = Resource(**resource)

            except ImportError as e:
                raise GraphSchemaException(
                    "Error deserializing graph schema. Can't "
                    "find class for graph component type "
                    f"'{serialized_node['uses']}'."
                ) from e

            nodes[node_name] = SchemaNode(**serialized_node)

        return GraphSchema(nodes)

    @property
    def target_names(self) -> List[Text]:
        """Returns the names of all target nodes."""
        return [node_name for node_name, node in self.nodes.items() if node.is_target]

    def minimal_graph_schema(self, targets: Optional[List[Text]] = None) -> GraphSchema:
        """Returns a new schema where all nodes are a descendant of a target."""
        dependencies = self._all_dependencies_schema(
            targets if targets else self.target_names
        )

        return GraphSchema(
            {
                node_name: node
                for node_name, node in self.nodes.items()
                if node_name in dependencies
            }
        )

    def _all_dependencies_schema(self, targets: List[Text]) -> List[Text]:
        required = []
        for target in targets:
            required.append(target)
            try:
                target_dependencies = self.nodes[target].needs.values()
            except KeyError:  # This can happen if the target is an input placeholder.
                continue
            for dependency in target_dependencies:
                required += self._all_dependencies_schema([dependency])

        return required

    def has_node(self, node_type: Type, include_subtypes: bool = True) -> bool:
        """Checks if the graph schema contains a node of the specified node type.
        By default, it also checks for subtypes of the specified node type.
        """
        for node in self.nodes.values():
            if (node.uses is node_type) or (
                include_subtypes and issubclass(node.uses, node_type)
            ):
                return True
        return False

    def count_nodes_of_a_given_type(
        self, node_type: Type, include_subtypes: bool = True
    ) -> int:
        """Counts the number of the nodes of specified class"""
        counter = 0
        for node in self.nodes.values():
            if (node.uses is node_type) or (
                include_subtypes and issubclass(node.uses, node_type)
            ):
                counter += 1
        return counter


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

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
        cls: Any,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """Creates a component using a persisted version of itself.

        If not overridden this method merely calls `create`.

        Args:
            config: The config for this graph component. This is the default config of
                the component merged with config specified by the user.
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

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config.

        Default config and user config are merged by the `GraphNode` before the
        config is passed to the `create` and `load` method of the component.

        Returns:
            The default config of the component.
        """
        return {}

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component cannot work with.

        Returns: A list of not supported languages, or
            `None` to signify all are supported.
        """
        return None

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []

    @classmethod
    def fingerprint_addon(cls: Any, config: Dict[str, Any]) -> Optional[str]:
        """Adds additional data to the fingerprint calculation.

        This is useful if a component uses external data that is not provided
        by the graph.
        """
        return None


class GraphNodeHook(ABC):
    """Holds functionality to be run before and after a `GraphNode`."""

    @abstractmethod
    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Runs before the `GraphNode` executes.

        Args:
            node_name: The name of the node being run.
            execution_context: The execution context of the current graph run.
            config: The node's config.
            received_inputs: Mapping from parameter name to input value.

        Returns:
            Data that is then passed to `on_after_node`

        """
        ...

    @abstractmethod
    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Runs after the `GraphNode` as executed.

        Args:
            node_name: The name of the node that has run.
            execution_context: The execution context of the current graph run.
            config: The node's config.
            output: The output of the node.
            input_hook_data: Data returned from `on_before_node`.
        """
        ...


@dataclass
class ExecutionContext:
    """Holds information about a single graph run."""

    graph_schema: GraphSchema = field(repr=False)
    model_id: Optional[Text] = None
    should_add_diagnostic_data: bool = False
    is_finetuning: bool = False
    # This is set by the `GraphNode` before it is passed to the `GraphComponent`.
    node_name: Optional[Text] = None

    def has_node(self, node_type: Type, include_subtypes: bool = True) -> bool:
        """Checks if the graph node of the given type is present in the graph schema.
        By default, it also checks for subtypes of the specified node type.
        """
        return self.graph_schema.has_node(node_type, include_subtypes)


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
        self._component_config: Dict[Text, Any] = rasa.utils.common.override_defaults(
            self._component_class.get_default_config(), component_config
        )
        self._fn_name: Text = fn_name
        self._fn: Callable = getattr(self._component_class, self._fn_name)
        self._inputs: Dict[Text, Text] = inputs
        self._eager: bool = eager

        self._model_storage = model_storage
        self._existing_resource = resource

        self._execution_context: ExecutionContext = dataclasses.replace(
            execution_context, node_name=self._node_name
        )

        self._hooks: List[GraphNodeHook] = hooks if hooks else []

        self._component: Optional[GraphComponent] = None
        if self._eager:
            self._load_component()

    def _load_component(self, **kwargs: Any) -> None:
        structlogger.debug(
            "graph.node.loading_component",
            node_name=self._node_name,
            clazz=self._component_class.__name__,
            constructor=self._constructor_name,
            kwargs=kwargs,
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
        except InvalidConfigException:
            # Pass through somewhat expected exception to allow more fine granular
            # handling of exceptions.
            raise
        except Exception as e:
            if not isinstance(e, RasaException):
                raise GraphComponentException(
                    f"Error initializing graph component for node {self._node_name}."
                ) from e
            else:
                structlogger.error(
                    "graph.node.error_loading_component",
                    node_name=self._node_name,
                )
                raise

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

    async def __call__(
        self, *inputs_from_previous_nodes: Union[Tuple[Text, Any], Text]
    ) -> Tuple[Text, Any]:
        """Calls the `GraphComponent` run method when the node executes in the graph.

        Args:
            *inputs_from_previous_nodes: The output of all parent nodes. Each is a
                dictionary with a single item mapping the node's name to its output.
                If the node couldn't be resolved and has no output, the node name is
                provided instead of a tuple.

        Returns:
            The node name and its output.
        """
        # filter out arguments that dask couldn't lookup
        received_inputs: Dict[Text, Any] = {}
        for i in inputs_from_previous_nodes:
            if isinstance(i, tuple):
                node_name, node_output = i
                received_inputs[node_name] = node_output
            else:
                structlogger.warning(
                    "graph.node.input_not_resolved",
                    node_name=self._node_name,
                    input_name=i,
                    event_info=(
                        "Node input was not resolved, there is no putput. "
                        "Another component should have provided this as an output."
                    ),
                )

        kwargs = {}
        for input_name, input_provider_node_name in self._inputs.items():
            if input_provider_node_name not in received_inputs:
                raise GraphRunError(
                    f"Missing input to run node '{self._node_name}'. "
                    f"Expected input '{input_provider_node_name}' to "
                    f"provide parameter '{input_name}'."
                )
            kwargs[input_name] = received_inputs[input_provider_node_name]

        input_hook_outputs = self._run_before_hooks(kwargs)

        if not self._eager:
            constructor_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self._load_component(**constructor_kwargs)
            run_kwargs = {
                k: v for k, v in kwargs.items() if k not in constructor_kwargs
            }
        else:
            run_kwargs = kwargs

        structlogger.debug(
            "graph.node.running_component",
            node_name=self._node_name,
            clazz=self._component_class.__name__,
            fn=self._fn_name,
        )

        try:
            if asyncio.iscoroutinefunction(self._fn):
                output = await self._fn(self._component, **run_kwargs)
            else:
                output = self._fn(self._component, **run_kwargs)
        except InvalidConfigException:
            # Pass through somewhat expected exception to allow more fine granular
            # handling of exceptions.
            raise
        except Exception as e:
            if not isinstance(e, RasaException):
                raise GraphComponentException(
                    f"Error running graph component for node {self._node_name}."
                ) from e
            else:
                structlogger.error(
                    "graph.node.error_running_component",
                    node_name=self._node_name,
                )
                raise

        self._run_after_hooks(input_hook_outputs, output)

        return self._node_name, output

    def _run_after_hooks(self, input_hook_outputs: List[Dict], output: Any) -> None:
        for hook, hook_data in zip(self._hooks, input_hook_outputs):
            try:
                structlogger.debug(
                    "graph.node.hook.on_after_node",
                    node_name=self._node_name,
                    hook_name=hook.__class__.__name__,
                )
                hook.on_after_node(
                    node_name=self._node_name,
                    execution_context=self._execution_context,
                    config=self._component_config,
                    output=output,
                    input_hook_data=hook_data,
                )
            except Exception as e:
                raise GraphComponentException(
                    f"Error running after hook for node '{self._node_name}'."
                ) from e

    def _run_before_hooks(self, received_inputs: Dict[Text, Any]) -> List[Dict]:
        input_hook_outputs = []
        for hook in self._hooks:
            try:
                structlogger.debug(
                    "graph.node.hook.on_before_node",
                    node_name=self._node_name,
                    hook_name=hook.__class__.__name__,
                )
                hook_output = hook.on_before_node(
                    node_name=self._node_name,
                    execution_context=self._execution_context,
                    config=self._component_config,
                    received_inputs=received_inputs,
                )
                input_hook_outputs.append(hook_output)
            except Exception as e:
                raise GraphComponentException(
                    f"Error running before hook for node '{self._node_name}'."
                ) from e
        return input_hook_outputs

    @classmethod
    def from_schema_node(
        cls,
        node_name: Text,
        schema_node: SchemaNode,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
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
            hooks=hooks,
        )


@dataclass()
class GraphModelConfiguration:
    """The model configuration to run as a graph during training and prediction."""

    train_schema: GraphSchema
    predict_schema: GraphSchema
    training_type: TrainingType
    assistant_id: Optional[Text]
    language: Optional[Text]
    core_target: Optional[Text]
    nlu_target: Optional[Text]
    spaces: Optional[Dict[Text, Text]] = None
