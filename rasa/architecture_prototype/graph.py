import copy
from collections import ChainMap
from functools import wraps
import inspect
from typing import (
    Any,
    Callable,
    Text,
    Dict,
    List,
    Tuple,
    Optional,
)

from rasa.architecture_prototype.graph_utils import (
    DaskGraph,
    GraphSchema,
    run_dask_graph,
    visualise_dask_graph,
)
from rasa.architecture_prototype.interfaces import (
    ComponentPersistorInterface,
    DaskGraphNode,
    GraphNodeComponent,
    ModelPersistorInterface,
    TrainingCacheInterface,
)
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
import rasa.shared.utils.io

# TODO: investigate persistence of metadata (especially ResponseSelector)


def fingerprint(f: Callable) -> Callable:
    """ Stores the fingerprint and caches the result of a node run."""

    @wraps(f)
    def decorated(self, *args: Any, **kwargs: Any) -> Any:
        fingerprint_key = None
        if self.cache:
            fingerprint_key = self.cache.calculate_fingerprint_key(
                self.node_name, self.config, [list(v.values())[0] for v in args],
            )

        result = f(self, *args, **kwargs)

        if self.cache:
            self.cache.store_fingerprint(
                fingerprint_key, output=result,
            )
        return result

    return decorated


class RasaComponent(GraphNodeComponent):
    """Wraps nodes in a dask graph.

     Provides reusable functionality such as class constructing, config passing,
     caching etc.
     """

    def __init__(
        self,
        component_class: Any,
        config: Dict[Text, Any],
        fn_name: Text,
        node_name: Text,
        inputs: Dict[Text, Text],
        constructor_name: Text = None,
        eager: bool = True,
        cache: Optional[TrainingCacheInterface] = None,
        persistor: Optional[ComponentPersistorInterface] = None,
    ) -> None:
        """Create a `RasaComponent`

        Args:
            component_class: The component class that will be instantiated and then
                called when the node is executed.
            config: Config to be passed to the component class constructor.
            fn_name: Name of the function to be called on the component when the node is
                executed.
            node_name: Name of the node. Used for building the edges in the graph.
            inputs: A mapping of parameters to node_names that determines this nodes
                graph dependencies.
            constructor_name: The name of the constructor method for the component 
                class.
            eager: If true, will call the constructor on instantiation of this class.
            cache: An optional cache to store fingerprints and results.
            persistor: An optional persitor to be passed to the component so that it can
                save state. 
        """
        super().__init__(config, node_name, inputs, cache)
        self._eager = eager
        self._constructor_name = constructor_name
        self._component_class = component_class
        self._fn_name = fn_name
        self._run_fn = getattr(self._component_class, fn_name)
        self._component = None
        self._persistor = persistor

        if self._constructor_name:
            self._constructor_fn = getattr(
                self._component_class, self._constructor_name
            )
        else:
            self._constructor_fn = self._component_class

        input_names = list(inputs.keys())
        self.validate_params_in_inputs(input_names, self._run_fn)
        if not eager:
            self.validate_params_in_inputs(input_names, self._constructor_fn)

        if self._eager:
            self.create_component(**self.config)

    def validate_params_in_inputs(self, input_names, func) -> None:
        """ Validate that all required parameters for a function are provided by the
            graph inputs."""
        params = inspect.signature(func).parameters
        for param_name, param in params.items():
            if param_name in ["self", "args", "kwargs", "persistor"]:
                continue
            if param.default is inspect._empty:
                if param_name not in input_names:
                    raise ValueError(
                        f"{param_name} for function {func} is missing from inputs"
                    )

    def __call__(self, *args: Any) -> Dict[Text, Any]:
        """This is called when the node is executed."""
        result = self.run(*args)
        return {self.node_name: copy.deepcopy(result)}

    @fingerprint
    def run(self, *args: Any) -> Any:
        """Call the run method of the component."""
        received_inputs = dict(ChainMap(*args))
        kwargs = {}
        for input, input_node in self.inputs.items():
            kwargs[input] = received_inputs[input_node]

        if not self._eager:
            const_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self.create_component(**const_kwargs, **self.config)

        run_kwargs = kwargs
        if "kwargs" not in rasa.shared.utils.common.arguments_of(self._run_fn):
            run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._run_fn)

        print(
            f"************** {self.node_name}: {self._component_class.__name__}.{self._run_fn.__name__}"
        )
        #  This alters the input
        return self._run_fn(self._component, **run_kwargs)

    def create_component(self, **const_kwargs: Any) -> None:
        """Creates the component using the correct constructor method."""
        if self._persistor:
            const_kwargs["persistor"] = self._persistor
        self._component = self._constructor_fn(**const_kwargs)

    def __repr__(self) -> Text:
        return f"RasaComponent({self.node_name}, {self._component_class})"


def convert_to_dask_graph(
    graph_schema: GraphSchema,
    cache: Optional[TrainingCacheInterface] = None,
    model_persistor: Optional[ModelPersistorInterface] = None,
) -> Tuple[DaskGraph, List[Text]]:
    """Converts a graph schema into a loaded dask graph."""
    dsk = {}
    targets = []
    for step_name, step_config in graph_schema.items():
        if step_name == "targets":
            targets = step_config
        else:
            dsk[step_name] = graph_component_for_config(
                step_name, step_config, cache=cache, model_persistor=model_persistor
            )
    return dsk, targets


def graph_component_for_config(
    step_name: Text,
    step_config: Dict[Text, Any],
    config_overrides: Dict[Text, Any] = None,
    cache: Optional[TrainingCacheInterface] = None,
    model_persistor: ModelPersistorInterface = None,
) -> DaskGraphNode:
    """Creates a dask graph node containing a RasaComponent from a node in a schema."""
    component_config = step_config["config"].copy()
    if config_overrides:
        component_config.update(config_overrides)

    persistor = None
    if step_config.get("persistor", True):
        persistor = model_persistor.create_component_persistor(step_name)

    return (
        RasaComponent(
            node_name=step_name,
            component_class=step_config["uses"],
            config=component_config,
            fn_name=step_config["fn"],
            inputs=step_config["needs"],
            constructor_name=step_config.get("constructor_name"),
            eager=step_config.get("eager", True),
            persistor=persistor,
            cache=cache,
        ),
        *step_config["needs"].values(),
    )


def run_as_dask_graph(
    graph_schema: GraphSchema,
    cache: Optional[TrainingCacheInterface] = None,
    model_persistor: Optional[ModelPersistorInterface] = None,
) -> Dict[Text, Any]:
    """Converts a schema to a dask graph and runs it."""
    dask_graph, targets = convert_to_dask_graph(
        graph_schema, cache=cache, model_persistor=model_persistor
    )
    return run_dask_graph(dask_graph, targets)


def visualise_as_dask_graph(graph_schema: Dict[Text, Any], filename: Text) -> None:
    """Converts a schema to a dask graph and visualises it."""
    dask_graph, _ = convert_to_dask_graph(graph_schema)
    visualise_dask_graph(dask_graph, filename)
