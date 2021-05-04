import copy
from collections import ChainMap
import inspect
from typing import Any, Text, Dict, List, Union, TYPE_CHECKING, Tuple, Optional

import dask

from rasa.architecture_prototype.graph_fingerprinting import TrainingCache
from rasa.architecture_prototype.persistence import AbstractModelPersistor
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
import rasa.shared.utils.io

if TYPE_CHECKING:
    pass


class RasaComponent:
    def __init__(
        self,
        component_class: Any,
        config: Dict[Text, Any],
        fn_name: Text,
        node_name: Text,
        inputs: Dict[Text, Text],
        constructor_name: Text = None,
        eager: bool = True,
        cache: Optional["TrainingCache"] = None,
        persistor: Optional["ComponentPersistor"] = None,
    ) -> None:
        self._eager = eager
        self._inputs = inputs
        self._constructor_name = constructor_name
        self._component_class = component_class
        self._config = config
        self._fn_name = fn_name
        self._run_fn = getattr(self._component_class, fn_name)
        self._component = None
        self._node_name = node_name
        self._persistor = persistor
        self._cache = cache

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
            self.create_component(**self._config)

    def validate_params_in_inputs(self, input_names, func):
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
        received_inputs = dict(ChainMap(*args))

        # TODO: make the cache stuff a decorator
        fingerprint_key = None
        if self._cache:
            fingerprint_key = self._cache.calculate_fingerprint_key(
                # TODO: do list parts nicer
                self._node_name,
                self._config,
                [list(v.values())[0] for v in args],
            )

        kwargs = {}

        for input, input_node in self._inputs.items():
            kwargs[input] = received_inputs[input_node]

        if not self._eager:
            const_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self.create_component(**const_kwargs, **self._config)

        run_kwargs = kwargs

        if "kwargs" not in rasa.shared.utils.common.arguments_of(self._run_fn):
            run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._run_fn)

        print(f"************** {self._node_name} ***************")
        #  This alters the input
        result = self._run_fn(self._component, **run_kwargs)

        if self._cache:
            self._cache.store_fingerprint(
                fingerprint_key, output=result,
            )

        return {self._node_name: copy.deepcopy(result)}

    def create_component(self, **const_kwargs: Any) -> None:
        if self._persistor:
            const_kwargs["persistor"] = self._persistor
        self._component = self._constructor_fn(**const_kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RasaComponent):
            return NotImplemented

        return (
            self._node_name == other._node_name
            and self._component_class == other._component_class
            and self._config == other._config
            and self._fn_name == other._fn_name
        )

    def __repr__(self) -> Text:
        return f"{self._component_class}.{self._fn_name}"


def minimal_dask_graph(
    dask_graph: Dict[Text, Tuple[RasaComponent, Text]], targets: List[Text]
) -> Dict[Text, Tuple[RasaComponent, Text]]:
    dependencies = _all_dependencies(dask_graph, targets)

    return {
        step_name: step
        for step_name, step in dask_graph.items()
        if step_name in dependencies
    }


def _all_dependencies(
    dask_graph: Dict[Text, Tuple[RasaComponent, Text]], targets: List[Text]
) -> List[Text]:
    required = []
    for target in targets:
        required.append(target)
        target_dependencies = dask_graph[target][1:]
        for dependency in target_dependencies:
            required += _all_dependencies(dask_graph, [dependency])

    return required


def _minimal_graph_schema(
    graph_schema: Dict[Text, Any], targets: List[Text]
) -> Dict[Text, Tuple[RasaComponent, Text]]:
    dependencies = _all_dependencies_schema(graph_schema, targets)

    return {
        step_name: step
        for step_name, step in graph_schema.items()
        if step_name in dependencies
    }


def _all_dependencies_schema(
    graph_schema: Dict[Text, Any], targets: List[Text]
) -> List[Text]:
    required = []
    for target in targets:
        required.append(target)
        target_dependencies = graph_schema[target]["needs"].values()
        for dependency in target_dependencies:
            required += _all_dependencies_schema(graph_schema, [dependency])

    return required


def convert_to_dask_graph(
    graph_schema: Dict[Text, Any],
    cache: Optional[TrainingCache] = None,
    model_persistor: Optional[AbstractModelPersistor] = None,
) -> Dict[Text, Tuple[RasaComponent, Text]]:
    model_persistor = model_persistor or AbstractModelPersistor.default()

    dsk = {}
    for step_name, step_config in graph_schema.items():
        dsk[step_name] = graph_component_for_config(
            step_name, step_config, cache=cache, model_persistor=model_persistor
        )
    return dsk


def graph_component_for_config(
    step_name: Text,
    step_config: Dict[Text, Any],
    config_overrides: Dict[Text, Any] = None,
    cache: Optional[TrainingCache] = None,
    model_persistor: AbstractModelPersistor = None,
) -> Tuple[RasaComponent, Text]:
    component_config = step_config["config"].copy()
    if config_overrides:
        component_config.update(config_overrides)

    persistor = None
    if step_config.get("persist", True):
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
    graph_schema: Dict[Text, Any],
    target_names: Union[Text, List[Text]],
    cache: Optional[TrainingCache] = None,
    model_persistor: Optional[AbstractModelPersistor] = None,
) -> Dict[Text, Any]:
    dask_graph = convert_to_dask_graph(
        graph_schema, cache=cache, model_persistor=model_persistor
    )
    return run_dask_graph(dask_graph, target_names)


def run_dask_graph(
    dask_graph: Dict[Text, Tuple[Union[RasaComponent, "FingerprintComponent"], Text]],
    target_names: Union[Text, List[Text]],
) -> Dict[Text, Any]:
    return dict(ChainMap(*dask.get(dask_graph, target_names)))


def visualise_as_dask_graph(graph_schema: Dict[Text, Any], filename: Text) -> None:
    dask_graph = convert_to_dask_graph(graph_schema)
    dask.visualize(dask_graph, filename=filename)
