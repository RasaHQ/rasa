import copy
from collections import ChainMap
import inspect
from pathlib import Path
from typing import Any, Text, Dict, List, Union, TYPE_CHECKING, Tuple, Optional

import dask

from rasa.architecture_prototype.graph_fingerprinting import TrainingCache
from rasa.core.channels import UserMessage
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
import rasa.shared.utils.io

if TYPE_CHECKING:
    from rasa.core.policies.policy import PolicyPrediction


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
        persist: bool = True,
        cache: Optional["TrainingCache"] = None,
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
        self._persist = persist
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
        if self._persist:
            const_kwargs["persistor"] = Persistor(
                self._node_name, parent_dir=Path("model")
            )
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


class Persistor:
    def __init__(self, node_name: Text, parent_dir: Path) -> None:
        self._node_name = node_name
        self._parent_dir = parent_dir
        self._dir_for_node = Path(parent_dir / node_name)

    def file_for(self, filename: Text) -> Text:
        self._dir_for_node.mkdir(exist_ok=True)
        return str(self._dir_for_node / filename,)

    def directory_for(self, dir_name: Text) -> Text:
        self._dir_for_node.mkdir(exist_ok=True)
        directory = self._dir_for_node / dir_name
        directory.mkdir()
        return str(directory)

    def get_resource(self, resource_name, filename) -> Text:
        return str(Path(self._parent_dir, resource_name, filename))

    def resource_name(self) -> Text:
        return self._node_name


class Model:
    def __init__(self, rasa_graph: Dict[Text, Any]) -> None:
        self._rasa_graph = rasa_graph
        self._predict_graph = convert_to_dask_graph(self._rasa_graph)

    def handle_message(
        self, tracker: DialogueStateTracker, message: Optional[UserMessage]
    ) -> "PolicyPrediction":
        graph = self._predict_graph.copy()

        # Insert user message into graph
        graph["load_user_message"] = _graph_component_for_config(
            "load_user_message",
            self._rasa_graph["load_user_message"],
            {"message": message},
        )

        # Insert dialogue history into graph
        graph["load_history"] = _graph_component_for_config(
            "load_history", self._rasa_graph["load_history"], {"tracker": tracker}
        )

        result = dask.get(graph, "select_prediction")

        return result["select_prediction"]

    def predict_next_action(self, tracker: DialogueStateTracker,) -> "PolicyPrediction":
        return self.handle_message(tracker, message=None)

    def get_domain(self) -> Domain:
        domain_graph = _minimal_graph(self._predict_graph, targets=["load_domain"])
        return dask.get(domain_graph, "load_domain")["load_domain"]


def _minimal_graph(
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
    graph_schema: Dict[Text, Any], cache: Optional[TrainingCache] = None,
) -> Dict[Text, Tuple[RasaComponent, Text]]:
    dsk = {}
    for step_name, step_config in graph_schema.items():
        dsk[step_name] = _graph_component_for_config(
            step_name, step_config, cache=cache
        )
    return dsk


def _graph_component_for_config(
    step_name: Text,
    step_config: Dict[Text, Any],
    config_overrides: Dict[Text, Any] = None,
    cache: Optional[TrainingCache] = None,
) -> Tuple[RasaComponent, Text]:
    component_config = step_config["config"].copy()
    if config_overrides:
        component_config.update(config_overrides)

    return (
        RasaComponent(
            node_name=step_name,
            component_class=step_config["uses"],
            config=component_config,
            fn_name=step_config["fn"],
            inputs=step_config["needs"],
            constructor_name=step_config.get("constructor_name"),
            eager=step_config.get("eager", True),
            persist=step_config.get("persist", True),
            cache=cache,
        ),
        *step_config["needs"].values(),
    )


def fill_defaults(graph_schema: Dict[Text, Any]):
    for step_name, step_config in graph_schema.items():
        component_class = step_config["uses"]

        if hasattr(component_class, "defaults"):
            step_config["config"] = {
                **component_class.defaults,
                **step_config["config"],
            }


def run_as_dask_graph(
    graph_schema: Dict[Text, Any],
    target_names: Union[Text, List[Text]],
    cache: Optional[TrainingCache] = None,
) -> Dict[Text, Any]:
    dask_graph = convert_to_dask_graph(graph_schema, cache=cache)
    return run_dask_graph(dask_graph, target_names)


def run_dask_graph(
    dask_graph: Dict[Text, Tuple[Union[RasaComponent, "FingerprintComponent"], Text]],
    target_names: Union[Text, List[Text]],
) -> Dict[Text, Any]:
    return dict(ChainMap(*dask.get(dask_graph, target_names)))


def visualise_as_dask_graph(graph_schema: Dict[Text, Any], filename: Text) -> None:
    dask_graph = convert_to_dask_graph(graph_schema)
    dask.visualize(dask_graph, filename=filename)
