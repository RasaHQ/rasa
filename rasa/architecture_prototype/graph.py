from collections import ChainMap
import inspect
from pathlib import Path
from typing import Any, Text, Dict, List, Union

import dask

from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training


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
        kwargs = {}
        for input, input_node in self._inputs.items():
            kwargs[input] = received_inputs[input_node]

        if not self._eager:
            const_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self.create_component(**const_kwargs)

        run_kwargs = kwargs

        if "kwargs" not in rasa.shared.utils.common.arguments_of(self._run_fn):
            run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._run_fn)

        return {self._node_name: self._run_fn(self._component, **run_kwargs)}

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


def convert_to_dask_graph(rasa_graph: Dict[Text, Any]):
    dsk = {}
    for step_name, step_config in rasa_graph.items():
        dsk[step_name] = (
            RasaComponent(
                node_name=step_name,
                component_class=step_config["uses"],
                config=step_config["config"],
                fn_name=step_config["fn"],
                inputs=step_config["needs"],
                constructor_name=step_config.get("constructor_name"),
                eager=step_config.get("eager", True),
                persist=step_config.get("persist", True),
            ),
            *step_config["needs"].values(),
        )
    return dsk


def fill_defaults(graph: Dict[Text, Any]):
    for step_name, step_config in graph.items():
        component_class = step_config["uses"]

        if hasattr(component_class, "defaults"):
            defaults = component_class.defaults
            step_config["config"].update(defaults)


def run_as_dask_graph(
    rasa_graph: Dict[Text, Any], target_names: Union[Text, List[Text]]
) -> Dict[Text, Any]:
    dask_graph = convert_to_dask_graph(rasa_graph)
    return dict(ChainMap(*dask.get(dask_graph, target_names)))


def visualise_as_dask_graph(rasa_graph: Dict[Text, Any], filename: Text) -> None:
    dask_graph = convert_to_dask_graph(rasa_graph)
    dask.visualize(dask_graph, filename=filename)
