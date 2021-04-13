import inspect
from typing import Any, Text, Dict, List, Union

import dask

from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.shared.nlu.training_data.training_data import TrainingData


class TrainingDataReader:
    def __init__(self, filename: Text) -> None:
        self._filename = filename

    def read(self) -> TrainingData:
        return RasaYAMLReader().read(self._filename)


class RasaComponent:
    def __init__(
        self,
        component_class: Any,
        config: Dict[Text, Any],
        fn_name: Text,
        node_name: Text,
        input_names: List[Text],
        constructor_name: Text = None,
        eager=True,
    ) -> None:
        self._constructor_name = constructor_name
        self._component_class = component_class
        self._config = config
        self._fn_name = fn_name
        self._run_fn = getattr(self._component_class, fn_name)
        if self._constructor_name:
            self._constructor_fn = getattr(
                self._component_class, self._constructor_name
            )

        sig = inspect.signature(self._run_fn)
        for param_name, param in sig.parameters.items():
            if param.default is not inspect._empty:
                if param_name not in input_names:
                    raise ValueError(f"{param_name} not in {input_names}")

        if eager:
            self.load_component()
        else:
            self._component = None

        self._node_name = node_name

    def load_component(self):
        if self._constructor_name:
            self._component = self._constructor_fn(**self._config)
        else:
            self._component = self._component_class(**self._config)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._component:
            self.load_component()
        return self._run_fn(self._component, *args, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RasaComponent):
            return NotImplemented

        return (
            self._node_name == other._node_name
            and type(self._component) == type(other._component)
            and self._config == other._config
            and self._fn_name == other._fn_name
        )

    def __repr__(self) -> Text:
        return f"{type(self._component)}.{self._fn_name}"


def run_as_dask_graph(
    rasa_graph: Dict[Text, Any], target_names: Union[Text, List[Text]]
):
    dask_graph = convert_to_dask_graph(rasa_graph)
    return dask.get(dask_graph, target_names)


def convert_to_dask_graph(rasa_graph: Dict[Text, Any]):
    dsk = {}
    for step_name, step_config in rasa_graph.items():
        dsk[step_name] = (
            RasaComponent(
                node_name=step_name,
                component_class=step_config["uses"],
                config=step_config["config"],
                fn_name=step_config["fn"],
                input_names=list(step_config["needs"].keys()),
                constructor_name=step_config.get("constructor_name"),
                eager=step_config.get("eager", True),
            ),
            *step_config["needs"].values(),
        )
    return dsk
