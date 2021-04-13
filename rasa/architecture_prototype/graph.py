from collections import ChainMap
import inspect
from typing import Any, Text, Dict, List, Union

import dask

from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.common


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
        inputs: Dict[Text, Text],
        constructor_name: Text = None,
        eager=True,
    ) -> None:
        self._eager = eager
        self._inputs = inputs
        self._constructor_name = constructor_name
        self._component_class = component_class
        self._config = config
        self._fn_name = fn_name
        self._run_fn = getattr(self._component_class, fn_name)
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
            self._component = self._constructor_fn(**self._config)

        self._node_name = node_name

    def validate_params_in_inputs(self, input_names, func):
        params = inspect.signature(func).parameters
        for param_name, param in params.items():
            if param_name in ['self', 'args', 'kwargs']:
                continue
            if param.default is inspect._empty:
                if param_name not in input_names:
                    raise ValueError(f"{param_name} for function {func} is missing from inputs")

    def __call__(self, *args: Any) -> Dict[Text, Any]:
        received_inputs = dict(ChainMap(*args))
        kwargs = {}
        for input, input_node in self._inputs.items():
            kwargs[input] = received_inputs[input_node]

        if not self._eager:
            const_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs,self._constructor_fn)
            self._component = self._constructor_fn(**const_kwargs)

        run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._run_fn)
        return {self._node_name: self._run_fn(self._component, **run_kwargs)}

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


def run_as_dask_graph(
    rasa_graph: Dict[Text, Any], target_names: Union[Text, List[Text]]
) -> Dict[Text, Any]:
    dask_graph = convert_to_dask_graph(rasa_graph)
    return dict(ChainMap(*dask.get(dask_graph, target_names)))


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
            ),
            *step_config["needs"].values(),
        )
    return dsk
