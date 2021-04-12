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
        self, component: Any, config: Dict[Text, Any], fn_name: Text, node_name: Text
    ) -> None:
        self._component = component(**config)
        self._config = config
        self._run = getattr(component, fn_name)
        self._fn_name = fn_name
        self._node_name = node_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(self._component, *args, **kwargs)

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
                component=step_config["uses"],
                config=step_config["config"],
                fn_name=step_config["fn"],
                node_name=step_name,
            ),
            *step_config["needs"],
        )
    return dsk
