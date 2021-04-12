from pathlib import Path
from typing import Any, Text, Dict, List, Union

import dask

import rasa.core.training
from rasa.shared.constants import DEFAULT_DOMAIN_PATH
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.utils.common


class ProjectReader:
    def __init__(self, project: Text) -> None:
        self._project = project

    def load_importer(self) -> TrainingDataImporter:
        return TrainingDataImporter.load_from_dict(
            domain_path=Path(self._project, DEFAULT_DOMAIN_PATH),
            training_data_paths=[self._project],
        )

    def read(self) -> Any:
        raise NotImplementedError()


class TrainingDataReader(ProjectReader):
    def read(self) -> TrainingData:
        importer = self.load_importer()
        return rasa.utils.common.run_in_loop(importer.get_nlu_data())


class DomainReader(ProjectReader):
    def read(self) -> Domain:
        importer = self.load_importer()
        return rasa.utils.common.run_in_loop(importer.get_domain())


class StoryReader(ProjectReader):
    def read(self) -> List[TrackerWithCachedStates]:
        importer = self.load_importer()
        domain = rasa.utils.common.run_in_loop(importer.get_domain())

        generated_coroutine = rasa.core.training.load_data(importer, domain,)
        return rasa.utils.common.run_in_loop(generated_coroutine)


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
