import abc
import copy
import json
import os
import tarfile
from abc import ABC
from pathlib import Path
from typing import Text, Dict, Any, Union


class ComponentPersistor:
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


class AbstractModelPersistor(ABC):
    @abc.abstractmethod
    def create_component_persistor(self, node_name: Text) -> "ComponentPersistor":
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def create_model_package(
        self, target: Text, predict_graph_schema: Dict[Text, Any],
    ) -> None:
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def load_model_package(self, persisted_model: Text) -> Dict[Text, Any]:
        raise NotImplementedError("Please implement this.")

    @classmethod
    def default(cls) -> "AbstractModelPersistor":
        return LocalModelPersistor(Path("model"))


class LocalModelPersistor(AbstractModelPersistor):
    def __init__(self, local_path: Path = Path("model")) -> None:
        self._dir = local_path

    def create_component_persistor(self, node_name: Text) -> "ComponentPersistor":
        return ComponentPersistor(node_name, self._dir)

    def create_model_package(
        self, target: Text, predict_graph_schema: Dict[Text, Any],
    ) -> None:
        graph = serialize_graph_schema(predict_graph_schema)
        (self._dir / "predict_graph.json").write_text(graph)

        with tarfile.open(target, "w:gz") as tar:
            for elem in os.scandir(self._dir):
                tar.add(elem.path, arcname=elem.name)

    def load_model_package(self, persisted_model: Text) -> "Dict[Text, Any]":
        if Path(persisted_model).is_file():
            with tarfile.open(persisted_model, mode="r:gz") as tar:
                tar.extractall(self._dir)
        elif Path(persisted_model).is_dir():
            self._dir = Path(persisted_model)
            # shutil.copytree(persisted_model, self._dir, dirs_exist_ok=True, symlinks=True)

        return deserialize_graph_schema((self._dir / "predict_graph.json").read_text())


def serialize_graph_schema(graph_schema: Dict[Text, Any]) -> Text:
    to_serialize = copy.deepcopy(graph_schema)
    for step_name, step_config in to_serialize.items():
        component_class = step_config["uses"]
        step_config["uses"] = component_class.name
    return json.dumps(to_serialize)


def deserialize_graph_schema(
    serialized_graph_schema: Union[Text, Path]
) -> Dict[Text, Any]:
    from rasa.architecture_prototype.graph_components import load_graph_component
    import rasa.nlu.registry
    import rasa.core.registry

    schema = json.loads(serialized_graph_schema)
    for step_name, step_config in schema.items():
        component_class_name = step_config["uses"]
        try:
            step_config["uses"] = rasa.nlu.registry.get_component_class(
                component_class_name
            )
        except Exception:
            try:
                step_config["uses"] = load_graph_component(component_class_name)
            except Exception:
                try:
                    step_config["uses"] = rasa.core.registry.policy_from_module_path(
                        component_class_name
                    )
                except:
                    raise ValueError(f"Unknown component: {component_class_name}")

    return schema
