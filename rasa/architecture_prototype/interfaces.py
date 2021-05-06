from __future__ import annotations
import abc
from abc import ABC
from typing import (
    Any,
    Optional,
    Text,
    Dict,
    List,
    Tuple,
)


from rasa.core.channels import UserMessage
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


class TrainingCacheInterface(ABC):
    """Stores the fingerprints and output values for a graph run."""

    @abc.abstractmethod
    def store_fingerprint(self, fingerprint_key: Text, output: Any) -> None:
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def calculate_fingerprint_key(
        self, node_name: Text, config: Dict, inputs: List[Any]
    ) -> "Text":
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def get_fingerprint(self, current_fingerprint_key: Text) -> Optional[Text]:
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def get_output(self, fingerprint_key: Text) -> Any:
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError("Please implement this.")

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: bytes) -> TrainingCacheInterface:
        raise NotImplementedError("Please implement this.")


class GraphNodeComponent(ABC):
    """Node in a dask graph."""

    def __init__(
        self,
        config: Dict[Text, Any],
        node_name: Text,
        inputs: Dict[Text, Text],
        cache: TrainingCacheInterface,
    ) -> None:
        self.inputs = inputs
        self.config = config
        self.node_name = node_name
        self.cache = cache

    @abc.abstractmethod
    def __call__(self, *args: Any) -> Dict[Text, Any]:
        raise NotImplementedError("Please implement this.")


GraphSchema = Dict[Text, Any]
DaskGraphNode = Tuple[GraphNodeComponent, Text]
DaskGraph = Dict[Text, DaskGraphNode]


class ModelInterface(ABC):
    """A model uses a dask graph to predict actions based on user messages."""

    @abc.abstractmethod
    def handle_message(
        self, tracker: DialogueStateTracker, message: Optional[UserMessage]
    ) -> Tuple["PolicyPrediction", Optional[UserUttered]]:
        raise NotImplementedError("Please implement this.")

    def predict_next_action(
        self, tracker: DialogueStateTracker,
    ) -> Tuple["PolicyPrediction", UserUttered]:
        raise NotImplementedError("Please implement this.")

    def get_domain(self) -> Domain:
        raise NotImplementedError("Please implement this.")

    def persist(self, target: Text) -> None:
        raise NotImplementedError("Please implement this.")

    @classmethod
    def load(cls, target: Text, persistor: ModelPersistorInterface) -> ModelInterface:
        raise NotImplementedError("Please implement this.")

    def train(self, project: Text) -> ModelInterface:
        raise NotImplementedError("Please implement this.")


class ModelPersistorInterface(ABC):
    """Persists and loads a Model."""

    @abc.abstractmethod
    def create_component_persistor(
        self, node_name: Text
    ) -> ComponentPersistorInterface:
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def create_model_package(self, target: Text, model: ModelInterface) -> None:
        raise NotImplementedError("Please implement this.")

    @abc.abstractmethod
    def load_model_package(self, persisted_model: Text) -> ModelInterface:
        raise NotImplementedError("Please implement this.")


class ComponentPersistorInterface(ABC):
    """Provides functionality for a RasaComponent to save and load artifacts."""

    @abc.abstractmethod
    def file_for(self, filename: Text) -> Text:
        raise NotImplementedError("Please implement this.")

    def directory_for(self, dir_name: Text) -> Text:
        raise NotImplementedError("Please implement this.")

    def get_resource(self, resource_name, filename) -> Text:
        raise NotImplementedError("Please implement this.")

    def resource_name(self) -> Text:
        raise NotImplementedError("Please implement this.")
