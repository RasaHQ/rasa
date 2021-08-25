from __future__ import annotations
from typing import Dict, Text, Any, Optional

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.loading import load_data

class NLUTrainingDataProvider(GraphComponent):
    """Provides NLU training data during training and inference time."""

    def __init__(
        self,
        model_storage: ModelStorage,
        resource: Resource,
        training_data: Optional[TrainingData] = None,
    ) -> None:
        """Creates NLU training data provider."""
        self._model_storage = model_storage
        self._resource = resource
        self._training_data = training_data

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUTrainingDataProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> NLUTrainingDataProvider:
        """Creates provider using a persisted version of itself."""
        with model_storage.read_from(resource) as resource_directory:
            training_data = load_data(resource_name=str(resource_directory), language=config['language'])
        return cls(model_storage, resource, training_data)

    def _persist(self, training_data: TrainingData) -> None:
        """Persists NLU training data to model storage."""
        with self._model_storage.write_to(self._resource) as resource_directory:
            training_data.persist(str(resource_directory), "nlu.yml")

    def provide_at_training(
        self,
        importer: TrainingDataImporter,
        language: Optional[Text],
        persist: Optional[bool],
    ) -> TrainingData:
        """Provides nlu training data during training."""
        training_data = importer.get_nlu_data(language=language)
        if persist:
            self._persist(training_data)
        return training_data

    def provide_at_inference(self) -> Optional[TrainingData]:
        """Provides the nlu training data during inference."""
        return self._training_data
