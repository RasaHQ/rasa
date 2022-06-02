from __future__ import annotations
from typing import Dict, Text, Any
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import (
    TrainingData,
    DEFAULT_TRAINING_DATA_OUTPUT_PATH,
)


class NLUTrainingDataProvider(GraphComponent):
    """Provides NLU training data during training."""

    def __init__(
        self, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource
    ) -> None:
        """Creates a new NLU training data provider."""
        self._config = config
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def get_default_config(cls) -> Dict[Text, Any]:
        """Returns the default config for NLU training data provider."""
        return {"persist": False, "language": None}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUTrainingDataProvider:
        """Creates a new NLU training data provider."""
        return cls(config, model_storage, resource)

    def _persist(self, training_data: TrainingData) -> None:
        """Persists NLU training data to model storage."""
        with self._model_storage.write_to(self._resource) as resource_directory:
            training_data.persist(
                dir_name=str(resource_directory),
                filename=DEFAULT_TRAINING_DATA_OUTPUT_PATH,
            )

    def provide(self, importer: TrainingDataImporter) -> TrainingData:
        """Provides nlu training data during training."""
        if "language" in self._config:
            training_data = importer.get_nlu_data(language=self._config["language"])
        else:
            training_data = importer.get_nlu_data()
        if self._config["persist"]:
            self._persist(training_data)
        return training_data
