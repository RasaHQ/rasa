from __future__ import annotations
from typing import Dict, Text, Any

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
)
from rasa.shared.importers.importer import TrainingDataImporter


class ProjectProvider(GraphComponent):
    """Provides domain and training data during training and inference time."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Default config for ProjectProvider."""
        return {
            "config": {},
            "config_path": DEFAULT_CONFIG_PATH,
            "domain_path": DEFAULT_DOMAIN_PATH,
            "training_data_paths": [DEFAULT_DATA_PATH],
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initializes the ProjectProvider."""
        self._config = config

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> ProjectProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(config)

    def provide(self) -> TrainingDataImporter:
        """Provides the TrainingDataImporter."""
        return TrainingDataImporter.load_from_dict(**self._config)
