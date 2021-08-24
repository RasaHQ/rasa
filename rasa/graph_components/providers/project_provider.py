from __future__ import annotations
from typing import Dict, Text, Any

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.importers.importer import TrainingDataImporter


class ProjectProvider(GraphComponent):
    """Provides domain during training and inference time."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Default config for ProjectProvider."""
        return {
            "config_path": "config.yml",
            "domain_path": "domain.yml",
            "training_data_paths": ["data/"],
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
        """Provides the domain during inference."""
        return TrainingDataImporter.load_from_config(
            self._config["config_path"],
            self._config["domain_path"],
            self._config["training_data_paths"],
        )
