from __future__ import annotations
from typing import Dict, Text, Any

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter


class DomainProvider(GraphComponent):
    """Provides domain during training and inference time."""

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Creates domain provider from config."""
        self._config = config

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default configuration (see parent class for full docstring)."""
        return {
            "remove_duplicates": True,
            "unique_last_num_states": None,
            "augmentation_factor": 50,
            "tracker_limit": None,
            "use_story_concatenation": True,
            "debug_plots": False,
        }

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DomainProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(config)

    @staticmethod
    def generate_domain(config_path: Text, domain_path: Text) -> Domain:
        """Generates loaded Domain of the bot."""
        importer = TrainingDataImporter.load_from_config(config_path, domain_path)
        return importer.get_domain()
