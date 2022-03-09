from __future__ import annotations
from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrackerWithCachedStates, TrainingDataGenerator
from rasa.shared.core.training_data.structures import StoryGraph


class TrainingTrackerProvider(GraphComponent):
    """Provides training trackers to policies based on training stories."""

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Creates provider from config."""
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
    ) -> TrainingTrackerProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(config)

    def provide(
        self, story_graph: StoryGraph, domain: Domain
    ) -> List[TrackerWithCachedStates]:
        """Generates the training trackers from the training data.

        Args:
            story_graph: The story graph containing the test stories and rules.
            domain: The domain of the model.

        Returns:
            The trackers which can be used to train dialogue policies.
        """
        generator = TrainingDataGenerator(story_graph, domain, **self._config)
        return generator.generate()
