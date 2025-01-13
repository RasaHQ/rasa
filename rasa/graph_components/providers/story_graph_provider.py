from __future__ import annotations

from typing import Any, Dict, List, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.core.training_data.structures import StoryGraph, StoryStep
from rasa.shared.importers.importer import TrainingDataImporter

STORIES_PERSISTENCE_FILE_NAME = "stories.yml"


class StoryGraphProvider(GraphComponent):
    """Provides the training data from stories."""

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        stories: StoryGraph = None,
    ) -> None:
        """Creates provider from config."""
        self._config = config
        self._model_storage = model_storage
        self._resource = resource
        self._stories = stories

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default configuration (see parent class for full docstring)."""
        return {"exclusion_percentage": None}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> StoryGraphProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> StoryGraphProvider:
        """Creates provider using a persisted version of itself."""
        with model_storage.read_from(resource) as resource_directory:
            reader = YAMLStoryReader()
            story_steps = reader.read_from_file(
                resource_directory / STORIES_PERSISTENCE_FILE_NAME
            )
        stories = StoryGraph(story_steps)
        return cls(config, model_storage, resource, stories)

    def _persist(self, story_steps: List[StoryStep]) -> None:
        """Persists flows to model storage."""
        with self._model_storage.write_to(self._resource) as resource_directory:
            writer = YAMLStoryWriter()
            writer.dump(
                resource_directory / STORIES_PERSISTENCE_FILE_NAME,
                story_steps,
            )

    def provide_train(self, importer: TrainingDataImporter) -> StoryGraph:
        """Provides the story graph from the training data.

        Args:
            importer: instance of TrainingDataImporter.

        Returns:
            The story graph containing stories and rules used for training.
        """
        stories = importer.get_stories(**self._config)
        self._persist(stories.story_steps)
        return stories

    def provide_inference(self) -> StoryGraph:
        """Provides the stories configuration during inference."""
        if self._stories is None:
            self._stories = StoryGraph([])
        return self._stories
