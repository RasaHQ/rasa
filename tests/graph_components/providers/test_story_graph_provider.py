from typing import Dict, Text, Any

import pytest
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.story_graph_provider import StoryGraphProvider
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter


@pytest.mark.parametrize(
    "config", [{}, {"exclusion_percentage": None}, {"exclusion_percentage": 25}]
)
def test_story_graph_provider_provide(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config: Dict[Text, Any],
    config_path: Text,
    domain_path: Text,
    stories_path: Text,
):
    component = StoryGraphProvider.create(
        {**StoryGraphProvider.get_default_config(), **config},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )
    importer = TrainingDataImporter.load_from_config(
        config_path, domain_path, [stories_path]
    )

    story_graph_from_component = component.provide(importer)
    assert isinstance(story_graph_from_component, StoryGraph)

    story_graph = importer.get_stories(**config)

    assert story_graph.fingerprint() == story_graph_from_component.fingerprint()
