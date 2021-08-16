from typing import Dict, Text, Any

import pytest
from rasa.components.providers.training_tracker_provider import TrainingTrackerProvider
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.structures import StoryGraph


@pytest.mark.parametrize(
    "config, expected_trackers",
    [
        ({}, 507),
        ({"augmentation_factor": 0}, 7),
        ({"use_story_concatenation": False}, 7),
        (
            {
                "remove_duplicates": True,
                "unique_last_num_states": None,
                "augmentation_factor": 50,
                "tracker_limit": None,
                "use_story_concatenation": True,
                "debug_plots": False,
            },
            507,
        ),
    ],
)
def test_generating_trackers(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config: Dict[Text, Any],
    expected_trackers: int,
):
    reader = YAMLStoryReader()
    steps = reader.read_from_file("data/test_yaml_stories/stories.yml")
    component = TrainingTrackerProvider.create(
        {**TrainingTrackerProvider.get_default_config(), **config},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )

    trackers = component.generate_trackers(
        story_graph=StoryGraph(steps), domain=Domain.empty()
    )

    assert len(trackers) == expected_trackers
    assert all(isinstance(t, TrackerWithCachedStates) for t in trackers)
