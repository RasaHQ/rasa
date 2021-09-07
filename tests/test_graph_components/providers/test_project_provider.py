from typing import Dict, Text, Any

import pytest
from rasa.graph_components.providers.project_provider import ProjectProvider
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.mark.parametrize(
    "config",
    [
        {
            "config_path": "examples/moodbot/config.yml",
            "domain_path": "examples/moodbot/domain.yml",
            "training_data_paths": ["examples/moodbot/data/"],
        }
    ],
)
def test_provide_importer(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config: Dict[Text, Any],
) -> None:
    project_provider = ProjectProvider.create(
        config, default_model_storage, Resource("xy"), default_execution_context
    )

    importer = project_provider.provide()

    assert isinstance(importer.get_config(), Dict)
    assert isinstance(importer.get_domain(), Domain)
    assert isinstance(importer.get_nlu_data(), TrainingData)
    assert len(importer.get_nlu_data().training_examples) > 0
    assert isinstance(importer.get_stories(), StoryGraph)


def test_provide_multiproject_importer(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    config_path = "data/test_multiproject/config.yml"
    importer_config = {"importers": [{"name": "MultiProjectImporter"}]}
    config = {
        "config_path": config_path,
        "config": importer_config,
        "domain_path": None,
        "training_data_paths": None,
    }
    project_provider = ProjectProvider.create(
        config, default_model_storage, Resource("xy"), default_execution_context
    )
    importer = project_provider.provide()

    training_data = importer.get_nlu_data()
    assert len(training_data.intents) == 4

    domain = importer.get_domain()
    assert len(domain.responses) == 4

    project_config = importer.get_config()
    assert len(project_config["policies"]) == 3
