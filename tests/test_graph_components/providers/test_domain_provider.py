from typing import Dict, Text, Any

import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.domain_provider import DomainProvider
from rasa.shared.core.domain import Domain


@pytest.mark.parametrize(
    "config",
    [
        ({}),
        ({"augmentation_factor": 0}),
        ({"use_story_concatenation": False}),
        (
            {
                "remove_duplicates": True,
                "unique_last_num_states": None,
                "augmentation_factor": 50,
                "tracker_limit": None,
                "use_story_concatenation": True,
                "debug_plots": False,
            }
        ),
    ],
)
def test_domain_provider_generates_domain(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config: Dict[Text, Any],
    config_path: Text,
    domain_path: Text,
):
    component = DomainProvider.create(
        {**DomainProvider.get_default_config(), **config},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )
    assert isinstance(component, DomainProvider)

    domain = component.generate_domain(config_path, domain_path)
    assert isinstance(domain, Domain)
