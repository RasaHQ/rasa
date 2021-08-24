from typing import Text

import pytest
from rasa.graph_components.providers.domain_without_response_provider import (
    DomainWithoutResponseProvider,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import KEY_E2E_ACTIONS, KEY_RESPONSES, Domain


@pytest.mark.parametrize(
    "domain_yml",
    [
        ("data/test_domains/travel_form.yml"),
        ("data/test_domains/mixed_retrieval_intents.yml"),
    ],
)
def test_copy_without_responses(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    domain_yml: Text,
):

    component = DomainWithoutResponseProvider.create(
        {"arbitrary-unused": 234},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )

    domain = Domain.from_file(path=domain_yml,)
    domain_copy = component.recreate_without_responses(domain=domain,)

    given_as_dict = domain.as_dict()
    copy_as_dict = domain_copy.as_dict()

    # all configurations not impacted by responses stay intact
    assert sorted(given_as_dict.keys()) == sorted(copy_as_dict.keys())
    for key in given_as_dict.keys():
        if key not in [KEY_RESPONSES, KEY_E2E_ACTIONS]:
            assert given_as_dict[key] == copy_as_dict[key]

    # Note: The given domains do contain responses and some actions for which no
    # responses are defined...
    assert domain.responses, "choose a different test config"
    assert domain.user_actions, "choose a different test config"
    assert set(domain.user_actions).difference(
        domain.responses
    ), "choose a different test config"

    # Assert that the recreated copy does not contain any response information
    assert not domain_copy.responses
    assert all(action in domain._custom_actions for action in domain_copy.user_actions)
    assert set(domain_copy.action_names_or_texts) < set(domain.action_names_or_texts)
    assert all(
        response not in domain_copy.action_names_or_texts
        for response in set(domain.responses).difference(domain._custom_actions)
    )
