from typing import Text

import pytest
from rasa.graph_components.providers.domain_without_response_provider import (
    DomainWithoutResponsesProvider,
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
def test_provide(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    domain_yml: Text,
):

    component = DomainWithoutResponsesProvider.create(
        {"arbitrary-unused": 234},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )

    original_domain = Domain.from_file(path=domain_yml)
    original_dict = original_domain.as_dict()

    modified_domain = component.provide(domain=original_domain)
    modified_dict = modified_domain.as_dict()

    # all configurations not impacted by responses stay intact
    assert sorted(original_dict.keys()) == sorted(modified_dict.keys())
    for key in original_dict.keys():
        if key not in [KEY_RESPONSES, KEY_E2E_ACTIONS]:
            assert original_dict[key] == modified_dict[key]

    # Note: The given domains do contain responses and some actions for which no
    # responses are defined...
    reminder = (
        f"This test needs to be adapted. The domain yamls have been changed so that "
        f"they no longer contain some actions and some responses - which do not "
        f"coincide (i.e. not all responses are listed as actions). Hence "
        f"{domain_yml} is no longer a good test case here. Please remove the yaml from "
        f"the parameterization or replace it with an different yaml."
    )
    assert original_domain.responses, reminder
    assert original_domain.user_actions, reminder
    assert set(original_domain.user_actions).difference(
        original_domain.responses
    ), reminder

    # The responses are empty
    for _, responses_for_action in modified_domain.responses.items():
        assert not responses_for_action

    # We still have all the same labels
    assert (
        original_domain.action_names_or_texts == modified_domain.action_names_or_texts
    )
