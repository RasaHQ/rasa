from typing import Text, Union, Dict

import pytest

from rasa.graph_components.providers.domain_for_core_training_provider import (
    DomainForCoreTrainingProvider,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import (
    KEY_RESPONSES,
    Domain,
    SESSION_CONFIG_KEY,
    KEY_SLOTS,
    KEY_FORMS,
    SESSION_EXPIRATION_TIME_KEY,
    CARRY_OVER_SLOTS_KEY,
)
from rasa.shared.core.constants import SLOT_MAPPINGS
from rasa.shared.constants import REQUIRED_SLOTS_KEY


@pytest.mark.parametrize(
    "input_domain",
    [
        "data/test_domains/conditional_response_variations.yml",  # responses
        "data/test_domains/default_with_slots.yml",  # slots
        "data/test_domains/default_with_mapping.yml",  # slot mappings
        {KEY_FORMS: {"form1": {REQUIRED_SLOTS_KEY: ["slot1"]}}},  # form
        {
            SESSION_CONFIG_KEY: {
                SESSION_EXPIRATION_TIME_KEY: (
                    2 * Domain.empty().session_config.session_expiration_time
                ),
                CARRY_OVER_SLOTS_KEY: (
                    not Domain.empty().session_config.carry_over_slots
                ),
            }
        },
    ],
)
def test_provide(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    input_domain: Union[Text, Dict],
):
    # prepare input
    if isinstance(input_domain, str):
        original_domain = Domain.from_file(path=input_domain)
    else:
        original_domain = Domain.from_dict(input_domain)

    # pass through component
    component = DomainForCoreTrainingProvider.create(
        {"arbitrary-unused": 234},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )
    modified_domain = component.provide(domain=original_domain)

    # convert to dict for comparison
    modified_dict = modified_domain.as_dict()
    original_dict = original_domain.as_dict()
    default_dict = Domain.empty().as_dict()

    assert sorted(original_dict.keys()) == sorted(modified_dict.keys())
    for key in original_dict.keys():

        # replaced with default values
        if key in ["config", SESSION_CONFIG_KEY]:
            assert modified_dict[key] == default_dict[key]

        # for slots, we drop the specification of how they are filled
        elif key == KEY_SLOTS:
            for slot_key in original_dict[key]:
                assert modified_dict[key][slot_key][SLOT_MAPPINGS] == []
                original_dict[key][slot_key][SLOT_MAPPINGS] = []  # drop for comparison
                assert original_dict[key][slot_key] == modified_dict[key][slot_key]

        # for responses, we only keep the keys
        elif key == KEY_RESPONSES:
            assert set(modified_dict[key].keys()) == set(original_dict[key].keys())
            for sub_key in original_dict[key]:
                assert modified_dict[key][sub_key] == []

        # for forms, we only keep the keys (and the Domain will add a default key)
        elif key == KEY_FORMS:
            assert set(modified_dict[key].keys()) == set(original_dict[key].keys())
            for sub_key in original_dict[key]:
                assert set(modified_dict[key][sub_key].keys()) == {REQUIRED_SLOTS_KEY}
                assert modified_dict[key][sub_key][REQUIRED_SLOTS_KEY] == {}

        # everything else remains unchanged
        else:
            assert original_dict[key] == modified_dict[key]
