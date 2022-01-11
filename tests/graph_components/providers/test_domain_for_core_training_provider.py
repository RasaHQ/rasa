from typing import Text, Union, Dict

import pytest
from pathlib import Path

from _pytest.tmpdir import TempPathFactory

from rasa import model_training
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import (
    KEY_RESPONSES,
    Domain,
    SESSION_CONFIG_KEY,
    KEY_FORMS,
    SESSION_EXPIRATION_TIME_KEY,
    CARRY_OVER_SLOTS_KEY,
)
from rasa.graph_components.providers.domain_for_core_training_provider import (
    DomainForCoreTrainingProvider,
)
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
def test_provide_removes_or_replaces_expected_information(
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
                assert modified_dict[key][sub_key][REQUIRED_SLOTS_KEY] == []

        # everything else remains unchanged
        else:
            assert original_dict[key] == modified_dict[key]


def test_train_core_with_original_or_provided_domain_and_compare(
    tmp_path_factory: TempPathFactory,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    # Choose an example where the provider will remove a lot of information:
    example = Path("examples/formbot/")
    training_files = [example / "data" / "rules.yml"]

    # Choose a configuration with a policy
    # Note: This is sufficient to illustrate that the component won't be re-trained
    # when the domain changes. We do *not* test here whether removing keys would/
    # should not have any effect.
    config = """
    recipe: default.v1
    language: en

    policies:
      - name: RulePolicy
    """
    config_dir = tmp_path_factory.mktemp("config dir")
    config_file = config_dir / "config.yml"
    with open(config_file, "w") as f:
        f.write(config)

    # Train with the original domain
    original_domain_file = example / "domain.yml"
    original_output_dir = tmp_path_factory.mktemp("output dir")
    model_training.train(
        domain=original_domain_file,
        config=str(config_file),
        training_files=training_files,
        output=original_output_dir,
    )

    # Let the provider create a modified domain
    original_domain = Domain.from_file(original_domain_file)
    component = DomainForCoreTrainingProvider.create(
        {"arbitrary-unused": 234},
        default_model_storage,
        Resource("xy"),
        default_execution_context,
    )
    modified_domain = component.provide(domain=original_domain)

    # Dry-run training with the modified domain
    modified_domain_dir = tmp_path_factory.mktemp("modified domain dir")
    modified_domain_file = modified_domain_dir / "modified_config.yml"
    modified_domain.persist(modified_domain_file)

    modified_output_dir = tmp_path_factory.mktemp("modified output dir")
    modified_result = model_training.train(
        domain=modified_domain_file,
        config=str(config_file),
        training_files=training_files,
        output=modified_output_dir,
        dry_run=True,
    )

    assert modified_result.dry_run_results["train_RulePolicy0"].is_hit
