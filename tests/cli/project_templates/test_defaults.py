from pathlib import Path

import pytest

from rasa.cli.project_templates.defaults import (
    RasaDefaults,
    _get_domain_from_importer,
    get_pattern_defaults,
    get_rasa_defaults,
)
from rasa.cli.scaffold import ProjectTemplateName, scaffold_path
from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.patterns.domain_for_patterns import (
    generate_domain_for_default_patterns,
)
from rasa.shared.constants import CONFIG_ADDITIONAL_LANGUAGES_KEY, CONFIG_LANGUAGE_KEY
from rasa.shared.core.constants import LANGUAGE_SLOT
from rasa.shared.utils.llm import (
    SystemPrompts,
    get_system_default_prompts,
)
from rasa.shared.utils.yaml import read_yaml


def test_get_rasa_defaults_contents_are_consistent():
    Configuration.initialise_empty()
    calm_dir = Path(scaffold_path(ProjectTemplateName.DEFAULT))
    config_yaml = (calm_dir / "config.yml").read_text(encoding="utf-8")
    endpoints_yaml = (calm_dir / "endpoints.yml").read_text(encoding="utf-8")

    result = get_rasa_defaults(
        config_yaml,
        endpoints_yaml,
    )
    assert isinstance(result, RasaDefaults)

    config = read_yaml(config_yaml)
    endpoints = read_yaml(endpoints_yaml)
    expected_prompts = get_system_default_prompts(config, endpoints)
    assert isinstance(result.prompts, SystemPrompts)
    assert result.prompts == expected_prompts

    pattern_domain = generate_domain_for_default_patterns()
    assert result.actions and result.actions == pattern_domain.actions
    assert result.intents and result.intents == pattern_domain.intents
    assert result.contexts and result.contexts == pattern_domain.contexts

    pattern_defaults = get_pattern_defaults(config)
    assert result.responses and result.responses == pattern_defaults.responses
    assert result.slots and result.slots == pattern_defaults.slots
    assert result.flows and result.flows == pattern_defaults.flows


def test_rasa_defaults_rejects_unknown_fields():
    with pytest.raises(Exception):
        RasaDefaults(
            prompts=get_system_default_prompts({}, {}),
            actions=["action_restart"],
            intents=["greet"],
            contexts={},
            responses={},
            slots={},
            flows={},
            unexpected="unexpected_value",
        )


def test_get_domain_from_importer_contains_builtin_slots():
    domain = _get_domain_from_importer({})
    assert any(slot for slot in domain.slots if slot.is_builtin)


def test_get_domain_from_importer_language_slot_config():
    config = {CONFIG_LANGUAGE_KEY: "en", CONFIG_ADDITIONAL_LANGUAGES_KEY: ["de", "it"]}
    domain = _get_domain_from_importer(config)
    language_slot = next(
        (slot for slot in domain.slots if slot.name == LANGUAGE_SLOT),
        None,
    )
    assert language_slot is not None
    assert language_slot.initial_value == "en"
    assert language_slot.values == ["de", "it", "en"]
