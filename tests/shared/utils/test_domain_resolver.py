import pytest
from pydot import frozendict

from rasa.shared.core.domain import KEY_INTENTS, KEY_ENTITIES, USE_ENTITIES_KEY, \
    IGNORE_ENTITIES_KEY
from rasa.shared.utils.domain_resolver import DomainResolver


def test_load_domain_yaml():
    domain_path = "data/test_domains/travel_form.yml"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    assert len(domain_yaml[KEY_INTENTS]) == 2
    assert all([isinstance(k, dict) for k in domain_yaml[KEY_INTENTS]])


def test_load_domain_yaml_from_multiple_files():
    domain_path = "data/test_domains/test_domain_from_directory_tree"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    assert set(domain_yaml[KEY_INTENTS]) == {"utter_subskill", "utter_subroot",
                                             "utter_skill_2", "utter_skill_1",
                                             "utter_root", "utter_root2"}


def test_collect_and_prefix_entities():
    domain_path = "data/test_domains/test_domain_from_directory_tree"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    prefixed_domain_yaml, entities = \
        DomainResolver.collect_and_prefix_entities("games", domain_yaml)

    assert entities == {"monopoly", "ball", "chess", "pandemic", "cluedo"}
    assert set(prefixed_domain_yaml[KEY_ENTITIES]) == {"games!monopoly", "games!ball",
                                                       "games!chess", "games!pandemic",
                                                       "games!cluedo"}


def test_collect_and_prefix_entities_with_entity_attributes():
    domain_path = "data/test_domains/travel_form.yml"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    prefixed_domain_yaml, entities = \
        DomainResolver.collect_and_prefix_entities("travel", domain_yaml)

    assert entities == {"GPE", "name"}
    assert len(prefixed_domain_yaml[KEY_ENTITIES]) == 2
    prefixed_name_index = prefixed_domain_yaml[KEY_ENTITIES].index("travel!name")
    assert prefixed_name_index != -1
    prefixed_gpe_entity = prefixed_domain_yaml[KEY_ENTITIES][1-prefixed_name_index]
    assert len(prefixed_gpe_entity.keys()) == 1
    assert list(prefixed_gpe_entity.keys())[0] == "travel!GPE"


def test_collect_and_prefix_intents():
    domain_path = "data/test_domains/test_domain_from_directory_tree"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    prefix = "games"
    prefixed_domain_yaml, entities = \
        DomainResolver.collect_and_prefix_entities(prefix, domain_yaml)
    prefixed_domain_yaml, intents = \
        DomainResolver.collect_and_prefix_intents(prefix, prefixed_domain_yaml,
                                                  entities)
    expected_intents = {"utter_subskill", "utter_subroot",
                       "utter_skill_2", "utter_skill_1",
                       "utter_root", "utter_root2"}
    assert intents == expected_intents
    expected_prefixed_intents = {f"{prefix}!{intent}" for intent in expected_intents}
    assert set(prefixed_domain_yaml[KEY_INTENTS]) == expected_prefixed_intents


def test_collect_and_prefix_intents_with_intent_attributes():
    domain_path = "data/test_domains/travel_form.yml"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    prefix = "travel"
    prefixed_domain_yaml, entities = \
        DomainResolver.collect_and_prefix_entities(prefix, domain_yaml)
    prefixed_domain_yaml, intents = \
        DomainResolver.collect_and_prefix_intents(prefix, prefixed_domain_yaml,
                                                  entities)

    assert intents == {"inform", "greet"}
    assert len(prefixed_domain_yaml[KEY_INTENTS]) == 2
    assert all([isinstance(i, dict) for i in prefixed_domain_yaml[KEY_INTENTS]])

    frozen_intent_dicts = {frozendict(i) for i in prefixed_domain_yaml[KEY_INTENTS]}
    expected_prefixed_intent_dicts = {
        frozendict({f"{prefix}!inform": {USE_ENTITIES_KEY: [f"{prefix}!GPE"]}}),
        frozendict({f"{prefix}!greet": {IGNORE_ENTITIES_KEY: [f"{prefix}!GPE"]}})
    }
    assert frozen_intent_dicts == expected_prefixed_intent_dicts


def test_collect_and_prefix_intents_with_intent_attributes_referencing_global_entity():
    domain_path = "data/test_spaces_domain_resolving/" \
                  "travel_form_reference_global_entity.yml"
    domain_yaml = DomainResolver.load_domain_yaml(domain_path)
    prefix = "travel"
    prefixed_domain_yaml, entities = \
        DomainResolver.collect_and_prefix_entities(prefix, domain_yaml)
    prefixed_domain_yaml, intents = \
        DomainResolver.collect_and_prefix_intents(prefix, prefixed_domain_yaml,
                                                  entities)

    frozen_intent_dicts = {frozendict(i) for i in prefixed_domain_yaml[KEY_INTENTS]}
    expected_prefixed_intent_dicts = {
        frozendict({f"{prefix}!inform": {USE_ENTITIES_KEY: [f"{prefix}!GPE",
                                                            "location"]}}),
        frozendict({f"{prefix}!greet": {IGNORE_ENTITIES_KEY: [f"{prefix}!GPE"]}})
    }
    assert frozen_intent_dicts == expected_prefixed_intent_dicts
