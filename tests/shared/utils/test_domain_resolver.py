from typing import Text

import pytest
import yaml
from pydot import frozendict

from rasa.shared.core.domain import KEY_INTENTS, KEY_ENTITIES, KEY_SLOTS, KEY_FORMS, \
    KEY_RESPONSES, KEY_ACTIONS
from rasa.shared.utils.domain_resolver import DomainResolver


def freeze_yaml_dict(yaml_text: Text) -> frozendict:
    return frozendict(yaml.load(yaml_text, None))


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
    prefix = "games"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    assert domain_info.entities == {"monopoly", "ball", "chess", "pandemic", "cluedo"}
    assert set(domain_yaml[KEY_ENTITIES]) == {"games!monopoly", "games!ball",
                                              "games!chess", "games!pandemic",
                                              "games!cluedo"}


def test_collect_and_prefix_entities_with_entity_attributes():
    domain_path = "data/test_domains/travel_form.yml"
    prefix = "travel"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    expected_yaml_text = f"""
        entities:
          - {prefix}!GPE:
              roles:
                - destination
                - origin
          - {prefix}!name
    """
    assert freeze_yaml_dict(expected_yaml_text) == \
           frozendict({KEY_ENTITIES: domain_yaml[KEY_ENTITIES]})

def test_collect_and_prefix_intents():
    domain_path = "data/test_domains/test_domain_from_directory_tree"
    prefix = "games"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    expected_intents = {"utter_subskill", "utter_subroot",
                       "utter_skill_2", "utter_skill_1",
                       "utter_root", "utter_root2"}
    assert domain_info.intents == expected_intents
    expected_prefixed_intents = {f"{prefix}!{intent}" for intent in expected_intents}
    assert set(domain_yaml[KEY_INTENTS]) == expected_prefixed_intents


def test_collect_and_prefix_intents_with_intent_attributes():
    domain_path = "data/test_domains/travel_form.yml"
    prefix = "travel"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    assert domain_info.intents == {"inform", "greet"}
    expected_yaml_text = f"""
            intents:
             - {prefix}!inform:
                 use_entities:
                   - {prefix}!GPE
             - {prefix}!greet:
                 ignore_entities:
                    - {prefix}!GPE
        """

    assert freeze_yaml_dict(expected_yaml_text) == \
           frozendict({KEY_INTENTS: domain_yaml[KEY_INTENTS]})


def test_collect_and_prefix_intents_with_intent_attributes_referencing_global_entity():
    domain_path = "data/test_spaces_domain_resolving/" \
                  "travel_form_reference_global_entity.yml"
    prefix = "travel"
    domain_yaml, _ = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    expected_yaml_text = f"""
        intents:
         - {prefix}!inform:
             use_entities:
               - {prefix}!GPE
               - location
         - {prefix}!greet:
             ignore_entities:
                - {prefix}!GPE
    """

    assert freeze_yaml_dict(expected_yaml_text) == \
           frozendict({KEY_INTENTS: domain_yaml[KEY_INTENTS]})


def test_collect_and_prefix_slots():
    domain_path = "data/test_domains/travel_form.yml"
    prefix = "travel"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    assert domain_info.slots == {"GPE_origin", "GPE_destination", "requested_slot"}

    expected_yaml_text = f"""
      {prefix}!GPE_origin:
        type: text
        mappings:
        - type: from_entity
          entity: {prefix}!GPE
          role: origin
      {prefix}!GPE_destination:
        type: text
        mappings:
        - type: from_entity
          entity: {prefix}!GPE
          role: destination
      {prefix}!requested_slot:
        type: text
        influence_conversation: false
        mappings:
        - type: custom
    """
    assert freeze_yaml_dict(expected_yaml_text) == frozendict(domain_yaml[KEY_SLOTS])


def test_collect_and_prefix_slots_with_mapping_referencing_global_intent():
    domain_path = "data/test_spaces_domain_resolving/money_reference_global_intent.yml"
    prefix = "money"
    domain_yaml, _ = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    # don't prefix the reference to greet
    expected_yaml_text = f"""
      {prefix}!amount:
        type: text
        initial_value: "0"
        mappings:
          - type: from_entity
            entity: {prefix}!amount
      {prefix}!has_said_hi:
        type: bool
        initial_value: false
        mappings:
          - type: from_intent
            intent: greet
    """
    assert freeze_yaml_dict(expected_yaml_text) == frozendict(domain_yaml[KEY_SLOTS])


def test_collect_and_prefix_forms():
    domain_path = "data/test_domains/restaurant_form.yml"
    prefix = "travel"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)
    assert domain_info.forms == {"restaurant_form"}
    expected_yaml_text = f"""
      {prefix}!restaurant_form:
       required_slots:
        - {prefix}!cuisine
        - {prefix}!people
    """

    assert freeze_yaml_dict(expected_yaml_text) == frozendict(domain_yaml[KEY_FORMS])


def test_collect_and_prefix_responses():
    domain_path = "data/test_spaces_domain_resolving" \
                  "/travel_form_reference_global_entity.yml"
    prefix = "travel"
    domain_yaml, _ = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    expected_yaml_text = f"""
        responses:
          utter_{prefix}!ask_GPE_origin:
            - text: "where are you leaving from?"
          utter_{prefix}!ask_GPE_destination:
            - text: "where are you going to?"
          utter_{prefix}!travel_plan:
            - text: "Alright, so you'll be flying from {{{prefix}!GPE_origin}} to {{{prefix}!GPE_destination}}."
    """
    prefixed_responses = {KEY_RESPONSES: domain_yaml[KEY_RESPONSES]}
    assert freeze_yaml_dict(expected_yaml_text) == \
           frozendict(prefixed_responses)


def test_collect_and_prefix_actions():
    domain_path = "data/test_spaces_domain_resolving" \
                  "/travel_form_reference_global_entity.yml"
    prefix = "travel"
    domain_yaml, domain_info = \
        DomainResolver.load_and_resolve(domain_path, prefix)

    expected_actions = {"action_search_travel"}
    assert expected_actions == domain_info.actions

    expected_yaml_text = f"""      
        actions:
          - {prefix}!action_search_travel
    """
    prefixed_actions = {KEY_ACTIONS: domain_yaml[KEY_ACTIONS]}
    assert freeze_yaml_dict(expected_yaml_text) == \
           frozendict(prefixed_actions)

