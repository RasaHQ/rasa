import copy
from typing import Dict, Text, Tuple, List, Set
import re
from rasa.shared.constants import DOMAIN_SCHEMA_FILE, REQUIRED_SLOTS_KEY, \
    IGNORED_INTENTS, RESPONSE_CONDITION
from rasa.shared.core.constants import SLOT_MAPPINGS, MAPPING_TYPE
from rasa.shared.core.domain import KEY_ENTITIES, Domain, KEY_INTENTS, \
    IGNORE_ENTITIES_KEY, USE_ENTITIES_KEY, KEY_SLOTS, KEY_FORMS, KEY_RESPONSES
import rasa.shared.data
import rasa.shared.utils.io
import rasa.shared.utils.validation
from rasa.shared.exceptions import YamlException
from rasa.shared.nlu.constants import INTENT, NOT_INTENT, ENTITY_ATTRIBUTE_TYPE, TEXT


class DomainResolver:

    @classmethod
    def load_domain_yaml(cls, domain_path: Text) -> Dict:
        """Load the domain yaml without doing internal logic consistency checks.

        Skipping the internal consistency checks is important here since a subspace
        might want to reference intents and entities from the main space. Referencing
        objects in a domain that this domain does not define itself usually results
        in an error. Note that the domain schema is still checked."""
        domain_files = rasa.shared.data.get_data_files(domain_path,
                                                       Domain.is_domain_file)
        combined_domain_dict = {}
        for domain_file in domain_files:
            domain_text = rasa.shared.utils.io.read_file(domain_file)
            try:
                rasa.shared.utils.validation.validate_yaml_schema(domain_text,
                                                                  DOMAIN_SCHEMA_FILE)
            except YamlException as e:
                e.filename = domain_file
                raise e
            domain_yaml = rasa.shared.utils.io.read_yaml(domain_text)
            combined_domain_dict = Domain.merge_domain_dicts(domain_yaml,
                                                             combined_domain_dict)
        return combined_domain_dict

    @classmethod
    def collect_and_prefix_entities(cls, prefix: Text,
                                    domain_yaml: Dict) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix the entities in the domain."""
        entities = set()
        prefixed_entity_section = []
        for entity in domain_yaml[KEY_ENTITIES]:
            if isinstance(entity, str):
                entities.add(entity)
                prefixed_entity_section.append(f"{prefix}!{entity}")
            if isinstance(entity, dict):
                entity_name = str(list(entity.keys())[0])
                entities.add(entity_name)
                prefixed_entity_name = f"{prefix}!{entity_name}"
                prefixed_entity_section.append({prefixed_entity_name:
                                                    entity[entity_name]})
        domain_yaml[KEY_ENTITIES] = prefixed_entity_section
        return domain_yaml, entities

    @classmethod
    def collect_and_prefix_intents(cls, prefix: Text,
                                   domain_yaml: Dict,
                                   space_entities: Set[Text]) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix the intents in the domain."""
        intents = set()
        prefixed_intent_section = []
        for intent in domain_yaml[KEY_INTENTS]:
            if isinstance(intent, str):
                intents.add(intent)
                prefixed_intent_section.append(f"{prefix}!{intent}")
            if isinstance(intent, dict):
                intent_name = str(list(intent.keys())[0])
                intent_attributes = intent[intent_name]
                prefixed_intent_attributes = copy.deepcopy(intent_attributes)
                intents.add(intent_name)
                prefixed_intent_name = f"{prefix}!{intent_name}"
                if IGNORE_ENTITIES_KEY in intent_attributes:
                    prefixed_intent_attributes[IGNORE_ENTITIES_KEY] = [
                        f"{prefix}!{entity}" if entity in space_entities else entity
                        for entity in intent_attributes[IGNORE_ENTITIES_KEY]
                    ]
                if USE_ENTITIES_KEY in intent_attributes:
                    prefixed_intent_attributes[USE_ENTITIES_KEY] = [
                        f"{prefix}!{entity}" if entity in space_entities else entity
                        for entity in intent_attributes[USE_ENTITIES_KEY]
                    ]
                prefixed_intent_section.append({prefixed_intent_name:
                                                prefixed_intent_attributes})
        domain_yaml[KEY_INTENTS] = prefixed_intent_section
        return domain_yaml, intents

    @classmethod
    def maybe_prefix_name_or_list(cls, prefix: Text, yaml: Dict, key: Text,
                                  comparing_set: Set[Text]) -> None:
        """Prefix a name or list of names in place if they are in the set."""
        if key in yaml:
            if isinstance(yaml[key], str) \
                    and yaml[key] in comparing_set:
                yaml[key] = f"{prefix}!{yaml[key]}"
            elif isinstance(yaml[key], list):
                yaml[key] = [f"{prefix}!{name}" if name in comparing_set else name
                             for name in yaml[key]]
    @classmethod
    def collect_and_prefix_slots(cls, prefix: Text,
                                 domain_yaml: Dict,
                                 intents: Set[Text],
                                 entities: Set[Text]) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix the slots and their attributes in the domain."""
        # TODO: slot mapping conditions
        # TODO: deal with common slots like `requested_slot`
        slots = set()
        prefixed_slot_section = {}
        for slot_name in domain_yaml[KEY_SLOTS]:
            slots.add(slot_name)
            prefixed_slot_name = f"{prefix}!{slot_name}"

            slot_attributes = domain_yaml[KEY_SLOTS][slot_name]
            prefixed_slot_attributes = copy.deepcopy(slot_attributes)
            if SLOT_MAPPINGS in slot_attributes:
                for mapping in prefixed_slot_attributes[SLOT_MAPPINGS]:
                    if INTENT in mapping:
                        cls.maybe_prefix_name_or_list(prefix, mapping, INTENT, intents)
                    if NOT_INTENT in mapping:
                        cls.maybe_prefix_name_or_list(prefix, mapping,
                                                      NOT_INTENT, intents)
                    if ENTITY_ATTRIBUTE_TYPE in mapping:
                        cls.maybe_prefix_name_or_list(prefix, mapping,
                                                      ENTITY_ATTRIBUTE_TYPE, entities)
            prefixed_slot_section[prefixed_slot_name] = prefixed_slot_attributes
        domain_yaml[KEY_SLOTS] = prefixed_slot_section
        return domain_yaml, slots

    @classmethod
    def collect_and_prefix_forms(cls, prefix: Text,
                                 domain_yaml: Dict,
                                 slots: Set[Text],
                                 intents: Set[Text]) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix the forms and their referenced slots in the domain."""
        forms = set()
        prefixed_form_section = {}
        for form_name in domain_yaml.get(KEY_FORMS, []):
            forms.add(form_name)
            prefixed_form_name = f"{prefix}!{form_name}"
            prefixed_form_attributes = copy.deepcopy(domain_yaml[KEY_FORMS][form_name])
            cls.maybe_prefix_name_or_list(prefix, prefixed_form_attributes,
                                          REQUIRED_SLOTS_KEY, slots)
            cls.maybe_prefix_name_or_list(prefix, prefixed_form_attributes,
                                          IGNORED_INTENTS, intents)
            prefixed_form_section[prefixed_form_name] = prefixed_form_attributes

        domain_yaml[KEY_FORMS] = prefixed_form_section
        return domain_yaml, forms

    @classmethod
    def collect_and_prefix_responses(cls, prefix: Text,
                                     domain_yaml: Dict,
                                     slots: Set[Text],
                                     forms: Set[Text]) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix responses and their referenced slots in the domain."""
        response_slot_regex = re.compile("{(" + "|".join(slots) + ")}")
        form_ask_response_regex = re.compile("utter_ask_(" + "|".join(forms) +
                                             ")_(" + "|".join(slots) + ")")
        responses = set()
        prefixed_response_section = {}
        for response_name in domain_yaml.get(KEY_RESPONSES, []):
            responses.add(response_name)
            if form_ask_response_regex.match(response_name):
                prefixed_response_name = \
                    form_ask_response_regex.sub(f"utter_ask_{prefix}"
                                                r"!\g<1>_"
                                                f"{prefix}!"
                                                r"\g<2>",
                                                response_name)
            else:
                prefixed_response_name = f"utter_{prefix}!{response_name[6:]}"
            prefixed_response_variations = \
                copy.deepcopy(domain_yaml[KEY_RESPONSES][response_name])
            for variation in prefixed_response_variations:
                if TEXT in variation:
                    variation[TEXT] = \
                        response_slot_regex.sub("{" + prefix + r"!\g<1>}",
                                                variation[TEXT])
                if RESPONSE_CONDITION in variation:
                    for condition in variation[RESPONSE_CONDITION]:
                        if condition[MAPPING_TYPE] == "slot":
                            cls.maybe_prefix_name_or_list(prefix, condition, "name",
                                                          slots)
            prefixed_response_section[prefixed_response_name] = \
                prefixed_response_variations
        domain_yaml[KEY_RESPONSES] = prefixed_response_section
        return domain_yaml, responses

    @classmethod
    def resolve(cls, prefix: Text,
                domain_yaml: Dict) -> Tuple[Dict, Set[Text],
                                            Set[Text], Set[Text],
                                            Set[Text], Set[Text]]:
        """Resolve a domain yaml, prefixing space-internal names."""
        domain_yaml = copy.deepcopy(domain_yaml)
        domain_yaml, entities = \
            DomainResolver.collect_and_prefix_entities(prefix, domain_yaml)
        domain_yaml, intents = \
            DomainResolver.collect_and_prefix_intents(prefix, domain_yaml,
                                                      entities)
        domain_yaml, slots = \
            DomainResolver.collect_and_prefix_slots(prefix, domain_yaml,
                                                    intents, entities)
        domain_yaml, forms = \
            DomainResolver.collect_and_prefix_forms(prefix, domain_yaml,
                                                    slots, intents)
        domain_yaml, responses = \
            DomainResolver.collect_and_prefix_responses(prefix, domain_yaml,
                                                        slots, forms)

        return domain_yaml, entities, intents, slots, forms, responses

    @classmethod
    def load_and_resolve(cls, domain_path: Text,
                         prefix: Text) -> Tuple[Dict, Set[Text],
                                                Set[Text], Set[Text],
                                                Set[Text], Set[Text]]:
        """Load domain yaml(s) from disc, and prefix space-internal names."""
        domain_yaml = cls.load_domain_yaml(domain_path)
        return cls.resolve(prefix, domain_yaml)


