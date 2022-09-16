from typing import Dict, Text, Tuple, List, Set

from rasa.shared.constants import DOMAIN_SCHEMA_FILE
from rasa.shared.core.domain import KEY_ENTITIES, Domain, KEY_INTENTS, \
    IGNORE_ENTITIES_KEY, USE_ENTITIES_KEY
import rasa.shared.data
import rasa.shared.utils.io
import rasa.shared.utils.validation
from rasa.shared.exceptions import YamlException


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
    def collect_and_prefix_entities(cls, prefix,
                                    domain_dict: Dict) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix the entities in the domain."""
        entities = set()
        prefixed_entity_section = []
        for entity in domain_dict[KEY_ENTITIES]:
            if isinstance(entity, str):
                entities.add(entity)
                prefixed_entity_section.append(f"{prefix}!{entity}")
            if isinstance(entity, dict):
                entity_name = str(list(entity.keys())[0])
                entities.add(entity_name)
                prefixed_entity_name = f"{prefix}!{entity_name}"
                prefixed_entity_section.append({prefixed_entity_name:
                                                    entity[entity_name]})
        domain_dict[KEY_ENTITIES] = prefixed_entity_section
        return domain_dict, entities

    @classmethod
    def collect_and_prefix_intents(cls, prefix,
                                   domain_dict: Dict,
                                   space_entities: Set[Text]) -> Tuple[Dict, Set[Text]]:
        """Collect and prefix the intents in the domain."""
        intents = set()
        prefixed_intent_section = []
        for intent in domain_dict[KEY_INTENTS]:
            if isinstance(intent, str):
                intents.add(intent)
                prefixed_intent_section.append(f"{prefix}!{intent}")
            if isinstance(intent, dict):
                intent_name = str(list(intent.keys())[0])
                intent_attributes = intent[intent_name]
                prefixed_intent_attributes = {}
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
        domain_dict[KEY_INTENTS] = prefixed_intent_section
        return domain_dict, intents
