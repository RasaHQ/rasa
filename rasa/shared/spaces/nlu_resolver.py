from typing import Text, Optional

from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.constants import INTENT, ENTITIES, ENTITY_ATTRIBUTE_TYPE, \
    INTENT_RESPONSE_KEY
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.spaces.domain_resolver import DomainInfo


class NLUResolver:
    # TODO: synonyms (needs adjustment for entity synonym mapper)
    @classmethod
    def prefix_intents(cls, prefix: Text, training_data: TrainingData,
                       domain_info: DomainInfo) -> None:
        """Prefix intents in the training data."""
        for msg in training_data.training_examples:
            intent = msg.get(INTENT)
            if intent and intent in domain_info.intents:
                msg.set(INTENT, f"{prefix}!{intent}")
            intent_response_key = msg.get(INTENT_RESPONSE_KEY)
            if intent_response_key:
                intent_part = intent_response_key.split("/")[0]
                if intent_part in domain_info.intents:
                    msg.set(INTENT_RESPONSE_KEY, f"{prefix}!{intent_response_key}")

    @classmethod
    def prefix_entities(cls, prefix: Text, training_data: TrainingData,
                        domain_info: DomainInfo):
        """Prefix entities in the training data."""
        for msg in training_data.training_examples:
            for entity in msg.get(ENTITIES, []):
                if entity[ENTITY_ATTRIBUTE_TYPE] in domain_info.entities:
                    entity[ENTITY_ATTRIBUTE_TYPE] = \
                        f"{prefix}!{entity[ENTITY_ATTRIBUTE_TYPE]}"

    @classmethod
    def prefix_regexes(cls, prefix, training_data: TrainingData) -> None:
        """Prefix name of regex features in the training data."""
        for regex_feature in training_data.regex_features:
            regex_feature["name"] = f"{prefix}!{regex_feature['name']}"

    @classmethod
    def prefix_lookups(cls, prefix, training_data: TrainingData) -> None:
        for lookup_table in training_data.lookup_tables:
            lookup_table["name"] = f"{prefix}!{lookup_table['name']}"

    @classmethod
    def prefix_responses(cls, prefix, training_data: TrainingData,
                         domain_info: DomainInfo) -> None:
        prefixed_responses = {}
        utter_prefix = "utter_"
        for response_name in training_data.responses:
            intent_part = response_name.split("/")[0][len(utter_prefix):]
            if intent_part in domain_info.intents:
                prefixed_response_name = f"{utter_prefix}{prefix}!" \
                                         f"{response_name[len(utter_prefix):]}"
                prefixed_responses[prefixed_response_name] = \
                    training_data.responses[response_name]
            else:
                prefixed_responses[response_name] = \
                    training_data.responses[response_name]
        training_data.responses = prefixed_responses

    @classmethod
    def load_nlu(cls, nlu_path: Text) -> TrainingData:
        """Load the nlu data from disc."""
        importer = RasaFileImporter(training_data_paths=nlu_path)
        return importer.get_nlu_data()

    @classmethod
    def load_and_resolve(cls, nlu_path: Optional[Text],
                         prefix: Text, domain_info: DomainInfo) -> TrainingData:
        """Load nlu data from disc and prefix accordingly"""
        training_data = cls.load_nlu(nlu_path)
        cls.prefix_intents(prefix, training_data, domain_info)
        cls.prefix_entities(prefix, training_data, domain_info)
        cls.prefix_lookups(prefix, training_data)
        cls.prefix_regexes(prefix, training_data)
        cls.prefix_responses(prefix, training_data, domain_info)
        return training_data

