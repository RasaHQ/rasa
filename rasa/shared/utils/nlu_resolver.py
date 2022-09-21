from typing import Text, Optional

from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.constants import INTENT, ENTITIES, ENTITY_ATTRIBUTE_TYPE
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.domain_resolver import DomainInfo


class NLUResolver:
    # TODO: lookups
    # TODO: regexes
    # TODO: synonyms
    # TODO: responses
    @classmethod
    def prefix_intents(cls, prefix: Text, training_data: TrainingData,
                       domain_info: DomainInfo) -> None:
        """Prefix intents in the training data."""
        for msg in training_data.training_examples:
            intent = msg.get(INTENT)
            if intent and intent in domain_info.intents:
                msg.set(INTENT, f"{prefix}!{intent}")

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
    def prefix_regexes(cls, prefix, training_data: TrainingData,
                       domain_info: DomainInfo):
        """Prefix name of regex features in the training data."""
        pass

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
        return training_data

