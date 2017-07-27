from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from builtins import range
import os

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


if typing.TYPE_CHECKING:
    import mitie


class MitieEntityExtractor(EntityExtractor):
    name = "ner_mitie"

    provides = ["entities"]

    requires = ["tokens"]

    def __init__(self, ner=None):
        self.ner = ner

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    def extract_entities(self, text, tokens, feature_extractor):
        ents = []
        tokens_strs = [token.text for token in tokens]
        if self.ner:
            entities = self.ner.extract_entities(tokens_strs, feature_extractor)
            for e in entities:
                if len(e[0]):
                    start = tokens[e[0][0]].offset
                    end = tokens[e[0][-1]].end

                    ents.append({
                        "entity": e[1],
                        "value": text[start:end],
                        "start": start,
                        "end": end
                    })

        return ents

    @staticmethod
    def find_entity(ent, text, tokens):
        offsets = [token.offset for token in tokens]
        ends = [token.end for token in tokens]
        if ent["start"] not in offsets:
            message = "Invalid entity {} in example '{}': entities must span whole tokens. Wrong entity start.".format(
                    ent, text)
            raise ValueError(message)
        if ent["end"] not in ends:
            message = "Invalid entity {} in example '{}': entities must span whole tokens. Wrong entity end.".format(
                    ent, text)
            raise ValueError(message)
        start = offsets.index(ent["start"])
        end = ends.index(ent["end"]) + 1
        return start, end

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig) -> None
        import mitie

        trainer = mitie.ner_trainer(config["mitie_file"])
        trainer.num_threads = config["num_threads"]
        found_one_entity = False
        for example in training_data.entity_examples:
            text = example.text
            tokens = example.get("tokens")
            sample = mitie.ner_training_instance([t.text for t in tokens])
            for ent in example.get("entities", []):
                start, end = MitieEntityExtractor.find_entity(ent, text, tokens)
                sample.add_entity(list(range(start, end)), ent["entity"])
                found_one_entity = True

            trainer.add(sample)
        # Mitie will fail to train if there is not a single entity tagged
        if found_one_entity:
            self.ner = trainer.train()

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception("Failed to train 'intent_featurizer_mitie'. Missing a proper MITIE feature extractor.")

        ents = self.extract_entities(message.text, message.get("tokens"), mitie_feature_extractor)
        extracted = self.add_extractor_name(ents)
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    @classmethod
    def load(cls, model_dir, model_metadata, cached_component, **kwargs):
        # type: (Text, Metadata, Optional[MitieEntityExtractor], **Any) -> MitieEntityExtractor
        import mitie

        if model_dir and model_metadata.get("entity_extractor_mitie"):
            entity_extractor_file = os.path.join(model_dir, model_metadata.get("entity_extractor_mitie"))
            extractor = mitie.named_entity_extractor(entity_extractor_file)
            return MitieEntityExtractor(extractor)
        else:
            return MitieEntityExtractor()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        if self.ner:
            entity_extractor_file = os.path.join(model_dir, "entity_extractor.dat")
            self.ner.save_to_disk(entity_extractor_file, pure_model=True)
            return {"entity_extractor_mitie": "entity_extractor.dat"}
        else:
            return {"entity_extractor_mitie": None}
