from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from builtins import range
import os
import re

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.training_data import TrainingData


if typing.TYPE_CHECKING:
    import mitie


class MitieEntityExtractor(EntityExtractor):
    name = "ner_mitie"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    def __init__(self, ner=None):
        self.ner = ner

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    def extract_entities(self, text, tokens, feature_extractor):
        ents = []
        offset = 0
        if self.ner:
            entities = self.ner.extract_entities(tokens, feature_extractor)
            for e in entities:
                _range = e[0]
                _regex = "\s*".join(re.escape(tokens[i]) for i in _range)
                expr = re.compile(_regex)
                m = expr.search(text[offset:])
                start, end = m.start() + offset, m.end() + offset
                entity_value = text[start:end]
                offset += m.end()
                ents.append({
                    "entity": e[1],
                    "value": entity_value,
                    "start": start,
                    "end": end
                })

        return ents

    @staticmethod
    def find_entity(ent, text):
        import mitie

        tk = MitieTokenizer()
        tokens, offsets = tk.tokenize_with_offsets(text)
        if ent["start"] not in offsets:
            message = "Invalid entity {} in example '{}': entities must span whole tokens".format(ent, text)
            raise ValueError(message)
        start = offsets.index(ent["start"])
        _slice = text[ent["start"]:ent["end"]]
        val_tokens = mitie.tokenize(_slice)
        end = start + len(val_tokens)
        return start, end

    def train(self, training_data, mitie_file, num_threads):
        # type: (TrainingData, Text, Optional[int]) -> None
        import mitie

        trainer = mitie.ner_trainer(mitie_file)
        trainer.num_threads = num_threads
        found_one_entity = False
        for example in training_data.entity_examples:
            text = example["text"]
            tokens = mitie.tokenize(text)
            sample = mitie.ner_training_instance(tokens)
            for ent in example["entities"]:
                start, end = MitieEntityExtractor.find_entity(ent, text)
                sample.add_entity(list(range(start, end)), ent["entity"])
                found_one_entity = True

            trainer.add(sample)
        # Mitie will fail to train if there is not a single entity tagged
        if found_one_entity:
            self.ner = trainer.train()

    def process(self, text, tokens, mitie_feature_extractor, entities):
        # type: (Text, List[Text], mitie.total_word_feature_extractor, List[Dict[Text, Any]]) -> Dict[Text, Any]

        extracted = self.add_extractor_name(self.extract_entities(text, tokens, mitie_feature_extractor))
        entities.extend(extracted)
        return {
            "entities": entities
        }

    @classmethod
    def load(cls, model_dir, entity_extractor_mitie):
        # type: (Text, Text) -> MitieEntityExtractor
        import mitie

        if model_dir and entity_extractor_mitie:
            entity_extractor_file = os.path.join(model_dir, entity_extractor_mitie)
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
