from __future__ import unicode_literals, print_function
from __future__ import division
from __future__ import absolute_import

from builtins import range, str
import os
import random
import io

import pathlib
import warnings

from typing import Optional

from rasa_nlu.components import Component
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import TrainingData


class SpacyEntityExtractor(Component, EntityExtractor):
    name = "ner_spacy"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    def __init__(self, ner=None, fine_tune_spacy_ner=False):
        self.ner = ner
        self.fine_tune_spacy_ner = fine_tune_spacy_ner

    def train(self, spacy_nlp, training_data, fine_tune_spacy_ner):
        # type: (Language, TrainingData, Optional[bool]) -> None
        from spacy.language import Language
        self.fine_tune_spacy_ner = fine_tune_spacy_ner
        if training_data.num_entity_examples > 0:
            train_data = self._convert_examples(training_data.entity_examples)
            ent_types = [[ent["entity"] for ent in ex["entities"]] for ex in training_data.entity_examples]
            entity_types = list(set(sum(ent_types, [])))

            self.ner = self._train_from_scratch(spacy_nlp, entity_types, train_data)

    def process(self, spacy_doc, spacy_nlp):
        # type: (Doc) -> dict
        from spacy.tokens import Doc

        return {
            "entities": self.extract_entities(spacy_doc, spacy_nlp)
        }

    def extract_entities(self, doc, nlp):
        # type: (Doc) -> [dict]
        from spacy.tokens import Doc

        if self.ner is not None:
            self.ner(doc)

            # REMOVE THIS, as soon as we are able again to fine tune spacy models instead of combining them
            if nlp.entity is not None and self.fine_tune_spacy_ner:
                sp_doc = nlp(doc.text)
                spacy_ents = sp_doc.ents
                for spacy_ent in spacy_ents:
                    for e in doc.ents:
                        if e.start_char <= spacy_ent.start_char < e.end_char or \
                              e.start_char <= spacy_ent.end_char < e.end_char:
                            break
                    else:
                        doc.ents += (spacy_ent,)

            entities = [
                {
                    "entity": ent.label_,
                    "value": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents]
            return entities
        else:
            return []

    @classmethod
    def load(cls, model_dir, entity_extractor, fine_tune_spacy_ner, spacy_nlp):
        # type: (str, str, Language) -> SpacyEntityExtractor
        from spacy.language import Language
        from spacy.pipeline import EntityRecognizer

        if model_dir and entity_extractor:
            ner_dir = os.path.join(model_dir, entity_extractor)
            ner = EntityRecognizer.load(pathlib.Path(ner_dir), spacy_nlp.vocab)
            return SpacyEntityExtractor(ner, fine_tune_spacy_ner)
        else:
            return SpacyEntityExtractor()

    def persist(self, model_dir):
        # type: (str) -> dict
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        import json

        if self.ner:
            ner_dir = os.path.join(model_dir, 'ner')
            if not os.path.exists(ner_dir):
                os.mkdir(ner_dir)

            entity_extractor_config_file = os.path.join(ner_dir, "config.json")
            entity_extractor_file = os.path.join(ner_dir, "model")

            with io.open(entity_extractor_config_file, 'w') as f:
                f.write(str(json.dumps(self.ner.cfg)))
            self.ner.model.dump(entity_extractor_file)
            return {
                "entity_extractor": "ner",
                "fine_tune_spacy_ner": self.fine_tune_spacy_ner,
            }
        else:
            return {"entity_extractor": None}

    def _convert_examples(self, entity_examples):
        def convert_entity(ent):
            return ent["start"], ent["end"], ent["entity"]

        def convert_example(ex):
            return ex["text"], [convert_entity(ent) for ent in ex["entities"]]

        return [convert_example(ex) for ex in entity_examples]

    def _train_from_scratch(self, nlp, entity_types, train_data):
        from spacy.pipeline import EntityRecognizer

        ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
        self._update_ner_model(ner, nlp, train_data)
        return ner

    # TODO: this is not used at the moment as there is an issue in the latest spacy version. Currently there is no way
    # TODO: to directly fine tune the entity model as that will result in a malloc error during parse time
    def _fine_tune(self, nlp, entity_types, train_data):
        if nlp.entity:
            ner = nlp.entity
            for entity_type in entity_types:
                if entity_type not in ner.cfg['actions']['1']:
                    ner.add_label(entity_type)
            self._update_ner_model(ner, nlp, train_data)
            return ner
        else:
            warnings.warn("Failed to fine tune model. There was no model to fine tune. ")
            return None

    def _update_ner_model(self, ner, nlp, train_data):
        from spacy.gold import GoldParse

        for itn in range(5):
            random.shuffle(train_data)
            for raw_text, entity_offsets in train_data:
                doc = nlp.make_doc(raw_text)
                nlp.tagger(doc)
                gold = GoldParse(doc, entities=entity_offsets)
                ner.update(doc, gold)
