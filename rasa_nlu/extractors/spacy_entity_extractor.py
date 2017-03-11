from __future__ import unicode_literals, print_function

import os
import random

import pathlib
import warnings

from rasa_nlu.components import Component


class SpacyEntityExtractor(Component):
    name = "ner_spacy"

    def __init__(self, ner=None):
        self.ner = ner

    def train(self, spacy_nlp, training_data, fine_tune_spacy_ner):
        if training_data.num_entity_examples > 0:
            train_data = self._convert_examples(training_data.entity_examples)
            ent_types = [[ent["entity"] for ent in ex["entities"]] for ex in training_data.entity_examples]
            entity_types = list(set(sum(ent_types, [])))

            if fine_tune_spacy_ner:
                self.ner = self._fine_tune(spacy_nlp, entity_types, train_data)
            else:
                self.ner = self._train_from_scratch(spacy_nlp, entity_types, train_data)

    def process(self, spacy_doc):
        return {
            "entities": self.extract_entities(spacy_doc)
        }

    def extract_entities(self, doc):
        if self.ner is not None:
            self.ner(doc)

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
    def load(cls, model_dir, entity_extractor, spacy_nlp):
        from spacy.pipeline import EntityRecognizer

        if model_dir and entity_extractor:
            ner_dir = os.path.join(model_dir, entity_extractor)
            ner = EntityRecognizer.load(pathlib.Path(ner_dir), spacy_nlp.vocab)
            return SpacyEntityExtractor(ner)
        else:
            return SpacyEntityExtractor()

    def persist(self, model_dir):
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import json
        if self.ner:
            ner_dir = os.path.join(model_dir, 'ner')
            if not os.path.exists(ner_dir):
                os.mkdir(ner_dir)

            entity_extractor_config_file = os.path.join(ner_dir, "config.json")
            entity_extractor_file = os.path.join(ner_dir, "model")

            with open(entity_extractor_config_file, 'w') as f:
                json.dump(self.ner.cfg, f)
            self.ner.model.dump(entity_extractor_file)
            return {"entity_extractor": "ner"}
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
                gold = GoldParse(doc, entities=entity_offsets)
                ner.update(doc, gold)
