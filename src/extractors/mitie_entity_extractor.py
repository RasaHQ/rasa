import os
import re

from rasa_nlu.components import Component
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer


class MitieEntityExtractor(Component):
    name = "ner_mitie"

    def __init__(self, ner=None):
        self.ner = ner

    def get_entities(self, text, tokens, feature_extractor):
        ents = []
        offset = 0
        if self.ner:
            entities = self.ner.extract_entities(tokens, feature_extractor)
            for e in entities:
                _range = e[0]
                _regex = u"\s*".join(tokens[i] for i in _range)
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
        # TODO: inject tokenizer results here!!!!
        tk = MitieTokenizer()
        tokens, offsets = tk.tokenize_with_offsets(text)
        if ent["start"] not in offsets:
            message = u"invalid entity {0} in example '{1}':".format(ent, text) + \
                      u" entities must span whole tokens"
            raise ValueError(message)
        start = offsets.index(ent["start"])
        _slice = text[ent["start"]:ent["end"]]
        val_tokens = mitie.tokenize(_slice)
        end = start + len(val_tokens)
        return start, end

    def train(self, training_data, mitie_file, num_threads):
        import mitie

        trainer = mitie.ner_trainer(mitie_file)
        trainer.num_threads = num_threads
        for example in training_data.entity_examples:
            text = example["text"]
            tokens = mitie.tokenize(text)
            sample = mitie.ner_training_instance(tokens)
            for ent in example["entities"]:
                start, end = MitieEntityExtractor.find_entity(ent, text)
                sample.add_entity(xrange(start, end), ent["entity"])

            trainer.add(sample)
        self.ner = trainer.train()

    def process(self, text, tokens, mitie_feature_extractor):
        return {
            "entities": self.extract_entities(text, tokens, mitie_feature_extractor)
        }

    @classmethod
    def load(cls, model_dir):
        import mitie

        if model_dir:
            entity_extractor_file = os.path.join(model_dir, "entity_extractor.dat")
            extractor = mitie.named_entity_extractor(entity_extractor_file)
            return MitieEntityExtractor(extractor)
        else:
            return None

    def persist(self, model_dir):
        entity_extractor_file = os.path.join(model_dir, "entity_extractor.dat")
        self.ner.save_to_disk(entity_extractor_file, pure_model=True)
        return {
            "entity_extractor": "entity_extractor.dat"
        }
