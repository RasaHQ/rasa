import re

from mitie import ner_trainer, tokenize, ner_training_instance, named_entity_extractor
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


class MITIEEntityExtractor(object):
    def __init__(self, ner=None):
        self.ner = ner

    def get_entities(self, text, tokens, featurizer):
        ents = []
        if self.ner:
            entities = self.ner.extract_entities(tokens, featurizer.feature_extractor)
            for e in entities:
                _range = e[0]
                _regex = u"\s*".join(re.escape(tokens[i]) for i in _range)
                expr = re.compile(_regex)
                m = expr.search(text)
                start, end = m.start(), m.end()
                entity_value = text[start:end]
                ents.append({
                    "entity": e[1],
                    "value": entity_value,
                    "start": start,
                    "end": end
                })

        return ents

    @staticmethod
    def find_entity(ent, text):
        tk = MITIETokenizer()
        tokens, offsets = tk.tokenize_with_offsets(text)
        if ent["start"] not in offsets:
            message = u"invalid entity {0} in example '{1}':".format(ent, text) + \
                      u" entities must span whole tokens"
            raise ValueError(message)
        start = offsets.index(ent["start"])
        _slice = text[ent["start"]:ent["end"]]
        val_tokens = tokenize(_slice)
        end = start + len(val_tokens)
        return start, end

    @staticmethod
    def train(entity_examples, fe_file, max_num_threads):
        trainer = ner_trainer(fe_file)
        trainer.num_threads = max_num_threads
        for example in entity_examples:
            text = example["text"]
            tokens = tokenize(text)
            sample = ner_training_instance(tokens)
            for ent in example["entities"]:
                start, end = MITIEEntityExtractor.find_entity(ent, text)
                sample.add_entity(xrange(start, end), ent["entity"])

            trainer.add(sample)
        ner = trainer.train()
        return MITIEEntityExtractor(ner)

    @staticmethod
    def load(path):
        if path:
            extractor = named_entity_extractor(path)
            return MITIEEntityExtractor(extractor)
        else:
            return None

    def persist(self, dir_name):
        import os

        entity_extractor_file = os.path.join(dir_name, "entity_extractor.dat")
        self.ner.save_to_disk(entity_extractor_file, pure_model=True)
        return {
            "entity_extractor": "entity_extractor.dat"
        }
