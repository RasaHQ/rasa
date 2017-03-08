from mitie import *

from extractors.mitie_entity_extractor import MITIEEntityExtractor
from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer

from rasa_nlu import Interpreter
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


class MITIEInterpreter(Interpreter):
    @staticmethod
    def load(meta, featurizer=None):
        """
        :type meta: rasa_nlu.model.Metadata
        :rtype: MITIEInterpreter
        """
        extractor = MITIEEntityExtractor.load(meta.entity_extractor_path)

        classifier = text_categorizer(meta.intent_classifier_path)

        if featurizer is None:
            featurizer = MITIEFeaturizer.load(meta.feature_extractor_path)

        entity_synonyms = Interpreter.load_synonyms(meta.entity_synonyms_path)

        return MITIEInterpreter(
            classifier,
            extractor,
            featurizer,
            entity_synonyms)

    def __init__(self,
                 intent_classifier=None,
                 entity_extractor=None,
                 featurizer=None,
                 entity_synonyms=None):
        self.extractor = entity_extractor
        self.featurizer = featurizer
        self.classifier = intent_classifier
        self.ent_synonyms = entity_synonyms
        self.tokenizer = MITIETokenizer()

    def get_intent(self, tokens):
        if self.classifier:
            label, score = self.classifier(tokens, self.featurizer.feature_extractor)
        else:
            label, score = "None", 0.0
        return label, score

    def parse(self, text):
        tokens = self.tokenizer.tokenize(text)
        intent, score = self.get_intent(tokens)
        entities = self.extractor.get_entities(text, tokens, self.featurizer)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': score}
