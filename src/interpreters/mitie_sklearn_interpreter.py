import cloudpickle
from mitie import named_entity_extractor

from rasa_nlu import Interpreter
from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
from rasa_nlu.interpreters.mitie_interpreter_utils import get_entities
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


class MITIESklearnInterpreter(Interpreter):
    def __init__(self,
                 intent_classifier=None,
                 entity_extractor=None,
                 feature_extractor=None,
                 entity_synonyms=None, **kwargs):
        self.extractor = None
        self.classifier = None
        if entity_extractor:
            self.extractor = named_entity_extractor(entity_extractor, feature_extractor)
        if intent_classifier:
            with open(intent_classifier, 'rb') as f:
                self.classifier = cloudpickle.load(f)
        self.featurizer = MITIEFeaturizer(feature_extractor)
        self.tokenizer = MITIETokenizer()
        self.ent_synonyms = None
        if entity_synonyms:
            self.ent_synonyms = Interpreter.load_synonyms(entity_synonyms)

    def get_intent(self, sentence_tokens):
        """Returns the most likely intent and its probability for the input text.

        :param sentence_tokens: text to classify
        :return: tuple of most likely intent name and its probability"""
        if self.classifier:
            X = self.featurizer.features_for_tokens(sentence_tokens).reshape(1, -1)
            intent_ids, probabilities = self.classifier.predict(X)
            intents = self.classifier.transform_labels_num2str(intent_ids)
            intent, score = intents[0], probabilities[0]
        else:
            intent, score = "None", 0.0

        return intent, score

    def parse(self, text):
        tokens = self.tokenizer.tokenize(text)
        intent, probability = self.get_intent(tokens)
        entities = get_entities(text, tokens, self.extractor)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
