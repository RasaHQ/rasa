from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier

from rasa_nlu.extractors.mitie_entity_extractor import MITIEEntityExtractor
from rasa_nlu import Interpreter
from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


class MITIESklearnInterpreter(Interpreter):
    @staticmethod
    def load(meta, featurizer=None):
        """
        :type meta: rasa_nlu.model.Metadata
        :rtype: MITIESklearnInterpreter
        """
        extractor = MITIEEntityExtractor.load(meta.entity_extractor_path)

        if featurizer is None:
            featurizer = MITIEFeaturizer.load(meta.feature_extractor_path)

        classifier = SklearnIntentClassifier.load(meta.intent_classifier_path)

        entity_synonyms = Interpreter.load_synonyms(meta.entity_synonyms_path)

        return MITIESklearnInterpreter(
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
        self.classifier = intent_classifier
        self.featurizer = featurizer
        self.tokenizer = MITIETokenizer()
        self.ent_synonyms = entity_synonyms

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
        entities = self.extractor.get_entities(text, tokens, self.featurizer)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
