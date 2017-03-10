from rasa_nlu.components import Component
from rasa_nlu.featurizers import Featurizer


class MitieFeaturizer(Featurizer, Component):
    name = "intent_featurizer_mitie"

    context_provides = ["intent_features"]

    def ndim(self, feature_extractor):
        return feature_extractor.num_dimensions

    def train(self, training_data, mitie_feature_extractor):
        sentences = [e["text"] for e in training_data.intent_examples]
        features = self.features_for_sentences(sentences, mitie_feature_extractor)
        return {
            "intent_features": features
        }

    def process(self, tokens, mitie_feature_extractor):
        features = self.features_for_tokens(tokens, mitie_feature_extractor)
        return {
            "intent_features": features
        }

    def features_for_tokens(self, tokens, feature_extractor):
        import numpy as np

        vec = np.zeros(self.ndim(feature_extractor))
        for token in tokens:
            vec += feature_extractor.get_feature_vector(token)
        if tokens:
            return vec / len(tokens)
        else:
            return vec

    def features_for_sentences(self, sentences, feature_extractor):
        import mitie
        import numpy as np

        X = np.zeros((len(sentences), self.ndim(feature_extractor)))

        for idx, sentence in enumerate(sentences):
            tokens = mitie.tokenize(sentence)
            X[idx, :] = self.features_for_tokens(tokens, feature_extractor)
        return X
