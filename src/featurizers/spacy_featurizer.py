from rasa_nlu.featurizers import Featurizer
from rasa_nlu.components import Component


class SpacyFeaturizer(Featurizer, Component):
    name = "intent_featurizer_spacy"

    context_provides = ["intent_features"]

    def ndim(self, nlp):
        return nlp.vocab.vectors_length

    def train(self, spacy_nlp, training_data):
        sentences = [e["text"] for e in training_data.intent_examples]
        features = self.features_for_sentences(sentences, spacy_nlp)
        return {
            "intent_features": features
        }

    def process(self, spacy_doc, spacy_nlp):
        features = self.features_for_doc(spacy_doc, spacy_nlp)
        return {
            "intent_features": features
        }

    def features_for_doc(self, doc, nlp):
        import numpy as np

        vec = []
        for token in doc:
            if token.has_vector:
                vec.append(token.vector)
        if vec:
            return np.sum(vec, axis=0) / len(vec)
        else:
            return np.zeros(self.ndim(nlp))

    def features_for_sentences(self, sentences, nlp):
        import numpy as np

        X = np.zeros((len(sentences), self.ndim(nlp)))
        for idx, sentence in enumerate(sentences):
            doc = nlp(sentence)
            X[idx, :] = self.features_for_doc(doc, nlp)
        return X
