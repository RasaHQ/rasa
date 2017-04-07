from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.components import Component
from rasa_nlu.training_data import TrainingData


class SpacyFeaturizer(Featurizer, Component):
    name = "intent_featurizer_spacy"

    context_provides = {
        "train": ["intent_features"],
        "process": ["intent_features"],
    }

    def ndim(self, nlp):
        # type: (Language) -> int
        from spacy.language import Language

        return nlp.vocab.vectors_length

    def train(self, spacy_nlp, training_data):
        # type: (Language, TrainingData) -> dict
        from spacy.language import Language

        sentences = [e["text"] for e in training_data.intent_examples]
        features = self.features_for_sentences(sentences, spacy_nlp)
        return {
            "intent_features": features
        }

    def process(self, spacy_doc, spacy_nlp):
        # type: (Doc, Language) -> dict
        from spacy.language import Language
        from spacy.tokens import Doc

        features = self.features_for_doc(spacy_doc, spacy_nlp)
        return {
            "intent_features": features
        }

    def features_for_doc(self, doc, nlp):
        # type: (Doc, Language) -> np.ndarray
        import numpy as np
        from spacy.language import Language
        from spacy.tokens import Doc

        vec = []
        for token in doc:
            if token.has_vector:
                vec.append(token.vector)
        if vec:
            return np.sum(vec, axis=0) / len(vec)
        else:
            return np.zeros(self.ndim(nlp))

    def features_for_sentences(self, sentences, nlp):
        # type: ([str], Language) -> np.ndarray
        import numpy as np
        from spacy.language import Language

        X = np.zeros((len(sentences), self.ndim(nlp)))
        for idx, sentence in enumerate(sentences):
            doc = nlp(sentence)
            X[idx, :] = self.features_for_doc(doc, nlp)
        return X
