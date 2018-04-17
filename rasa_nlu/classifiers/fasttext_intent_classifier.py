from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.utils import PY3

import logging
import cloudpickle
from builtins import zip
import os
import io
from typing import Any, Optional
from typing import List
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)

# How many intents are at max put into the output intent ranking, everything else will be cut off
INTENT_RANKING_LENGTH = 10

class FastTextIntentClassifier(Component):
    """Intent classifier using the sklearn framework
    Here are the references for later use and specs adds
    https://github.com/facebookresearch/fastText/blob/master/python/fastText/FastText.py"""


    name = "intent_classifier_fasttext"

    provides = ["intent", "intent_ranking"]

    def __init__(self, clf=None, le=None):
        # type: (fastText) -> None
        """Construct a new intent classifier using the fasttext framework."""

        self.clf = clf

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "fastText"]

    def transform_labels(self, labels):
        # type: (List[Text]) -> np.ndarray
        """Transforms a list of strings into numeric label representation.
        :param labels: List of labels to convert to numeric representation"""

        return [x.replace('__label__', '') for x in labels]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train the intent classifier on a data set.
        :param num_threads: number of threads used during training time
        We keep it here for later use."""


    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_classifier_fasttext": "intent_classifier.pkl"
        }


    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SklearnIntentClassifier
        from fastText import load_model

        if model_dir and model_metadata.get("intent_classifier_fasttext"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_classifier_fasttext"))

            with io.open(classifier_file, 'rb') as f:  # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    # Load the clf outside pickle to avoid segmentation fault error (compiler side)
                    component = cloudpickle.load(f)
                    component.clf = load_model(model_metadata.get("model_fasttext"))
                    component.language = model_metadata.get("language")
                    return component
        else:
            return cls()

    def preprocess(self, raw_text, language=None):
        """
        Preprocess a message before feeding to classification model.
        :param raw_text
        :return: processed message
        """
        import re
        import nltk
        from nltk.corpus import stopwords

        raw_text = re.sub(r'br / ', '', raw_text)

        if language=="en_EN":
            word_list = nltk.word_tokenize(raw_text.decode('utf-8'), 'english')
        elif language=="fr_FR":
            word_list = nltk.word_tokenize(raw_text.decode('utf-8'), 'french')
        else:
            raise ValueError('nltk.word_tokenize: language in model_metadata not covered.')

        if language=="en_EN":
            stopword_set = set(stopwords.words("english"))
        elif language=="fr_FR":
            stopword_set = set(stopwords.words("french"))
        else:
            raise ValueError('stopwords: language in model_metadata not covered.')

        meaningful_words = [w.lower() for w in word_list if w not in stopword_set]
        cleaned_word_list = " ".join(meaningful_words)
        return cleaned_word_list

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        if not self.clf:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            nb_labels = len(self.clf.get_labels())
            preprocessed_message = self.preprocess(raw_text=message.text, language=self.language)
            intent_ids, probabilities = self.clf.predict(preprocessed_message, k=nb_labels)
            intents = self.transform_labels(intent_ids)
            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten

            if len(intents) > 0 and len(probabilities) > 0:
                ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
                intent = {"name": intents[0], "confidence": probabilities[0]}
                intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
