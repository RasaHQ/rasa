from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import os
from typing import Any, Optional
from typing import List
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)
logger.info("lskjgfqsegpzqdgn")

# How many intents are at max put into the output intent ranking, everything else will be cut off
INTENT_RANKING_LENGTH = 10

if typing.TYPE_CHECKING:
    import fasttext
    import numpy as np


class FastTextIntentClassifier(Component):
    """Intent classifier using the sklearn framework"""

    name = "intent_classifier_fasttext"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "fasttext"]

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SklearnIntentClassifier

        if model_dir and model_metadata.get("intent_classifier_fasttext"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_classifier_fasttext"), "model.ftz")
            model = fasttext.load_model(classifier_file)
            return model
        else:
            return cls()

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        X = message.get("text_features").reshape(1, -1)
        aa = self.predict(X)
        logger.info(aa)
        intents = self.transform_labels_num2str(intent_ids)
        # `predict` returns a matrix as it is supposed
        # to work for multiple examples as well, hence we need to flatten
        intents, probabilities = intents.flatten(), probabilities.flatten()

        if intents.size > 0 and probabilities.size > 0:
            ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
            intent = {"name": intents[0], "confidence": probabilities[0]}
            intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
        else:
            intent = {"name": None, "confidence": 0.0}
            intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
