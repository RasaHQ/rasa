# -*-coding: utf-8-*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import collections
import os

try:
    import pickle
except ImportError:
    import cPickle as pickle


from builtins import map

import typing
from typing import Any
from typing import Dict
from typing import Text
from typing import Optional

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message

PERSIST_MODEL_FILE = 'exact_classifier.pkl'

if typing.TYPE_CHECKING:
    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.config import RasaNLUConfig


class ExactIntentClassifier(Component):

    name = "intent_classifier_exact"

    provides = ["intent"]

    def __init__(self):
        self._intent_data = collections.defaultdict(list)

        super(ExactIntentClassifier, self).__init__()

    def set_persist_data(self, intent_data):
        self._intent_data = intent_data

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        msg_text = message.text
        msg_text = msg_text.strip() # with leading and trailing whitespace removed

        if msg_text in self._intent_data:
            predicted_intents = self._intent_data[msg_text]

            # if there two or more predicted intents,
            # take first one as most confident one
            most_likely_intent = predicted_intents[0]

            # replace previous classifier's decision, using this one only
            message.set("intent", most_likely_intent, add_to_output=True)

            # replace previous classifier's intent_ranking
            message.data['intent_ranking'] = predicted_intents

            # no more work
            return None

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]

        model_data = os.path.join(model_dir, PERSIST_MODEL_FILE)
        with open(model_data, 'wb') as fd:
            pickle.dump(self._intent_data, fd)

        return {
            "intent_classifier_keyword": PERSIST_MODEL_FILE,
        }

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        # split intent to different text
        for e in training_data.intent_examples:
            text = e.text
            intent = e.data['intent']

            question_mark = config.get("trailing_punctuation_marks", "")

            augmented_text = self._augment_text(text, question_mark)

            for m in augmented_text:
                self._intent_data[m].append(
                    {"name": intent, "confidence": 1.0}
                )

        # re-calculate the score
        for intents in self._intent_data.values():
            mean_score = 1 / len(intents)
            for i in intents:
                i['confidence'] = mean_score

    @staticmethod
    def _augment_text(text, question_mark):
        # with leading and trailing whitespace removed
        text = text.strip()

        cleaned_message = text.rstrip(question_mark)
        patched_message = ["".join([text, i]) for i in question_mark]

        augmented_msg = set([cleaned_message] + patched_message)
        return augmented_msg

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> Component
        """Load this component from file.

        After a component got trained, it will be persisted by calling `persist`. When the pipeline gets loaded again,
         this component needs to be able to restore itself. Components can rely on any context attributes that are
         created by `pipeline_init` calls to components previous to this one."""

        model_data = os.path.join(model_dir, PERSIST_MODEL_FILE)

        with open(model_data, 'rb') as fd:
            intent_data = pickle.load(fd)

        new_instance = cls()
        new_instance.set_persist_data(intent_data)

        return cached_component if cached_component else new_instance
