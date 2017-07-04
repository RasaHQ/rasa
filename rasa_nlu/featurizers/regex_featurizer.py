from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
import re
import warnings
import io
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from future.utils import PY3
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    import numpy as np
    from rasa_nlu.model import Metadata


class RegexFeaturizer(Component):
    name = "intent_featurizer_regex"

    provides = ["text_features"]

    requires = ["text_features", "spacy_doc"]

    def __init__(self, regex_dict):
        self.regex_dict = regex_dict

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy", "numpy", "sklearn", "cloudpickle"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        for example in training_data.training_examples:
            updated = self._text_features_with_regex(example)
            example.set("text_features", updated)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated = self._text_features_with_regex(message)
        message.set("text_features", updated)

    def _text_features_with_regex(self, message):
        import numpy as np

        if self.regex_dict is not None:
            extras = self._regexes_match_sentence(message)
            return np.hstack((message.get("text_features"), extras))
        else:
            return message.get("text_features")

    def _regexes_match_sentence(self, example):
        """Given a sentence, returns a vector of {1,0} values indicating which regexes match"""

        import numpy as np
        found = [re.search(exp, example) is not None for exp in sorted(self.regex_dict.keys())]
        return np.array(found).astype('float')

    @classmethod
    def load(cls, model_dir, model_metadata, cached_component, **kwargs):
        # type: (Text, Metadata, Optional[RegexFeaturizer], **Any) -> RegexFeaturizer
        import cloudpickle

        if model_dir and model_metadata.get("regex_featurizer"):
            file = os.path.join(model_dir, model_metadata.get("regex_featurizer"))
            with io.open(file, 'rb') as f:   # pramga: no cover
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return RegexFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        import cloudpickle

        file = os.path.join(model_dir, "regex_featurizer.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "regex_featurizer": "regex_featurizer.pkl"
        }

