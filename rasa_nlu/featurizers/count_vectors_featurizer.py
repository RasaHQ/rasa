from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import typing
import os
import io
from future.utils import PY3
from typing import Any, Dict, List, Optional, Text

from rasa_nlu import utils
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


class CountVectorsFeaturizer(Featurizer):
    """Bag of words featurizer"""

    name = "intent_featurizer_count_vectors"

    provides = ["text_features"]

    requires = []

    defaults = {
        # min number of word occurancies in the document to add to vocabulary
        "min_df": 1,

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        "max_df": 1.0,

        # set ngram range
        "min_ngram": 1,
        "max_ngram": 1
    }

    def __init__(self, component_config=None):
        # type: (RasaNLUModelConfig) -> None
        """Construct a new count vectorizer using the sklearn framework."""

        super(CountVectorsFeaturizer, self).__init__(component_config)

        # default parameters
        # min number of word occurancies in the document to add to vocabulary
        self.min_df = self.component_config['min_df']

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = self.component_config['max_df']

        # set ngram range
        self.min_ngram = self.component_config['min_ngram']
        self.max_ngram = self.component_config['max_ngram']

        # declare class instance for CountVect
        self.vect = None

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn"]

    def train(self, training_data, cfg, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """Take parameters from config and
            construct a new count vectorizer using the sklearn framework."""
        from sklearn.feature_extraction.text import CountVectorizer

        self.vect = CountVectorizer(ngram_range=(self.min_ngram, self.max_ngram),
                                    max_df=self.max_df,
                                    min_df=self.min_df)

        lem_exs = [self._lemmatize(example)
                   for example in training_data.intent_examples]

        X = self.vect.fit_transform(lem_exs).toarray()

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set("text_features", X[i])

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        bag = self.vect.transform([self._lemmatize(message)]).toarray()
        message.set("text_features", bag)

    @staticmethod
    def _lemmatize(message):
        if message.get("spacy_doc"):
            return ' '.join([t.lemma_ for t in message.get("spacy_doc")])
        else:
            return message.text

    @classmethod
    def load(cls, model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> CountVectorsFeaturizer

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("featurizer_file"):
            file_name = meta.get("featurizer_file")
            featurizer_file = os.path.join(model_dir, file_name)
            return utils.pycloud_unpickle(featurizer_file)
        else:
            raise Exception("Failed to load featurizer. Path {} "
                            "doesn't exist".format(os.path.abspath(model_dir)))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        utils.pycloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}
