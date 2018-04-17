from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import typing
import os
import io
import re
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
    """Bag of words featurizer

    Creates bag-of-words representation of intent features
    using sklearn's `CountVectorizer`.
    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature."""

    name = "intent_featurizer_count_vectors"

    provides = ["text_features"]

    requires = []

    defaults = {
        # the parameters are taken from
        # sklearn's CountVectorizer

        # regular expression for tokens
        "token_pattern": r'(?u)\b\w\w+\b',

        # remove accents during the preprocessing step
        "strip_accents": None,  # {'ascii', 'unicode', None}

        # list of stop words
        "stop_words": None,  # string {'english'}, list, or None (default)

        # min document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "min_df": 1,  # float in range [0.0, 1.0] or int

        # max document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "max_df": 1.0,  # float in range [0.0, 1.0] or int

        # set range of ngrams to be extracted
        "min_ngram": 1,
        "max_ngram": 1,

        # limit vocabulary size
        "max_features": None
    }

    def __init__(self, component_config=None):
        """Construct a new count vectorizer using the sklearn framework."""

        super(CountVectorsFeaturizer, self).__init__(component_config)

        # regular expression for tokens
        self.token_pattern = self.component_config['token_pattern']

        # remove accents during the preprocessing step
        self.strip_accents = self.component_config['strip_accents']

        # list of stop words
        self.stop_words = self.component_config['stop_words']

        # min number of word occurancies in the document to add to vocabulary
        self.min_df = self.component_config['min_df']

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = self.component_config['max_df']

        # set ngram range
        self.min_ngram = self.component_config['min_ngram']
        self.max_ngram = self.component_config['max_ngram']

        # limit vocabulary size
        self.max_features = self.component_config['max_features']

        # declare class instance for CountVect
        self.vect = None

        # preprocessor
        self.preprocessor = lambda s: re.sub(r'\b[0-9]+\b', 'NUMBER', s)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn"]

    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """Take parameters from config and
            construct a new count vectorizer using the sklearn framework."""
        from sklearn.feature_extraction.text import CountVectorizer

        # use even single character word as a token
        self.vect = CountVectorizer(token_pattern=self.token_pattern,
                                    strip_accents=self.strip_accents,
                                    stop_words=self.stop_words,
                                    ngram_range=(self.min_ngram,
                                                 self.max_ngram),
                                    max_df=self.max_df,
                                    min_df=self.min_df,
                                    max_features=self.max_features,
                                    preprocessor=self.preprocessor)

        lem_exs = [self._lemmatize(example)
                   for example in training_data.intent_examples]

        try:
            X = self.vect.fit_transform(lem_exs).toarray()
        except ValueError:
            self.vect = None
            return

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set("text_features", X[i])

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        if self.vect is None:
            logger.error("There is no trained CountVectorizer: "
                         "component is either not trained or "
                         "didn't receive enough training data")
        else:
            bag = self.vect.transform([self._lemmatize(message)]).toarray()
            message.set("text_features", bag)

    @staticmethod
    def _lemmatize(message):
        if message.get("spacy_doc"):
            return ' '.join([t.lemma_ for t in message.get("spacy_doc")])
        else:
            return message.text

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> CountVectorsFeaturizer

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("featurizer_file"):
            file_name = meta.get("featurizer_file")
            featurizer_file = os.path.join(model_dir, file_name)
            return utils.pycloud_unpickle(featurizer_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return CountVectorsFeaturizer(meta)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        utils.pycloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}
