from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import typing
import os
import io
from future.utils import PY3

from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    import sklearn

logger = logging.getLogger(__name__)


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
        # type: () -> None
        """Construct a new count vectorizer using the sklearn framework."""

        super(CountVectorsFeaturizer, self).__init__(component_config)

        # default parameters
        # min number of word occurancies in the document to add to vocabulary
        self.min_df = 1

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = 1.0

        # set ngram range
        self.min_ngram = 1
        self.max_ngram = 1

        # declare class instance for CountVect
        self.vect = None

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn", "cloudpickle"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None
        """Take parameters from config and
            construct a new count vectorizer using the sklearn framework."""
        from sklearn.feature_extraction.text import CountVectorizer

        # overwrite parameters from config if they are there
        params_dict = config.get("intent_featurizer_count_vectors", {})

        self.min_df = params_dict.get("min_df", self.min_df)
        self.max_df = params_dict.get("max_df", self.max_df)
        self.min_ngram = params_dict.get("min_ngram", self.min_ngram)
        self.max_ngram = params_dict.get("max_ngram", self.max_ngram)

        self.vect = CountVectorizer(ngram_range=(self.min_ngram, self.max_ngram),
                                    max_df=self.max_df,
                                    min_df=self.min_df)

        lem_exs = [self._lemmetize(example)
                   for example in training_data.intent_examples]

        X = self.vect.fit_transform(lem_exs).toarray()

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set("text_features", X[i])

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        bag = self.vect.transform([self._lemmetize(message)]).toarray()
        message.set("text_features", bag)

    def _lemmetize(self, message):
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
        import cloudpickle

        if model_dir and model_metadata.get("intent_featurizer_count_vectors"):
            featurizer_file = os.path.join(model_dir,
                                           model_metadata.get("intent_featurizer_count_vectors"))
            with io.open(featurizer_file, 'rb') as f:   # pramga: no cover
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return CountVectorsFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        import cloudpickle

        featurizer_file = os.path.join(model_dir, "intent_featurizer_count_vectors.pkl")
        with io.open(featurizer_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
                "intent_featurizer_count_vectors": "intent_featurizer_count_vectors.pkl"
                }
