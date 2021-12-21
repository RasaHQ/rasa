import logging
from typing import Any, Text, Dict, List, Type

from sklearn.feature_extraction.text import TfidfVectorizer
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
from joblib import dump, load
from rasa.shared.nlu.constants import (
    TEXT,
    TEXT_TOKENS,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class TfIdfFeaturizer(SparseFeaturizer, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            **SparseFeaturizer.get_default_config(),
            "analyzer": "word",
            "min_ngram": 1,
            "max_ngram": 1,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """Constructs a new tf/idf vectorizer using the sklearn framework."""
        super().__init__(name, config)
        # Initialize the tfidf sklearn component
        self.tfm = TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=(config["min_ngram"], config["max_ngram"]),
        )

        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the component from training data."""
        texts = [e.get(TEXT) for e in training_data.training_examples if e.get(TEXT)]
        self.tfm.fit(texts)
        self.persist()
        return self._resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, execution_context.node_name, model_storage, resource)

    def _set_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Sets the features on a single message. Utility method."""
        tokens = message.get(TEXT_TOKENS)

        # If the message doesn't have tokens, we can't create features.
        if not tokens:
            return None

        # Make distinction between sentence and sequence features
        text_vector = self.tfm.transform([message.get(TEXT)])
        word_vectors = self.tfm.transform([t.text for t in tokens])

        final_sequence_features = Features(
            word_vectors,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming message and compute and set features."""
        for message in messages:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_features(message, attribute)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place."""
        self.process(training_data.training_examples)
        return training_data

    def persist(self) -> None:
        """
        Persist this model into the passed directory.

        Returns the metadata necessary to load the model again. In this case; `None`.
        """
        with self._model_storage.write_to(self._resource) as model_dir:
            dump(self.tfm, model_dir / "tfidfvectorizer.joblib")

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Loads trained component from disk."""
        try:
            with model_storage.read_from(resource) as model_dir:
                tfidfvectorizer = load(model_dir / "tfidfvectorizer.joblib")
                component = cls(
                    config, execution_context.node_name, model_storage, resource
                )
                component.tfm = tfidfvectorizer
        except (ValueError, FileNotFoundError):
            logger.debug(
                f"Couldn't load metadata for component '{cls.__name__}' as the persisted "
                f"model data couldn't be loaded."
            )
        return component

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
